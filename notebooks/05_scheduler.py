# %% [markdown]
# # 第 05 章 · 调度器：静态批处理 vs 连续批处理
#
# > 同一张 GPU，换一个调度策略，吞吐能翻倍、尾延迟能降一个数量级。
# > 这一章我们把「GPU 一步」抽象成成本模型，亲手写出两种调度器，并用它们跑出趋势。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 75 分钟 |
# | 运行环境 | 🏠 Mac（CPU/MPS）· 无需 GPU（调度器是模拟器） |
# | 前置章节 | 02（KV cache 与 prefill/decode 两阶段）· 03（分页注意力） |
# | 本章接口 | 定义 `Request`，供第 06 章复用 |
# | 交付物 | 两种调度器 + 甘特图 + 吞吐/延迟曲线 + 参数扫描表 |
#
# **学完你能回答：**
# 1. 连续批处理（continuous batching）到底比静态批处理多做了什么？
# 2. 为什么 `max_num_seqs` 和 `max_num_batched_tokens` 一个管「同时几个请求」、一个管「一步算多少 token」？
# 3. 静态批处理在真实对话负载下吞吐低，padding 和「等最长请求」各占多少责任？

# %% [markdown]
# ## 05.1 问题与直觉
#
# 先看一张图。上半图是**静态批处理**：攒一批请求，整批一起 prefill，然后一起 decode，直到这批里**最长**的那个结束，才接下一批。
# 下半图是**连续批处理**（iteration-level scheduling，Orca 论文的核心思想）：每一个 decode step 结束，都检查有没有空出来的 slot，有空位就让新请求进来。
#
# ![](../assets/figures/ch05_gantt.svg)
#
# 图里有三处浪费，正是本章要量化的东西：
#
# - **padding**：一批请求的 prompt 长度不同，朴素实现要 pad 到最长那条再一起算，短 prompt 的算力白烧。
# - **等最长请求**：8 个 token 就结束的请求，它的 slot 要空等 512 个 token 的邻居，GPU 仍在为邻居算，但短请求的 slot 是死的。
# - **批间空窗**：整批没结束前，新到的请求只能排队；对话负载的输出长度是重尾的，这个空窗经常占掉一半时间。
#
# 反过来，连续批处理也不是免费的：新请求的 prefill 会插进 decode 的节奏里，让尾部 ITL 变差。
# 第 06 章的 chunked prefill 就是来修这个问题的——本章先把问题测出来。

# %% [markdown]
# 为什么不做真机实验？因为本章要回答的是**调度策略**的因果，而不是某张卡的绝对性能。
# 我们从 01/02 章知道：prefill 是算力受限（耗时 ∝ token 数），decode 是带宽受限（每步读一遍权重，但权重开销被 batch 摊薄，单步耗时增长远慢于 batch 大小）。
# 于是把「GPU 一步」写成两行公式，就能跑出可信趋势，换任何模型只要改三个常数。
#
# 成本模型参数（量级参照 A100 上的 7B 模型，单位为毫秒）：
#
# | 参数 | 含义 | 默认值 |
# | --- | --- | --- |
# | `base_ms` | 一步的固定成本 ≈ 读一遍权重 | 6.0 |
# | `prefill_per_token_ms` | 每 token 的 prefill 计算 | 0.08 |
# | `decode_per_seq_ms` | 每序列的 decode 附加成本 | 0.12 |
# | `prefill_launch_ms` | 多一次 prefill 的 kernel 发射 | 1.0 |
#
# 三个指标先定义清楚，后面所有表都用同一口径：
#
# - **TTFT**：请求到达到第一个 token 产出（prefill 完的时刻）。
# - **ITL / TPOT**：相邻两个输出 token 的间隔；我们在全量 token 上取 p95。
# - **吞吐**：整个 trace 里产出的 token 数 ÷ makespan；**利用率**：活跃 slot 数 ÷ `max_num_seqs`，按时间加权。

# %% [markdown]
# ## 05.2 从零实现
#
# 先定接口。`Request` 的字段与 vLLM 的 `Request` 对齐：`prompt_len`、`max_new_tokens`、`arrival`。
# 运行期我们只维护两个计数器：`num_computed`（KV cache 里已有的 context 长度）和 `num_generated`（已产出的 token 数）。

# %%
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch

sys.path.insert(0, str(Path.cwd().parent))

from aiinfra.bench import bench, pick_device, sync
from aiinfra.plotting import COLORS, save, setup_style
from aiinfra.tinygpt import GPTConfig, KVCache, TinyGPT

setup_style()
device = pick_device()
torch.manual_seed(0)
np.random.seed(0)
print("device =", device)

# %% [markdown]
# ### 数据结构：`Request` 与成本模型

# %%
@dataclass
class Request:
    """一个推理请求。字段命名与 vLLM 的 Request 对齐。"""

    id: int
    prompt_len: int
    max_new_tokens: int
    arrival: float  # 到达时刻（秒）
    num_computed: int = 0  # KV cache 里已有的 context token 数
    num_generated: int = 0  # 已经产出的 token 数
    t_first: float = -1.0
    t_done: float = -1.0
    token_times: list[float] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.num_generated >= self.max_new_tokens

    def target_ctx(self) -> int:
        """下一步 decode 之前，KV 里必须有的 context 长度。"""
        return self.prompt_len + max(self.num_generated - 1, 0)

    def needs_prefill(self) -> bool:
        return self.num_computed < self.target_ctx()

    def prefill_need(self) -> int:
        return self.target_ctx() - self.num_computed


# %%
@dataclass(frozen=True)
class CostModel:
    """把「GPU 一步」抽象成可配置耗时（毫秒）。"""

    base_ms: float = 6.0
    prefill_per_token_ms: float = 0.08
    decode_per_seq_ms: float = 0.12
    prefill_launch_ms: float = 1.0

    def step_ms(self, prefill_tokens: int, decode_seqs: int) -> float:
        t = self.base_ms + self.prefill_per_token_ms * prefill_tokens + self.decode_per_seq_ms * decode_seqs
        if prefill_tokens > 0:
            t += self.prefill_launch_ms
        return t


cost = CostModel()
# 两条关键假设，直接写成断言
assert cost.step_ms(0, 32) < 2 * cost.step_ms(0, 1)  # decode 对 batch 亚线性
marginal = cost.step_ms(2048, 0) - cost.step_ms(1024, 0)
assert abs(marginal - 1024 * cost.prefill_per_token_ms) < 1e-9  # prefill 的边际成本对 token 线性
print("decode b=1 / b=16 / b=32:", [round(cost.step_ms(0, b), 2) for b in (1, 16, 32)], "ms")

# %% [markdown]
# ### 一次模拟的结果容器
#
# `Run` 里同时保留**给表格的指标**和**给甘特图的轨迹**：`segments` 是 `(slot, t0, t1, 类型)`，
# 类型只有 `prefill` / `decode` 两种；`steps` 用来算利用率；`batch_info` 用来拆 padding 与 slot 浪费。

# %%
@dataclass
class Run:
    metrics: dict
    reqs: list[Request]
    segments: list[tuple[int, float, float, str]]
    steps: list[tuple[float, float, int]]
    makespan: float
    batch_info: list[tuple[int, int, int, int]]  # (padded, useful, slot_capacity, slot_useful)


def summarize(name, reqs, segments, steps, makespan, max_num_seqs, batch_info) -> Run:
    busy = sum(t1 - t0 for t0, t1, _ in steps)
    ttft = np.array([r.t_first - r.arrival for r in reqs])
    e2e = np.array([r.t_done - r.arrival for r in reqs])
    itl = np.concatenate([np.diff(r.token_times) for r in reqs])
    metrics = {
        "策略": name,
        "吞吐(tokens/s)": sum(r.max_new_tokens for r in reqs) / makespan,
        "吞吐(req/s)": len(reqs) / makespan,
        "TTFT p50(ms)": np.quantile(ttft, 0.5) * 1000,
        "TTFT p95(ms)": np.quantile(ttft, 0.95) * 1000,
        "E2E 均值(ms)": np.mean(e2e) * 1000,
        "E2E p95(ms)": np.quantile(e2e, 0.95) * 1000,
        "ITL p95(ms)": np.quantile(itl, 0.95) * 1000,
        "GPU空闲率": 1 - busy / makespan,
        "平均batch利用率": sum(n * (t1 - t0) for t0, t1, n in steps) / busy / max_num_seqs,
    }
    metrics["prefill填充浪费"] = 1 - sum(x[1] for x in batch_info) / sum(x[0] for x in batch_info) if batch_info else 0.0
    metrics["decode槽位浪费"] = 1 - sum(x[3] for x in batch_info) / sum(x[2] for x in batch_info) if batch_info else 0.0
    return Run(metrics, reqs, segments, steps, makespan, batch_info)

# %% [markdown]
# ### 策略 A：静态批处理
#
# 逻辑只有三段：GPU 空闲时从等待队列取一批 → 整批 prefill（耗时按 pad 到最长 prompt 算）→ 整批 decode 到最长结束。
# 注意 decode 时**已完成的请求不再占算力，但它的 slot 也不能被新请求复用**——这就是静态批处理的死穴。

# %%
def run_static(reqs, cost, max_num_seqs=16, pad=True):
    """静态批处理：整批 prefill → 一起 decode 到最长结束。"""
    pending = sorted(reqs, key=lambda r: r.arrival)
    waiting, running = [], []
    now, segments, steps, batch_info = 0.0, [], [], []

    while pending or waiting or running:
        while pending and pending[0].arrival <= now:
            waiting.append(pending.pop(0))
        if not running:
            if not waiting and pending:
                now = pending[0].arrival
                continue
            if not waiting:
                break
            batch, waiting = waiting[:max_num_seqs], waiting[max_num_seqs:]
            padded, useful = max(r.prompt_len for r in batch) * len(batch), sum(r.prompt_len for r in batch)
            batch_info.append((padded, useful, max(r.max_new_tokens for r in batch) * len(batch),
                               sum(r.max_new_tokens for r in batch)))
            dur = cost.step_ms(padded if pad else useful, 0) / 1000
            for lane, r in enumerate(batch):
                r.num_computed, r.num_generated = r.prompt_len, 1
                r.t_first = now + dur; r.token_times.append(r.t_first)
                segments.append((lane, now, now + dur, "prefill"))
            steps.append((now, now + dur, len(batch)))
            now += dur
            running = [(r, lane) for lane, r in enumerate(batch) if not r.finished]
            finished = [r for r in batch if r.finished]
            for r in finished: r.t_done = now
            continue
        dur = cost.step_ms(0, len(running)) / 1000
        for r, lane in running:
            r.num_computed += 1; r.num_generated += 1
            r.token_times.append(now + dur)
            segments.append((lane, now, now + dur, "decode"))
        steps.append((now, now + dur, len(running)))
        now += dur
        finished = [r for r, _ in running if r.finished]
        for r in finished: r.t_done = now
        running = [(r, lane) for r, lane in running if not r.finished]
    return summarize("静态批处理", reqs, segments, steps, now, max_num_seqs, batch_info)


# %% [markdown]
# ### 策略 B：连续批处理
#
# 每个 iteration 做三件事：
#
# 1. 把已到达的请求放进 `waiting`；
# 2. **先给正在 decode 的请求各排 1 个 token**，剩余 token 预算再用来接纳新请求的 prefill；
# 3. 一步执行完，完成的请求释放 slot，下一个 iteration 立刻有新请求补进来。
#
# 这里有一个重要限制：本章还没有 chunked prefill，所以一条 prompt 必须**整体塞进当前一步的 token 预算**，
# 否则只能等。这正是第 06 章要拆掉的天花板。
#
# %%
def run_continuous(reqs, cost, max_num_seqs=16, max_num_batched_tokens=4096):
    """连续批处理：每个 iteration 接纳新请求，decode 与 prefill 混批。"""
    pending = sorted(reqs, key=lambda r: r.arrival)
    waiting, running = [], []
    now, segments, steps = 0.0, [], []
    lane_free, lane_of = list(range(max_num_seqs)), {}

    while pending or waiting or running:
        while pending and pending[0].arrival <= now:
            waiting.append(pending.pop(0))
        admits, budget = [], max_num_batched_tokens - len(running)
        while waiting and len(running) + len(admits) < max_num_seqs:
            if waiting[0].prompt_len > budget:
                break  # 没有 chunked prefill：整条 prompt 塞不进本步就只能等
            r = waiting.pop(0); admits.append(r); budget -= r.prompt_len
        if not running and not admits:
            if not pending:
                break
            now = pending[0].arrival
            continue
        dur = cost.step_ms(sum(r.prompt_len for r in admits), len(running)) / 1000
        end = now + dur
        for r in admits:
            lane_of[r.id] = lane_free.pop(0)
            r.num_computed, r.num_generated = r.prompt_len, 1
            r.t_first = end
            r.token_times.append(end)
            segments.append((lane_of[r.id], now, end, "prefill"))
        for r in running:
            r.num_computed += 1; r.num_generated += 1
            r.token_times.append(end)
            segments.append((lane_of[r.id], now, end, "decode"))
        steps.append((now, end, len(running) + len(admits))); now = end
        done = [r for r in running + admits if r.finished]
        running = [r for r in running + admits if not r.finished]
        for r in done:
            r.t_done = now; lane_free.append(lane_of.pop(r.id))
        lane_free.sort()
    return summarize("连续批处理", reqs, segments, steps, now, max_num_seqs, [])

# %% [markdown]
# ### 固定到达序列
#
# 两种策略必须吃**同一条**到达序列，否则不可比。这里用固定 seed 的 Poisson 到达，prompt 长度 128–2048，
# 输出长度是重尾分布（中位数 32，长尾到 1024）——真实对话负载的样子。

# %%
def chat_workload(n=64, rate=16.0, seed=0):
    rng = np.random.default_rng(seed)
    t, reqs = 0.0, []
    for i in range(n):
        t += rng.exponential(1.0 / rate)
        reqs.append(
            Request(
                id=i,
                prompt_len=int(rng.choice([128, 256, 512, 1024, 2048], p=[0.25, 0.3, 0.25, 0.15, 0.05])),
                max_new_tokens=int(rng.choice([8, 16, 32, 64, 128, 256, 512, 1024],
                                              p=[0.20, 0.20, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03])),
                arrival=float(t),
            )
        )
    return reqs


workload = chat_workload()
print("请求数 =", len(workload), "| 最后到达 =", round(workload[-1].arrival, 2), "s")
pd.DataFrame([{"prompt_len": r.prompt_len, "max_new_tokens": r.max_new_tokens} for r in workload[:8]])

# %% [markdown]
# ### 断言即测试
#
# 用 4 条同构请求手算静态批处理的 makespan，和模拟器对账；再验证连续批处理不会比静态批处理慢，
# 且每个请求恰好产出 `max_new_tokens` 个 token。

# %%
def fresh(reqs):
    """复制一批请求：保留到达序列，清空运行期状态（模拟器会就地修改 Request）。"""
    return [Request(r.id, r.prompt_len, r.max_new_tokens, r.arrival) for r in reqs]


same = [Request(id=i, prompt_len=64, max_new_tokens=8, arrival=0.0) for i in range(4)]
expected = (cost.step_ms(4 * 64, 0) + 7 * cost.step_ms(0, 4)) / 1000
assert abs(run_static(fresh(same), cost, max_num_seqs=4).makespan - expected) < 1e-9
# 同到达、同长度时两种策略应该完全等价
assert abs(run_continuous(fresh(same), cost, max_num_seqs=4).makespan - expected) < 1e-9

# 关键差异：在静态批处理跑到一半时到达的短请求，必须等整批结束
pair = [Request(id=0, prompt_len=64, max_new_tokens=64, arrival=0.0),
        Request(id=1, prompt_len=64, max_new_tokens=4, arrival=0.05)]
a, b = run_static(fresh(pair), cost, max_num_seqs=2), run_continuous(fresh(pair), cost, max_num_seqs=2)
assert all(len(r.token_times) == r.max_new_tokens for r in b.reqs)
assert b.metrics["TTFT p50(ms)"] * 2 < a.metrics["TTFT p50(ms)"]
print(f"中途到达的短请求: 静态 TTFT p50 {a.metrics['TTFT p50(ms)']:.0f}ms vs 连续 {b.metrics['TTFT p50(ms)']:.0f}ms")

# %% [markdown]
# ### 标定：先验证「decode 对 batch 亚线性」这条假设
#
# 成本模型最关键的一条是「decode 一步的耗时几乎不随 batch 增大」。用本机的 TinyGPT 实测一下：
# 固定权重、固定上下文，batch 从 1 拉到 32，测每步耗时。

# %%
cfg = GPTConfig()
tiny = TinyGPT(cfg).to(device).eval()


@torch.no_grad()
def decode_loop(batch: int, steps: int = 20, ctx: int = 32):
    cache = KVCache(cfg, batch_size=batch, max_seq_len=ctx + steps + 1, device=device)
    tiny(torch.randint(0, cfg.vocab_size, (batch, ctx), device=device), start_pos=0, kv_cache=cache)
    tok = torch.randint(0, cfg.vocab_size, (batch, 1), device=device)
    pos = cache.seq_len

    def fn():
        for _ in range(steps):
            tiny(tok, start_pos=pos, kv_cache=cache)
    return fn


rows = []
for b in [1, 2, 4, 8, 16, 32]:
    r = bench(decode_loop(b), warmup=2, reps=5, name=f"decode_b{b}", device=device)
    rows.append({"batch": b, "实测单步(ms)": r.median_ms / 20, "模型单步(ms)": cost.step_ms(0, b)})
df_cal = pd.DataFrame(rows)
df_cal["实测/模型"] = df_cal["实测单步(ms)"] / df_cal["模型单步(ms)"]
df_cal.round(3)

# %% [markdown]
# **怎么读**：batch 从 1 到 32，TinyGPT 的每步耗时只涨了约 5%（中间点有 ±25% 的噪声），而线性增长会是 32 倍。
# 绝对值当然对不上（TinyGPT 只有 2.8M 参数，本机 MPS 也不是 A100），但对不上的部分是**一个固定常数**，
# 不影响「batch↑ → 单步耗时几乎不变 → 吞吐↑↑」这个结论。趋势可信，绝对值只用于算相对倍数。

# %%
fig, ax = plt.subplots(figsize=(7.2, 4.2))
norm = df_cal["实测单步(ms)"] / df_cal["实测单步(ms)"].iloc[0]
model_norm = df_cal["模型单步(ms)"] / df_cal["模型单步(ms)"].iloc[0]
ax.plot(df_cal["batch"], df_cal["batch"], ls=":", color="#EF4444", label="Linear reference")
ax.plot(df_cal["batch"], model_norm, "s--", color="#F59E0B", label="Cost model (6 + 0.12/seq ms)")
ax.plot(df_cal["batch"], norm, "o-", color="#2563EB", label="TinyGPT measured (MPS)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Decode batch size (sequences)")
ax.set_ylabel("Step time (normalized to batch=1)")
ax.set_title("Decode step time is sub-linear in batch size")
ax.legend()
save(fig, "ch05_decode_scaling")

# %% [markdown]
# ## 05.3 实测
#
# 跑同一条到达序列（64 个请求，λ=16 req/s，`max_num_seqs=16`），对比两种策略。
# 每行都是同一份 `chat_workload`，模拟器里没有随机性，结果可复现。

# %%
static = run_static(chat_workload(), cost, max_num_seqs=16)
continuous = run_continuous(chat_workload(), cost, max_num_seqs=16)  # 各自新建一份，互不污染
table = pd.DataFrame([static.metrics, continuous.metrics])
ratio = continuous.metrics["吞吐(tokens/s)"] / static.metrics["吞吐(tokens/s)"]
print(f"连续/静态 吞吐倍数 = {ratio:.2f}x")
table.round(2)

# %% [markdown]
# 读表，先看四个数字：
#
# - **吞吐 2.55x**：同一块「GPU」、同一条到达序列，只是换调度策略。
# - **TTFT p95 从 21.3s 降到 1.2s（约 18x）**：静态批处理里，请求到达后要等当前整批跑完才有机会进来。
# - **E2E p95 从 22.6s 降到 6.2s（约 3.6x）**：尾部请求不用排长队。
# - **batch 利用率 0.37 vs 0.61**：GPU 并不是「闲着」（空闲率两行都是 0），而是**忙而无效**——静态批处理里
#   大量 slot 在等最长请求，算力被 padding 和空转吃掉。
#
# 再看两个浪费指标：静态批处理 **57% 的 prefill 算力花在 padding 上**，**82% 的 decode slot-time 是空转**。
# 但注意：padding 可以靠 03 章的分页注意力消掉，而「等最长请求」消不掉——后面参数实验会验证这一点。

# %%
def merge_segments(segments):
    """把同一 slot、同类型、首尾相接的段合并，甘特图才画得快。"""
    out = []
    for lane, t0, t1, kind in sorted(segments):
        if out and out[-1][0] == lane and out[-1][3] == kind and abs(out[-1][2] - t0) < 1e-9:
            out[-1] = (lane, out[-1][1], t1, kind)
        else:
            out.append((lane, t0, t1, kind))
    return out


def draw_gantt(ax, run, title, n_slots=16, color_of=None):
    color_of = color_of or {"prefill": COLORS["primary"], "decode": COLORS["success"]}
    for lane in range(n_slots):  # 灰底 = 该 slot 空闲
        ax.axhspan(lane - 0.5, lane + 0.5, color=COLORS["bg"], zorder=0, lw=0)
    for lane, t0, t1, kind in merge_segments(run.segments):
        ax.add_patch(plt.Rectangle((t0, lane - 0.42), max(t1 - t0, 1e-4), 0.84,
                                   color=color_of[kind], lw=0, zorder=2))
    ax.set_xlim(0, max(run.makespan, 1e-3))
    ax.set_ylim(n_slots - 0.5, -0.5)
    ax.set_yticks([0, 4, 8, 12, 15])
    ax.set_ylabel("Slot index")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)


fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.9), sharex=True)
draw_gantt(axes[0], static, f"Static batching: slot util {static.metrics['平均batch利用率']:.2f}, many slots idling")
draw_gantt(axes[1], continuous, f"Continuous batching: freed slots refilled, slot util {continuous.metrics['平均batch利用率']:.2f}")
axes[1].set_xlabel("Time (s)")
handles = [Patch(color=COLORS["primary"], label="prefill"), Patch(color=COLORS["success"], label="decode"),
           Patch(color=COLORS["bg"], label="idle slot")]
fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.015), fontsize=9, frameon=False)
fig.suptitle("Scheduling trace on the same arrival sequence (x: time, y: slot)", y=1.00)
fig.tight_layout()
save(fig, "ch05_gantt")

# %% [markdown]
# 甘特图里最值得看的是**下半图**：slot 一旦空出来，下一个 iteration 就有新请求填进去，所以色块几乎不断档。
# 上半图里每一批的右边界都被最长的请求拖住，短请求结束后 slot 变成灰色，直到整批结束。
# 这就是「等最长请求」的可视化：GPU 一直在跑，但大部分 slot 在陪跑。

# %% [markdown]
# ### 扫描到达率：吞吐-延迟曲线
#
# 把 λ 从 2 拉到 64，每一步都重新生成到达序列（只改 rate，其余参数固定）。
# 横轴取实际达到的吞吐，纵轴取 E2E p95——越靠右下越好。

# %%
rows = []
for rate in [2, 4, 8, 16, 32, 64]:
    for fn in (run_static, run_continuous):
        m = fn(chat_workload(rate=rate), cost, max_num_seqs=16).metrics
        rows.append({"策略": m["策略"], "到达率(req/s)": rate, "吞吐(req/s)": m["吞吐(req/s)"],
                     "吞吐(tokens/s)": m["吞吐(tokens/s)"], "E2E p95(s)": m["E2E p95(ms)"] / 1000,
                     "batch利用率": m["平均batch利用率"]})
df_rate = pd.DataFrame(rows)
df_rate.round(2)

# %%
fig, ax = plt.subplots(figsize=(7.2, 4.2))
plot_df = df_rate.rename(columns={"吞吐(req/s)": "throughput", "E2E p95(s)": "p95", "到达率(req/s)": "arrival"})
styles = {"静态批处理": ("o", "#EF4444", "Static batching"), "连续批处理": ("s", "#10B981", "Continuous batching")}
for name, grp in plot_df.groupby("策略"):
    marker, color, label = styles[name]
    ax.plot(grp["throughput"], grp["p95"], marker=marker, color=color, label=label)
    for _, row in grp.iterrows():
        ax.annotate(f"λ={row['arrival']}", (row["throughput"], row["p95"]),
                    textcoords="offset points", xytext=(4, -11), fontsize=8, color="#6B7280")
ax.set_xlabel("Achieved throughput (req/s)")
ax.set_ylabel("E2E p95 latency (s)")
ax.set_title("Throughput-latency trade-off: continuous batching is lower-right")
ax.legend()
save(fig, "ch05_throughput_latency")

# %% [markdown]
# 读图：**静态批处理在 ~2.2 req/s 就撞墙**，λ 再大也只增加排队时间（点往上走、不往右走）。
# 连续批处理一路涨到 ~6 req/s，p95 仍只有静态的 1/3。
# 两条曲线的「拐点」就是 GPU 真正饱和的位置；拐点左侧是延迟受限，右侧是吞吐受限。
# 生产上调参的目标就是：让工作点尽量靠近拐点、但别越过 SLO 线。

# %% [markdown]
# ### 扫描 `max_num_seqs` 与 `max_num_batched_tokens`
#
# 这两个参数是 vLLM 调度器的两个「闸门」：
#
# - `max_num_seqs` 限制**一个 iteration 里最多有几条序列**（并发上限）；
# - `max_num_batched_tokens` 限制**一个 iteration 里最多算多少个 token**（prefill + decode 都算）。
#
# 前者小 → batch 不满、吞吐低；后者小 → 长 prompt 塞不进一步，调度器只能干等（没有 chunked prefill 时）。

# %%
rows = []
for seqs in [1, 2, 4, 8, 16, 32]:
    m = run_continuous(chat_workload(), cost, max_num_seqs=seqs).metrics
    rows.append({"max_num_seqs": seqs, "吞吐(tokens/s)": m["吞吐(tokens/s)"], "TTFT p95(ms)": m["TTFT p95(ms)"],
                 "E2E p95(ms)": m["E2E p95(ms)"], "batch利用率": m["平均batch利用率"]})
df_seqs = pd.DataFrame(rows)
df_seqs.round(2)

# %%
rows = []
for budget in [2048, 4096, 8192, 16384]:
    m = run_continuous(chat_workload(), cost, max_num_seqs=16, max_num_batched_tokens=budget).metrics
    rows.append({"max_num_batched_tokens": budget, "吞吐(tokens/s)": m["吞吐(tokens/s)"],
                 "TTFT p95(ms)": m["TTFT p95(ms)"], "E2E p95(ms)": m["E2E p95(ms)"],
                 "batch利用率": m["平均batch利用率"]})
df_budget = pd.DataFrame(rows)
df_budget.round(2)

# %% [markdown]
# 两张表各有一条「断崖」：
#
# - `max_num_seqs` 从 1 加到 16，吞吐从 154 涨到 792 tokens/s（5.1x），TTFT p95 从 51s 降到 1.2s。
#   继续加到 32 吞吐不再涨（prefill 成为瓶颈），利用率却掉到 0.34——加并发只是在摊薄 batch 的有效负载。
# - `max_num_batched_tokens=2048` 时吞吐掉到 319：负载里有 2048 token 的 prompt，而一步预算被 decode 占掉一部分，
#   这些请求永远凑不齐「一整条 prompt 进同一步」，只能等 GPU 跑空。这就是第 06 章 chunked prefill 要解的限制：
#   把一条 prompt 拆成多步算，预算再小也能前进。
#
# 记两个经验公式（后面章节会反复用）：
#
# - 吞吐 ≈ `min(max_num_seqs / 单请求每步成本 × 每步 token 数, 显存带宽 / 每 token 字节)`；
# - 不开 chunked prefill 时，`max_num_batched_tokens ≥ 最长 prompt + 并发 decode 数`，否则会出现「永远排不进」。
#
# 再回答本章开头的问题：**静态批处理为什么吞吐低**？
# 1. padding：一批 prompt 长度不同，短的要陪长的算——本实验里占掉 57% 的 prefill 算力；
# 2. 等最长请求：输出重尾时，一批的时长由最长输出决定，短输出的 slot 全程空转——占掉 82% 的 decode slot-time；
# 3. 批间空窗：整批不结束，新请求进不来，队列越压越长（TTFT p95 = 21s 就是这么来的）。
# 前两条是「一个 batch 内部」的浪费，第三条是「batch 之间」的浪费。连续批处理同时消灭了第 2、3 条；
# 第 1 条要靠分页注意力（03 章）——这也解释了为什么两个技术要一起用。

# %% [markdown]
# ## 05.4 对照工业实现（vLLM）
#
# 本课程锁定 **vLLM 0.28.x**（V1 engine）。我们这 120 行模拟器，在 vLLM 里对应这些真实位置：
#
# | 我们的实现 | vLLM 中的对应物 | 文件（0.28.x） |
# | --- | --- | --- |
# | `Request`（prompt_len/max_new_tokens/arrival） | `Request` / `EngineCoreRequest` | `vllm/v1/request.py` |
# | `run_continuous` 的主循环 | `Scheduler.schedule()` 每次返回一个 `SchedulerOutput` | `vllm/v1/core/sched/scheduler.py` |
# | `waiting` / `running` 两个队列 | `self.waiting` / `self.running` | 同上 |
# | `max_num_seqs` | `SchedulerConfig.max_num_seqs`（CLI `--max-num-seqs`） | `vllm/config/scheduler.py` |
# | `max_num_batched_tokens` | `SchedulerConfig.max_num_batched_tokens`（`--max-num-batched-tokens`） | 同上 |
# | 每步 1 token 的 decode | `_schedule_running`（V1 后期版本；早期版本内联在 `schedule()` 里） | `vllm/v1/core/sched/scheduler.py` |
# | 接纳新请求的 prefill | `_schedule_prefills` / `_schedule_new_requests` | 同上 |
# | slot 与 KV 分配 | `KVCacheManager.allocate_slots()` | `vllm/v1/core/kv_cache_manager.py` |
# | 我们的 `summarize` | `vllm bench serve` 的 TTFT/TPOT/吞吐统计 | `vllm/benchmarks/serve.py` |
#
# 差距在哪：
#
# 1. **KV 是分页的**（03 章）：vLLM 的 `KVCacheManager` 按 block 分配，所以同样的 `max_num_seqs`
#    能装下更多请求，也没有 padding；我们的模拟器只记「token 数」。
# 2. **V1 默认就是 chunked prefill**：`_schedule_running` 给 running 排 token 时，会把长 prompt 拆成
#    `max_num_batched_tokens - num_decode_tokens` 的 chunk，与 decode 同批。第 06 章手写这个逻辑。
# 3. **抢占**：显存不够时 vLLM 会把 running 队尾的请求 preempt（recompute 或 swap），
#    我们的模拟器在第 06 章补上。
# 4. **优先级 / FCFS / 多租户**：vLLM 有 `priority`、`--scheduling-policy`，本章只做 FCFS。
#
# 注：V1 在 0.8–0.10 期间重构过 scheduler 的方法拆分，类名与路径以课程锁定的 0.28.x 源码为准；
# 用 `grep -n "def schedule" vllm/v1/core/sched/scheduler.py` 打开文件对一遍最稳。
#
# 你在 vLLM 里能直接看到的证据：`--max-num-seqs` 调小后 TTFT 下降、吞吐下降；
# `--max-num-batched-tokens` 调到小于最长 prompt 时，日志会警告长请求排不进来。本章的两张表就是这两句话的手写版。

# %% [markdown]
# ## 05.5 练习与小结
#
# **练习**
#
# 1. ⭐（10 分钟）把 `chat_workload` 的输出分布改成「全部 64 token」的均匀分布，重跑对比表。
#    静态批处理的「decode 槽位浪费」应该接近 0，但吞吐提升倍数还剩多少？为什么省不掉？
# 2. ⭐（10 分钟）把 `CostModel.decode_per_seq_ms` 从 0.12 调到 0.5，重跑吞吐-延迟曲线。
#    连续批处理的优势变大还是变小？用「decode 是带宽受限」解释。
# 3. ⭐⭐（20 分钟）给 `run_continuous` 加一个「prefill 优先」的变体：一步里先排完 prefill 再排 decode，
#    而不是混批。对比 ITL p95 和吞吐，说明混批和分步各自的代价。
# 4. ⭐⭐（20 分钟）实现「按剩余输出长度排序」的静态批处理（SJF），看它能挽回多少吞吐。
#    提示：到达序列不变，但同一批里按 `max_new_tokens` 升序打包；想想为什么生产上不敢这么做（公平性/饥饿）。
# 5. ⭐⭐⭐（30 分钟）设计一个调度策略，同时优化吞吐和 p95 TTFT。评分要点：
#    (a) 有明确的准入条件（不是「先到先得」）；(b) 给出与 FCFS 的对比数字；(c) 说明在重尾负载下的饥饿风险；
#    (d) 说明实现需要动 vLLM 的哪个文件、哪个方法。
#
# **你现在应该能回答：**
#
# - 连续批处理比静态批处理多做了什么？——在**每个 iteration 的边界**重新组批，而不是每个 batch 的边界。
# - `max_num_seqs` 与 `max_num_batched_tokens` 分别限制什么？调小各自会先伤哪个指标？
# - 静态批处理吞吐低的三笔浪费：padding、等最长请求、批间空窗——各自对应哪个指标？
#
# **小结与下一章**
#
# 本章的连续批处理把吞吐拉起来，但它的 ITL p95（29ms）比静态批处理（7.6ms）更差：
# 因为一条 2048 token 的 prefill 一旦和 decode 同批，所有正在 decode 的请求都要多等约 170ms。
# 第 06 章用两种手段修这个问题：**chunked prefill**（把 prefill 切小、与 decode 交替）与**抢占**（KV 不够时让请求让位）。
#
# **延伸阅读**
#
# - Orca 论文（OSDI'22）：iteration-level scheduling 与 selective batching 的原始出处
# - vLLM V1 scheduler：`vllm/v1/core/sched/scheduler.py` 的 `schedule()` / `_schedule_running` / `_schedule_prefills`
# - vLLM 文档：`--max-num-seqs`、`--max-num-batched-tokens` 的调参建议
