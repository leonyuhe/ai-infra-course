# %% [markdown]
# # 第 06 章 · Chunked Prefill 与抢占
#
# > 第 05 章的连续批处理把吞吐拉起来了，却让 decode 被长 prefill 顶住；
# > 这一章把 prefill 切碎、把 KV 不够的请求请出去，修掉它的尾延迟。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 80 分钟 |
# | 运行环境 | 🏠 Mac（CPU/MPS）· 无需 GPU（调度器是模拟器） |
# | 前置章节 | 05（`Request` / 连续批处理调度器） |
# | 本章接口 | 复用 05 章的 `Request` 语义（prompt_len / max_new_tokens / arrival / num_computed / num_generated） |
# | 交付物 | chunked prefill 调度器 + 抢占（recompute / swap）+ 扫描表 + 时间轴图 |
#
# **学完你能回答：**
# 1. chunked prefill 为什么能降低 TPOT 的尾部？代价是什么？
# 2. 显存不够时，recompute 与 swap 各付什么代价？为什么 vLLM 默认 recompute？
# 3. `max_num_batched_tokens` 与 chunk 大小是什么关系？调小 chunk 为什么反而更慢？

# %% [markdown]
# ## 06.1 问题与直觉
#
# 第 05 章的实验留下一个尾巴：连续批处理的吞吐是静态批处理的 2.55 倍，但它的 **ITL p95 = 29ms**，
# 而静态批处理只有 7.6ms。原因是：一条 2048 token 的 prompt 如果和 decode 混在同一步，这一步要算 170ms，
# 所有正在 decode 的请求都要等这 170ms 才能吐下一个 token。
#
# ![](../assets/figures/ch06_chunked_timeline.svg)
#
# 上半图：**关闭 chunked prefill**。长 prompt 的 prefill 是一个 170ms 的大蓝块，期间所有 decode 都停住，
# 时间轴上就是一条条红色「断点」。下半图：**开启 chunked prefill**。同一条长 prompt 被切成 4 个 512 token 的 chunk，
# 每个 iteration 只算一个 chunk，剩下的时间照常 decode；尾部的 170ms 断点变成 50ms 的小台阶。
#
# 这就是 vLLM `--enable-chunked-prefill` 的核心：**用更频繁的小 prefill，换 decode 的平稳输出**。
# 代价有两个：每一步都要多付一次 kernel 发射/权重读取的固定成本（chunk 越小、开销占比越高），
# 以及长请求自己的 TTFT 会略微变长。

# %% [markdown]
# 第二个问题来自显存。KV cache 是按 token 线性增长的：一个 7B 模型（32 层、8 个 KV head、head_dim=128、fp16）每个 token 的 KV 是
# `2 × 32 × 8 × 128 × 2 = 131072 B ≈ 128 KB`。A100 80GB 去掉 14GB 权重与激活后，大约能放 40 万 token 的 KV——
# 听起来很多，但 1 万条 40 token 的并发对话就能把它吃光。
#
# 显存不够时调度器只有两个选择：
#
# - **recompute**：丢掉这个请求的 KV，放回等待队列，下次重新 prefill（prompt + 已生成的 token 全部重算）。
#   代价是**浪费 GPU 算力**；我们统计「重算了多少 token」。
# - **swap**：把 KV 拷到 CPU 内存，显存先腾出来，恢复时再拷回来。代价是**浪费 PCIe 带宽**与 host 内存。
#
# 本章的实验会把 KV 预算从大到小扫一遍，量出「预算 → 抢占次数 → 重算 token → 吞吐/延迟」这条链。
# 为了让抢占在几秒内发生，模拟器把 KV 预算缩小到 4096–16384 token（真实 A100 约为 50 万 token，
# 但抢占发生的条件和触发机制完全相同，只是来得更晚）。

# %% [markdown]
# ## 06.2 从零实现
#
# 为了「章节自包含」，本章内联一份最小实现；`Request` 的字段与第 05 章**完全一致**
# （`prompt_len` / `max_new_tokens` / `arrival` / `num_computed` / `num_generated`），
# 因此 05 章的任何到达序列都能直接喂进来。

# %%
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import pandas as pd
import torch

sys.path.insert(0, str(Path.cwd().parent))

from aiinfra.bench import bench, pick_device
from aiinfra.plotting import COLORS, save, setup_style
from aiinfra.tinygpt import GPTConfig, KVCache, TinyGPT

setup_style()
device = pick_device()
torch.manual_seed(0)
np.random.seed(0)
print("device =", device)

# %% [markdown]
# ### 数据结构：与 05 章同构
#
# `target_ctx()` 是本章新增的关键概念：**下一步 decode 之前，KV 里必须有多少 context token**。
# 新请求的 target 是 prompt 长度；生成过 k 个 token 后，target 是 `prompt + k - 1`（因为最后生成的那个 token 还没进 KV）。
# 有了它，抢占后的「重算」就是「把 `num_computed` 归零，重新 prefill 到 target」。

# %%
@dataclass
class Request:
    """一个推理请求；字段与 05 章一致。"""

    id: int
    prompt_len: int
    max_new_tokens: int
    arrival: float
    num_computed: int = 0  # KV cache 里已有的 context token 数
    num_generated: int = 0  # 已经产出的 token 数
    t_first: float = -1.0
    t_done: float = -1.0
    token_times: list[float] = field(default_factory=list)
    preempted: int = 0
    swapped: bool = False

    @property
    def finished(self) -> bool:
        return self.num_generated >= self.max_new_tokens

    def target_ctx(self) -> int:
        return self.prompt_len + max(self.num_generated - 1, 0)

    def needs_prefill(self) -> bool:
        return self.num_computed < self.target_ctx()

    def prefill_need(self) -> int:
        return self.target_ctx() - self.num_computed

# %% [markdown]
# 成本模型沿用第 05 章的常数，另外加一个 `swap_ms_per_token`：128KB/token 的 KV 在 25GB/s 的 PCIe 上，
# 单程约 `128KB / 25GB/s ≈ 0.005ms`——这就是 swap 的单价。

# %%
@dataclass(frozen=True)
class CostModel:
    base_ms: float = 6.0
    prefill_per_token_ms: float = 0.08
    decode_per_seq_ms: float = 0.12
    prefill_launch_ms: float = 1.0
    swap_ms_per_token: float = 0.005  # 128KB/token ÷ 25GB/s ≈ 5µs

    def step_ms(self, prefill_tokens: int, decode_seqs: int) -> float:
        t = self.base_ms + self.prefill_per_token_ms * prefill_tokens + self.decode_per_seq_ms * decode_seqs
        return t + (self.prefill_launch_ms if prefill_tokens else 0.0)


cost = CostModel()
decode_step = cost.step_ms(0, 16)
chunk_step = cost.step_ms(512, 16)
assert decode_step < chunk_step < cost.step_ms(2048, 16) / 3  # 切 4 块后，单步代价降到 1/3 以下

# %% [markdown]
# ### 调度器：5 个函数
#
# 调度器拆成「状态 + 5 个纯函数」，每一段只做一件事，方便对照 vLLM 的 `schedule()`：
#
# | 函数 | 职责 | vLLM 对应 |
# | --- | --- | --- |
# | `new_state` | 全部可变状态 | `Scheduler` 的成员变量 |
# | `admit` | 从 `waiting` 接纳到 `running`（查 slot / KV 余量） | `_schedule_prefills` 的准入部分 |
# | `build_plan` | 排 decode + 分 prefill chunk；KV 不够就抢占 | `_schedule_running` + `_preempt` |
# | `step` | 执行一步、推进时间、记录轨迹 | `SchedulerOutput` + `execute_model` |
# | `run_chunked` | 事件循环 + 指标 | `EngineCore` 的 step 循环 |

# %%
def new_state(reqs, *, max_num_seqs, max_num_batched_tokens, chunk_size,
              max_kv_tokens, preempt_mode, swap_ms_per_token):
    """调度器的全部可变状态。函数式写法，每个函数只改自己负责的字段。"""
    return {
        "pending": deque(sorted(reqs, key=lambda r: r.arrival)),
        "waiting": deque(),
        "running": [],
        "now": 0.0,
        "segments": [],  # (请求 id, t0, t1, "prefill"/"decode")，画时间轴用
        "steps": [],
        "prefill_per_step": [],
        "lane_free": list(range(max_num_seqs)),
        "lane_of": {},
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "chunk_size": chunk_size,
        "max_kv_tokens": max_kv_tokens,
        "preempt_mode": preempt_mode,
        "swap_ms_per_token": swap_ms_per_token,
        "swap_ms": 0.0,
        "stats": {"preemptions": 0, "recompute_tokens": 0, "swap_ms": 0.0},
    }

# %% [markdown]
# **准入（admit）**：被 swap 出去的请求回来时，必须一次性把整段 KV 拷回显存，所以它需要 `num_computed` 的空间；
# 新请求只需要下一个 chunk 的空间。`kv_used` 只统计 `running` 里的请求——被抢占的请求不占显存。

# %%
def kv_used(st):
    return sum(r.num_computed for r in st["running"])


def admit(st):
    while st["waiting"] and len(st["running"]) < st["max_num_seqs"]:
        r = st["waiting"][0]
        first = r.num_computed if r.swapped else min(st["chunk_size"] or r.prefill_need(), r.prefill_need())
        first = max(first, 1)
        if kv_used(st) + first > st["max_kv_tokens"] and st["running"]:
            break
        st["waiting"].popleft()
        if r.swapped:  # swap-in：把 KV 从 CPU 拷回 GPU
            t = r.num_computed * st["swap_ms_per_token"]
            st["stats"]["swap_ms"] += t
            st["swap_ms"] += t
            r.swapped = False
        st["running"].append(r)

# %% [markdown]
# **抢占（preempt）**：选 `running` 队尾的请求（最近进入的优先被牺牲，和 vLLM 的 FCFS 语义一致），
# 放回 `waiting` 队首，保证它下次能被优先恢复，不会被持续饿死。

# %%
def preempt(st, r):
    st["stats"]["preemptions"] += 1
    if st["preempt_mode"] == "recompute":
        st["stats"]["recompute_tokens"] += r.num_computed
        r.num_computed = 0  # 丢掉 KV：下次从 0 开始重新 prefill
    else:  # swap-out：KV 拷到 CPU，显存先腾出来
        t = r.num_computed * st["swap_ms_per_token"]
        st["stats"]["swap_ms"] += t
        st["swap_ms"] += t
        r.swapped = True
    st["running"].remove(r)
    if r.id in st["lane_of"]:
        st["lane_free"].append(st["lane_of"].pop(r.id))
        st["lane_free"].sort()
    st["waiting"].appendleft(r)  # 回队首，保持 FCFS 优先

# %% [markdown]
# **组批（build_plan）**：这是本章最核心的 20 行。
#
# 1. 先给所有「已经 prefill 完」的 running 请求各排 1 个 decode token；
# 2. 剩余 token 预算全部给 prefill，但每条请求一步最多吃 `chunk_size` 个 token（chunked prefill 的「切」就在这里）；
# 3. 如果本步要写的 KV 超过预算，就反复抢占 running 队尾，直到放得下——注意抢占后要**重新组批**，因为被抢占的请求可能已经在 plan 里。

# %%
def build_plan(st):
    """返回 (plan, prefill_tokens, decode_seqs)。plan 里每项是 (request, prefill_n, decode_n)。"""
    while True:
        n_dec = sum(1 for r in st["running"] if not r.needs_prefill())
        budget = st["max_num_batched_tokens"] - n_dec
        plan = [(r, 0, 1) for r in st["running"] if not r.needs_prefill()]
        prefill_tokens = 0
        cap = budget if st["chunk_size"] is None else min(st["chunk_size"], budget)
        for r in st["running"]:
            if not r.needs_prefill():
                continue
            need = r.prefill_need()
            n = need if st["chunk_size"] is None else min(need, cap - prefill_tokens)
            if n <= 0 or n > cap - prefill_tokens:
                break  # 没有 chunked prefill 时，整条 prompt 塞不进本步就只能等
            plan.append((r, n, 0))
            prefill_tokens += n
        fits = kv_used(st) + prefill_tokens + n_dec <= st["max_kv_tokens"]
        if fits or len(st["running"]) <= 1:
            return plan, prefill_tokens, n_dec
        victim = next((r for r in reversed(st["running"]) if r.num_computed > 0), None)
        if victim is None:
            return plan, prefill_tokens, n_dec
        preempt(st, victim)

# %% [markdown]
# **执行一步（step）**：推进时间、更新请求状态、记录甘特图 segment。
# 注意首个 token 是在 prefill 完成的那一步产出的（和 vLLM 一致），之后就进入每步 1 token 的 decode。

# %%
def step(st):
    admit(st)
    plan, prefill_tokens, n_dec = build_plan(st)
    st["prefill_per_step"].append(prefill_tokens)
    dur = st["cost"].step_ms(prefill_tokens, n_dec) / 1000 + st["swap_ms"] / 1000
    st["swap_ms"] = 0.0
    end = st["now"] + dur
    for r, n, _ in plan:
        if r.id not in st["lane_of"]:
            st["lane_of"][r.id] = st["lane_free"].pop(0)
        if n > 0:
            r.num_computed += n
            st["segments"].append((r.id, st["now"], end, "prefill"))
            if r.num_computed >= r.target_ctx() and r.num_generated == 0:
                r.num_generated = 1
                r.t_first = end
                r.token_times.append(end)
        else:
            r.num_computed += 1
            r.num_generated += 1
            r.token_times.append(end)
            st["segments"].append((r.id, st["now"], end, "decode"))
    st["steps"].append((st["now"], end, len(plan)))
    st["now"] = end
    done = [r for r in st["running"] if r.finished]
    st["running"] = [r for r in st["running"] if not r.finished]
    for r in done:
        r.t_done = st["now"]
        st["lane_free"].append(st["lane_of"].pop(r.id))
    st["lane_free"].sort()

# %% [markdown]
# **事件循环 + 指标（run_chunked）**：和 05 章一样，时间只在「有请求在跑」或「跳到下一个到达」时前进；
# 指标口径也保持一致（TTFT / TPOT=相邻 token 间隔 / 吞吐 / makespan）。

# %%
def run_chunked(reqs, cost, *, max_num_seqs=16, max_num_batched_tokens=4096, chunk_size=None,
                max_kv_tokens=10**9, preempt_mode="recompute", swap_ms_per_token=0.005):
    assert max_num_seqs <= max_num_batched_tokens, "每步至少要给每个 decode 序列留 1 token"
    st = new_state(reqs, max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens,
                   chunk_size=chunk_size, max_kv_tokens=max_kv_tokens,
                   preempt_mode=preempt_mode, swap_ms_per_token=swap_ms_per_token)
    st["cost"] = cost
    while st["pending"] or st["waiting"] or st["running"]:
        while st["pending"] and st["pending"][0].arrival <= st["now"]:
            st["waiting"].append(st["pending"].popleft())
        if not st["running"] and not st["waiting"]:
            if not st["pending"]:
                break
            st["now"] = st["pending"][0].arrival
            continue
        step(st)
    st["metrics"] = summarize(st, reqs)
    st["reqs"] = reqs
    return st


def summarize(st, reqs):
    ttft = np.array([r.t_first - r.arrival for r in reqs])
    e2e = np.array([r.t_done - r.arrival for r in reqs])
    itl = np.concatenate([np.diff(r.token_times) for r in reqs])
    return {
        "吞吐(tokens/s)": sum(r.max_new_tokens for r in reqs) / st["now"],
        "TTFT p95(ms)": np.quantile(ttft, 0.95) * 1000,
        "TPOT p95(ms)": np.quantile(itl, 0.95) * 1000,
        "E2E p95(ms)": np.quantile(e2e, 0.95) * 1000,
        "preemptions": st["stats"]["preemptions"],
        "重算token": st["stats"]["recompute_tokens"],
        "swap(ms)": st["stats"]["swap_ms"],
        "makespan(s)": st["now"],
    }

# %% [markdown]
# ### 负载与断言
#
# 实验负载故意做成「短对话 + 长上下文」混合：短请求 prompt=128、输出 64 token，按 λ=6 到达；
# 长请求 prompt=1024/2048、输出 64 token，按 λ=8 到达。长 prompt 是制造 decode 断点的元凶。

# %%
def chat_workload(long_len=2048, n_short=24, n_long=32, seed=3):
    """短对话（128 prompt）+ 长上下文请求（long_len prompt）按 Poisson 到达。"""
    rng = np.random.default_rng(seed)
    reqs, t = [], 0.0
    for i in range(n_short):
        t += rng.exponential(1 / 6.0)
        reqs.append(Request(i, 128, 64, float(t)))
    t = 0.0
    for j in range(n_long):
        t += rng.exponential(1 / 8.0)
        reqs.append(Request(100 + j, long_len, 64, float(t)))
    return sorted(reqs, key=lambda r: r.arrival)


def preempt_workload(n=12, prompt=256, out=512, rate=12.0, seed=2):
    """长输出请求：每个请求最终要占 prompt+out 个 KV token，用来制造显存压力。"""
    rng = np.random.default_rng(seed)
    reqs, t = [], 0.0
    for i in range(n):
        t += rng.exponential(1 / rate)
        reqs.append(Request(i, prompt, out, float(t)))
    return sorted(reqs, key=lambda r: r.arrival)


# %% [markdown]
# 四条不变量直接写成 `assert`：
#
# 1. 所有请求都必须跑完，且恰好产出 `max_new_tokens` 个 token；
# 2. chunked prefill 每步的 prefill token 数不能超过 `chunk_size`；
# 3. KV 预算充足时抢占次数必须为 0；
# 4. chunk 大小 ≥ 最长 prompt 时，结果应与「关闭 chunking」完全一致。

# %%
off = run_chunked(chat_workload(), cost, max_num_seqs=16, chunk_size=None)
on = run_chunked(chat_workload(), cost, max_num_seqs=16, chunk_size=512)
assert all(len(r.token_times) == r.max_new_tokens for r in on["reqs"])
assert all(len(r.token_times) == r.max_new_tokens for r in off["reqs"])
assert max(on["prefill_per_step"]) <= 512  # chunk 的硬约束
assert on["metrics"]["TPOT p95(ms)"] < off["metrics"]["TPOT p95(ms)"] / 2  # chunk 真正切开了 prefill
same_b = run_chunked(chat_workload(), cost, max_num_seqs=16, chunk_size=2048)
assert same_b["metrics"]["TPOT p95(ms)"] > 0.99 * off["metrics"]["TPOT p95(ms)"]  # chunk 太大 = 退化为不切
loose = run_chunked(preempt_workload(), cost, max_num_seqs=12, max_num_batched_tokens=2048,
                    chunk_size=1024, max_kv_tokens=65536)
assert loose["stats"]["preemptions"] == 0  # 预算充足时不应抢占
print(f"chunk=512 每步最多 prefill {max(on['prefill_per_step'])} token；宽松 KV 下抢占 0 次")

# %% [markdown]
# ### 标定：chunked prefill 的固定开销
#
# chunk 不是越小越好。每切一刀，都要多付一次 kernel 发射/权重读取的固定成本。
# 用 TinyGPT 实测：同样 prefill 2048 token，切成 4 段/8 段的耗时与不切相比如何。

# %%
cfg = GPTConfig()
tiny = TinyGPT(cfg).to(device).eval()


@torch.no_grad()
def prefill_chunked(T, chunk):
    x = torch.randint(0, cfg.vocab_size, (1, T), device=device)

    def fn():
        cache = KVCache(cfg, batch_size=1, max_seq_len=T + 8, device=device)
        for s in range(0, T, chunk):
            tiny(x[:, s:s + chunk], start_pos=s, kv_cache=cache)
            cache.advance(chunk)
    return fn


rows = []
for chunk in [256, 512, 1024, 2048]:
    r = bench(prefill_chunked(2048, chunk), warmup=2, reps=7, name=f"chunk{chunk}", device=device)
    rows.append({"chunk": chunk, "prefill 2048 token 总耗时(ms)": r.median_ms})
df_chunk_ms = pd.DataFrame(rows)
df_chunk_ms["相对不切的倍数"] = df_chunk_ms["prefill 2048 token 总耗时(ms)"] / df_chunk_ms["prefill 2048 token 总耗时(ms)"].iloc[-1]
df_chunk_ms.round(3)

# %% [markdown]
# **怎么读**（M3 Ultra / MPS / TinyGPT，reps=7 取中位数）：
#
# - chunk=1024：总耗时约 6ms，比不切（约 5.7ms）多不到 10%——只多一次发射，几乎可以忽略。
# - chunk=512：总耗时约 7–8ms，多 20–40%——4 次发射的固定成本开始显现。
# - chunk=256：总耗时 35–42ms，是**不切的 6–8 倍**——8 次发射的固定开销彻底压倒计算。
#
# TinyGPT 很小，固定开销占的比例比 7B 模型夸张；但趋势是通用的：**chunk 太小，省下的尾延迟会被额外 prefill 开销吃掉**；
# chunk 太大（等于 prompt 长度），又退化成「关闭 chunked prefill」。
#
# 于是「该选多大的 chunk」不是拍脑袋，而是**在 decode 尾延迟与 prefill 吞吐之间取平衡点**。
# 下面用模拟器把这个平衡点扫出来。

# %% [markdown]
# ## 06.3 实测
#
# 先看主线：同一份混合负载，关闭 vs 开启 chunked prefill（chunk=512）。
# 所有实验的 `max_num_seqs=16`、`max_num_batched_tokens=4096`，长 prompt 长度 2048。

# %%
rows = []
for name, chunk in [("关闭 chunked prefill", None), ("chunk = 512", 512), ("chunk = 1024", 1024)]:
    m = run_chunked(chat_workload(), cost, max_num_seqs=16, chunk_size=chunk)["metrics"]
    rows.append({"配置": name, **m})
df_main = pd.DataFrame(rows)
df_main.round(1)

# %% [markdown]
# 读表：
#
# - **TPOT p95：172.6ms → 49.8ms（3.5x）**。chunk=1024 只能降到 90.7ms——因为一步 prefill 仍然是 1024 token（约 90ms）。
#   想压低尾部，chunk 必须显著小于长 prompt 的长度。
# - **吞吐：471.9 → 456.0 tokens/s（-3.3%）**。pipeline 里多了几次 prefill 发射，但换来的是尾延迟的 3.5 倍改善。
# - **TTFT p95 略升（2.41s → 2.63s）**：长请求的 prefill 被切碎后，最后一段到得稍晚。
# - **E2E p95 略升 8%**：这是真实的 trade-off，调参时要按 SLO 决定值不值。

# %% [markdown]
# ### 时间轴：谁被谁阻塞
#
# 图里是主线实验的完整负载（24 短 + 32 长，长 prompt 2048）。
# 横轴是**产出 token 的时刻**，纵轴是**这个 token 与上一个 token 的间隔（ITL）**：
# 每个点都是一次「decode 被卡了多久」，点越高，用户等待越久。

# %%
def itl_points(st):
    """把每个请求的相邻 token 间隔摊平成点：(产出时刻, ITL ms, 请求类型)。"""
    xs, ys, kinds = [], [], []
    for r in st["reqs"]:
        for a, b in zip(r.token_times[:-1], r.token_times[1:]):
            xs.append(b)
            ys.append((b - a) * 1000)
            kinds.append("long prompt" if r.prompt_len >= 1024 else "short chat")
    return np.array(xs), np.array(ys), np.array(kinds)


demo_off = run_chunked(chat_workload(), cost, max_num_seqs=16, chunk_size=None)
demo_on = run_chunked(chat_workload(), cost, max_num_seqs=16, chunk_size=512)
fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.8), sharex=True)
for ax, st, title in [
    (axes[0], demo_off, f"Chunked prefill OFF: ITL spikes to {demo_off['metrics']['TPOT p95(ms)']:.0f} ms"),
    (axes[1], demo_on, f"Chunked prefill ON (chunk=512): ITL capped near {demo_on['metrics']['TPOT p95(ms)']:.0f} ms"),
]:
    xs, ys, kinds = itl_points(st)
    for kind, color, size in [("long prompt", COLORS["primary"], 8), ("short chat", COLORS["success"], 14)]:
        mask = kinds == kind
        ax.scatter(xs[mask], ys[mask], s=size, color=color, alpha=0.8, label=kind)
    ax.axhline(100, color=COLORS["danger"], ls="--", lw=1.2, label="100 ms")
    ax.set_ylabel("Inter-token latency (ms)")
    ax.set_title(title)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.015), fontsize=9, frameon=False)
axes[1].set_xlabel("Time (s)")
fig.suptitle("Decode stalls over time: one long prefill step vs 512-token chunks", y=1.00)
save(fig, "ch06_timeline")

# %% [markdown]
# 读图：关闭 chunking 时，值在 100ms 红线之上的点连成一簇，高度约等于「一条 2048 token prefill 的耗时」
# ——每来一个长请求，所有正在 decode 的请求都被按停一次。开启 chunk=512 后，大部分点落到 50ms 以下；
# 注意点的分布从「少而长」变成「多而短」：这就是 chunked prefill 的交易——用更频繁的小停顿，换掉偶发的大停顿。

# %% [markdown]
# ### 扫描：chunk 大小 × 长 prompt 长度
#
# 两个维度一起扫：chunk ∈ {关闭, 512, 1024, 2048}，长 prompt ∈ {1024, 2048}。
# 每格都跑完整的 24 短 + 32 长负载，指标取全局 TPOT p95。

# %%
rows = []
for long_len in [1024, 2048]:
    for chunk in [None, 512, 1024, 2048]:
        m = run_chunked(chat_workload(long_len=long_len), cost, max_num_seqs=16, chunk_size=chunk)["metrics"]
        rows.append({"长 prompt": long_len, "chunk": "关闭" if chunk is None else chunk,
                     "TPOT p95(ms)": m["TPOT p95(ms)"], "吞吐(tokens/s)": m["吞吐(tokens/s)"],
                     "长请求 TTFT p95(ms)": m["TTFT p95(ms)"]})
df_scan = pd.DataFrame(rows)
df_scan.round(1)

# %% [markdown]
# 这张表是整个第 06 章最重要的数字：
#
# - **收益随长 prompt 变长而变大**：长 = 2048 时，chunk 512 把 TPOT p95 从 172.6 压到 49.8（3.5x）；
#   长 = 1024 时只有 90.7 → 49.8（1.8x）。因为 chunk 必须小于 prompt 的「一次 prefill 时长」才有意义。
# - **chunk = prompt 长度等于没开**：2048/2048 两行的 TPOT 都是 172.6ms、吞吐只差 0.1%——「切了但没切开」。
# - **吞吐几乎不动**（±3%），说明 chunked prefill 主要买的是**尾延迟**，不是吞吐。
# - **chunk 太小也有下限**：标定实验里 chunk=256 的 prefill 总耗时是不切的 6–8 倍；而在模拟里，
#   256 相比 512 只把 TPOT p95 再压 20ms，却多付了这部分 prefill 开销——512–1024 是常见甜点。

# %%
fig, ax = plt.subplots(figsize=(7.2, 4.2))
plot_df = df_scan.rename(columns={"长 prompt": "long_len", "TPOT p95(ms)": "tpot_p95", "chunk": "chunk"})
pos = {None: 0, 512: 1, 1024: 2, 2048: 3}
labels = ["off", "512", "1024", "2048"]
for long_len, marker, color in [(1024, "o", "#7C3AED"), (2048, "s", "#2563EB")]:
    sub = plot_df[plot_df["long_len"] == long_len]
    xs = [pos[None if c == "关闭" else c] for c in sub["chunk"]]
    ax.plot(xs, sub["tpot_p95"], marker=marker, color=color, label=f"long prompt = {long_len}")
ax.set_xticks(list(pos.values()))
ax.set_xticklabels(labels)
ax.set_xlabel("Chunk size (tokens)")
ax.set_ylabel("TPOT p95 (ms)")
ax.set_title("Chunked prefill: tail latency vs chunk size")
ax.legend()
save(fig, "ch06_chunk_scan")

# %% [markdown]
# ### 抢占：KV 预算打满之后
#
# 固定 12 个长输出请求（prompt 256、输出 512，每个最终占用约 768 个 KV token），
# 把它们放到不同大小的 KV 预算里：4096 → 8192 → 16384 → 65536。
# 预算 < 工作集（约 9000 token）时必然发生抢占。

# %%
rows = []
for kv in [4096, 8192, 16384, 65536]:
    for mode in ["recompute", "swap"]:
        st = run_chunked(preempt_workload(), cost, max_num_seqs=12, max_num_batched_tokens=2048,
                         chunk_size=1024, max_kv_tokens=kv, preempt_mode=mode)
        m = st["metrics"]
        rows.append({"KV 预算(token)": kv, "抢占策略": mode, "抢占次数": m["preemptions"],
                     "重算token": m["重算token"], "swap(ms)": m["swap(ms)"],
                     "吞吐(tokens/s)": m["吞吐(tokens/s)"], "E2E p95(ms)": m["E2E p95(ms)"]})
df_pre = pd.DataFrame(rows)
df_pre.round(1)

# %% [markdown]
# 读表（这组数字要结合成本模型看）：
#
# - **预算 65536（远大于工作集）**：0 次抢占，吞吐 1383 tokens/s，E2E p95 4.0s——这是基线。
# - **预算 4096（约工作集的 44%）**：recompute 抢占 23 次，**重算了 8645 个 token**（相当于总输出 6144 的 1.4 倍），
#   吞吐掉到 804.6（-42%），E2E p95 涨到 6.8s。这是「显存不够」的真实代价：不是慢一点，而是慢一半。
# - **swap**：同样预算下只抢占 10 次、0 token 重算，但要付 43.8ms 的 KV 搬运时间，吞吐 886.8，略好于 recompute。
#
# 为什么 vLLM V1 默认（也是唯一）用 recompute 而不是 swap？因为 swap 需要 host 侧 KV 备份、pinned memory 与额外的
# PCIe 带宽，抢占频繁时 CPU↔GPU 拷贝会和别的请求抢带宽；而 recompute 只占 GPU、实现简单，
# 并且重算的 token 数是一个可以直接暴露给你的指标。V0 两种都支持（`--swap-space`），V1 精简掉了 swap。
#
# **结论**：显存不够时，吞吐和尾延迟同时恶化；调度的第一优先级永远是「让 KV 预算装得下工作集」。
# 这也解释了为什么 `--gpu-memory-utilization`、`max_model_len` 是生产上最敏感的两个参数。

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
pre_plot = df_pre.rename(columns={"KV 预算(token)": "kv_budget", "吞吐(tokens/s)": "tps",
                                  "重算token": "wasted", "抢占策略": "mode"})
for mode, marker, color in [("recompute", "o", "#EF4444"), ("swap", "s", "#10B981")]:
    sub = pre_plot[pre_plot["mode"] == mode]
    ax1.plot(sub["kv_budget"], sub["tps"], marker=marker, color=color, label=mode)
ax1.set_xscale("log", base=2)
ax1.set_xlabel("KV budget (tokens)")
ax1.set_ylabel("Throughput (tokens/s)")
ax1.set_title("Preemption hurts throughput")
ax1.legend()
sub = pre_plot[pre_plot["mode"] == "recompute"]
ax2.bar(range(len(sub)), sub["wasted"], color="#F59E0B")
ax2.set_xticks(range(len(sub)))
ax2.set_xticklabels([str(int(v)) for v in sub["kv_budget"]], rotation=30)
ax2.set_xlabel("KV budget (tokens)")
ax2.set_ylabel("Recomputed tokens")
ax2.set_title("Wasted work under recompute")
fig.suptitle("KV budget vs preemption cost", y=1.02)
save(fig, "ch06_preemption")

# %% [markdown]
# ## 06.4 对照工业实现（vLLM）
#
# 本章的两个机制，在 vLLM 里对应这些位置（课程锁定 **vLLM 0.28.x**）：
#
# | 我们的实现 | V0（≤0.7.x） | V1（0.8+，0.28.x） |
# | --- | --- | --- |
# | 把 prompt 切 chunk | `--enable-chunked-prefill` + `_schedule_chunked_prefill`（`vllm/core/scheduler.py`） | 默认开启，`Scheduler.schedule()` 把 `max_num_batched_tokens` 当每步预算，长 prefill 自动切分 |
# | 每步 token 上限 | `--max-num-batched-tokens` | 同名参数，`SchedulerConfig.max_num_batched_tokens` |
# | 长 prompt 的 chunk 上限 | `long_prefill_token_threshold`（CLI `--long-prefill-token-threshold`） | 并入上面的 token 预算，无单独开关 |
# | 组批 / 排 token | `_schedule_prefills` / `_schedule_decode` | `_schedule_running` / `_schedule_prefills`（方法名见下方版本注） |
# | 抢占 | `_preempt` + `--swap-space`（recompute / swap） | `Scheduler` 内的 `_preempt`：默认 recompute（swap 已移除） |
# | KV 分配 | `BlockSpaceManager`（block table） | `KVCacheManager.allocate_slots()`；返回 None 就触发抢占 |
#
# 版本注：`_schedule_chunked_prefill` 是 **V0** 的方法名；V1 把 chunked prefill 做成了默认路径，
# 逻辑合并进 `schedule()`/`_schedule_running`。在 0.28.x 源码里 grep 不到前者是正常的，
# 打开 `vllm/v1/core/sched/scheduler.py` 从 `def schedule` 往下读即可。
#
# 参数与本章实验的对应关系：
#
# - `--max-num-batched-tokens` 决定「一步能算多少 token」：调大 → 长 prefill 更少被切、TTFT 更短，但 decode 断点更宽；
#   调小 → 尾延迟更好，但 chunk 太小时固定开销上升（标定实验已量化）。
# - V0 的 `long_prefill_token_threshold`（CLI：`--long-prefill-token-threshold`）用来限制
#   「长 prompt 一次最多算多少」，等价于本章的 `chunk_size`；两者都指向同一个权衡。
# - `--enable-chunked-prefill` 在 V1 已无需开启（默认行为），V0 需要显式打开。
# - 抢占没有 CLI 开关：显存不够时自动发生。你能观测到的是 `vllm:num_requests_swapped`/preemption 计数与
#   「重算 token」的间接表现（TTFT 抖动、吞吐下降）。
#
# 我们 200 行 vs 工业实现的差距：vLLM 还要处理前缀缓存（04 章）、block 级 KV 分配（03 章）、
# 多请求同批的 attention metadata、CUDA graph、优先级调度——但「切 chunk、抢占、重算」这三件事的内核就是本章这 200 行。

# %% [markdown]
# ## 06.5 练习与小结
#
# **练习**
#
# 1. ⭐（10 分钟）把 `chunk_size` 改成 256，重跑主线对比表。TPOT p95 继续下降了吗？为什么吞吐下降得更明显？
# 2. ⭐（10 分钟）把 `max_num_batched_tokens` 从 4096 调到 1024，重跑 chunk=512 的扫描。
#    解释为什么「预算」和「chunk」不是一回事，以及两者同时变小时谁会先成为瓶颈。
# 3. ⭐⭐（25 分钟）实现「抢占时优先选最短输出」的策略（preempt shortest-first），对比 recompute 浪费的 token 数。
#    提示：改 `preempt` 的 victim 选择；想想为什么 vLLM 不这么做（公平性/饥饿）。
# 4. ⭐⭐（25 分钟）给 `build_plan` 加一个「decode 优先但每 N 步至少放一个 prefill chunk」的节奏控制，
#    扫描 N = 1/2/4，画出 TPOT p95 与吞吐的权衡曲线。
# 5. ⭐⭐⭐（40 分钟）设计一个自适应 chunk 策略：根据当前 running 队列长度与 TPOT SLO 动态调整 chunk 大小。
#    评分要点：(a) 决策规则可解释；(b) 与固定 chunk 的对比数字；(c) 过载/空载两种极端下不退化；
#    (d) 说明要改 vLLM 的哪个类、哪个方法。
#
# **你现在应该能回答：**
#
# - chunked prefill 为什么能降低 TPOT 尾部？为什么 chunk 太大没用、太小又亏？
# - recompute 与 swap 的代价分别是什么？为什么 V1 只保留 recompute？
# - 显存不足时，为什么吞吐和延迟会一起恶化？第一优先级该调哪个参数？
#
# **小结与下一章**
#
# 05 章让 GPU 忙起来，06 章让忙得「平稳」。至此推理内核（KV cache → 分页 → 前缀缓存 → 调度 → chunked prefill）
# 的主线走完。接下来 07 章进入优化：量化能把权重和 KV 的体积压下来——这直接缓解本章末尾暴露的
# 「显存装不下工作集」问题。
#
# **延伸阅读**
#
# - vLLM 文档：Chunked Prefill、`max_num_batched_tokens`、preemption 三节
# - vLLM V0 scheduler：`vllm/core/scheduler.py::_schedule_chunked_prefill`（历史实现）
# - vLLM V1 scheduler：`vllm/v1/core/sched/scheduler.py`、`vllm/v1/core/kv_cache_manager.py`
# - Sarathi-Serve 论文：chunked prefill 与 stall-free batching 的系统化描述
