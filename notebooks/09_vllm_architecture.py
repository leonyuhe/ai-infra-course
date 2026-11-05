# %% [markdown]
# # 第 09 章 · vLLM 架构与源码地图
#
# > 02–06 章我们手写了 KV cache、分页、前缀缓存、调度器和 chunked prefill；这一章把这些零件装回工业实现，回答「它们到底放在哪、怎么连起来、我们少写了什么」。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 60–75 分钟 |
# | 运行环境 | 🏠 Mac（CPU/MPS）· 无需 GPU、无需安装 vLLM |
# | 前置章节 | 02–06（KV cache / PagedAttention / 前缀缓存 / 调度器 / chunked prefill） |
# | 锁定版本 | vLLM 0.28.x（v1 引擎）；源码路径按 v1 目录结构整理，不确定处标「待核对版本」 |
# | 本章产物 | 分层架构图、请求生命周期、逐项对照表、源码地图、vLLM 默认参数下的调度模拟 |
#
# **学完你能回答：**
# 1. 一个 HTTP 请求从进来到吐出 token 经过哪几层？层与层之间传的是什么对象？
# 2. `Scheduler`、`KVCacheManager`、`BlockPool`、`Executor`、`ModelRunner` 各自负责什么，边界在哪？
# 3. 我们 02–06 章写的 200 行，对应 vLLM 里的哪些文件？差距是「多写了什么」？

# %% [markdown]
# ## 09.1 问题与直觉
#
# 你在 05 章写过一个几十行的连续批处理调度器，跑得挺好。那为什么工业实现需要一层层拆开？因为教学版把三件难事藏起来了：
#
# 1. **边界**：请求从哪来、到哪去——API 层、进程边界、流式返回、异常；
# 2. **钱**：KV cache 的块必须被精确分配、复用、回收，块表还要交给 attention kernel；
# 3. **规模**：256 个请求同时 decode 时，每步的 token 预算、抢占、采样都要可解释、可观测。
#
# 先看整体分层：左边是数据流，右边是每一层的源码位置。
#
# ![](../assets/figures/ch09_architecture.svg)
#
# 再看一个请求的一生：admission 决定它何时进来，prefill 决定 TTFT，decode 决定 TPOT，finish/preempt 决定它会不会被踢出去重算。
#
# ![](../assets/figures/ch09_lifecycle.svg)
#
# **反例**：如果像静态批处理那样「凑满一批再跑、跑完再收」，并发里只要有一个长请求，就会把同批所有请求的 TTFT/TPOT 一起拖长。vLLM 的连续批处理每个 iteration 都重新决定「这一步带谁玩」。
#
# 一句话记住：**vLLM = 分页 KV cache（02–04 章）+ 每步重排的调度器（05–06 章）+ 把它们包成 OpenAI API 的工程外壳。**

# %% [markdown]
# ## 09.2 从零实现：一个 vLLM 同构的调度模拟器
#
# 本节不抄源码，而是用 120 行复刻调度器的**决策逻辑**，变量名与 vLLM 对齐（`max_num_seqs`、`max_num_batched_tokens`、`prefill_left`、`preempt`）。09.3 用它跑出 vLLM 默认参数下的轨迹。
#
# | 概念 | 本实验里的名字 | vLLM v1 中的位置 |
# | --- | --- | --- |
# | 请求 | `Req` | `vllm/v1/request.py` 的 `Request` |
# | 每步 token 预算 | `max_num_batched_tokens` | `Scheduler` 构造参数 |
# | 同时运行请求上限 | `max_num_seqs` | `Scheduler` 构造参数 |
# | 物理块池 | `BlockPool` | `vllm/v1/core/block_pool.py` |
# | 一步的调度结果 | `scheduled` 列表 | `SchedulerOutput` |
# | 抢占 | `preempted` 集合 | `Scheduler` 内部的抢占逻辑（方法名待核对版本） |
#
# 规则先写死，读到后面再解释：
# - 每个 iteration 先喂 running 请求（decode 发 1 个 token，prefill 发剩余 chunk），再收 waiting 请求；
# - 请求需要新块而块池为空 → 抢占「最近被调度」的请求，释放块并让它 recompute；
# - 一步耗时 = prefill token 数 × 单 token 成本 + decode 固定开销 + 每请求成本。

# %%
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent))

from aiinfra.bench import bench
from aiinfra.plotting import save, setup_style

setup_style()
np.random.seed(0)
ROOT = Path.cwd().parent
FIG = ROOT / "assets" / "figures"
print("repo root:", ROOT)
assert (FIG / "ch09_architecture.svg").exists(), "概念图缺失"

# %%
@dataclass
class Req:
    """一个请求。字段名尽量与 vLLM 的 Request 对齐。"""
    rid: int
    prompt_len: int
    max_new: int
    arrival: float            # 到达时刻（ms）
    prefill_left: int = 0     # 还没算的 prompt token 数（chunked prefill 用）
    num_generated: int = 0
    kv_tokens: int = 0        # 已写入 KV cache 的 token 数
    blocks: int = 0           # 当前持有的物理块数
    ttft: float | None = None
    finish_t: float | None = None

    def __post_init__(self):
        self.prefill_left = self.prompt_len


@dataclass
class SimConfig:
    """调度器参数 + 一个极简的耗时模型（8B 模型 / A100 级算力的粗略值）。"""
    name: str
    max_num_seqs: int
    max_num_batched_tokens: int
    block_size: int = 16
    prefill_ms_per_token: float = 0.04
    decode_base_ms: float = 8.0
    decode_ms_per_seq: float = 0.12


def make_workload(n: int = 80, rate: float = 40.0, seed: int = 0) -> list[Req]:
    """80 个请求、平均 40 req/s 到达；prompt 128–1024，输出 32–128。"""
    rng = np.random.RandomState(seed)
    prompt = rng.randint(128, 1025, n)
    max_new = rng.randint(32, 129, n)
    arrival = np.cumsum(rng.exponential(1000.0 / rate, n))
    return [Req(i, int(prompt[i]), int(max_new[i]), float(arrival[i])) for i in range(n)]


CFG_VLLM = SimConfig("vLLM 默认 256/8192", max_num_seqs=256, max_num_batched_tokens=8192)
CFG_SMALL = SimConfig("连续批处理 8/2048", max_num_seqs=8, max_num_batched_tokens=2048)
wl = make_workload()
print(f"{len(wl)} 个请求；prompt 均值 {np.mean([r.prompt_len for r in wl]):.0f} tokens，"
      f"到达跨度 {wl[-1].arrival:.0f} ms")
assert all(r.prefill_left == r.prompt_len for r in wl)

# %%
class BlockPool:
    """最小版 BlockPool：只记账、不碰真实显存；对应 vllm/v1/core/block_pool.py。"""

    def __init__(self, total_blocks: int, block_size: int = 16):
        self.total = total_blocks
        self.free = total_blocks
        self.block_size = block_size
        self.peak_used = 0

    def ensure(self, req: Req, new_tokens: int) -> bool:
        """保证 req 能容纳 kv_tokens + new_tokens；不够返回 False，由调度器决定抢占。"""
        need = math.ceil((req.kv_tokens + new_tokens) / self.block_size)
        delta = need - req.blocks
        if delta <= 0:
            return True
        if delta > self.free:
            return False
        self.free -= delta
        req.blocks += delta
        self.peak_used = max(self.peak_used, self.total - self.free)
        return True

    def release(self, req: Req) -> None:
        self.free += req.blocks
        req.blocks = 0
        req.kv_tokens = 0

    @property
    def used(self) -> int:
        return self.total - self.free

# %%
def schedule_step(cfg: SimConfig, pool: BlockPool, active: list, waiting: deque):
    """一步调度：先 running、再 waiting。对应 Scheduler.schedule() 的核心循环。"""
    budget, scheduled, preempted = cfg.max_num_batched_tokens, [], set()
    for r in list(active):
        n = 1 if r.prefill_left == 0 else min(r.prefill_left, budget)
        if budget <= 0:
            break
        if not pool.ensure(r, n):
            pool.release(r)                       # 抢占：释放块、回到 waiting、之后 recompute
            r.prefill_left, r.num_generated = r.prompt_len, 0
            active.remove(r)
            waiting.appendleft(r)
            preempted.add(r.rid)
            continue
        budget -= n
        scheduled.append((r, n))
    while waiting and len(active) < cfg.max_num_seqs and budget > 0:
        r = waiting[0]
        if r.rid in preempted:                    # 本步刚被抢占的请求不再立刻塞回去，避免抖动
            break
        n = min(r.prefill_left, budget)
        if not pool.ensure(r, n):
            break
        waiting.popleft()
        active.append(r)
        budget -= n
        scheduled.append((r, n))
    return scheduled, preempted


def step_cost(cfg: SimConfig, scheduled: list):
    """一步的耗时：prefill 按 token 线性，decode 有固定开销 + 每请求成本。"""
    n_prefill = sum(n for r, n in scheduled if r.prefill_left > 0)
    n_decode = sum(1 for r, n in scheduled if r.prefill_left == 0)
    cost = n_prefill * cfg.prefill_ms_per_token
    if n_decode:
        cost += cfg.decode_base_ms + cfg.decode_ms_per_seq * n_decode
    return n_prefill, n_decode, cost

# %%
def apply_step(scheduled, active, pool, t, step_ms):
    """执行一步：写 KV、生成 token、回收已完成的请求。"""
    for r, n in scheduled:
        r.kv_tokens += n
        if r.prefill_left > 0:
            r.prefill_left -= n
            if r.prefill_left == 0:               # prefill 完成即采样出第一个 token
                r.num_generated, r.ttft = 1, t + step_ms
        else:
            r.num_generated += 1
    done = []
    for r in list(active):
        if r.num_generated >= r.max_new:
            r.finish_t = t + step_ms
            pool.release(r)
            active.remove(r)
            done.append(r)
    return done


# %%
def simulate(cfg: SimConfig, workload: list[Req], total_blocks: int = 24000,
             trace: bool = False) -> dict:
    """按 iteration 推进的离散事件模拟：admission → prefill → decode → finish/preempt。"""
    pool = BlockPool(total_blocks, cfg.block_size)
    arrivals = deque(sorted(workload, key=lambda r: r.arrival))
    waiting, active, done, rows = deque(), [], [], []
    t, preempts, steps = 0.0, 0, 0
    for _ in range(20000):
        if not active and not waiting and arrivals:
            t = max(t, arrivals[0].arrival)       # 空转：跳到下一个到达时刻
        while arrivals and arrivals[0].arrival <= t:
            waiting.append(arrivals.popleft())
        if not (active or waiting):
            break
        scheduled, preempted = schedule_step(cfg, pool, active, waiting)
        preempts += len(preempted)
        if not scheduled:
            break
        n_prefill, n_decode, step_ms = step_cost(cfg, scheduled)
        done.extend(apply_step(scheduled, active, pool, t, step_ms))
        t += step_ms
        steps += 1
        if trace:
            rows.append({"step": steps, "running": len(active), "waiting": len(waiting),
                         "prefill_tok": n_prefill, "decode_req": n_decode,
                         "step_ms": round(step_ms, 2), "t_ms": round(t, 1),
                         "kv_blocks": pool.used, "preempts": preempts})
    return {"finished": done, "makespan": t, "preemptions": preempts,
            "steps": steps, "peak_blocks": pool.peak_used, "trace": rows}

# %%
def static_batch(workload: list[Req], batch_size: int = 8):
    """05 章的反例：凑满一批 → 整批 decode 完 → 下一批。"""
    reqs = sorted(workload, key=lambda r: r.arrival)
    t = 0.0
    for i in range(0, len(reqs), batch_size):
        batch = reqs[i:i + batch_size]
        start = max(t, batch[-1].arrival)          # 等最后一条到齐才开跑
        step = CFG_VLLM.decode_base_ms + CFG_VLLM.decode_ms_per_seq * len(batch)
        prefill = sum(r.prompt_len for r in batch) * CFG_VLLM.prefill_ms_per_token
        for r in batch:
            r.ttft = start + prefill + step
            r.finish_t = r.ttft + (r.max_new - 1) * step
        t = max(r.finish_t for r in batch)
    return reqs, t


def summarize(result: dict, label: str) -> dict:
    done = result["finished"]
    lat = np.array([r.finish_t - r.arrival for r in done])
    ttft = np.array([r.ttft for r in done])
    tokens = sum(r.max_new for r in done)
    ms = max(result["makespan"], 1e-9)
    return {"方案": label, "完成请求": len(done), "makespan_s": round(ms / 1000, 2),
            "吞吐(tok/s)": round(tokens / ms * 1000, 1),
            "TTFT_p50_ms": round(float(np.percentile(ttft, 50)), 1),
            "延迟_p50_ms": round(float(np.percentile(lat, 50)), 1),
            "延迟_p99_ms": round(float(np.percentile(lat, 99)), 1),
            "抢占次数": result["preemptions"], "KV峰值块": result["peak_blocks"]}

# %% [markdown]
# 模拟器就绪。注意 `schedule_step` 里的两段循环，就是 vLLM `Scheduler.schedule()` 的骨架：
# 先处理 running（decode 与未完成的 prefill），再按 `max_num_seqs` 和 token 预算收 waiting。
#
# ## 09.3 实测：vLLM 默认参数 vs 我们手写的两种策略
#
# 同一份负载（80 请求、40 req/s 到达）跑四种方案。前两种是 05/06 章的手写策略，后两种是 vLLM 默认参数在「KV 充足」与「KV 不足」两种机器上的表现。

# %%
rows = []
static_done, static_makespan = static_batch(make_workload())
rows.append(summarize({"finished": static_done, "makespan": static_makespan,
                       "preemptions": 0, "peak_blocks": 0}, "静态批处理 b=8（05 章）"))

res_small = simulate(CFG_SMALL, make_workload())
rows.append(summarize(res_small, "连续批处理 8/2048（05 章）"))

res_vllm = simulate(CFG_VLLM, make_workload(), total_blocks=24000)   # A100 80G ≈ 24k 块
rows.append(summarize(res_vllm, "vLLM 默认 256/8192（KV 充足）"))

res_poor = simulate(CFG_VLLM, make_workload(), total_blocks=400)     # 约 6400 token 的 KV
rows.append(summarize(res_poor, "vLLM 默认 256/8192（KV 不足）"))

df_compare = pd.DataFrame(rows)
df_compare["scheme_en"] = ["Static batch b=8", "Continuous 8/2048",
                           "vLLM defaults 256/8192", "vLLM defaults, KV short"]
df_compare

# %%
res_trace = simulate(CFG_VLLM, make_workload(), total_blocks=24000, trace=True)
trace_df = pd.DataFrame(res_trace["trace"])
print(f"总迭代 {len(trace_df)} 步；KV 峰值 {res_trace['peak_blocks']} 块")
trace_df.head(12)

# %%
assert len(res_vllm["finished"]) == 80, "所有请求都应完成"
assert res_vllm["peak_blocks"] <= 24000, "块数不能超过块池容量"
assert df_compare.loc[2, "吞吐(tok/s)"] > df_compare.loc[0, "吞吐(tok/s)"], "连续批处理应快于静态批处理"
assert df_compare.loc[3, "抢占次数"] > 0, "KV 不足时应发生抢占"

overhead = bench(lambda: simulate(CFG_VLLM, make_workload(), 24000),
                 name="simulate(80 reqs)", warmup=1, reps=5)
pd.DataFrame([overhead.as_dict()])

# %% [markdown]
# 上表最后一列是**在本机测得的**调度决策总耗时（纯 Python，含 80 个请求的完整模拟）。
# 它比一步 GPU 前向（毫秒级）还小两个数量级——这说明「调度器本身很便宜，贵的永远是模型前向和 KV 搬运」。
#
# 下面扫一遍 `max_num_seqs`：固定 token 预算 8192、KV 充足，只改同时运行的请求数上限。

# %%
rows = []
for seqs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    cfg = SimConfig(f"seqs={seqs}", max_num_seqs=seqs, max_num_batched_tokens=8192)
    res = simulate(cfg, make_workload(), total_blocks=24000)
    row = summarize(res, cfg.name)
    row["max_num_seqs"] = seqs
    rows.append(row)
sweep_df = pd.DataFrame(rows)
sweep_df[["max_num_seqs", "吞吐(tok/s)", "TTFT_p50_ms", "延迟_p50_ms", "延迟_p99_ms", "KV峰值块"]]

# %%
import matplotlib.pyplot as plt

x_seqs = sweep_df["max_num_seqs"].to_numpy()
tok_s = sweep_df["吞吐(tok/s)"].to_numpy()
p99_ms = sweep_df["延迟_p99_ms"].to_numpy()
fig, ax1 = plt.subplots()
ax1.plot(x_seqs, tok_s, "o-", color="#2563EB", label="throughput")
ax1.set_xscale("log", base=2)
ax1.set_xlabel("max_num_seqs (cap on concurrently running requests)")
ax1.set_ylabel("Throughput (tok/s)", color="#2563EB")
ax2 = ax1.twinx()
ax2.plot(x_seqs, p99_ms, "s--", color="#EF4444", label="p99 latency")
ax2.set_ylabel("p99 latency (ms)", color="#EF4444")
ax2.grid(False)
ax1.set_title("max_num_seqs: throughput saturates near 64; p99 falls then flattens (underload)")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
save(fig, "ch09_scheduler_sweep")

# %%
x_names = df_compare["scheme_en"].tolist()
tok_cmp = df_compare["吞吐(tok/s)"].to_numpy()
fig, ax = plt.subplots()
colors = ["#9CA3AF", "#7C3AED", "#2563EB", "#EF4444"]
ax.bar(x_names, tok_cmp, color=colors)
ax.set_ylabel("Throughput (tok/s)")
ax.set_title("Same workload, four strategies: the gap is how many requests run at once")
ax.tick_params(axis="x", rotation=10)
save(fig, "ch09_policy_compare")

# %% [markdown]
# **读图**：
#
# - **调度轨迹**（`trace_df`）：前 6 步把 8192 token 预算几乎全花在 prefill 上，单步 300–340 ms；第 7 步后进入纯 decode，一步 17.6 ms ≈ `8.0 + 0.12 × 80`。这就是「prefill 贵、decode 按并发摊薄」的量化版本。
# - **方案对比**：静态批处理 488 tok/s → 连续批处理 677 → vLLM 默认 1634，约 **3.3 倍**；KV 不足时掉回 641，并出现 21 次抢占。
# - **原因**：decode 一步成本 ≈ `固定开销 + 每请求成本 × 并发数`。并发上不去就摊不薄固定开销；而并发被 `max_num_seqs` 和 **KV 块数**双重限制。
# - **TTFT 1222 ms 不是 bug**：这份负载 prompt 均值 608、到达率 40 req/s，prefill 需求约 2.4 万 token/s，已经贴着模拟器的 prefill 容量（8192 / 335 ms ≈ 2.4 万 token/s）。第 11 章的瓶颈判定就从这类现象开始。
# - **max_num_seqs 扫描**：吞吐从 120 涨到 1660 tok/s；64 之后并发翻倍只涨 5.5%，随后完全持平。注意这份负载只有 80 个请求，128 已能把它们全部并发，**末尾的持平是负载上限而非硬件上限**；真实的硬件拐点要用持续的到达流压测（第 11 章）。p99 在欠载下先降后平，过载时才会随并发上升。真正的容量天花板是 KV：见「KV 不足」那一行。
#
# **对 vLLM 参数的启示**：`max_num_seqs` 是吞吐旋钮，但不是越大越好；先确认 KV 够（`gpu_memory_utilization`、`max_model_len`），再按 SLO 的 p99 反推并发上限。

# %% [markdown]
# ## 09.4 对照工业实现（vLLM 0.28.x）
#
# ### 逐项对照：我们的类 vs vLLM 的类
#
# | 我们的实现（章） | vLLM v1 对应物 | 文件（0.28.x） | vLLM 多做了什么 |
# | --- | --- | --- | --- |
# | `KVCache`（02） | `KVCacheManager` + `BlockPool` + attention backend 的 KV 读写 | `vllm/v1/core/kv_cache_manager.py` / `block_pool.py` | 块表由 kernel 直接寻址；支持多种 attention 后端与异构层 |
# | `BlockManager`（03） | `KVCacheManager.allocate_slots()` / `free()` | 同上 + `vllm/v1/core/kv_cache_utils.py` | 按 `KVCacheSpec` 自动算块数，支持 sliding window / Mamba 等异构层 |
# | `HashPrefixCache`（04） | 前缀缓存（block hash）+ `--enable-prefix-caching` | `vllm/v1/core/kv_cache_manager.py`（`get_computed_blocks`） | 跨请求/跨进程共享，且与抢占、调度器协同 |
# | `Scheduler`（05） | `Scheduler.schedule()` | `vllm/v1/core/sched/scheduler.py` | 优先级/FCFS 队列、token 预算、抢占、结构化输出约束 |
# | `chunked prefill` + 抢占（06） | chunked prefill + `PreemptionMode.RECOMPUTE` | 同上 | 与前缀缓存、spec decode、LoRA 同时工作 |
# | `Request` 数据类（05） | `Request` + `RequestStatus` | `vllm/v1/request.py` | 状态机、多模态输入、spec token、采样参数 |
# | 手写 gather attention（03） | `AttentionBackend` 实现 | `vllm/v1/attention/backends/`（路径待核对版本） | FlashAttention / FlashInfer / Triton kernel，变长 batch |
# | 无（07 章） | 量化层 | `vllm/model_executor/layers/quantization/` | fp8/int8/AWQ/GPTQ，权重加载与 kernel 融合 |
# | 无（08 章） | 投机解码 | `vllm/v1/spec_decode/`（路径待核对版本） | draft model / n-gram / EAGLE，验证与回滚 |

# %% [markdown]
# ### 源码地图：先看这 6 个位置
#
# | 路径 | 关键类/函数 | 职责 | 核对状态（0.28.x） |
# | --- | --- | --- | --- |
# | `vllm/v1/engine/` | `LLMEngine`、`AsyncLLM`、`EngineCore`、`EngineCoreClient`、`Processor`、`OutputProcessor` | 请求编排、进程边界、流式输出 | 目录结构自 v1 起稳定；单文件拆分随版本变化，**待核对版本** |
# | `vllm/v1/core/sched/scheduler.py` | `Scheduler`（`schedule()`、`_preempt_request` 等） | 每步调度、token 预算、抢占 | **待核对版本**（方法名可能调整） |
# | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager`（`allocate_slots`、`free`、`get_computed_blocks`） | 块分配/回收/前缀缓存 | **待核对版本** |
# | `vllm/v1/core/block_pool.py` | `BlockPool` | 物理块池、块哈希 | **待核对版本** |
# | `vllm/v1/worker/` | `Worker`、`GPUModelRunner`（`execute_model`、`prepare_inputs`） | 前向、输入准备、采样 | 文件名可能拆成 `gpu_worker.py` / `gpu_model_runner.py`，**待核对版本** |
# | `vllm/model_executor/` | 各模型实现（如 Qwen3 的 `ForCausalLM`） | 模型结构与权重加载 | 文件可能叫 `qwen3.py`，**待核对版本** |

# %% [markdown]
# ### 一个请求的调用链（按真实函数名，可以照着下断点）
#
# 1. `vllm/entrypoints/openai/api_server.py` 收到 `POST /v1/chat/completions`，转给 `OpenAIServingChat`；
# 2. `AsyncLLM.add_request()` → `Processor` 做 tokenize / 多模态处理，生成 `EngineCoreRequest`；
# 3. 请求通过 `EngineCoreClient` 送进 **独立的 EngineCore 进程**（ZMQ），进入 `Scheduler` 的 waiting 队列；
# 4. 每个 iteration：`Scheduler.schedule()` → 调 `KVCacheManager.allocate_slots()` 分配块 → 产出 `SchedulerOutput`；
# 5. `Executor.execute_model(SchedulerOutput)` → `GPUModelRunner` 拼 batch、准备输入（slot mapping / block table）→ 各层 forward + attention backend；
# 6. 采样出 token，`OutputProcessor` detokenize，按 SSE 流式返回；同时把新 token 写回请求状态；
# 7. 请求结束（`finish_reason`）→ `KVCacheManager.free()` 归还块；若中途块不足 → 抢占 → recompute。
#
# 这 7 步就是 09.1 生命周期图里的箭头。想验证自己在读对版本，跑下面五条命令，比背路径可靠：

# %% [markdown]
# ```bash
# python -c "import vllm; print(vllm.__version__)"            # 期望 0.28.x
# python -c "from vllm.v1.core.sched.scheduler import Scheduler; print(Scheduler)"
# python -c "from vllm.v1.core.kv_cache_manager import KVCacheManager; print(KVCacheManager)"
# python -c "from vllm.v1.core.block_pool import BlockPool; print(BlockPool)"
# find vllm/v1/engine -maxdepth 1 -name "*.py" | sort        # 对照 09.4 的目录表
# ```
#
# 如果 0.28.x 的拆分与本章表格不一致，**以上面命令的输出为准**，把差异回填到表格；不要按记忆硬编码路径。
# 本章的架构与方法论（分层、对象流、调度规则）跨版本稳定，具体文件名才需要逐版本核对。

# %% [markdown]
# ### 差距在哪：200 行 vs 上万行
#
# 我们的模拟器只做了**决策**，没做**执行**。工业实现多出来的部分可以归成三类：
#
# 1. **边界工程**：多进程、ZMQ、Ray、HTTP/SSE、鉴权、异常与超时——这些一行都不产生 token，但缺了就上不了线；
# 2. **异构**：不同模型（Qwen/Llama/Mixtral/Mamba）、不同 KV 规格（全注意力 / sliding window / MLA）、不同 attention 后端，都要在 `KVCacheSpec` 与 `AttentionBackend` 两层被抹平；
# 3. **可观测与可运维**：Prometheus 指标（`vllm:num_requests_running`、`vllm:kv_cache_usage_perc`）、请求日志、抢占告警——第 11 章整章都在用它们。
#
# 所以「读源码」的正确姿势不是从第一行读到最后一行，而是**带着一个现象，沿本章的调用链找到那 1–2 个文件**。

# %% [markdown]
# ## 09.5 练习与小结
#
# **练习**（⭐ 改参数 / ⭐⭐ 改实现 / ⭐⭐⭐ 开放设计）
#
# 1. ⭐ 把 `CFG_VLLM.max_num_batched_tokens` 从 8192 改成 2048，重跑 09.3 的两张表。吞吐和 TTFT 各变多少？先用 token 预算解释，再跑一遍验证。
# 2. ⭐ 把 `decode_ms_per_seq` 从 0.12 改成 0.02（模拟更快的卡），观察 sweep 的拐点位置向哪边移动。为什么？
# 3. ⭐⭐ 给 `schedule_step` 加一个「优先级队列」：短 prompt 优先。在长 prompt 混短 prompt 的负载下比较 p50 / p99 延迟，并解释为什么吞吐基本不变。
# 4. ⭐⭐ 把抢占从 recompute 改成 swap（把被抢请求的块「换出」而不是丢弃，稍后再换回），对比 `抢占次数 × prefill 成本` 与换入换出的搬运成本。需要给 `Req` 加字段。
# 5. ⭐⭐⭐ 设计一条读码路线：线上现象是「首 token 很慢，但 TPOT 正常」。你会在 09.4 的哪几个文件加日志/下断点？给出验证步骤与预期证据。（评分要点：定位到 prefill/调度层、给出可观测指标、区分 KV 不足与排队的证据、有回滚方案）
#
# **你现在应该能回答：**
#
# - 为什么 `Scheduler` 和 `KVCacheManager` 要拆成两个类？各自的不变量是什么？
# - `max_num_seqs` 和 `max_num_batched_tokens` 分别卡住了什么？调大哪个会把 p99 推高？
# - 为什么 KV 不足时吞吐会「掉回去」，而不仅是变慢？
#
# **小结与下一章**
#
# 本章我们把 02–06 章的零件映射到了 vLLM 的真实文件与类名，并用一个同构模拟器验证了调度规则。下一章我们把服务真正跑起来（Mac 用 `vllm-metal`，云上用 Docker），用 OpenAI 兼容 API 发请求，看看这些内部对象在 HTTP 层长什么样。

# %% [markdown]
# **延伸阅读**
#
# - vLLM v1 引擎设计说明（`docs/` 下的 v1 架构文档，版本随 0.28.x）；
# - `vllm/v1/core/sched/scheduler.py` 的 `schedule()`：对照本章 `schedule_step` 读；
# - `vllm/v1/core/kv_cache_manager.py` 的 `allocate_slots()`：对照 03 章的 `BlockManager.allocate` 读；
# - 第 10 章：OpenAI 兼容 API、LoRA、结构化输出与工具调用。
