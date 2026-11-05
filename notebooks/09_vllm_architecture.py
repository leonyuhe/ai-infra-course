# %% [markdown]
# # 第 09 章 · vLLM 架构与源码地图
#
# > 前六章我们手写了推理引擎的每个零件；这一章把它们装回工业实现，回答「它们放在哪、怎么连起来、我们少写了什么」。
#
# 你接手了一个别人写的推理服务，日志里个别请求的耗时飞到几秒。同事丢下一句「去看 vLLM 源码」，
# 你打开 `vllm/v1/`——几十个文件，不知道该从哪个开始读。
#
# 这一章不教你背路径，而是给你一张**能照着下断点的地图**：先看哪几个文件、按什么顺序看、
# 一个请求按真实函数名怎么走完。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 75–90 分钟 |
# | 运行环境 | 🏠 Mac（CPU/MPS）· 无需 GPU、无需安装 vLLM |
# | 前置章节 | 02–06（KV cache / PagedAttention / 前缀缓存 / 调度器 / chunked prefill） |
# | 本章产物 | 分层架构图、请求生命周期、可运行的调度模拟器、四组对比实验、源码地图 |
# | 本章接口 | 模拟器变量名与 vLLM 对齐（`max_num_seqs`、`max_num_batched_tokens`）；不依赖 `vllm` 包 |
#
# **学完你能回答：**
# 1. 一个 HTTP 请求从进来到吐出 token 经过哪几层？层与层之间传的是什么对象？
# 2. `Scheduler`、`KVCacheManager`、`BlockPool`、`Executor`、`ModelRunner` 的边界在哪？
# 3. 我们 02–06 章写的几百行，对应 vLLM 里的哪些文件？差距是「多写了什么」？
# 4. 怎么用约 120 行模拟器验证 vLLM 的调度规则，而不需要真的装 vLLM？

# %% [markdown]
# ## 09.0 本章目标与路线
#
# **一句话目标**：把 02–06 章手写的零件映射到 vLLM v1 的分层架构与真实源码位置，
# 并用一个同构模拟器验证「vLLM 默认参数下的调度规则」；不改变任何已有实验结论。
#
# **代价与边界**：
# - 本章不安装 vLLM、不下载模型；模拟器只复刻**决策逻辑**（谁进 batch、谁被抢占），不复刻 GPU 执行；
# - 源码路径按 vLLM 0.28.x 的 v1 目录整理，离线无法确认的条目显式标「待核对版本」；
# - 模拟数字用于解释趋势，不代表真实 GPU 吞吐（真实数字在第 11 章的压测里）。
#
# **本章交付物**：
#
# | # | 交付物 | 在哪一节 | 怎么验收 |
# | --- | --- | --- | --- |
# | 1 | 分层架构图 + 请求生命周期图 | 09.2 | 能指着图说出每一层传什么对象 |
# | 2 | 约 120 行的 vLLM 同构模拟器 | 09.3 Step 1–4 | 四组实验的断言全过（请求全完成、块不超限） |
# | 3 | 四种调度策略对比 | 09.4 实验 A | 吞吐 488 → 1634 tok/s，约 3.3 倍 |
# | 4 | 调度轨迹与 `max_num_seqs` 扫描 | 09.4 实验 B/C | 读数与「prefill 贵、decode 摊薄」一致 |
# | 5 | 逐项对照表 + 源码地图 + 调用链 | 09.5 | 每条路径都有核对状态与核对命令 |
#
# **阅读路线**：09.1 问题 → 09.2 高阶设计（先看两张图）→ 09.3 分四步实现（每步有验收）
# → 09.4 实验 A/B/C → 09.5 源码地图终检 → 09.6 练习。
#
# **如果只有 30 分钟**：读 09.1 的反例、09.2 的两张图，再跳到 09.5 的「一个请求的调用链」——
# 那一段是本章最能立刻用上的部分，也是下次排查时你会回来查的地方。

# %% [markdown]
# ## 09.1 问题与直觉：教学代码与生产引擎差在哪
#
# 05 章你写过一个几十行的连续批处理调度器，跑得挺好。那为什么工业实现要一层层拆开？
#
# **先猜一下**：02–06 章我们手写的推理引擎，核心逻辑两百来行；vLLM v1 的调度与 KV 管理
# 是上万行。这 50 倍的差距，差的是 (a) 更多的边界情况　(b) 更快更多的 kernel
# 　(c) 根本不在一个维度——我们写的是「决策」，它写的是「决策 + 执行 + 边界」。
#
# 答案是 (c)。选 (a) 的人会以为「把边界补一补就追上了」；补完你会发现，产生 token 的还是
# 那两百行，剩下的一万行全是为了让这两百行在生产里不出事。
#
# 教学版具体把三件难事藏起来了：
#
# 1. **边界**：请求从哪来、到哪去——HTTP 层、多进程、流式返回、异常与超时；
# 2. **钱**：KV cache 的块必须被精确分配、复用、回收，块表还要交给 attention kernel；
# 3. **规模**：256 个请求同时 decode 时，每步的 token 预算、抢占、采样都要可解释、可观测。
#
# 两张关键图（分层架构与请求生命周期）放在 09.2，先记住一个反例：
#
# **反例**：如果像静态批处理那样「凑满一批再跑、跑完再收」，并发里只要有一个长请求，
# 就会把同批所有请求的 TTFT/TPOT 一起拖长。vLLM 的连续批处理每个 iteration 都重新决定
# 「这一步带谁玩」——**「每一步重排」这四个字，是本章所有实验的共同原因**。

# %% [markdown]
# | 维度 | 教学版（02–06 章） | 工业版 vLLM v1 |
# | --- | --- | --- |
# | 边界 | 一个 Python 进程内调用 | API Server + 独立 EngineCore 进程 + Executor/Worker |
# | KV 管理 | 连续预分配 / 手写 block manager | `KVCacheManager` + `BlockPool` + 前缀缓存 + 多 attention 后端 |
# | 调度 | 一个 while 循环 | `Scheduler`：队列、token 预算、抢占、结构化输出约束 |
# | 可观测 | `print` | Prometheus 指标 + 请求日志 + `vllm bench serve` |
#
# 表里最右一列值得多看一眼：`KVCacheManager`、`BlockPool`、`Scheduler` 这三个名字，
# 和我们三章的三个类一一对应，一个不多、一个不少——差别全在每一格后面的实现。
#
# 本章要解决的两个子问题：
#
# - **正确性**：vLLM 的调度规则（预算 / 准入 / 抢占）能不能用 100 行复刻，且行为一致？
# - **成本**：这些规则分别影响吞吐、TTFT、p99 的哪一部分？
#
# **先给结论**：vLLM = 分页 KV cache（02–04 章）+ 每步重排的调度器（05–06 章）+ 把它们包成 OpenAI API 的工程外壳。

# %% [markdown]
# ## 09.2 高阶设计：分层、生命周期与三个组件
#
# 先看整体分层：左边是数据流，右边是每一层的源码位置。
#
# ![](../assets/figures/ch09_architecture.svg)
#
# 看图的方法：先看右半边（谁在哪个进程里），再看左半边（箭头上传的是什么对象）。
# 第一张图回答「有哪些层」，第二张图回答「一个请求怎么在层之间移动」——两张都看懂，
# 09.3 的代码就只是把图上的方框填成函数。
#
# 再看一个请求的一生：admission 决定它何时进来，prefill 决定 TTFT，decode 决定 TPOT，
# finish/preempt 决定它会不会被踢出去重算。
#
# ![](../assets/figures/ch09_lifecycle.svg)

# %% [markdown]
# 动手前先把结构定下来。整章只有一个模拟对象和三个组件。
#
# **组件 1：请求与预算（存什么）**
# `Req` 记录 prompt 长度、已生成 token、已占块数、到达时刻；`SimConfig` 记录 vLLM 的两个关键预算：
# `max_num_seqs`（同时跑多少请求）与 `max_num_batched_tokens`（一步最多算多少 token）。
#
# **组件 2：块池（钱在哪）**
# `BlockPool` 按 `block_size=16` 记账：`ensure()` 检查能否为新 token 开出块，`release()` 把块还回池子。
# 它对应 `vllm/v1/core/block_pool.py`，不碰真实显存，只维护「谁占了多少块」。
# 顺便把池子的大小换算成人话：实验 A 里的 24,000 块 × 16 token = **38.4 万 token 的 KV 容量**，
# 「A100 80G ≈ 24k 块」这个数字就是这么落进模拟器的。
#
# **组件 3：调度器（谁先跑）**
# 每个 iteration 做三件事：
# 1. 先喂 running 请求：decode 发 1 个 token，未完成的 prefill 发剩余 chunk；
# 2. 再按 token 预算和 `max_num_seqs` 从 waiting 收新请求；
# 3. 块不够时抢占「最近被调度」的请求，释放块并让它 recompute。
#
# 这个顺序不是随手排的。已经在跑的请求，KV 块已经占在手里；这一步如果不给它算，
# 块就白占着却没有任何进展。所以 **running 优先的真正理由是「别让已经花掉的钱闲置」**，
# 公平性排在它后面。
#
# 数据流：
#
# ```text
# arrivals → waiting 队列 → Scheduler.schedule() → SchedulerOutput → 执行 → token / 完成 / 抢占
#                               ↑ 关键：每步都重排，而不是一批跑到底
# ```

# %% [markdown]
# ### 接口契约
#
# | 模拟器里的名字 | 语义 | vLLM v1 对应物 |
# | --- | --- | --- |
# | `Req` / `SimConfig` | 请求状态 + 调度参数 | `vllm/v1/request.py` 的 `Request`、`Scheduler` 构造参数 |
# | `BlockPool.ensure/release` | 块分配与回收 | `vllm/v1/core/block_pool.py` |
# | `schedule_step()` | 一步的调度决策 | `Scheduler.schedule()`（`vllm/v1/core/sched/scheduler.py`） |
# | `step_cost()` | 一步的耗时模型 | 真实 GPU 上 prefill/decode 的执行时间 |
# | `apply_step()` | 写 KV、采样、回收 | `Executor.execute_model()` + `GPUModelRunner` |
# | `preempted` | 本步被抢占的请求 | `Scheduler` 内部抢占逻辑（方法名待核对版本） |
#
# 为什么把 `schedule_step` 和 `apply_step` 分开？前者只做决策（可单测、可记录轨迹），
# 后者改变状态（写 KV、生成 token、释放块）。vLLM 也是这个分界：`Scheduler` 产出
# `SchedulerOutput`，`Worker` 才真正执行。
#
# 这个分界还有一层工程价值：**决策可以被单独录制、回放、断言，不需要 GPU**。
# 本章那 120 行模拟器能在一台 Mac 上验证 vLLM 的调度规则，靠的就是这条缝——
# 如果决策和执行缠在一起，你得先有一张 A100 才知道调度器写得对不对。
#
# ### 验收标准
#
# | 步骤 | 验收 |
# | --- | --- |
# | Step 1 数据结构 | 80 个请求都满足 `prefill_left == prompt_len` |
# | Step 2 块池 | `ensure` 失败时不占块；`release` 后 `free + used == total` |
# | Step 3 调度 | 每步 token 总数 ≤ `max_num_batched_tokens`；running ≤ `max_num_seqs` |
# | Step 4 模拟 | 所有请求都完成；峰值块 ≤ 块池容量；KV 不足时抢占次数 > 0 |
# | 实验 A/B/C | 四策略吞吐排序正确；轨迹第 7 步后进入纯 decode；sweep 单调 |

# %% [markdown]
# ## 09.3 从零实现：分四步
#
# 不抄源码，用约 120 行复刻 vLLM 调度器的决策逻辑，变量名与源码对齐。每步先讲目标和做法，
# 再给代码，最后给验收。
#
# | Step | 做什么 | 代码 |
# | --- | --- | --- |
# | 1 | 定义请求、配置与可复现负载 | `Req` / `SimConfig` / `make_workload` |
# | 2 | 写块池：分配、回收、峰值统计 | `BlockPool` |
# | 3 | 写一步调度：running 优先、预算、抢占 | `schedule_step` / `step_cost` / `apply_step` |
# | 4 | 推进时间轴并写对照基线 | `simulate` / `static_batch` / `summarize` |
#
# 先做环境准备：引入依赖、固定 seed，并确认概念图存在。

# %%
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
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

# %% [markdown]
# ### Step 1 · 数据结构与负载
#
# 目标：把「请求」和「调度参数」变成可迭代的对象，并造一份可复现的负载
# （80 请求、平均 40 req/s 到达、prompt 128–1024、输出 32–128）。
#
# 做法：`Req` 的字段与 vLLM `Request` 对齐（`prefill_left`、`num_generated`、`blocks`）；
# `SimConfig` 把调度参数与耗时模型放在一起，方便第 09.4 节做参数扫描。

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

# %% [markdown]
# **Step 1 验收**：80 个请求都满足 `prefill_left == prompt_len`；负载 prompt 均值约 608 token、
# 到达跨度约 1.8 s（见上方打印），后续实验可复现。

# %% [markdown]
# ### Step 2 · 块池
#
# 目标：用「16 token 一块」的记账方式回答「这个请求还能不能再来一个 token」。
#
# 做法：`ensure` 只在块够时扣块，失败必须保证块数不变（否则调度器会误判）；
# `release` 把块全部还回。`peak_used` 用于后面验证「峰值不超过容量」。

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

# %% [markdown]
# **Step 2 验收**：`free + used == total` 恒成立；`ensure` 失败时不改变块数。
# 后面实验 C 的 `peak_blocks <= total_blocks` 断言会再次覆盖这一点。

# %% [markdown]
# ### Step 3 · 一步调度
#
# 目标：复刻 `Scheduler.schedule()` 的两个阶段——先 running、再 waiting，并在块不足时抢占。
#
# 做法：`schedule_step` 只做决策，返回 `(请求, 本步 token 数)` 列表与被抢占集合；
# `step_cost` 把「算多少 token」翻译成毫秒；`apply_step` 负责写 KV、采样、回收。
# 三个函数各司其职，正好对应 vLLM 的 Scheduler / 时间模型 / Worker。

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

# %% [markdown]
# **Step 3 验收**：每步 token 总数不超过 `max_num_batched_tokens`，running 不超过 `max_num_seqs`；
# 被抢占的请求进入 `preempted` 集合，且不会在同一步被重新塞回 waiting。

# %% [markdown]
# ### Step 4 · 推进时间轴与对照基线
#
# 目标：把「一步」循环成「一次完整服务」，并补一个 05 章的静态批处理基线用于对照。
#
# 做法：`simulate` 只做推进与记录（可开关 trace）；`summarize` 统一计算吞吐、TTFT、
# p50/p99 延迟，保证四种方案口径一致；`static_batch` 是「凑满一批、跑完再收」的反例。

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
# **Step 4 验收**：`simulate` 的 `finished` 覆盖全部 80 个请求，`peak_blocks <= total_blocks`；
# 模拟器与静态批处理基线共用 `summarize` 口径，四种方案可以直接比较。

# %% [markdown]
# ## 09.4 实测
#
# 三个实验回答三个问题，全部基于 09.3 的模拟器（复刻调度决策，不是 GPU 执行）：
#
# | 实验 | 问题 | 产出 |
# | --- | --- | --- |
# | A | 四种调度策略差多少？ | 对比表 + 柱状图 |
# | B | 一步里 prefill / decode / KV 怎么变？ | 调度轨迹表 |
# | C | `max_num_seqs` 从 1 到 256，吞吐和尾延迟怎么走？ | 扫描表 + 双轴图 |
#
# 测量协议：负载 80 请求、平均 40 req/s 到达、固定 seed；时间模型
# `step_ms = 0.04 × prefill_tokens + 8.0 + 0.12 × decode_requests`；
# 调度决策开销用 `aiinfra.bench.bench(warmup=1, reps=5)` 在本机实测。
# **绝对值是模拟量级，趋势才是结论。**

# %% [markdown]
# ### 实验 A · 四种调度策略
#
# **先猜一下**：同一份 80 请求的负载，只把「静态批处理」换成「连续批处理」，吞吐能涨多少？
# (a) 20% 左右　(b) 1.5 倍　(c) 3 倍以上。猜完再看下面的表。
#
# 协议：同一份负载跑四种方案——静态批处理 b=8（05 章反例）、连续批处理 8/2048（05 章手写）、
# vLLM 默认 256/8192 且 KV 充足（A100 80G ≈ 24k 块）、vLLM 默认但 KV 只剩 400 块。
# 四种方案的指标都由 `summarize` 统一计算。

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

# %% [markdown]
# 把吞吐画成柱状图，差距更直观：

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
# **读数**：
#
# - **方案对比**：回收开头的猜测，答案是 (c)。静态批处理 488 tok/s → 连续批处理 677 → vLLM 默认 1634，
#   约 **3.3 倍**；KV 不足时掉回 641，并出现 21 次抢占。选 (a) 的直觉来自「不过是换了组批方式」——
#   但静态批处理在等最慢那条请求时，GPU 和 KV 都在空转；连续批处理把这段空转变成了别人的 token。
# - **原因**：decode 一步成本 ≈ `固定开销 + 每请求成本 × 并发数`。并发上不去就摊不薄固定开销；
#   而并发被 `max_num_seqs` 和 **KV 块数**双重限制。
# - **TTFT 1222 ms 不是 bug**：这份负载 prompt 均值 608、到达率 40 req/s，prefill 需求约
#   2.4 万 token/s，已经贴着模拟器的 prefill 容量（8192 / 335 ms ≈ 2.4 万 token/s）。
#   第 11 章的瓶颈判定就从这类现象开始。
#
# > **⚠️ 坑：把 `max_num_seqs` 当并发上限来规划容量。**
# > 表里 1634 tok/s 那一行的前提是 KV 充足。同样的参数把块池砍到 400 块，吞吐掉回 641（少 61%），
# > 还多出 21 次抢占。只记参数不记 KV，容量规划会高估一倍以上——参数是旋钮，KV 才是电源。

# %% [markdown]
# ### 实验 B · 调度轨迹：一步里发生了什么
#
# **先猜一下**：前几步单步 300–340 ms，这段时间主要在干什么？
# (a) decode 那几十个请求　(b) 8192 token 的 prefill 预算　(c) 调度器在算。
#
# 协议：用 vLLM 默认参数跑同一负载，记录前 12 步的 running / waiting / prefill token /
# decode 请求数 / 单步耗时 / KV 块数 / 抢占次数。
#
# 读数看三件事：prefill 与 decode 是否混批、每步时间由谁主导、KV 何时达到峰值。

# %%
res_trace = simulate(CFG_VLLM, make_workload(), total_blocks=24000, trace=True)
trace_df = pd.DataFrame(res_trace["trace"])
print(f"总迭代 {len(trace_df)} 步；KV 峰值 {res_trace['peak_blocks']} 块")
trace_df.head(12)

# %% [markdown]
# **读数**：回收上个猜测——答案是 (b)。前 6 步把 8192 token 预算几乎全花在 prefill 上，
# 8192 × 0.04 ms = 327.7 ms，和单步 300–340 ms 对得上；第 7 步后进入纯 decode，
# 一步 17.6 ms ≈ `8.0 + 0.12 × 80`。这就是「prefill 贵、decode 按并发摊薄」的量化版本，
# 也是 TTFT 和 TPOT 由不同阶段决定的直接证据。选 (a) 的直觉是「几十个请求都在等，肯定在
# decode」——实际上前几步连一个 decode 都塞不进去，8192 token 的预算全被 prefill 吃掉了。
#
# 更值得记的是 decode 这一行：80 个请求同时 decode，一步 17.6 ms；而单独一个请求，一步是
# 8.0 + 0.12 = 8.12 ms。把 80 个请求打包之后，单请求单 token 的成本从 8.12 ms 摊到
# 17.6 ÷ 80 ≈ 0.22 ms，便宜约 37 倍——**decode 不是「把 80 份活干完」，而是「同一份权重伺候 80 个人」**。

# %% [markdown]
# ### 实验 C · `max_num_seqs` 扫描与调度开销
#
# **先猜一下**：把 `max_num_seqs` 从 1 一路开到 256，吞吐会一路涨吗？会停的话，停在 32、64，
# 还是 256？
#
# 协议：固定 token 预算 8192、KV 充足，只改 `max_num_seqs`（1/2/4/8/16/32/64/128/256），
# 每个点跑完整模拟；再用 `bench` 实测 80 请求模拟的调度决策总耗时。
#
# 先看断言与调度开销——它是「调度器很便宜，贵的永远是前向和 KV 搬运」的证据：

# %%
assert len(res_vllm["finished"]) == 80, "所有请求都应完成"
assert res_vllm["peak_blocks"] <= 24000, "块数不能超过块池容量"
assert df_compare.loc[2, "吞吐(tok/s)"] > df_compare.loc[0, "吞吐(tok/s)"], "连续批处理应快于静态批处理"
assert df_compare.loc[3, "抢占次数"] > 0, "KV 不足时应发生抢占"

overhead = bench(lambda: simulate(CFG_VLLM, make_workload(), 24000),
                 name="simulate(80 reqs)", warmup=1, reps=5)
pd.DataFrame([overhead.as_dict()])

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

# %% [markdown]
# **读数**：
#
# - **调度开销**：本机实测 80 请求的完整模拟约 2 ms（纯 Python），比一步 GPU 前向（毫秒级）
#   还小两个数量级——调度本身不是瓶颈。
# - **`max_num_seqs` 扫描**：吞吐从 120 涨到 1660 tok/s，**13.8 倍**，靠的只是把并发上限拧开；
#   64 之后并发翻倍只涨 5.5%，随后完全持平。
# - **回收猜测**：吞吐确实停了，停在 64。猜 32 的人把它和拐点混了（拐点问的是「再加并发值不值」，
#   不是「吞吐何时封顶」）；猜 256 的人把参数上限当成了吞吐上限。这个 64 **不是硬件上限，是这份负载的上限**——它只有
#   80 个请求，128 个并发槽位已经能一次全装下；真实的硬件拐点要用持续的到达流压测（第 11 章）。
#   p99 在欠载下先降后平，过载时才会随并发上升。真正的容量天花板是 KV：见实验 A 的「KV 不足」那一行。
#
# **对 vLLM 参数的启示**：`max_num_seqs` 是吞吐旋钮，但不是越大越好；先确认 KV 够
# （`gpu_memory_utilization`、`max_model_len`），再按 SLO 的 p99 反推并发上限。
#
# 到这里我们知道了「参数该调多大」，还不知道「这些决策在真实 vLLM 里写在哪个文件」。
# 09.5 把模拟器的每个函数对到真实源码上，包括一条可以照着下断点的调用链。

# %% [markdown]
# ## 09.5 对照工业实现（vLLM 0.28.x）
#
# 09.3 的模拟器只覆盖「决策」，这一节做**源码地图终检**：把我们的每个类映射到 vLLM v1 的真实文件，
# 并给出离线也能执行的版本核对命令。
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
#
# 这 6 个位置不是「重要性排序」，而是**读码顺序**：从上往下，每读完一个文件，
# 你都能回答一个具体问题，下一个文件的问题正好是它引出来的。
#
# 1. `vllm/v1/engine/` —— 请求从哪进来、怎么变成 `EngineCoreRequest`（先看 `AsyncLLM.add_request`）；
# 2. `vllm/v1/core/sched/scheduler.py` —— 谁被选进这一步（`schedule()` 的两个循环）；
# 3. `vllm/v1/core/kv_cache_manager.py` —— 选中之后，块从哪来（`allocate_slots` / `get_computed_blocks`）；
# 4. `vllm/v1/core/block_pool.py` —— 池里还剩几块、块哈希怎么算（抢占失败就发生在这一层）；
# 5. `vllm/v1/worker/` —— 决策怎么变成张量（`execute_model` 拼 batch、`prepare_inputs` 建 block table）；
# 6. `vllm/model_executor/` —— 张量走进具体模型，这时才开始真正的矩阵乘法。
#
# 前四个文件决定「算谁」，后两个决定「怎么算」。**绝大多数性能与正确性问题，答案在前四个里**。

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
# 这 7 步就是 09.2 生命周期图里的箭头。想验证自己在读对版本，跑下面五条命令，比背路径可靠：
#
# 想动手的话，三个断点足够把这条链走通，而且不用先读完任何文件：
#
# 1. `Scheduler.schedule()` 入口：看这一步的 `SchedulerOutput` 里带了几条请求、多少 prefill token；
# 2. `KVCacheManager.allocate_slots()`：看一个新请求要几个块、块池还剩多少——抢占就是在这里失败的；
# 3. `GPUModelRunner.execute_model()`：看 `SchedulerOutput` 被翻译成什么形状的张量。
#
# 一个新手常犯的选择：第一反应去模型 `forward` 里下断点。那里每层都会停一次，你会被自己
# 打断几十次，而真正决定行为的东西（谁进 batch、块从哪来）在它上游早就定完了。
#
# ```bash
# python -c "import vllm; print(vllm.__version__)"            # 期望 0.28.x
# python -c "from vllm.v1.core.sched.scheduler import Scheduler; print(Scheduler)"
# python -c "from vllm.v1.core.kv_cache_manager import KVCacheManager; print(KVCacheManager)"
# python -c "from vllm.v1.core.block_pool import BlockPool; print(BlockPool)"
# find vllm/v1/engine -maxdepth 1 -name "*.py" | sort        # 对照 09.5 的目录表
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
#
# 换个说法：那上万行里，真正产生 token 的部分，就是我们这两百行写过的那些；
# 其余的部分都在做同一件事——**让那两百行在生产里不把事情搞砸**。

# %% [markdown]
# ## 09.6 练习与小结
#
# **练习**（⭐ 改参数 / ⭐⭐ 改实现 / ⭐⭐⭐ 开放设计）
#
# 1. ⭐ 把 `CFG_VLLM.max_num_batched_tokens` 从 8192 改成 2048，重跑 09.4 实验 A 与实验 C 的表。吞吐和 TTFT 各变多少？先用 token 预算解释，再跑一遍验证。
# 2. ⭐ 把 `decode_ms_per_seq` 从 0.12 改成 0.02（模拟更快的卡），观察 09.4 实验 C 的拐点位置向哪边移动。为什么？
# 3. ⭐⭐ 给 `schedule_step` 加一个「优先级队列」：短 prompt 优先。在长 prompt 混短 prompt 的负载下比较 p50 / p99 延迟，并解释为什么吞吐基本不变。
# 4. ⭐⭐ 把抢占从 recompute 改成 swap（把被抢请求的块「换出」而不是丢弃，稍后再换回），对比 `抢占次数 × prefill 成本` 与换入换出的搬运成本。需要给 `Req` 加字段。
# 5. ⭐⭐⭐ 设计一条读码路线：线上现象是「首 token 很慢，但 TPOT 正常」。你会在 09.5 的哪几个文件加日志/下断点？给出验证步骤与预期证据。（评分要点：定位到 prefill/调度层、给出可观测指标、区分 KV 不足与排队的证据、有回滚方案）
#
# **你现在应该能回答：**
#
# - 为什么 `Scheduler` 和 `KVCacheManager` 要拆成两个类？各自的不变量是什么？
# - `max_num_seqs` 和 `max_num_batched_tokens` 分别卡住了什么？调大哪个会把 p99 推高？
# - 为什么 KV 不足时吞吐会「掉回去」，而不仅是变慢？
#
# **自检 3 问**（不看笔记能答出来才算过）
#
# 1. `Scheduler` 和 `KVCacheManager` 各自的不变量是什么？如果只能给其中一个加日志，你选哪个？
# 2. `max_num_seqs=256` 一定比 64 快吗？在什么条件下这个不等式会反过来？
# 3. 一个请求的 TTFT 很高、TPOT 正常，你先查调度、查 KV，还是查网络？为什么？
#
# > **一句话记住**：vLLM 多出来的那一万行，几乎都在保证「每一步的决策不把钱花错位置」。
#
# **真实工作场景题**
#
# 线上告警：`vllm:num_requests_waiting` 长时间大于 0，p99 TTFT 抬头，但吞吐没涨。
# 手上有 09.5 的调用链和 11 章的五分支决策树。写出你的排查顺序（先看哪两个指标、
# 怎么区分「并发上限太小」和「KV 不够」），以及每一步的预期证据。
# **小结与下一章**
#
# 本章我们把 02–06 章的零件映射到了 vLLM 的真实文件与类名，并用一个同构模拟器验证了调度规则。
# 下一章我们把服务真正跑起来（Mac 用 `vllm-metal`，云上用 Docker），用 OpenAI 兼容 API 发请求，
# 看看这些内部对象在 HTTP 层长什么样。

# %% [markdown]
# **延伸阅读**
#
# - vLLM v1 引擎设计说明（`docs/` 下的 v1 架构文档，版本随 0.28.x）；
# - `vllm/v1/core/sched/scheduler.py` 的 `schedule()`：对照本章 `schedule_step` 读；
# - `vllm/v1/core/kv_cache_manager.py` 的 `allocate_slots()`：对照 03 章的 `BlockManager.allocate` 读；
# - 第 10 章：OpenAI 兼容 API、LoRA、结构化输出与工具调用。
