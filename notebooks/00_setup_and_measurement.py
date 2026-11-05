# %% [markdown]
# # 第 00 章 · 环境、硬件与度量方法
#
# > AI Infra 的第一课不是跑模型，而是**怎么测**。测不准，后面所有调优都是玄学。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 60 分钟 |
# | 运行环境 | 🏠 Mac（CPU 或 MPS）· 无需 GPU |
# | 前置章节 | 无（先读 `STYLE_GUIDE.md` 了解本课程的实验口径） |
# | 本章产物 | 一个可复用的 `bench()`、环境快照、以及对"测量误差"的直觉 |
#
# **学完你能回答：**
# 1. 为什么同一段代码跑两次，延迟能差 2 倍？哪些误差是可控的？
# 2. TTFT、TPOT、p50、p95 分别在描述什么？什么时候该看哪个？
# 3. 并发从 1 加到 16，吞吐为什么不是线性涨？用 Little's law 解释。

# ## 00.0 本章目标与路线
#
# **一句话目标**：建立一套**可信、可复用**的测量方法——后面 14 章的所有结论都靠它。
# 没有统一口径，调优就是玄学。
#
# **本章交付**：
#
# | 交付物 | 在哪一节 | 怎么验收 |
# | --- | --- | --- |
# | 环境快照（芯片、内存、版本） | 00.2 | 报告第一行能复现 |
# | 可信计时器（预热/同步/分位数） | 00.3 | 与 `aiinfra.bench.bench()` 结果一致 |
# | 延迟分布与尾部 | 00.4 | p90 与 p50 的差距可解释 |
# | Little's law 推演 | 00.5 | 吞吐-并发曲线符合公式 |
#
# **阅读路线**：00.1 为什么测量会骗人 → 00.2 环境快照 → 00.3 从零写计时器 →
# 00.4 分位数 → 00.5 吞吐与延迟 → 00.6 对照 vLLM → 00.7 练习。

# %% [markdown]
# ## 00.1 问题与直觉
#
# 假设你测出 vLLM 的 TTFT 是 120ms，这个数字可信吗？在回答之前，先看它经过了多少层"污染"：
#
# ![](../assets/figures/ch00_measurement.svg)
#
# 一次可信的测量 = **预热**（让 kernel 编译、缓存、频率稳定）+ **同步**（GPU 是异步的，不同步等于只测了"下单时间"）+ **重复**（单次采样没有统计意义）+ **分位数**（均值会被长尾污染）+ **环境快照**（换台机器数字就不可比）。
#
# 这一章我们把这些做成工具，后面 14 章都用同一把尺子。

# %% [markdown]
# ## 00.2 环境快照
#
# 任何实验报告的第一行都应该是「在哪台机器上测的」。硬件、torch 版本、设备类型不同，数字不可比。
#
# %%
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path.cwd().parent))  # 从 notebooks/ 回到仓库根目录

from aiinfra.bench import env_report, human_bytes, pick_device, sync
from aiinfra.plotting import save, setup_style

setup_style()
device = pick_device()
torch.manual_seed(0)
np.random.seed(0)

info = env_report()
pd.DataFrame(sorted(info.items()), columns=["项目", "值"])

# %% [markdown]
# 把这张表贴在实验报告开头，跨机器比较时才不会得出"A 比 B 快 3 倍"这种错觉。
#
# 注意 `device`：Mac 上是 `mps`，云上是 `cuda`。**MPS 与 CUDA 都是异步执行**，这正是下一个坑。

# %% [markdown]
# ### 坑 1：不同步 = 只测到了"发射时间"
#
# GPU/MPS 的算子调用是"下单即返回"，真正计算在设备上排队。如果不在结束前同步，CPU 计时器会提前停表。

# %%
x = torch.randn(2048, 2048, device=device)


def one_matmul() -> torch.Tensor:
    return x @ x


for _ in range(5):  # 先预热，排除首轮 kernel 编译的干扰
    one_matmul()
sync(device)

N = 50


def time_loop(sync_at_end: bool) -> float:
    sync(device)
    t0 = time.perf_counter()
    for _ in range(N):
        one_matmul()
    if sync_at_end:
        sync(device)  # 等设备真正算完
    return (time.perf_counter() - t0) / N * 1000


no_sync_ms = time_loop(sync_at_end=False)
with_sync_ms = time_loop(sync_at_end=True)

print(f"不同步: {no_sync_ms:8.3f} ms/次")
print(f"已同步: {with_sync_ms:8.3f} ms/次")
print(f"低估倍数: {with_sync_ms / max(no_sync_ms, 1e-9):6.1f}x")
assert with_sync_ms > 0
# 在 MPS / CUDA 上，不同步的版本只测到了 CPU 发射算子的时间，
# 实测通常低估 10–50 倍（取决于算子大小与队列深度）；CPU 上两者接近。

# %% [markdown]
# **结论**：凡是报告 GPU/MPS 时间的地方，`sync()` 必须出现。本课程的 `aiinfra.bench.bench()` 已经内置了这件事。

# %% [markdown]
# ## 00.3 从零实现一个可信的计时器
#
# 需求很简单，但细节决定成败：
#
# | 设计点 | 做法 | 不这么做会怎样 |
# | --- | --- | --- |
# | 预热 | 先空跑 3 次 | 第一次包含 kernel 编译/内存分配，虚高 |
# | 同步 | 每次计时前后 `sync()` | 测成 CPU 发射时间 |
# | 重复 | 至少 5 次 | 单次结果就是噪声 |
# | 统计 | 中位数 + p90 + min | 均值被偶发卡顿拉高 |
# | 命名 | 每个结果带 name | 一堆数字分不清谁是谁 |

# %%
import statistics
import time


def my_bench(fn, warmup: int = 3, reps: int = 5, name: str = "op") -> dict:
    """最小可用版：预热 → 重复计时 → 分位数。"""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        sync(device)  # 开始前确保设备空闲
        t0 = time.perf_counter()
        fn()
        sync(device)  # 结束后确保真的算完
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return {
        "name": name,
        "median_ms": round(statistics.median(samples), 3),
        "p90_ms": round(samples[int(len(samples) * 0.9) - 1], 3),
        "min_ms": round(samples[0], 3),
        "reps": reps,
    }


result = my_bench(one_matmul, warmup=3, reps=10, name="2048x2048 matmul")
pd.DataFrame([result])

# %% [markdown]
# 正式版在 `aiinfra/bench.py` 的 `bench()`，接口一致，多了 p90/min/mean 的整理。后面所有章节都用它，**不要**再手写 `time.time()`。

# %% [markdown]
# ## 00.4 实测：分位数与噪声
#
# 同一段代码跑 50 次，画出分布。你会看到：大部分样本挤在左边，少数样本拖出长尾——用户感知到的"卡顿"恰恰来自右尾。

# %%
samples = []
for _ in range(50):
    sync(device)
    t0 = time.perf_counter()
    one_matmul()
    sync(device)
    samples.append((time.perf_counter() - t0) * 1000)

s = pd.Series(samples)
summary = pd.DataFrame(
    {
        "mean": [s.mean()],
        "p50": [s.quantile(0.50)],
        "p90": [s.quantile(0.90)],
        "p99": [s.quantile(0.99)],
        "max": [s.max()],
        "cv(σ/μ)": [s.std() / s.mean()],
    }
).round(3)
summary

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.hist(s, bins=20, color="#2563EB", alpha=0.75, edgecolor="white")
ax.axvline(s.median(), color="#10B981", ls="--", lw=2, label=f"p50 = {s.median():.1f} ms")
ax.axvline(s.quantile(0.90), color="#F59E0B", ls="--", lw=2, label=f"p90 = {s.quantile(0.90):.1f} ms")
ax.set_xlabel("Latency per matmul (ms)")
ax.set_ylabel("Count")
ax.set_title("Latency over 50 runs: the tail defines user experience")
ax.legend()
save(fig, "ch00_latency_dist");

# %% [markdown]
# 读图方法：
#
# - **p50 和中位数接近** → 分布稳定，可以用中位数代表典型情况。
# - **p90 远大于 p50** → 存在抖动（频率切换、内存回收、其他进程），生产上要按 p90 做 SLO。
# - **cv > 0.2** → 这台机器/这个负载的测量结果不可直接横向比较，先找噪声源。
#
# 生产环境里，vLLM 暴露的也是 `ttft`/`tpot` 的直方图而不是均值，原因相同。
#
# > **小模型测量地板**：在 MPS 上，一次小 forward 的固定开销约 1–1.5ms（kernel 启动 + 命令缓冲）。
# > 这个地板会吃掉所有"减少计算量"的优化收益：前缀缓存理论上能省 65x，实测可能只有 2–4x。
# > 所以后面每章都同时给**理论值**和**实测值**，并注明固定开销占比；必要时加 CPU 对照。

# %% [markdown]
# ## 00.5 实测：吞吐与延迟（Little's law）
#
# 吞吐（req/s）= 并发请求数 ÷ 平均延迟（s）。这是排队论里最简单的恒等式，却解释了 AI Infra 的一半现象：
#
# - 加并发 → 吞吐涨 → 但单请求延迟也涨 → 涨到某个点，吞吐不再涨，延迟失控。
# - 因为 GPU 的算力/带宽是固定的，队列只会把等待时间变长。

# %%
rows = []
for concurrency in [1, 2, 4, 8, 16, 32]:
    latency_ms = 20 + 1.2 * concurrency  # 排队模型示意：并发越高，单请求越慢
    throughput = concurrency / (latency_ms / 1000)
    rows.append(
        {
            "concurrency": concurrency,
            "latency_ms": latency_ms,
            "throughput_rps": round(throughput, 2),
        }
    )
df = pd.DataFrame(rows)
df

# %%
fig, ax1 = plt.subplots()
ax1.plot(df["concurrency"], df["throughput_rps"], "o-", color="#2563EB", label="Throughput")
ax1.set_xlabel("Concurrent requests")
ax1.set_ylabel("Throughput (req/s)", color="#2563EB")
ax1.set_xscale("log", base=2)

ax2 = ax1.twinx()
ax2.plot(df["concurrency"], df["latency_ms"], "s--", color="#EF4444", label="Latency")
ax2.set_ylabel("Latency per request (ms)", color="#EF4444")
ax2.grid(False)

ax1.set_title("Little's law: throughput climbs, latency starts to rise")
save(fig, "ch00_littles_law");

# %% [markdown]
# 注意这张图是**模型推演**，不是实测。它给你一个预期：真实系统会在某个并发点突然"拐弯"（KV cache 用完、调度排队、显存不足），
# 第 05 章的连续批处理实验会亲手测出那个拐点。

# %% [markdown]
# ## 00.6 对照工业实现（vLLM）
#
# 本课程锁定 **vLLM 0.28.x**。它把上面的口径固化成了 CLI 与指标：
#
# | 我们的概念 | vLLM 中的名字 | 位置 |
# | --- | --- | --- |
# | 预热 | benchmark 的 warmup 阶段 | `vllm/benchmarks/serve.py` |
# | TTFT | `ttft`（time to first token） | `vllm/engine/metrics.py` |
# | TPOT | `tpot`（time per output token） | 同上 |
# | p50/p90/p99 | `--metric-percentiles 50,90,99` | `vllm bench serve` 参数 |
# | 实时并发 | `vllm:num_requests_running` | Prometheus 指标 |
#
# 云上（或装了 `vllm-metal` 的 Mac）可以直接跑：
#
# ```bash
# vllm bench serve --model Qwen/Qwen3-0.6B \
#   --num-prompts 200 --max-concurrency 16 \
#   --metric-percentiles 50,90,99
# ```
#
# 它输出的 `Mean TTFT` / `P99 TPOT` 就是本章 `my_bench()` 思想的工业版：
# 预热、分位数、并发扫描，一个都不少。

# %% [markdown]
# ## 00.7 练习与小结
#
# **练习**
#
# 1. ⭐ 把 `my_bench` 的 `reps` 从 5 改成 50，观察 p90 与 p50 的差距怎么变。为什么重复越多，p90 越"稳定"？
# 2. ⭐ 在 MPS 上分别用 `torch.float32` 和 `torch.bfloat16` 跑同一个 matmul，记录延迟。bfloat16 一定更快吗？为什么？
# 3. ⭐⭐ 写一个 `bench_steady(fn)`：先跑 2 秒"热身"，再用滑动窗口检测频率是否稳定，不稳定就丢弃前 N 次。
# 4. ⭐⭐ 用 Little's law 反推：如果业务要求 p95 延迟 < 200ms、QPS = 50，至少需要多少并发容量？假设单请求串行服务。
# 5. ⭐⭐⭐ 设计一个实验，区分"测量噪声"和"真实的性能退化"。提示：固定随机种子、同一进程内交替 A/B、看置信区间。
#
# **你现在应该能回答：**
#
# - 为什么 `sync()` 是计时的一部分，而不是可选项？
# - 为什么生产 SLO 用 p95/p99，而调优时先看 p50？
# - 吞吐与延迟为什么不能同时无限优化？
#
# **延伸阅读**
#
# - vLLM 官方 benchmark 文档：`vllm bench serve` 的参数与输出字段
# - Little's law 的原始表述（排队论 / 利特尔法则）
# - 下一章：我们要真正跑一次 Transformer 前向，看看「算力」和「带宽」谁先成为瓶颈。
