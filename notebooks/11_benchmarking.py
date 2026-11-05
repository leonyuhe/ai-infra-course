# %% [markdown]
# # 第 11 章 · 压测与调优
#
# > 服务能跑起来只是起点。这一章用回放的压测数据 + 一个排队模型，回答三个问题：**拐点在哪、瓶颈是谁、该调哪个参数**。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 75–90 分钟 |
# | 运行环境 | 🏠 Mac（纯回放，无需 GPU / 无需服务） |
# | 前置章节 | 09（架构与源码地图）、10（服务与接口契约） |
# | 数据来源 | `cloud/fixtures/benchmark_results.json`（示例录制，命令与字段同 `vllm bench serve`；非本机实测） |
# | 本章产物 | 指标定义表、拐点判据与 SLO 容量、并发扫描表/曲线、排队模型、调优清单与决策树 |
# | 本章接口 | `percentiles / itl_ms / request_metrics`；`df_sweep` / `df_all` 数据表 |
#
# **学完你能回答：**
# 1. TTFT、TPOT、ITL、吞吐、p50/p90/p99 分别怎么定义？各自对应 vLLM 的哪个指标？
# 2. 并发从 1 加到 32、64，吞吐什么时候到拐点？拐点的判据是什么？
# 3. 吞吐上不去时，怎么用几条命令区分「算力 / 带宽 / KV / 排队 / 网络」？
# 4. 给定 SLO，怎么从压测数据反推容量，并知道该调哪个参数？

# %% [markdown]
# ## 11.0 本章目标与路线
#
# **一句话目标**：用「指标定义 → 并发扫描 → 拐点判据 → 排队模型 → 决策树」把压测数据
# 变成可执行的调优动作；不改变任何实验结论与断言。
#
# **代价与边界**：
# - 数据来自 `cloud/fixtures/benchmark_results.json`（示例录制，非本机实测），只用于趋势与判据演示；
#   生产结论必须用自己的 `vllm bench serve` 数据重录；
# - 排队模型是一阶近似（`step(n) ≈ a + b·n`），只在低并发段可信，高并发会低估延迟；
# - 本章不启动服务、不联网、不需要 GPU。
#
# **本章交付物**：
#
# | # | 交付物 | 在哪一节 | 怎么验收 |
# | --- | --- | --- | --- |
# | 1 | 指标定义表（TTFT/TPOT/ITL/吞吐/分位/KV/队列） | 11.2 | 每个指标能说出定义、单位、诊断什么 |
# | 2 | 拐点判据 + SLO 容量 | 11.3 Step 3 | 拐点 = 32，SLO 容量 = 32 |
# | 3 | 并发扫描表与三张曲线 | 11.4 实验 A–C | 吞吐 52 → 862 tok/s，p99 TTFT 68 → 2600 ms |
# | 4 | 排队模型与误差 | 11.4 实验 B | `step ≈ 16.49 + 0.663n`，R² = 0.993 |
# | 5 | 调优清单 + 五分支决策树 | 11.5 | 每个参数写明调大/调小代价 |
#
# **阅读路线**：11.1 问题 → 11.2 高阶设计（指标 + 决策树）→ 11.3 分四步实现
# → 11.4 实验 A/B/C → 11.5 对照 vLLM → 11.6 练习。

# %% [markdown]
# ## 11.1 问题与直觉：为什么「平均延迟」和「单点压测」会骗人
#
# 压测最常见的两个错误：
#
# 1. **只看平均延迟**：均值把长尾稀释掉，而用户投诉的恰恰是 p99；
# 2. **只压一个并发**：单点数字不告诉你「拐点在哪、再加并发会不会亏」。
#
# **反例**：只看 `nvidia-smi` 的 SM 利用率，看到 95% 就以为「卡到极限了」。
# decode 阶段 SM 高但有效算力低，瓶颈可能是显存带宽；调大 batch 只会更慢。
#
# 本章的整章路线是一棵诊断树：从「吞吐上不去 / 尾延迟超标」出发，五条分支按
# 「从便宜到昂贵」排序，每条都给一条能直接粘贴的命令（诊断树见 11.2）。
#
# 判断之前先要有数据，而数据的口径取决于压测方式：
# - **闭环**（固定并发、等响应再发下一条）：测的是「这个并发下的稳态」；
# - **开环**（按固定速率发）：测的是「系统能否扛住到达率」。
# `vllm bench serve --max-concurrency` 是闭环；本章的扫描都用闭环。

# %% [markdown]
# | 维度 | 错误做法 | 正确做法 | 对应指标 |
# | --- | --- | --- | --- |
# | 延迟 | 只看均值 | 看 p50/p90/p99 三件套 | TTFT / TPOT / ITL 分位数 |
# | 负载 | 只压 1 个并发 | 并发扫描 1 → 64，找拐点 | 吞吐、p99、队列长度 |
# | 瓶颈 | 凭感觉调参 | 五分支决策树 + 命令取证 | SM / DRAM / KV / waiting |
# | 容量 | 取吞吐最高点 | 取 SLO 内的最大并发 | goodput、SLO 命中 |
#
# 本章要解决的两个子问题：
#
# - **正确性**：回放数据是否自洽（Little's law / 吞吐恒等式）？拐点判据是否可复现？
# - **成本**：每加一档并发，吞吐赚多少、尾延迟付多少？该调哪个参数？

# %% [markdown]
# ## 11.2 高阶设计：指标、决策树与验收标准
#
# 动手前先把「量什么」和「怎么判断」定下来。整章只有三个组件。
#
# **组件 1：指标定义（先定义，再测量）**
#
# | 指标 | 定义 | 单位 | 诊断什么 |
# | --- | --- | --- | --- |
# | TTFT | 请求发出 → 首 token 返回 | ms | prefill 慢 / 排队 / 网络 |
# | TPOT | 首 token 后平均每个输出 token 的耗时 `(末−首)/(n−1)` | ms | decode 是否被 batch 拖慢 |
# | ITL | 相邻输出 token 的时间差序列 | ms | 卡顿、抢占、调度抖动 |
# | 吞吐 | 单位时间输出 token 数 / 请求数 | tok/s、req/s | 卡有没有被喂饱 |
# | p50 / p90 / p99 | 延迟分位数 | ms | SLO 与容量规划 |
# | KV 使用率 | 已用块 / 总块 | % | 是否 KV 不足、要不要抢占 |
# | 队列长度 | 等待中请求数 | 个 | 调度排队 / 并发上限太小 |
# | 运行中请求 | 正在 prefill/decode 的请求数 | 个 | 实际并发 vs `max_num_seqs` |
# | 抢占次数 | 因 KV 不足被抢占的累计次数 | 次 | KV 不足 / 长请求挤压短请求 |

# %% [markdown]
# **组件 2：瓶颈判定决策树**
#
# ![](../assets/figures/ch11_decision_tree.svg)
#
# 五条分支按「先排除客户端与网络（最便宜）→ 再看排队 → 最后才动模型与并行度」排序，
# 每条都给一条可粘贴的诊断命令：
#
# 1. **算力 bound**（SM > 90% 且功耗贴顶）：`nvidia-smi dmon -s u -d 1`；对策：量化、TP、换更小模型。
# 2. **带宽 bound**（DRAM > 85%、SM 不高，decode 典型）：`nvidia-smi --query-gpu=utilization.memory --format=csv -l 1`；对策：fp8/int8、加大 batch 摊薄权重读取。
# 3. **KV 不足**（`kv_cache_usage_perc > 0.9` 或日志有 Preemption）：`curl -s localhost:8000/metrics | grep kv_cache_usage`；对策：调高显存占比、减 `max-model-len`、开 prefix caching、加卡。
# 4. **调度排队**（`num_requests_waiting > 0` 且 running 到上限）：`curl -s localhost:8000/metrics | grep num_requests_waiting`；对策：调大 `max-num-seqs` / `max-num-batched-tokens`。
# 5. **网络 / 客户端**（服务端 TPOT 正常但首包抖）：`curl -o /dev/null -s -w 'connect=%{time_connect} ttft=%{time_starttransfer}' localhost:8000/health`；对策：异步 + keep-alive、流式、排查 TLS/代理。
#
# 一个经验法则：**先有数字，再改参数**。没有 metrics 或 `nvidia-smi` 证据的调优，只是在赌。

# %% [markdown]
# **组件 3：三个计算件（指标 → 判据 → 模型）**
#
# - `percentiles / itl_ms / request_metrics`：把 token 时间戳变成指标；
# - 拐点判据：吞吐单步增益 < 10% 且 p99 TTFT 增幅 > 50%；
# - 排队模型：`step(n) ≈ a + b·n`，拟合前 5 个并发点后预测吞吐与延迟。
#
# 数据流：
#
# ```text
# token 时间戳 → 指标（TTFT/TPOT/ITL）→ Little's law 自检 → 并发扫描
#             → 拐点 / SLO → 排队模型 → 调参动作（决策树）
# ```
#
# ### 验收标准
#
# | 步骤 | 验收 |
# | --- | --- |
# | Step 1 指标函数 | 128 个 token 的 TPOT ∈ (16, 20) ms；p50 < p99 |
# | Step 2 回放自检 | Little's law 与吞吐恒等式误差 < 2% |
# | Step 3 拐点 / SLO | 拐点 = 32；SLO 容量 = 32 |
# | Step 4 排队模型 | R² > 0.98；并发 ≤ 32 时平均误差 < 8% |
# | 实验 A/B/C | 表 / 图 / 读数一一对应，拐点与 SLO 结论一致 |

# %% [markdown]
# ## 11.3 从零实现：分四步
#
# | Step | 做什么 | 代码 |
# | --- | --- | --- |
# | 1 | 定义指标函数并验证单请求 | `percentiles` / `itl_ms` / `request_metrics` |
# | 2 | 加载回放数据并做 Little's law 自检 | fixture + `np.allclose` |
# | 3 | 拐点判据与 SLO 容量 | `pct_change` + 阈值 |
# | 4 | 排队模型拟合与误差 | `np.polyfit` + 预测 |
#
# 先做环境准备：引入依赖、固定 seed，并确认回放数据文件存在。

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent))

from aiinfra.bench import bench
from aiinfra.plotting import save, setup_style

setup_style()
np.random.seed(0)
ROOT = Path.cwd().parent
FIXTURES = ROOT / "cloud" / "fixtures"
print("repo root:", ROOT)
assert (FIXTURES / "benchmark_results.json").exists()

# %% [markdown]
# ### Step 1 · 指标函数
#
# 目标：把「token 时间戳」变成 TTFT / TPOT / ITL 三个指标，并证明计算本身很便宜。
#
# 做法：`itl_ms` 取相邻 token 时间差；`request_metrics` 用 ITL 均值作为 TPOT；
# `percentiles` 统一分位数口径。先造一条 128 token 的模拟时间戳验证定义，
# 再在本机实测 1 万个 ITL 的分位数计算开销。

# %%
def percentiles(samples, qs=(50, 90, 99)) -> dict:
    arr = np.asarray(samples, dtype=float)
    return {f"p{q}": round(float(np.percentile(arr, q)), 1) for q in qs}


def itl_ms(token_times_ms) -> np.ndarray:
    """ITL：相邻输出 token 的时间差（毫秒）。"""
    gaps = np.diff(np.asarray(token_times_ms, dtype=float))
    assert gaps.size >= 1, "至少要有 2 个 token 才能算 ITL"
    return gaps


def request_metrics(ttft_ms: float, token_times_ms) -> dict:
    gaps = itl_ms(token_times_ms)
    return {"ttft_ms": round(ttft_ms, 1), "tpot_ms": round(float(gaps.mean()), 1),
            "itl_p50_ms": round(float(np.percentile(gaps, 50)), 1),
            "itl_p99_ms": round(float(np.percentile(gaps, 99)), 1)}

# %%
rng = np.random.RandomState(0)
gaps = np.clip(rng.normal(18.0, 2.0, 127), 12.0, 30.0)   # 128 个输出 token、TPOT≈18ms
token_times = 180.0 + np.cumsum(gaps)                    # 首 token 出现在 180ms
m = request_metrics(180.0, token_times)
print("单请求指标（模拟 128 个 token 的时间戳）：", m)
assert 16 < m["tpot_ms"] < 20
assert m["itl_p50_ms"] < m["itl_p99_ms"]
pd.DataFrame([m])

# %% [markdown]
# 指标计算本身很便宜——便宜到可以对每个请求都算一遍、不需要抽样。下面在本机实测：

# %%
import time

many_gaps = np.clip(rng.normal(18.0, 2.0, 9999), 12.0, 30.0)
overhead = bench(lambda: percentiles(many_gaps), name="p50/p90/p99 of 9999 ITL",
                 warmup=3, reps=10)
pd.DataFrame([overhead.as_dict()])

# %% [markdown]
# **Step 1 验收**：128 个 token 的 TPOT 落在 16–20 ms，ITL 的 p50 < p99；
# 1 万个 ITL 的分位数计算在本机是亚毫秒级，逐请求计算指标完全可行。

# %% [markdown]
# ### Step 2 · 加载回放数据并自检
#
# 目标：把压测结果加载成 DataFrame，并证明它值得分析。
#
# 做法：`provenance` 字段会明确告诉你数据来源（本章是示例录制，不是本机实测）；
# 一条压测数据如果连 `吞吐 = 并发 ÷ 平均延迟` 都对不上，就不值得分析——把这两条恒等式
# 写成断言，是压测报告的第一道质检。

# %%
BENCH = json.loads((FIXTURES / "benchmark_results.json").read_text(encoding="utf-8"))
OUT_LEN = BENCH["config"]["output_len"]
df_sweep = pd.DataFrame(BENCH["sweep"]).copy()
df_ext = pd.DataFrame(BENCH["extended"]).copy()
df_param = pd.DataFrame(BENCH["param_sweep"]).copy()

print("数据来源:", BENCH["provenance"]["note"][:56], "...")
print("录制命令:", BENCH["provenance"]["command"])
print(f"模型={BENCH['config']['model']} 输入={BENCH['config']['input_len']} "
      f"输出={OUT_LEN} 卡={BENCH['provenance']['hardware']}")
assert len(df_sweep) == 5 and set(df_sweep["concurrency"]) == {1, 4, 8, 16, 32}

# %% [markdown]
# 自检口径：
#
# - Little's law：`request/s ≈ 并发 ÷ 平均延迟`；
# - 吞吐恒等式：`output tok/s ≈ request/s × 输出长度`。

# %%
df_sweep["req_s_little"] = df_sweep["concurrency"] / df_sweep["mean_latency_s"]
df_sweep["tok_s_from_req"] = df_sweep["request_throughput_req_s"] * OUT_LEN
assert np.allclose(df_sweep["req_s_little"], df_sweep["request_throughput_req_s"], rtol=0.02)
assert np.allclose(df_sweep["tok_s_from_req"], df_sweep["output_throughput_tok_s"], rtol=0.02)
print("Little's law 与吞吐恒等式自检通过（容差 2%）")
df_sweep[["concurrency", "mean_latency_s", "request_throughput_req_s", "output_throughput_tok_s"]]

# %%
cols = ["concurrency", "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms", "tpot_p50_ms",
        "itl_p99_ms", "output_throughput_tok_s", "kv_cache_usage_pct_peak",
        "num_requests_waiting_max", "gpu_sm_util_pct", "gpu_dram_util_pct"]
df_sweep[cols]

# %% [markdown]
# **Step 2 验收**：5 个并发点 + 1 个加测点加载成功；Little's law 与吞吐恒等式在 2% 容差内通过，
# 数据自洽，可以进入拐点分析。

# %% [markdown]
# ### Step 3 · 拐点判据与 SLO 容量
#
# 目标：从扫描曲线里找出「再翻倍并发就亏」的那个点，并算出 SLO 内能开多少并发。
#
# 做法：拐点不是吞吐最高点，而是满足两条判据的点——吞吐单步增益 < 10%，且 p99 TTFT 增幅 > 50%；
# SLO 容量是另一个问题：给定 `p99 TTFT ≤ 1000ms`、`TPOT ≤ 40ms`，取仍能命中的最大并发。

# %%
df_all = pd.concat([df_sweep, df_ext], ignore_index=True)
df_all["吞吐增益"] = df_all["output_throughput_tok_s"].pct_change()
df_all["p99TTFT增幅"] = df_all["ttft_p99_ms"].pct_change()
is_knee = (df_all["吞吐增益"] < 0.10) & (df_all["p99TTFT增幅"] > 0.5)
assert is_knee.sum() >= 1, "数据应至少出现一次拐点"
knee_row = df_all[is_knee].iloc[0]
knee = int(df_all.loc[knee_row.name - 1, "concurrency"])
print(f"拐点：并发 {knee} → {int(knee_row['concurrency'])} 时，"
      f"吞吐只涨 {knee_row['吞吐增益'] * 100:.1f}%，p99 TTFT 涨 {knee_row['p99TTFT增幅'] * 100:.0f}%")
assert knee == 32

# %%
SLO_TTFT_P99, SLO_TPOT_P50 = 1000.0, 40.0
df_all["满足SLO"] = (df_all["ttft_p99_ms"] <= SLO_TTFT_P99) & (df_all["tpot_p50_ms"] <= SLO_TPOT_P50)
df_all["goodput_req_s"] = np.where(df_all["满足SLO"], df_all["request_throughput_req_s"], 0.0)
slo_cap = int(df_all.loc[df_all["满足SLO"], "concurrency"].max())
assert slo_cap == 32, "按给定 SLO，32 并发是上限"
print(f"SLO 容量：最多 {slo_cap} 并发；64 并发时 p99 TTFT={df_all.iloc[-1]['ttft_p99_ms']:.0f}ms、"
      f"TPOT={df_all.iloc[-1]['tpot_p50_ms']:.0f}ms，双双越线")
df_all[["concurrency", "output_throughput_tok_s", "goodput_req_s", "ttft_p99_ms",
        "tpot_p50_ms", "kv_cache_usage_pct_peak", "满足SLO"]]

# %% [markdown]
# **Step 3 验收**：拐点 = 32（32→64 吞吐仅 +6.2%，p99 TTFT +182%）；
# SLO 容量 = 32，64 并发时 TTFT 与 TPOT 双双越线。

# %% [markdown]
# ### Step 4 · 排队模型
#
# 目标：解释「为什么吞吐到拐点后延迟一定上升」，并知道模型什么时候失效。
#
# 做法：把一步 decode 的耗时写成 `step(n) ≈ a + b × n`：`a` 是每步固定开销（kernel 启动、
# 权重读取），`b × n` 是每多一个并发的代价。于是吞吐 `≈ n / step(n) × 1000` 接近上界
# `1000 / b`，而单请求延迟随 n 线性上涨——两者不可能同时优化。
# 用前 5 个并发点拟合，再拿模型去对回放数据。

# %%
xs = df_all["concurrency"].to_numpy(float)[:5]
ys = df_all["tpot_p50_ms"].to_numpy(float)[:5]
slope, intercept = np.polyfit(xs, ys, 1)
pred = intercept + slope * xs
r2 = 1 - ((ys - pred) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
print(f"每步耗时 ≈ {intercept:.2f} + {slope:.3f} × 并发（ms），R² = {r2:.4f}")
print(f"模型给出的吞吐上界 ≈ {1000 / slope:.0f} tok/s")
assert r2 > 0.98 and slope > 0

# %%
n_grid = np.arange(1, 97)
df_model = pd.DataFrame({"concurrency": n_grid})
df_model["step_ms"] = intercept + slope * df_model["concurrency"]
df_model["tok_per_s"] = df_model["concurrency"] / df_model["step_ms"] * 1000
df_all["模型_吞吐"] = df_all["concurrency"] / (intercept + slope * df_all["concurrency"]) * 1000
err = (df_all["模型_吞吐"] - df_all["output_throughput_tok_s"]).abs() / df_all["output_throughput_tok_s"]
print(f"模型误差：并发≤32 平均 {err[:5].mean():.1%}；并发 64 为 {err.iloc[5]:.1%}")
assert err[:5].mean() < 0.08
df_all[["concurrency", "tpot_p50_ms", "output_throughput_tok_s", "模型_吞吐"]]

# %% [markdown]
# **Step 4 验收**：`R² = 0.9929 > 0.98`；并发 ≤ 32 时平均误差 3.8% < 8%；
# 64 并发误差 26%，正好标出「线性模型失效、必须实测」的边界。

# %% [markdown]
# ## 11.4 实测
#
# 三个实验，全部基于 `cloud/fixtures/benchmark_results.json`（示例录制，非本机实测；
# 来源与录制命令见 Step 2 的打印）。协议：闭环压测，`--num-prompts 200`，
# 输入 512 / 输出 128，并发 1/4/8/16/32（加测 64）；分位数由
# `vllm bench serve --metric-percentiles 50,90,99` 给出。
#
# | 实验 | 问题 | 产出 |
# | --- | --- | --- |
# | A | 并发加到多少，吞吐到拐点、延迟起飞？ | 扫描表 + 双轴曲线 |
# | B | 排队模型能不能解释拐点？哪里开始失灵？ | 模型 vs 回放图 + 误差表 |
# | C | 尾延迟和 KV/队列怎么一起恶化？参数怎么调？ | 分位数图 + 参数扫描表 |

# %% [markdown]
# ### 实验 A · 并发扫描
#
# 协议：5 个并发点 + 1 个加测点。三列一起看：吞吐、p99 TTFT、KV 使用率。

# %%
cols = ["concurrency", "output_throughput_tok_s", "request_throughput_req_s",
        "ttft_p99_ms", "tpot_p50_ms", "kv_cache_usage_pct_peak",
        "num_requests_waiting_max", "gpu_sm_util_pct", "gpu_dram_util_pct"]
pd.concat([df_sweep[cols], df_ext[cols]], ignore_index=True)

# %%
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax1.plot(df_all["concurrency"], df_all["output_throughput_tok_s"], "o-",
         color="#2563EB", label="throughput (tok/s)")
ax1.axvline(knee, color="#F59E0B", ls="--", lw=1.5, label=f"knee = {knee}")
ax1.set_xscale("log", base=2)
ax1.set_xlabel("Concurrency (closed-loop max-concurrency)")
ax1.set_ylabel("Output throughput (tok/s)", color="#2563EB")
ax2 = ax1.twinx()
ax2.plot(df_all["concurrency"], df_all["ttft_p99_ms"], "s--", color="#EF4444", label="p99 TTFT")
ax2.set_ylabel("p99 TTFT (ms)", color="#EF4444")
ax2.grid(False)
ax1.set_title("Concurrency sweep: past 32, throughput +6% while p99 TTFT +182%")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
save(fig, "ch11_concurrency_sweep")

# %% [markdown]
# **读数**：
#
# - **吞吐**：1→32 并发涨了 15.6 倍（52→812 tok/s），但 32→64 只涨 6%（812→862）。拐点判据在 32 命中。
# - **延迟**：p99 TTFT 从 68ms 涨到 921ms（32 并发）、2600ms（64 并发）。
#   p99 比 p50 早得多地失控，所以 SLO 必须按分位数定。

# %% [markdown]
# ### 实验 B · 排队模型 vs 回放
#
# 协议：用 Step 4 拟合出的线性模型预测 1–96 并发的吞吐，与回放点对比；同时画出上界 `1000/b`。

# %%
fig, ax = plt.subplots()
ax.plot(df_model["concurrency"], df_model["tok_per_s"], "-", color="#2563EB",
        label="queueing model n/(a+bn)")
ax.plot(df_all["concurrency"], df_all["output_throughput_tok_s"], "o", color="#EF4444",
        label="replay data")
ax.axhline(1000 / slope, color="#9CA3AF", ls=":", label="model bound 1000/b")
ax.axvline(knee, color="#F59E0B", ls="--", lw=1.5, label=f"knee = {knee}")
ax.set_xlabel("Concurrency")
ax.set_ylabel("Throughput (tok/s)")
ax.set_title("Queueing model saturates; measured throughput tops out earlier")
ax.legend()
save(fig, "ch11_model_vs_replay")

# %% [markdown]
# **读数**：模型在 32 并发以内误差不到 5%，但 64 并发时明显高估（26%）——说明高并发下每步耗时
# **超线性**增长（KV 读取、调度、抢占开始收税）。这正是「模型只能解释趋势、实测才能定容量」的原因。

# %% [markdown]
# ### 实验 C · 尾延迟、KV 与参数扫描
#
# 协议：把 TTFT 的 p50/p90/p99 画成柱状图，看尾延迟何时失控；再固定并发 32，
# 只改 `--max-num-seqs`（8/16/32/64/128/256），看排队与 KV 的联动。

# %%
fig, ax = plt.subplots()
xpos = np.arange(len(df_all))
width = 0.26
ax.bar(xpos - width, df_all["ttft_p50_ms"], width, color="#2563EB", label="p50")
ax.bar(xpos, df_all["ttft_p90_ms"], width, color="#7C3AED", label="p90")
ax.bar(xpos + width, df_all["ttft_p99_ms"], width, color="#EF4444", label="p99")
ax.set_xticks(xpos, df_all["concurrency"])
ax.set_xlabel("Concurrency")
ax.set_ylabel("TTFT (ms)")
ax.set_title("TTFT percentiles: p99 runs away long before p50")
ax.legend(ncol=3)
save(fig, "ch11_ttft_percentiles")

# %% [markdown]
# 参数扫描的结论先列成表：
#
# | `max_num_seqs` | TTFT p50 / p99 (ms) | TPOT (ms) | 吞吐 (tok/s) | KV 峰值 | 等待峰值 | 结论 |
# | --- | --- | --- | --- | --- | --- | --- |
# | 8 | 760 / 2100 | 44.0 | 302 | 18% | 24 | 并发被截断，排队最严重 |
# | 16 | 340 / 1100 | 36.0 | 545 | 30% | 16 | 仍排队 |
# | 32 | 250 / 780 | 38.0 | 700 | 51% | 8 | 接近拐点 |
# | 64 | 260 / 900 | 40.0 | 800 | 72% | 4 | 收益递减 |
# | 128 | 270 / 930 | 39.0 | 812 | 81% | 4 | 基本持平 |
# | 256 | 268 / 921 | 38.2 | 812 | 55% | 4 | 默认值，本负载已饱和 |
#
# `max_num_seqs` 低于并发时，排队直接吃 TTFT；超过拐点后吞吐不再涨，只会把 KV 和显存顶满。
# **上限 ≈ 并发拐点 × 1.5** 是个稳妥起点。

# %%
df_param[["max_num_seqs", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms",
          "output_throughput_tok_s", "kv_cache_usage_pct_peak", "num_requests_waiting_max", "note"]]

# %% [markdown]
# **读数**：
#
# - **分位数**：p99 比 p50 早得多地失控（64 并发时 p99 2600ms，p50 只有 705ms），
#   所以 SLO 必须按分位数定。
# - **KV 与队列**：并发 32 时 KV 用到 55%、`num_requests_waiting` 峰值 4；64 时 KV 87%、等待 17。
#   两个信号都在说「别再加了」。
# - **SM / DRAM**：64 并发时 SM 98%、DRAM 96%——两条都贴顶，说明已经同时受算力和带宽限制，
#   不是单纯调参数能救的。

# %% [markdown]
# ## 11.5 对照工业实现（vLLM 0.28.x）
#
# ### 指标映射：本章的指标 vs vLLM 的指标名
#
# | 本章指标 | vLLM 对应（0.28.x，名称待核对） | 形式 |
# | --- | --- | --- |
# | TTFT | `vllm:time_to_first_token_seconds` | 直方图 |
# | TPOT | `vllm:time_per_output_token_seconds` | 直方图 |
# | ITL | `vllm:inter_token_latency_seconds`（待核对） | 直方图 |
# | 吞吐 | `vllm:generation_tokens_total`、`vllm:prompt_tokens_total` | 计数器 |
# | p50/p90/p99 | `vllm bench serve --metric-percentiles 50,90,99` | 压测输出 |
# | KV 使用率 | `vllm:kv_cache_usage_perc` | Gauge |
# | 队列长度 | `vllm:num_requests_waiting` | Gauge |
# | 运行中请求 | `vllm:num_requests_running` | Gauge |
# | 抢占次数 | `vllm:num_preemptions_total`（待核对） | 计数器 |

# %% [markdown]
# ### 参数调优清单：每个参数的「调大 / 调小」代价
#
# | 参数 | 默认（0.28.x 待核对） | 调大的代价 | 调小的代价 | 看哪个指标 |
# | --- | --- | --- | --- | --- |
# | `--max-num-seqs` | 256 | KV 与显存压力上升，长尾 TTFT/TPOT 变差；超过拐点吞吐不涨 | 并发被截断，`num_requests_waiting` 上升，TTFT 暴涨 | waiting、TTFT p99、KV% |
# | `--max-num-batched-tokens` | 8192 | 单步变长，decode 的 ITL/TPOT 变差（长 prompt 会阻塞 decode） | prefill 被切碎，TTFT 上升，调度开销变大 | TTFT、ITL p99、prefill 日志 |
# | `--gpu-memory-utilization` | 0.90 | KV 变大、并发更高；但越接近 1 越容易 OOM / 碎片失败 | KV 不足 → 抢占 → 吞吐崩 | KV%、preemption、OOM 日志 |
# | `--enable-prefix-caching` | 视版本（部分 v1 默认开，待核对） | 缓存块占用 KV；缓存污染时命中率低、白占显存 | 重复 system prompt 每次都重算，TTFT 高 | 缓存命中率、TTFT |
# | `--enable-chunked-prefill` | 视版本（v1 常默认开，待核对） | 长 prompt 被切片，单步更碎，TTFT 略升 | 长 prompt 与 decode 混批，短请求 TPOT 抖动 | ITL p99、长短请求 TPOT 分位 |
# | `--tensor-parallel-size` | 1 | 卡间通信增加，小 batch 收益低；需要卡数整除 | 单卡放不下模型或 KV 不够，只能降精度 | 每卡 SM/DRAM、通信占比 |

# %% [markdown]
# ### 差距在哪：本章的指标是「离线副本」
#
# vLLM 把上面这些指标做成了持续采集的 Prometheus 埋点（直方图、Gauge、计数器），
# 再配 `vllm bench serve` 输出分位数；本章只是把它们录成 JSON 后离线复算。
# 这样做的好处是完全可复现、无 GPU 也能跑；代价是看不到实时抖动，也无法验证「指标名随版本变化」。
#
# ### 怎么把本章变成你自己的数据
#
# 1. 云上起服务后跑 `bash cloud/scripts/bench_sweep.sh`，它会按 1/4/8/16/32 并发各跑一次 `vllm bench serve --save-result`；
# 2. 把 `cloud/results/` 里的 JSON 汇总成 `cloud/fixtures/benchmark_results.json` 的同构结构（字段见本章加载的 fixture）；
# 3. 重跑本 notebook，所有表和图会变成你自己机器的数字——趋势不变，绝对值才有意义。

# %% [markdown]
# ## 11.6 练习与小结
#
# **练习**（⭐ 改参数 / ⭐⭐ 改实现 / ⭐⭐⭐ 开放设计）
#
# 1. ⭐ 把 SLO 改成 `p99 TTFT ≤ 500ms`，重跑 11.3 Step 3 的 SLO 容量表。容量掉到多少？这说明 SLO 怎么影响成本？
# 2. ⭐ 把拐点判据从「吞吐增益 < 10%」改成 5% / 20%，分别重新找拐点。判据的选择会怎样改变容量结论？
# 3. ⭐⭐ 给 `percentiles` 加 bootstrap 置信区间，用 200 次重采样给出 p99 的 95% CI。样本量多少时 p99 才可信？
# 4. ⭐⭐ 把闭环扫描改成开环：按固定到达率（如 10/20/40 req/s）发请求，模拟 Little's law 里的到达率视角，对比两种压测方式下的「拐点」差异。
# 5. ⭐⭐⭐ 设计一份容量规划：业务要求 p95 TTFT < 300ms、QPS = 20、输入 512 / 输出 128。给定本章数据，选卡型、并发上限、实例数，并写出成本与压测验证方案。（评分要点：SLO 拆解、并发与 KV 账本、成本、验证步骤、回滚阈值）
#
# **你现在应该能回答：**
#
# - 为什么「平均延迟」会骗人？p99 高时应该先看哪三个服务端指标？
# - 拐点的两条判据是什么？为什么吞吐到拐点后延迟必然上升？
# - 算力 bound 与带宽 bound 在 `nvidia-smi` 上的区别是什么？分别该怎么救？
#
# **小结与下一章**
#
# 本章把「压测 → 指标 → 拐点 → 决策树 → 调参」串成闭环：先用 Little's law 自检数据，
# 再用排队模型解释趋势，最后用五条分支定位瓶颈。下一章进入多卡：TP/PP/DP 怎么切、
# 通信为什么是新的瓶颈，以及本章的 `--tensor-parallel-size` 会带来什么。

# %% [markdown]
# **延伸阅读**
#
# - `vllm bench serve` 的输出字段与 `--metric-percentiles`（版本随 0.28.x）；
# - Prometheus 指标全集：`vllm/engine/metrics.py` 与 `docs/design/metrics.md`（路径与指标名待核对）；
# - `cloud/scripts/bench_sweep.sh`：把本章的扫描固化成脚本；
# - 第 12 章：TP/PP/DP 与通信开销。
