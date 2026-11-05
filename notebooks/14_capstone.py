# %% [markdown]
# # 第 14 章 · Capstone：给定 SLO，算清要几张卡、多少钱
#
# > 前面 13 章都在回答「为什么慢」；这一章回答老板唯一关心的问题：「要花多少钱，能不能达标」。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 90 分钟 |
# | 运行环境 | 🧪 纯 Python 推算（Mac 可跑，无需 GPU；云上压测见第 11 章脚本） |
# | 前置章节 | 13（训练侧 infra） |
# | 本章产物 | 可改参数的 `plan_capacity()`、三方案成本/SLO 对比、QPS 敏感度曲线、上云校准清单 |
# | 本章接口 | `aiinfra.plotting.save`；复用第 01/02 章的显存与带宽公式 |
# | 参考实现 | vLLM 0.28.x（`--gpu-memory-utilization`、`--max-num-seqs`、`--max-model-len`） |
#
# **学完你能回答：**
# 1. 100 QPS、p95 TTFT < 500 ms、TPOT < 50 ms、2K 上下文，需要几台什么卡？
# 2. 8B fp16、8B int4、27B int4 三个方案，每百万 token 成本差多少？为什么 int4 不一定最便宜？
# 3. 哪些假设最危险？上云第一天应该先量哪三个数来校准这份表格？
# 4. QPS 或上下文翻倍时，成本和实例数会怎么变？哪个方案最先撑不住？

# %% [markdown]
# ## 14.0 本章目标与路线
#
# **一句话目标**：把「业务 SLO + 预算」翻译成一张可交付的容量方案表——
# **模型 × 精度 × 卡型 × 实例数 × 成本 × 风险**，并给出推荐方案与上云校准方法。
#
# **代价与边界**：
#
# - 本章是**纯推算**：用 p50 口径的公式估 TTFT/TPOT，不含排队毛刺、突发流量、
#   prefill/decode 相互抢占，也不含 int4 反量化的实际算力损失。
# - 卡型只取 A100 80GB（2 TB/s HBM、150 TFLOP/s 有效 fp16、$1.9/h）作为统一口径；
#   其他卡改 `GPU` 参数即可，结论逻辑不变。
# - 所有成本都是**模型状态级**估算；真实账单还要加存储、网络、闲置与折扣。
#
# **本章交付四样东西**：
#
# | # | 交付物 | 在哪一节 | 怎么验收 |
# | --- | --- | --- | --- |
# | 1 | `plan_capacity()`：从模型/精度/卡型推出并发、实例、成本 | 14.3 Step 1–2 | 方案 A 的并发、TTFT、TPOT 均 > 0 且 SLO 达标 |
# | 2 | 单方案账本（每步中间量） | 14.4 实验 A | 表里能逐项对上公式 |
# | 3 | 三方案对比 + 推荐 | 14.4 实验 B | 三方案延迟 SLO 全过；B 的 $/M 最低 |
# | 4 | QPS 敏感度曲线与预算线 | 14.4 实验 C | 实例数随 QPS 单调不减；成本是阶梯函数 |
#
# **阅读路线**：14.1 问题与直觉 → 14.2 高阶设计 → 14.3 分四步实现（每步有验收）→
# 14.4 实验 A/B/C → 14.5 对照 vLLM。
#
# **本地推算 vs 云上实测**：本章全部是推算；真实数字要用第 11 章的 `vllm bench serve` 在
# 选定实例上跑 2K/300 负载，把 `output_throughput`、`mean_ttft`、`p99_tpot` 填回 14.4 的
# 复算模板。14.4 会给出「先量哪三个数」。

# %% [markdown]
# ## 14.1 问题与直觉：SLO 与预算常常冲突
#
# 容量规划不是「拍一个卡数」，而是一个**闭环**：先用公式估，再用压测校准，最后用成本否决。
# 顺序错了会白花很多钱——先算「一张卡放不放得下」，再算「要几张卡」，最后才看「多少钱」。
#
# ![](../assets/figures/ch14_capacity_flow.svg)
#
# 三个反例，对应流程里最容易被跳过的三步：
#
# - **跳过显存账本**：直接按「100 QPS ÷ 单卡 QPS」算实例数。结果单卡 KV cache 只能放几十个
#   请求，`--max-num-seqs` 一调大就 OOM，真实并发根本到不了公式里的数。
# - **跳过 TTFT 分解**：预填 2K context 的耗时被忽略。decode 再快，首 token 也要等 prefill，
#   长上下文场景下 prefill 往往才是 TTFT 的大头。
# - **跳过成本复验**：8B fp16 方案 SLO 全达标，但 8 张 A100 一个月 $11k，超预算 11%——
#   方案「技术上可行、商业上不可行」，必须在表里标红。
#
# %% [markdown]
# ### 全章固定的业务输入
#
# 场景是企业内部知识库问答（RAG），参数如下：
#
# | 参数 | 取值 | 备注 |
# | --- | --- | --- |
# | 峰值 QPS | 100 | 工作时段峰值；日均约为峰值的 1/3 |
# | p95 TTFT | < 500 ms | 首 token 延迟，用户感知「有没有在动」 |
# | TPOT | < 50 ms | 每输出 token 延迟，用户感知「读得顺不顺」 |
# | 平均上下文 | 2K tokens | RAG 检索片段 + system prompt |
# | 平均输出 | 300 tokens | 答案长度 |
# | 月度预算上限 | $10,000 | 含全部 GPU 实例费，不含存储/网络 |
#
# ### 从「拍脑袋」到「闭环推算」
#
# 「拍脑袋」与「闭环推算」的差别：
#
# | 维度 | 拍脑袋 | 本章闭环 |
# | --- | --- | --- |
# | 起点 | 直接按 QPS 除单卡 QPS | SLO + 单卡显存账本 |
# | 约束 | 只看卡数和报价 | 显存 → 并发 → 实例 → 成本 → SLO 复验 |
# | 失败模式 | 并发到不了 / OOM / 超预算而不自知 | 每一步都能被否决（④ 或 ⑦） |
# | 产物 | 一个数字 | 方案表 + 风险清单 + 压测验收计划 |
#
# **本章要解决两个子问题**：
#
# 1. **可行性**：单卡放得下吗？在放得下的前提下，TTFT/TPOT 能不能达标？
# 2. **经济性**：要几台实例？每月多少钱？每百万 token 多少钱？预算够不够？
#
# 四个公式就能推出大部分结论（后面实验会逐个填参数）：
#
# ```
# 单卡并发上限 = min( KV 显存能放几个请求 , TPOT 预算内一步能搬多少字节 )
# 单实例 QPS   = 并发上限 ÷ (TTFT + 输出长度 × 单步 TPOT)
# 实例数       = ceil(峰值 QPS ÷ (单实例 QPS × 目标利用率))
# 每月成本     = 实例数 × TP 卡数 × 单卡小时价 × 730 小时
# ```

# %% [markdown]
# ## 14.2 高阶设计
#
# 整章有三个组件、一条推算链和一张验收表。
#
# **组件 1：输入参数化（要算什么）**
#
# - `GPU`：显存、HBM 带宽、有效 fp16 算力、小时价；
# - `Workload`：QPS、TTFT/TPOT 预算、上下文与输出长度、峰值/均值比、月预算。
#
# 把输入和公式分开，是为了让学员改一个参数就能重算，而不需要改 40 行代码。
#
# **组件 2：容量推算链（怎么算）**
#
# ```text
# 权重字节 → 单卡权重 → KV/请求 → KV 预算 → 并发上限
#        → 单步 TPOT → TTFT(prefill) → 单实例 QPS → 实例数 → 月成本 → $/M tokens
# ```
#
# 三个容易写错的地方：KV cache 的 dtype 永远是 fp16（量化只压权重）；TP 会同时切权重和 KV；
# TTFT 用 `2 × 参数量 × 上下文 ÷ (TP × 有效算力)` 再加快捷开销，是 p50 估算。
#
# **组件 3：三个候选方案（比什么）**
#
# | 方案 | 模型 | 精度 | 拓扑 | 关键假设 |
# | --- | --- | --- | --- | --- |
# | A | 8B | fp16（2 B/参数） | 1×A100 | 质量基线，显存最紧 |
# | B | 8B | int4（0.5 B/参数） | 1×A100 | 权重省 12 GB，KV 不变 |
# | C | 27B | int4（0.5 B/参数） | 2×A100（TP=2） | 质量更好，卡数翻倍 |
#
# 27B 的 KV 按 46 层、8 KV head、head_dim=128 估；int4 用 0.5 B/参数（含零点/缩放开销，
# 比理论 0.25 B 更保守）。判定口径有三条：延迟 SLO、月预算、每百万 token 成本。
#
# %% [markdown]
# ### 数据流与验收标准
#
# 数据流：
#
# ```text
# Workload ─┐
# GPU ──────┼→ plan_capacity(模型/精度/TP) → {并发, TPOT, TTFT, 实例, 月成本, $/M}
# 模型规格 ─┘                                      ↓
#                                     SLO 达标? ──否──→ 改模型/精度/TP 重算
#                                          │是
#                                          ↓
#                                    预算达标? ──否──→ 降规格或接受超预算
#                                          │是
#                                          ↓
#                                    方案表 + 风险 + 压测计划
# ```
#
# **验收标准**：
#
# | 步骤 | 验收 |
# | --- | --- |
# | Step 1 输入参数 | GPU/Workload 可实例化，场景打印正确 |
# | Step 2 规划函数 | 方案 A 返回完整字段（并发/TTFT/TPOT/实例/成本） |
# | Step 3 单方案 | 并发 > 0；TTFT/TPOT 达标 |
# | Step 4 三方案 | 每个方案都能放进单卡（并发 > 0） |
# | 实验 A/B/C | 账本表可对公式；三方案 SLO 全过且 B 最便宜；QPS 扫描单调 |

# %% [markdown]
# ## 14.3 从零实现：分四步
#
# 分成 4 步，每步先讲目标和做法，再给代码，最后给验收：
#
# | Step | 做什么 | 代码 |
# | --- | --- | --- |
# | 1 | 定义卡型、负载与场景 | `GPU`、`Workload` |
# | 2 | 写容量规划函数 | `plan_capacity()` |
# | 3 | 跑通单方案 A，检查 SLO | `plan_capacity(**PLAN_A)` |
# | 4 | 定义三个候选方案 | `PLANS` |

# %%
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd().parent))

from aiinfra.plotting import save, setup_style

setup_style()


@dataclass(frozen=True)
class GPU:
    name: str
    vram_gb: float          # 显存
    hbm_bw_gbps: float      # 实测可达 HBM 带宽（GB/s）
    tflops_fp16: float      # 有效 fp16 算力（TFLOP/s）
    hourly_usd: float       # 单卡小时价（美元）


@dataclass(frozen=True)
class Workload:
    qps: float = 100.0
    ttft_budget_ms: float = 500.0
    tpot_budget_ms: float = 50.0
    context: int = 2048
    output: int = 300
    peak_to_avg: float = 3.0      # 峰值 QPS / 日均 QPS
    budget_usd_month: float = 10_000.0


A100 = GPU("A100 80GB", vram_gb=80.0, hbm_bw_gbps=2000.0, tflops_fp16=150.0, hourly_usd=1.9)
WL = Workload()
print(f"场景：{WL.qps:.0f} QPS，TTFT p95 < {WL.ttft_budget_ms:.0f} ms，TPOT < {WL.tpot_budget_ms:.0f} ms，"
      f"{WL.context} 上下文 + {WL.output} 输出，预算 ${WL.budget_usd_month:,.0f}/月")

# %% [markdown]
# ### Step 1 · 输入参数化
#
# **目标**：把「卡」和「负载」变成两个不可变 dataclass，后续所有公式只读这两个对象。
#
# **做法**：`GPU` 只保留 4 个规格 + 价格；`Workload` 保留 SLO 与流量形状。A100 的参数取
# 公开规格的近似值：2 TB/s（实测可达，不是理论峰值）、150 TFLOP/s 有效 fp16、$1.9/h。
#
# **验收**：上面的打印行能正确输出 100 QPS / 500 ms / 50 ms / 2K / 300 / $10k。

# %% [markdown]
# ### Step 2 · 容量规划函数
#
# **目标**：给定模型/精度/卡型，返回单实例容量、实例数与成本。
#
# **做法**：按「权重 → KV → 并发 → 实例 → 成本」的顺序算。三个容易写错的地方：
#
# 1. **KV cache 的 dtype 永远是 fp16**：量化只压权重，不压 KV（除非显式开 KV 量化）。
# 2. **TP 会同时切权重和 KV**：每张卡只存 1/TP 的 KV，所以单步读的字节数要除以 TP。
# 3. **TTFT 用 prefill 计算量估**：`2 × 参数量 × 上下文长度 ÷ (TP × 有效算力)`，再加
#    20 ms 的固定开销（调度、网络、首 token 采样）。这是 p50；p95 还要留排队余量。
#
# **验收**：函数在方案 A 上返回完整字段，且并发/TTFT/TPOT 均为正（Step 3 验证）。

# %%
def plan_capacity(
    params: float, weight_bytes: float, n_layer: int, n_kv_head: int, head_dim: int,
    tp: int = 1, gpu: GPU = A100, wl: Workload = WL, util: float = 0.9, workspace_gb: float = 2.0,
) -> dict:
    """给定模型/精度/卡型，返回单实例容量、实例数与成本（不含激活值峰值与排队毛刺）。"""
    weights_bytes = params * weight_bytes
    kv_per_token = 2 * n_layer * n_kv_head * head_dim * 2        # K+V，fp16
    kv_per_req = kv_per_token * (wl.context + wl.output)         # 按最长序列算容量
    kv_read = kv_per_token * (wl.context + wl.output / 2)        # 按平均序列算单步搬运
    usable = gpu.vram_gb * util * 1e9
    weights_per_gpu = weights_bytes / tp
    kv_budget = usable - weights_per_gpu - workspace_gb * 1e9    # 激活/cudagraph 预留
    max_conc_kv = max(0, int(kv_budget / kv_per_req))
    tpot_bytes = wl.tpot_budget_ms / 1000 * gpu.hbm_bw_gbps * 1e9
    max_conc_tpot = max(0, int((tpot_bytes * tp - weights_bytes) / kv_read))
    concurrency = min(max_conc_kv, max_conc_tpot)
    step_ms = (weights_bytes + concurrency * kv_read) / tp / (gpu.hbm_bw_gbps * 1e9) * 1000
    prefill_ms = 2 * params * wl.context / (tp * gpu.tflops_fp16 * 1e12) * 1000 + 20
    qps_per_instance = concurrency / (prefill_ms / 1000 + wl.output * step_ms / 1000)
    instances = max(1, int(np.ceil(wl.qps / (qps_per_instance * 0.8))))
    monthly = instances * tp * gpu.hourly_usd * 730
    monthly_tokens_m = wl.qps / wl.peak_to_avg * 3600 * 730 * (wl.context + wl.output) / 1e6
    return {
        "单卡权重(GB)": weights_per_gpu / 1e9,
        "每 token KV(KB)": kv_per_token / 1024,
        "每请求 KV(MB)": kv_per_req / 1e6,
        "KV 可容纳并发": max_conc_kv,
        "TPOT 可容纳并发": max_conc_tpot,
        "并发上限": concurrency,
        "单步 TPOT(ms)": step_ms,
        "TTFT 估计(ms)": prefill_ms,
        "单实例 QPS": qps_per_instance,
        "实例数": instances,
        "GPU 总数": instances * tp,
        "月成本($)": monthly,
        "每百万 token 成本($)": monthly / monthly_tokens_m,
        "TTFT 达标": prefill_ms <= wl.ttft_budget_ms,
        "TPOT 达标": step_ms <= wl.tpot_budget_ms,
        "预算达标": monthly <= wl.budget_usd_month,
    }

# %% [markdown]
# ### Step 3 · 跑通单方案 A
#
# **目标**：先只算方案 A（8B fp16，单卡 A100），把中间量摊开看，确认函数跑得通。
# 模型结构取 Llama-3-8B 的近似：32 层、8 个 KV head（GQA）、head_dim=128，
# 每 token KV = `2 × 32 × 8 × 128 × 2B = 128 KB`。
#
# **做法**：调用 `plan_capacity`，断言「单卡放得下（并发 > 0）」且「延迟 SLO 达标」。
#
# **验收**：三个断言全部通过；下一步在实验 A 里打印完整账本。

# %%
PLAN_A = dict(params=8e9, weight_bytes=2.0, n_layer=32, n_kv_head=8, head_dim=128, tp=1)
plan_a = plan_capacity(**PLAN_A)
assert plan_a["并发上限"] > 0, "并发为 0 说明单卡放不下权重，方案直接不可行"
assert plan_a["TTFT 达标"] and plan_a["TPOT 达标"]
print(f"方案 A：并发 {plan_a['并发上限']}，TPOT {plan_a['单步 TPOT(ms)']:.1f} ms，"
      f"TTFT {plan_a['TTFT 估计(ms)']:.0f} ms，实例 {plan_a['实例数']}，月成本 ${plan_a['月成本($)']:,.0f}")

# %% [markdown]
# ### Step 4 · 定义三个候选方案
#
# **目标**：把 A/B/C 三组模型规格写成字典，供实验 B 批量对比。
#
# **做法**：每组只给 6 个参数（参数量、每参数字节、层数、KV head、head_dim、TP）。
# B 与 A 只差精度（0.5 vs 2 B/参数）；C 换成 27B、46 层、TP=2。
#
# **验收**：三个方案都能放进单卡（`并发上限 > 0`），否则方案在 14.1 的 ④ 步就被否决。

# %%
PLANS = {
    "A 8B fp16 ×1": dict(params=8e9, weight_bytes=2.0, n_layer=32, n_kv_head=8, head_dim=128, tp=1),
    "B 8B int4 ×1": dict(params=8e9, weight_bytes=0.5, n_layer=32, n_kv_head=8, head_dim=128, tp=1),
    "C 27B int4 ×2": dict(params=27e9, weight_bytes=0.5, n_layer=46, n_kv_head=8, head_dim=128, tp=2),
}
for name, kw in PLANS.items():
    r = plan_capacity(**kw)
    assert r["并发上限"] > 0, f"{name} 单卡放不下，应该先换精度/TP"
print("三方案均可放入单卡；完整对比见 14.4 实验 B")

# %% [markdown]
# ## 14.4 实测
#
# 三个实验，分别回答三个问题：
#
# | 实验 | 问题 | 产出 |
# | --- | --- | --- |
# | A | 单方案 A 的账本长什么样？每个中间量怎么来的？ | 单方案指标表 |
# | B | 三个方案谁满足 SLO/预算？$/M tokens 差多少？ | 三方案对比表 + 推荐 |
# | C | QPS 从 40 扫到 160，实例数和成本怎么变？ | 敏感度表 + 阶梯曲线 |
#
# **测量协议**：纯公式推算，无随机性；参数口径 = A100 80GB / 2 TB/s / 150 TFLOP/s / $1.9/h、
# 目标利用率 80%、峰值/均值 = 3。同一份输入下结果可完全复现。
#
# **本地推算与云上实测的分工**：
#
# | 实验 | 本地推算（本章） | 云上实测（第 11 章 + `12_tp_bench.sh`）要量的数 |
# | --- | --- | --- |
# | A 单方案账本 | 公式 p50 | `mean_ttft`、`mean_tpot`、`output_throughput` |
# | B 三方案对比 | 实例数 + 成本 | 每方案在选定实例上的真实吞吐与账单 |
# | C 敏感度 | 实例数阶梯 + 预算线 | 压测不同 QPS 下的 p95 与排队行为 |

# %% [markdown]
# ### 实验 A · 单方案账本
#
# 把方案 A 的每个中间量打印成表。这张表就是流程图的 ④⑤⑥⑦ 步，每一行都能用 14.2 的推算链复算。

# %%
plan_table = pd.DataFrame(
    [{"指标": k, "数值": round(v, 2) if isinstance(v, float) else v} for k, v in plan_a.items()]
)
plan_table

# %% [markdown]
# **读数**：
#
# - **A100 80GB 放得下 8B fp16**：16 GB 权重 + 约 175 个请求 × 0.3 GB KV ≈ 70 GB，刚好。
# - **并发上限由 KV 显存而不是 TPOT 预算决定**（175 vs 291）：decode 阶段每步读的是
#   「权重 + 全部在途请求的 KV」，KV 越长，单步越慢；这里 KV 先把显存吃满了。
#
# 把单实例 QPS 乘以实例数，就得到 100 QPS 需要的 8 张卡——但 8 × $1.9 × 730 = **$11,096/月，
# 超预算 11%**。方案 A 的判定是「SLO ✅、预算 ❌」。

# %% [markdown]
# ### 实验 B · 三方案对比
#
# 用同一个函数跑 A/B/C，比较单卡权重、每请求 KV、并发、TPOT、TTFT、实例数、GPU 总数、
# 月成本与 $/M tokens。断言：三方案延迟 SLO 全过；B 的 $/M 最低。

# %%
rows = [{"方案": name, **{k: round(v, 2) if isinstance(v, float) else v for k, v in plan_capacity(**kw).items()}}
        for name, kw in PLANS.items()]
compare_df = pd.DataFrame(rows)[
    ["方案", "单卡权重(GB)", "每请求 KV(MB)", "并发上限", "单步 TPOT(ms)", "TTFT 估计(ms)",
     "实例数", "GPU 总数", "月成本($)", "每百万 token 成本($)", "TTFT 达标", "TPOT 达标", "预算达标"]
]
per_m = dict(zip(compare_df["方案"], compare_df["每百万 token 成本($)"]))
assert compare_df[["TTFT 达标", "TPOT 达标"]].all().all(), "三个方案都应满足延迟 SLO"
assert per_m["B 8B int4 ×1"] < per_m["A 8B fp16 ×1"] < per_m["C 27B int4 ×2"]
compare_df

# %% [markdown]
# **读数**（数字来自上表，随假设变化，**逻辑不变**）：
#
# | 方案 | 月成本 | $/M tokens | SLO | 预算 | 结论与风险 |
# | --- | --- | --- | --- | --- | --- |
# | A 8B fp16 ×1 | $11,096 | $0.054 | ✅ | ❌ 超 11% | 质量基线；减 QPS 或换 int4 才可行 |
# | B 8B int4 ×1 | $8,322 | $0.040 | ✅ | ✅ 余 17% | **推荐**；风险：int4 精度损失需评测集验证 |
# | C 27B int4 ×2 | $16,644 | $0.081 | ✅ | ❌ 超 66% | 质量上限；TP=2 通信与 12 卡故障域是额外风险 |
#
# 三个反直觉点：
#
# 1. **int4 的收益不只是「权重省 12 GB」**：省下的显存变成了 KV 并发（175 → 214），
#    所以实例数从 8 降到 6，成本降 25%。量化在**容量受限**场景的杠杆比「提速」更大。
# 2. **27B 的 TPOT 反而更低**（18 ms vs 33 ms）：TP=2 把每步要读的 KV 也切了一半。
#    但它用 12 张卡换来的吞吐并没有线性提升，$/M tokens 是 B 的 2 倍。
# 3. **预算否决发生在最后一步**：A 和 C 的 SLO 都过了，但商业上不可行。真实评审时
#    「SLO 达标」只是入场券，成本与风险才是决策依据。

# %% [markdown]
# ### 实验 C · QPS 敏感度与预算线
#
# 容量规划必须回答「QPS 变了怎么办」。把峰值 QPS 从 40 扫到 160，看实例数与月成本怎么变——
# 因为实例数是 `ceil`，所以成本是阶梯函数，而不是直线。

# %%
QPS_LIST = [40, 60, 80, 100, 120, 160]
sweep = []
for qps in QPS_LIST:
    wl = Workload(qps=qps)
    row = {"Peak QPS": qps}
    for name, kw in PLANS.items():
        r = plan_capacity(**kw, wl=wl)
        row[f"inst_{name.split()[0]}"] = r["实例数"]
        row[f"cost_{name.split()[0]}"] = r["月成本($)"]
    sweep.append(row)
sweep_df = pd.DataFrame(sweep)
# 单调性断言：QPS 越高，实例数不可能更少
for col in [c for c in sweep_df.columns if c.startswith("inst_")]:
    assert sweep_df[col].is_monotonic_increasing, f"{col} 应随 QPS 单调不减"
sweep_df

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
colors = {"A": "#2563EB", "B": "#10B981", "C": "#F59E0B"}
for key, color in colors.items():
    axes[0].step(sweep_df["Peak QPS"], sweep_df[f"inst_{key}"], where="post", color=color, label=f"Plan {key}")
    axes[1].plot(sweep_df["Peak QPS"], sweep_df[f"cost_{key}"], "o-", color=color, label=f"Plan {key}")
axes[1].axhline(WL.budget_usd_month, color="#EF4444", ls="--", lw=1.5, label="Monthly budget $10k")
axes[0].set_xlabel("Peak QPS")
axes[0].set_ylabel("Instances required")
axes[0].set_title("Instance count is a step function")
axes[1].set_xlabel("Peak QPS")
axes[1].set_ylabel("Monthly cost (USD)")
axes[1].set_title("Monthly cost: plan B stays under budget up to 160 QPS")
for ax in axes:
    ax.legend(fontsize=8)
save(fig, "ch14_capacity_curve");

# %% [markdown]
# **读图**：**B 方案在 160 QPS 才接近预算线**（约 $13.9k），而 A 在 100 QPS 就超了，
# C 在 60 QPS 就超了。这个交叉点就是谈判筹码：如果业务承诺「峰值不超过 80 QPS」，
# A 也能压进预算；如果能接受 8B 的质量，B 是唯一有成本余量的方案。
#
# 还要记住这张图的**三个未建模项**，上云后必须用压测校准：
#
# 1. **排队**：我们按 80% 利用率算实例数，但 p95 TTFT 还取决于到达过程（突发流量）。
# 2. **prefill/decode 干扰**：连续批处理里 prefill 会抢占 decode，实际 TPOT 会比单步估算高。
# 3. **int4 的实际算力**：反量化有额外开销，有效 TFLOP/s 可能低于 fp16 的 150。
#
# 校准方法：用第 11 章的 `vllm bench serve` 在你选的实例上跑 2K/300 负载，
# 把实测 `output_throughput`、`mean_ttft`、`p99_tpot` 填回本表，重算实例数。

# %% [markdown]
# ## 14.5 对照工业实现（vLLM）
#
# | 我们的参数 | vLLM 中的对应物 | 说明（vLLM 0.28.x） |
# | --- | --- | --- |
# | `util=0.9` | `--gpu-memory-utilization 0.9` | 决定 `profile_run` 后能开多少 KV block |
# | `context+output` | `--max-model-len` | 单请求最长序列；直接决定每请求 KV 上限 |
# | `并发上限` | `--max-num-seqs` | 实际生效值取 min(显存, 该参数) |
# | `prefill_ms` | `--max-num-batched-tokens` / chunked prefill | prefill 切块会拉长 TTFT，但降低 TPOT 抖动 |
# | `weight_bytes` | `--quantization awq/gptq/fp8` | int4 权重、fp8 KV 是不同开关，别混 |
# | `tp` | `--tensor-parallel-size` | 必须整除 head 数；单机内才划算（第 12 章） |
# | 压测校准 | `vllm bench serve` + `--metric-percentiles 50,95,99` | 第 11 章的完整流程 |
# | KV 使用率 | `vllm:gpu_cache_usage_perc` 指标 | 超过 0.9 说明并发到顶，该加实例了 |
#
# 差距在哪：vLLM 的这些参数**互相耦合**。
#
# - 调大 `--max-num-seqs` 会提高吞吐，但 KV block 不够时会触发抢占（preemption），
#   被抢占的请求要重算 prefill，TTFT 反而变差。
# - `--max-model-len` 调大能接更长的上下文，但每请求 KV 线性增长，并发上限线性下降——
#   这就是本表里「上下文 2K → 8K，实例数翻几倍」的根源（练习 3）。
# - `--gpu-memory-utilization` 不是越高越好：留一点余量给激活值峰值和 CUDA Graph，
#   否则长 prompt 一来就 OOM。
#
# 所以容量规划的正确姿势是：**先用本章的表框定 2–3 个候选，再上云压测校准，最后用账单复验**。
# 不要跳过任何一步。

# %% [markdown]
# ## 14.6 练习与小结
#
# **练习**
#
# 1. ⭐ 把 `WL.qps` 改成 60，重跑三方案。谁进入预算？A 方案最多能撑多少 QPS？
# 2. ⭐ 把 `tpot_budget_ms` 收紧到 30 ms，哪些方案会失败？失败的是 TPOT 还是并发不够？
# 3. ⭐⭐ 把 `context` 从 2048 改成 8192，重跑三方案并解释：为什么 KV 显存是上下文长度的
#    线性函数，而实例数几乎是线性增长？哪个方案最先撑不住？
# 4. ⭐⭐ 给 `plan_capacity` 加一个 `prefix_cache_hit` 参数（RAG 的 system prompt 复用率，
#    例如 50%）。命中前缀的请求 prefill 只需算未命中的部分，重算 TTFT 与实例数。
# 5. ⭐⭐⭐ 写一页给老板的容量规划：3 方案对比 + 推荐 + 风险 + 上云验证计划。
#    评分要点（每点 20 分）：SLO 是否量化、成本是否含假设、风险是否有应对、验证是否可执行。
#
# **你现在应该能回答：**
#
# - 为什么「单卡放得下」不等于「单卡能扛 QPS」？两个约束分别是什么？
# - 为什么 int4 能让每百万 token 成本降 25%，但 27B int4 反而更贵？
# - 如果 p95 TTFT 实测比估算高 2 倍，你应该先查哪三个参数？
#
# **小结**
#
# 走完 14 章，你应该能把任何一个 LLM 服务需求翻译成一张表：**模型 × 精度 × 卡型 × 实例数 ×
# 成本 × 风险**。这张表的数字会变，但推导链条不会：显存账本 → 并发 → 实例 → 成本 → SLO 复验。
# 剩下的就是把表贴到云上，用真实压测把它校准成「可交付」的容量方案。
#
# **延伸阅读**
#
# - vLLM 文档：Optimization / Conserving Memory / `--gpu-memory-utilization` 与 `--max-num-seqs`
# - Little's law 与排队论基础（第 00 章）；GPU 利用率与 p95 的关系
# - 各云厂商 A100/H100/L40S 按小时计价页（本章用 $1.9/h 作为 A100 假设价）
