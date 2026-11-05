# %% [markdown]
# # 第 05 章 · 调度器：静态批处理 vs 连续批处理
#
# > 同一块 GPU、同一串请求，换一种调度方式，吞吐为什么能翻倍、尾延迟为什么能差一个数量级。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 90 分钟 |
# | 运行环境 | 🏠 Mac（CPU/MPS）· 无需 GPU（调度器是模拟器）· torch 2.15 |
# | 前置章节 | 02（prefill/decode 两阶段与瓶颈）· 03（分页注意力，理解 padding 可被消除） |
# | 本章产物 | `Request` 接口、事件驱动模拟器、静态/连续两种调度器、甘特图、吞吐-延迟曲线、参数扫描表 |
# | 本章接口 | `Request`（`prompt_len` / `max_new_tokens` / `arrival` / `num_computed` / `num_generated`），第 06 章直接复用 |
#
# **学完你能回答：**
# 1. 连续批处理（continuous batching）比静态批处理多做了什么？
# 2. 为什么 decode 单步耗时对 batch 亚线性？这一性质如何决定 batching 的收益？
# 3. `max_num_seqs` 与 `max_num_batched_tokens` 分别限制什么？调小各自会先伤哪个指标？
# 4. 静态批处理在真实对话负载下吞吐低，padding、等最长请求、批间空窗各占多少责任？

# %% [markdown]
# ## 05.0 本章目标与路线
#
# **一句话目标**：把「攒一批、整批跑完再换人」的静态批处理，换成「每个 decode step 都重新组批」的连续批处理；
# 在同一到达序列、同样 `max_num_seqs=16` 下，吞吐从 310 提到 792 tokens/s（**2.55x**），TTFT p95 从 21.3s 降到 1.2s，
# 并且不改变任何请求的输出内容。
#
# 21.3 秒是什么体感？够你把一段 60 行的报错从头读到尾；1.2 秒差不多是刷新一次页面的时间。
# 这两行数字来自同一个成本模型、同一条到达序列，唯一的变量是**先看谁**。
#
# **代价与边界**：
# - 调度器不省算力，它只让 GPU 的每个 slot 尽量不空转；代价是 decode 的 ITL 变差（29.3ms，而静态是 7.6ms）——
#   这是第 06 章 chunked prefill 要还的债。
# - 本章用**成本模型 + 事件驱动模拟器**而不是真机：要隔离「调度策略」这一个变量，真机的 kernel 与带宽噪声会把它淹没。
# - 结论是趋势与相对倍数，绝对值随硬件变化；成本模型的两条假设会在实验 A 用本机 TinyGPT 实测验证。
#
# **本章交付四样东西**：
#
# | # | 交付物 | 在哪一节 | 怎么验收 |
# | --- | --- | --- | --- |
# | 1 | `Request` 数据结构 + 成本模型 | 05.3 Step 1–2 | 字段语义断言；模型两条假设断言 |
# | 2 | 事件驱动模拟器与统一指标 | 05.3 Step 3 | 指标定义与表列一致；同一输入结果可复现 |
# | 3 | 静态 / 连续两种调度器 | 05.3 Step 4–5 | 同到达同长度时两者结果相等；错峰时连续不慢于静态 |
# | 4 | 三组实验：标定、对比、参数扫描 | 05.4 实验 A/B/C | 3 张图 + 5 张结果表，趋势与成本模型吻合 |
#
# **阅读路线**：05.1 问题与直觉 → 05.2 高阶设计 → 05.3 分步实现（6 步，每步有验收）→ 05.4 实验 A/B/C → 05.5 对照 vLLM → 05.6 练习。

# %% [markdown]
# ## 05.1 问题与直觉：同一块 GPU，两种时间轴
#
# 打开监控面板，你会看到一幅很别扭的画面：GPU 利用率一直很高，可用户还在问
# 「为什么我按了回车十几秒才看到第一个字」。算力没闲着、钱没少花，体验却不像花了钱的样子。
#
# 问题出在**调度器**：它让 GPU 一直在算，但相当一部分 slot 只是在陪跑。
# 调度器不改变模型，只改变「GPU 下一步算什么」。所以它造成的差异全部写在时间轴上。
# 先看同一到达序列在两种策略下的轨迹：
#
# ![](../assets/figures/ch05_gantt.svg)
#
# 上半图是**静态批处理**：攒一批请求，整批一起 prefill，然后一起 decode，直到这批里**最长**的那个结束，才接下一批。
# 下半图是**连续批处理**（continuous batching，iteration-level scheduling，Orca 论文的核心思想）：
# 每个 decode step 结束都重新检查一次 slot，一有空位就让新请求进来。
#
# 图里标出了三笔浪费，也是本章要量化的三件事：
#
# - **padding**：一批 prompt 长短不一，朴素实现要 pad 到最长再一起算，短 prompt 的算力白烧。
# - **等最长请求**：只生成 8 个 token 的请求，slot 要空等到 512 个 token 的邻居结束；GPU 仍在为邻居算，但这个 slot 是死的。
# - **批间空窗**：整批没结束前新请求进不来；对话负载的输出长度是重尾的，这个空窗会吃掉可观的 GPU 时间。

# %% [markdown]
# **先猜一下**：同一块卡、同一串请求，只把「攒一批跑完再换人」改成「每一步都重新组批」，
# 吞吐最多能涨多少？
#
# (a) 10% 左右——组批只是调度开销，不产生算力
# (b) 2–3 倍——同一份权重被更充分地摊薄
# (c) 10 倍以上——把 GPU 从半闲拉到满负荷
#
# 记下你的答案，实验 B 的读表会回来对账。

# %% [markdown]
# 用一个具体反例看清差别：一批 16 个请求，输出长度从 8 到 1024 token 不等。
# 把它们的经历画在时间轴上（步数是示意，量级按本章的负载）：
#
# ```text
# 静态批处理：16 个请求同批进来，输出 8 ~ 1024 token 不等
# 第   0 步 ┃ 整批 prefill：pad 到最长 prompt，短 prompt 的算力在这里白烧
# 第   1 步 ┃ 16 条序列一起 decode，16 个 slot 全绿
# 第   8 步 ┃ 输出最短的请求结束 → 它的 slot 从此变灰
# 第  32 步 ┃ 一半 slot 已空转，GPU 还在为最长的邻居算
# 第1024 步 ┃ 最后一个请求结束，整批才一起下班
# 第1025 步 ┃ 此间到达的请求，这时才第一次被看见
# ```
#
# 三个空转区间正好对应三笔浪费：第 0 步的 padding、第 8–1024 步的「等最长请求」、
# 第 1024 步之后的「批间空窗」。不换模型、不换卡，只把换人时机挪到每一步的边界：
#
# ```text
# 连续批处理：同一条到达序列
# 第   0 步 ┃ 组批：凑满 16 条，整批 prefill（padding 仍然存在）
# 第   8 步 ┃ 短请求结束 → 下一步，它的 slot 立刻被 waiting 队首的请求接管
# 第  32 步 ┃ 又换一茬短请求；slot 始终接近满员
# 第1024 步 ┃ 最长请求结束——它身后已经换过好几茬人
# ```
#
# 座位数（`max_num_seqs`）一个没加。变的是**换人的时机**：静态只在批边界换，连续在每一步边界换。
# 同样的 slot-time，连续批处理装下了更多 token。

# %% [markdown]
# 打个比方：静态批处理像一桌必须**所有人在同一站下车**的拼车，中途有人到站了也只能坐着等；
# 连续批处理像公交，每一站都能上下客。车还是那辆车、座位还是那么多，差别只在「什么时候允许换人」。
# 换算成本章的数字：同样是 16 个 slot，实验 B 里两种策略的吞吐差了 2.55 倍——
# 多出来的不是算力，而是原本被空转吃掉的那些 slot-time。

# %% [markdown]
# 关键概念对比：
#
# | 维度 | 静态批处理 | 连续批处理 |
# | --- | --- | --- |
# | 组批时机 | 整批跑完才重新组批 | 每个 iteration 边界都重新组批 |
# | 新请求何时进入 | 下一批 | 有空 slot 的下一步 |
# | prefill 方式 | 整批 pad 到最长 prompt 一起算 | 新请求单独 prefill，与 decode 混批 |
# | 输出长度不一 | 全批等最长的那个 | 各自结束、立即释放 slot |
# | 主要浪费 | padding + 槽位空转 + 批间空窗 | 几乎没有 slot 浪费 |
# | 代价 | 吞吐低、排队久 | decode 被 prefill 顶住，ITL 尾部变差 |
#
# 上面两张时间轴是同一个到达序列的两种走法。差异不在于「算得快不快」，而在于
# 「谁在什么时刻被允许进来」。**本章要解决的两个子问题**：
#
# 1. **吞吐**：同一到达序列、同一块 GPU，如何让 batch 尽量满？（答案：iteration-level scheduling）
# 2. **代价**：换策略之后延迟怎么变？哪些指标变好、哪些变差？怎样用数字把这笔交易说清楚？
#
# **为什么用模拟器**：本章要回答的是调度策略的因果，不是某张卡的绝对性能。01/02 章已经确立两件事：
# prefill 是算力受限（耗时 ∝ token 数），decode 是带宽受限（每步读一遍权重，权重开销被 batch 摊薄，单步耗时增长远慢于 batch）。
# 于是把「GPU 一步」写成可配置的两行公式，就能隔离调度变量、跑出可信趋势；换任何模型只要改三个常数。

# %% [markdown]
# ## 05.2 高阶设计
#
# 动手前先把方案定下来。本章只有三个组件和一条数据流：`Request` 记录请求状态，成本模型把 GPU 一步抽象成耗时，
# 调度器决定每一步把哪些请求拼成一批。
#
# **组件 1：Request（存什么）**
#
# 每个请求只有三个静态字段（`prompt_len`、`max_new_tokens`、`arrival`）和两个运行期计数器：
# `num_computed`（KV 里已有的 context 长度）与 `num_generated`（已产出的 token 数）。
# `t_first`、`t_done`、`token_times` 是给指标用的观测值。
# `target_ctx()` 是贯穿全章的不变量：下一步 decode 之前，KV 里必须有多少 context token。
#
# **组件 2：成本模型（一步有多贵）**
#
# 把一次模型前向抽象成三项之和（单位 ms）：
#
# - 固定项 `base_ms`：一步读一遍权重；
# - prefill 项：`prefill_per_token_ms × 本步 prefill 的 token 数`；
# - decode 项：`decode_per_seq_ms × 本步 decode 的序列数`。
#
# 再加一笔 `prefill_launch_ms` 的发射成本。这个形状来自 01/02 章的 roofline：prefill 算力受限，所以对 token 线性；
# decode 带宽受限，所以对 batch 亚线性——后面实验 A 会用 TinyGPT 实测验证后一条。

# %% [markdown]
# **组件 3：调度策略（谁决定下一步）**
#
# 两种策略共用同一套「批次执行」逻辑，差别只在组批时机：
#
# - **静态批处理**：GPU 空闲时取 `max_num_seqs` 个请求 → pad 后整批 prefill → 一起 decode 到最长结束 → 才组下一批。
# - **连续批处理**：每个 iteration 先给 running 的请求各排 1 个 decode token，再用剩余 token 预算从 waiting 里接纳新请求；
#   完成的请求当步释放 slot，下一步就能被新请求复用。
#
# **数据流**：
#
# ```text
# 到达序列 → waiting 队列 → [组批] → 一步执行（prefill / decode）→ 更新 KV 计数与 token 时间
#                               ↑                                        │
#                               └──────── 完成的请求离开、slot 释放 ←────┘
#
# 执行过程同时记录：分段轨迹（画甘特图）+ 每步 batch 大小（算利用率）+ 每请求 token 时间（算 TTFT/ITL）
# ```
#
# 把连续批处理的「一步」摊开看，它其实只有五个动作，而前三个动作在静态批处理里每批只做一次：
#
# ```text
# 一个 iteration 的决策顺序（连续批处理）
# ① 到点了吗：把 pending 里 arrival ≤ now 的请求搬进 waiting
# ② 先排 decode：running 里每条序列各 1 个 token，先占掉这么多预算
# ③ 再排 prefill：用剩余预算从 waiting 队首整条塞（注意：整条，不能切）
# ④ 一步执行：耗时 = 固定项 + prefill token × 单价 + decode 序列数 × 单价
# ⑤ 边界换人：finished 的请求释放 slot，下一步就能被新请求复用
# ```
#
# ②③ 的顺序藏着一个陷阱：decode 先占预算，prefill 只能吃剩下的。第 06 章的 chunked prefill
# 就是来拆 ③ 里那个「整条」的。
#
# **指标口径**（全章统一，也是后面所有表的列）：
#
# - **TTFT**：请求到达到第一个 token 产出；**ITL / TPOT**：相邻输出 token 的间隔，取全量 token 的 p95；
# - **吞吐**：trace 产出的 token 数 ÷ makespan；**GPU 空闲率**：没有请求在跑的时间占比；
# - **batch 利用率**：活跃 slot 数 ÷ `max_num_seqs`，按步时长加权。

# %% [markdown]
# **接口契约**：
#
# | 接口 | 语义 | 谁用 |
# | --- | --- | --- |
# | `Request(prompt_len, max_new_tokens, arrival)` | 一个请求的静态信息 | 两种调度器 |
# | `target_ctx()` / `prefill_need()` | 下一步前 KV 需要多长 / 还差多少 | 连续调度、第 06 章 chunked prefill |
# | `run_static(reqs, cost, max_num_seqs)` | 静态批处理，返回指标 + 轨迹 | 实验 B |
# | `run_continuous(reqs, cost, max_num_seqs, max_num_batched_tokens)` | 连续批处理，返回指标 + 轨迹 | 实验 B/C、第 06 章 |
# | `Run.metrics / segments / steps` | 表与图的数据源 | 实验 B/C |
#
# **验收标准**（每一步都配一条可执行验收，写不出来说明没理解）：
#
# | 步骤 | 验收 |
# | --- | --- |
# | Step 1 Request | `target_ctx` / `needs_prefill` 的语义断言通过 |
# | Step 2 成本模型 | decode 对 batch 亚线性、prefill 边际成本对 token 线性 |
# | Step 3 模拟器与指标 | 指标定义与表列一致；`Run` 字段齐全 |
# | Step 4 静态批处理 | 同构请求的 makespan == prefill + (max_new−1) × decode |
# | Step 5 连续批处理 | 与静态在同到达同长度时结果相等；错峰时不比静态慢 |
# | Step 6 整体验收 | 每个请求 token 数正确；中途到达的短请求 TTFT 至少好 2x |

# %% [markdown]
# ## 05.3 从零实现：分六步
#
# | Step | 做什么 | 代码 |
# | --- | --- | --- |
# | 1 | 定义 `Request`：静态字段 + 运行期计数 + 语义方法 | `Request` |
# | 2 | 把 GPU 一步抽象成可配置耗时 | `CostModel` |
# | 3 | 事件驱动模拟器的结果容器与统一指标 | `Run` / `summarize` |
# | 4 | 静态批处理：整批 prefill、整批 decode 到最长 | `run_static` |
# | 5 | 连续批处理：每个 iteration 重新组批 | `run_continuous` |
# | 6 | 用断言验收两条路径的不变量 | 关键不变量断言 |
#
# 先准备依赖与设备：

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
# ### Step 1 · Request：一个请求需要记住什么
#
# 目标：把「请求从到达到结束」的状态压缩成几个计数字段，让调度器只依赖这些字段做决策。
#
# 做法：
# - 静态字段（`prompt_len` / `max_new_tokens` / `arrival`）来自负载，运行期不改；
# - `num_computed` 记录 KV 里已有的 context，`num_generated` 记录已产出的 token 数；
# - `target_ctx()` 回答「下一步 decode 前 KV 必须有多长」，`prefill_need()` 回答「还差多少」。
#
# 验收：新请求必须需要 prefill；prefill 完成并产出第一个 token 后，应该可以直接 decode。

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


# 验收：fresh 请求需要 prefill；prefill 完成、产出第一个 token 后即可 decode
r = Request(id=0, prompt_len=8, max_new_tokens=3, arrival=0.0)
assert r.target_ctx() == 8 and r.prefill_need() == 8 and r.needs_prefill()
r.num_computed, r.num_generated = 8, 1
assert r.target_ctx() == 8 and r.prefill_need() == 0 and not r.needs_prefill()

# %% [markdown]
# 字段少带来一个好处：两种调度器、以及第 06 章的 chunked prefill 都只跟这 5 个字段打交道，接口保持稳定。

# %% [markdown]
# ### Step 2 · CostModel：把「GPU 一步」写成两行公式
#
# 目标：用三个常数把一步前向的耗时表达出来，既不依赖真实模型，又能复现 01/02 章的瓶颈结论。
#
# 做法：`step_ms(prefill_tokens, decode_seqs)` 把固定权重读取、prefill 计算、decode 附加计算相加；
# prefill token 非零时再加一笔 kernel 发射成本。
#
# 验收：先把两条关键假设写成断言——decode 对 batch 亚线性，prefill 的边际成本对 token 线性。

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
# 两条断言通过，说明成本模型在数量关系上是对的：batch 从 1 到 32，decode 单步只从 6.12ms 涨到 9.84ms（1.6x），
# 远小于 32x；而 prefill 每多 1024 个 token，耗时正好多 `1024 × 0.08 = 81.92ms`。
# 这种「固定成本 + 线性增量」的形状，是后面所有吞吐结论的来源。

# %% [markdown]
# ### Step 3 · 模拟器骨架：结果容器与统一指标
#
# 目标：一次模拟要同时产出「给表格的指标」和「给图的轨迹」，并且指标口径只定义一次。
#
# 做法：
# - `Run` 保存 4 类数据：`metrics`（表）、`segments`（甘特图）、`steps`（利用率）、`batch_info`（浪费拆解）；
# - `summarize` 从请求的 `token_times` 与 `t_first/t_done` 计算 TTFT、ITL、E2E、吞吐、空闲率、利用率。
#
# 验收：后面 Step 4/5 的任何一次运行都能直接生成表与图，不需要再写第二套统计。

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
# ### Step 4 · 静态批处理：整批 prefill、整批 decode 到最长
#
# 目标：实现最朴素的「request-level batching」，它也是后面所有对比的基线。
#
# 做法：GPU 空闲时从 waiting 取一批 → pad 到最长 prompt 后整批 prefill（`batch_info` 记录 padding 浪费）→
# 一起 decode 到最长请求结束。已完成的请求不再占算力，但它们的 slot 也不会被新请求复用——
# 这正是「等最长请求」的来源，也是连续批处理要消灭的东西。

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
# ### Step 5 · 连续批处理：每个 iteration 重新组批
#
# 目标：让 batch 在每个 decode step 的边界都能换人。
#
# 做法：
# 1. 把已到达的请求放进 `waiting`；
# 2. **先给正在 decode 的请求各排 1 个 token**，剩余 token 预算再用来接纳新请求的 prefill；
# 3. 一步执行完，完成的请求释放 slot，下一个 iteration 立刻有新请求补进来。
#
# 这里有一个必须记住的限制：本章还没有 chunked prefill，一条 prompt 必须**整体塞进当前一步的 token 预算**，
# 否则只能等。这条限制就是第 06 章要拆掉的天花板。

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
# ### Step 6 · 验收：两条路径的不变量
#
# 目标：用可执行的断言证明实现正确，而不是靠肉眼看图。
#
# 四条验收：
# 1. 同构请求的静态 makespan 等于手算值 `prefill + (max_new−1) × decode`；
# 2. 同到达、同长度时，连续批处理与静态批处理结果完全相等；
# 3. 错峰到达时，连续批处理的中途短请求 TTFT 至少好 2x（新请求不必等整批）；
# 4. 每个请求的 token 时间数量恰好等于 `max_new_tokens`。

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
# 四条验收全过：静态 makespan 与手算一致；同构场景下两种策略等价；把第二个请求的到达时间推迟到 0.05s 后，
# 静态要等第一个请求整批跑完（TTFT p50 186ms），连续批处理在下一个 iteration 就接住了（15ms）。
#
# 到这里，调度器实现部分结束。接下来用三组实验回答「到底差多少、差在哪、参数怎么影响」。

# %% [markdown]
# ## 05.4 实测
#
# 三个实验，分别回答三个问题：
#
# | 实验 | 问题 | 产出 |
# | --- | --- | --- |
# | A | 成本模型的两条假设站得住吗？ | TinyGPT 实测 vs 模型（decode 对 batch 亚线性） |
# | B | 连续批处理到底赢在哪？ | 吞吐/延迟/空闲率对比表 + 甘特图 + 三笔浪费拆解 |
# | C | 负载与参数怎么改变工作点？ | 到达率扫描曲线 + `max_num_seqs` / `max_num_batched_tokens` 扫描表 |
#
# 测量协议：模拟器用固定 seed 生成到达序列；标定实验用本机 TinyGPT（M3 Ultra / MPS / torch 2.15，
# `warmup=2, reps=5` 取中位数）。模拟器的成本常数按 A100 上的 7B 模型量级设置，
# 绝对值不跨机器比较，只比较相对趋势。

# %% [markdown]
# ### 实验 A · 标定：decode 一步真的与 batch 无关吗
#
# 问题：整个模拟器建立在「decode 单步耗时对 batch 亚线性」这条假设上，必须先验证。
#
# **先猜一下**：batch 从 1 拉到 32，decode 的单步耗时应该变多少倍？
#
# (a) 1.6 倍——权重只读一遍，多出来的只是几十个 token 的乘加
# (b) 5 倍——请求多了，总要多花点时间
# (c) 32 倍——每条序列都要单独算一遍
#
# 提示：先想清楚 decode 每步的大头是「读权重」还是「算 token」。
#
# 协议：固定权重与上下文（ctx=32），batch 从 1 拉到 32；每个配置测 20 个连续 decode step 的平均耗时，
# 与成本模型的预测并排比较。

# %%

cfg = GPTConfig()
tiny = TinyGPT(cfg).to(device).eval()


@torch.no_grad()
def decode_loop(batch: int, steps: int = 20, ctx: int = 32):
    def fn():
        cache = KVCache(cfg, batch_size=batch, max_seq_len=ctx + steps + 1, device=device)
        tiny(torch.randint(0, cfg.vocab_size, (batch, ctx), device=device), start_pos=0, kv_cache=cache)
        cache.advance(ctx)                     # prefill 的 32 个 token 必须先推进长度
        tok = torch.randint(0, cfg.vocab_size, (batch, 1), device=device)
        for _ in range(steps):
            tiny(tok, start_pos=cache.seq_len, kv_cache=cache)
            cache.advance(1)                   # 每步只追加 1 个 token
    return fn


rows = []
for b in [1, 2, 4, 8, 16, 32]:
    r = bench(decode_loop(b), warmup=2, reps=5, name=f"decode_b{b}", device=device)
    rows.append({"batch": b, "实测单步(ms)": r.median_ms / 20, "模型单步(ms)": cost.step_ms(0, b)})
df_cal = pd.DataFrame(rows)
df_cal["实测/模型"] = df_cal["实测单步(ms)"] / df_cal["模型单步(ms)"]
df_cal.round(3)

# %% [markdown]
# **读表**：batch 从 1 到 32，TinyGPT 的每步耗时从 3.3 ms 只涨到 3.7 ms（约 12%，中间点有 ±25% 的噪声），
# 而线性增长会是 32 倍。模型预测的 6.1–9.8 ms 系统性高于实测——因为 MPS 上小模型每步有约 2–3 ms 的
# 固定开销（Python 调用 + kernel 启动 + 同步），这部分「理论单步」里根本没算。**绝对值对不上不影响结论**：
# 两者都远低于线性增长，亚线性假设成立，后面只比较相对倍数。
#
# （回到上面的先猜）答案是 (a)：按成本模型，6.12ms → 9.84ms 约 1.6 倍，实测更极端，1→32 只涨了约 12%。
# 选 (c) 看起来对，是因为把 decode 想成了「32 条序列各跑一遍模型」；实际上 32 条序列挤在同一次前向里，
# 权重只读一遍，多出来的只是 32 份 KV 读取与乘加。选 (b) 则是拿「请求数」当「工作量」——
# decode 的工作量是**读多少字节**，不是有多少条请求。

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
# 图里三条线：红线是「如果每条请求串行」的线性增长，橙线是成本模型，蓝线是 TinyGPT 实测。
# 蓝线与橙线都贴在下方，斜率远小于红线——这就是 batching 收益的物理来源。

# %% [markdown]
# ### 实验 B · 静态 vs 连续：三笔浪费，换回 2.55 倍吞吐与 18 倍 TTFT
#
# 问题：同一到达序列、同一 `max_num_seqs=16`，连续批处理能赢多少？赢在哪个指标上？
#
# 协议：64 个请求，Poisson 到达 λ=16 req/s；prompt 长度 128–2048，输出重尾（中位数 32，长尾到 1024）。
# 两边吃**同一条**到达序列（各自重新生成，模拟器会就地修改请求对象，不能共用实例）。

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

# %%

static = run_static(chat_workload(), cost, max_num_seqs=16)
continuous = run_continuous(chat_workload(), cost, max_num_seqs=16)  # 各自新建一份，互不污染
table = pd.DataFrame([static.metrics, continuous.metrics])
ratio = continuous.metrics["吞吐(tokens/s)"] / static.metrics["吞吐(tokens/s)"]
print(f"连续/静态 吞吐倍数 = {ratio:.2f}x")
table.round(2)

# %% [markdown]
# **读表**，先看四个数字：
#
# - **吞吐 2.55x**：同一块「GPU」、同一条到达序列，只换调度策略。
# - **TTFT p95 从 21.3s 降到 1.2s（约 18x）**：静态批处理里请求到达后要等当前整批跑完才有机会进来。
# - **E2E p95 从 22.6s 降到 6.2s（约 3.6x）**：尾部请求不用排长队。
# - **batch 利用率 0.37 vs 0.61**：GPU 并不是「闲着」（空闲率两行都是 0），而是**忙而无效**——
#   静态批处理里大量 slot 在等最长请求，算力被 padding 和空转吃掉。
#
# 再看两个浪费指标：静态批处理 **57% 的 prefill 算力花在 padding 上**，**82% 的 decode slot-time 是空转**。
# 但注意：padding 可以靠 03 章的分页注意力消掉，而「等最长请求」消不掉——后面的参数实验会继续验证这一点。
#
# （回到 05.1 的先猜）答案落在 **(b) 2–3 倍**：实测 2.55x。选 (a) 的人把调度当成「顺序上的微调」——
# 它确实不增加任何算力，但它把 82% 的 decode slot-time 从空转里捞了回来；选 (c) 的人假设 GPU 原本有一半闲着，
# 可这两行的 GPU 空闲率都是 0，它的问题是**忙而无效**，不是没在忙。
#
# 这两个浪费指标还能换算成更直观的账：16 个 slot、平均 batch 利用率 0.37，
# 意味着同一时刻真正在产出 token 的只有**约 6 个**（0.37 × 16），另外 10 个在等最长的邻居。
# 这就像 16 个收银台只开了 6 个，剩下 10 个店员站在台后，等最后一位顾客慢慢装袋。
# padding 那笔同样直白：57% 的 prefill 算力在替别人的 prompt 长度付费。

# %% [markdown]
# 把这组指标换成用户的体感。同一个尾部请求，在两种策略下的经历：
#
# ```text
# 静态批处理（p95 那一档请求）
# t=0      ┃ 按下回车，请求进 waiting —— 但当前整批还没跑完，只能等
# t=21.3s  ┃ 第一颗字出现（TTFT p95）——之前的 21 秒屏幕上什么都没有
# 之后     ┃ 一旦轮到自己就很顺：ITL p95 = 7.6ms
#
# 连续批处理（同一个请求）
# t=0      ┃ 按下回车
# t=1.2s   ┃ 第一颗字出现（TTFT p95，约 18x 改善）
# 之后     ┃ 每颗字之间的等待却变长了：ITL p95 = 29.3ms
# ```
#
# 最后一行是这一章的账单：连续批处理为了把等待摊平，把 prefill 和 decode 挤进同一步，
# 长 prefill 每来一次，正在 decode 的请求就被按住一次。
#
# 换句话说：感觉上吞吐翻 2.55 倍应该让所有人都更快；实际上被改善的是**排队**，
# 已经在 decode 的请求反而更慢了。第 06 章来还这笔债。

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
# 把三笔浪费拆开看：
#
# 1. **padding**：静态批处理 57% 的 prefill 算力花在 pad 到最长 prompt 上；这一笔可以用 03 章的分页注意力消掉。
# 2. **等最长请求**：82% 的 decode slot-time 是空转；只有「每个 iteration 重新组批」才能消掉这一笔。
# 3. **批间空窗**：整批不结束，新请求进不来，队列越压越长——TTFT p95 = 21.3s 就是这么堆出来的。
#
# 连续批处理同时消灭第 2、3 笔，对第 1 笔无能为力（它仍然逐条 prefill）。
# 这也解释了为什么工业实现里「连续批处理」和「分页注意力」要一起用：它们修的是不同的浪费。

# %% [markdown]
# ### 实验 C · 工作点：到达率与两个调度参数把拐点推到哪里
#
# 问题：连续批处理的优势在不同负载下是否稳定？`max_num_seqs` 与 `max_num_batched_tokens`
# 分别怎么改变工作点？
#
# **先猜一下**：把 `max_num_batched_tokens` 从 4096 砍到 2048（负载里最长的 prompt 恰好 2048），
# 吞吐会掉多少？
#
# (a) 0——预算只是个上限，够用就行
# (b) 10–20%——偶尔有几条请求要多等一步
# (c) 直接腰斩——长 prompt 可能永远凑不齐一步
#
# 答案在下面两张扫描表之后。
#
# 协议：到达率 λ ∈ {2, 4, 8, 16, 32, 64}，每个 λ 重新生成 64 个请求；固定 `max_num_seqs=16`。
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
# **读图**：静态批处理在 ~2.2 req/s 就撞墙，λ 再大也只增加排队时间（点往上走、不往右走）；
# 连续批处理一路涨到 ~6 req/s，p95 仍只有静态的 1/3。
# 两条曲线的「拐点」就是 GPU 真正饱和的位置：拐点左侧是延迟受限，右侧是吞吐受限。
# 生产上调参的目标，就是让工作点尽量靠近拐点、但不越过 SLO 线。

# %% [markdown]
# 接着把两个调度参数分别扫一遍。它们是人话里的「两个闸门」：
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
#   继续加到 32，吞吐不再涨（prefill 成为瓶颈），利用率却掉到 0.34——加并发只是在摊薄 batch 的有效负载。
# - `max_num_batched_tokens=2048` 时吞吐掉到 319：负载里有 2048 token 的 prompt，而一步预算被 decode 占掉一部分，
#   这些请求永远凑不齐「一整条 prompt 进同一步」，只能等 GPU 跑空。
#
# （回到先猜）答案是 **(c)**：吞吐 792 → 319，掉 60%。
# (b) 看起来最合理，因为 2048 恰好等于负载里最长的 prompt——直觉上「预算够长，偶尔多等一步而已」。
# 但 decode 每步先占走一部分预算，剩下的永远小于 2048，长 prompt 于是**不是多等一步，是永远排不进**。
#
# 记两个经验公式（后面章节会反复用）：
#
# - 吞吐 ≈ `min(max_num_seqs / 单请求每步成本 × 每步 token 数, 显存带宽 / 每 token 字节)`；
# - 不开 chunked prefill 时，`max_num_batched_tokens ≥ 最长 prompt + 并发 decode 数`，否则会出现「永远排不进」。
#
# > **⚠️ 坑：把「连续批处理」理解成「把 batch 调大」**
# > 看到吞吐翻倍，第一反应常是把 `--max-num-seqs` 从 16 调到 32。实测结果正好相反：
# > 吞吐不涨（prefill 顶住了），平均 batch 利用率从 0.61 掉到 0.34，TTFT 还被拉长。
# > 连续批处理改的是**组批时机**，不是 batch 大小；调大并发只是把同一份 GPU 时间摊得更薄。
#
# 最后把实验 B/C 连起来回答本章的问题：静态批处理吞吐低，是因为它同时付了 padding、等最长请求、批间空窗三笔钱；
# 连续批处理修掉后两笔，padding 留给分页注意力；而 `max_num_batched_tokens` 的断崖说明，
# **不做 chunked prefill 就永远要被「整条 prompt」绑架**——这就是下一章的入口。

# %% [markdown]
# ## 05.5 对照工业实现（vLLM）
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
# **差距在哪**：
#
# 1. **KV 是分页的**（03 章）：vLLM 的 `KVCacheManager` 按 block 分配，同样的 `max_num_seqs` 能装下更多请求，
#    也没有 padding；我们的模拟器只记「token 数」。
# 2. **V1 默认就是 chunked prefill**：`_schedule_running` 给 running 排 token 时，会把长 prompt 拆成
#    `max_num_batched_tokens - num_decode_tokens` 的 chunk，与 decode 同批。第 06 章手写这个逻辑。
# 3. **抢占**：显存不够时 vLLM 会把 running 队尾的请求 preempt（recompute 或 swap），本章模拟器在第 06 章补上。
# 4. **优先级 / 多租户**：vLLM 有 `priority`、`--scheduling-policy`，本章只做 FCFS。

# %% [markdown]
# 注：V1 在 0.8–0.10 期间重构过 scheduler 的方法拆分，类名与路径以课程锁定的 0.28.x 源码为准；
# 用 `grep -n "def schedule" vllm/v1/core/sched/scheduler.py` 打开文件对一遍最稳。
#
# 在 vLLM 里能直接看到的证据：`--max-num-seqs` 调小后 TTFT 下降、吞吐下降；
# `--max-num-batched-tokens` 调到小于最长 prompt 时，日志会警告长请求排不进来。
# 本章的两张扫描表就是这两句话的手写版。

# %% [markdown]
# ## 05.6 练习与小结
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
# - 为什么 `max_num_batched_tokens` 会有断崖？它和 chunked prefill 是什么关系？

# %% [markdown]
# > **一句话记住**：连续批处理不是让 GPU 更忙，而是让「忙」的每一格都有用。
#
# **自检 3 问**（不看笔记答出来才算过）
#
# 1. 连续批处理与静态批处理差在「哪个边界换人」？为什么这一个时机的差别能让同样的 16 个 slot 吞吐翻倍？
# 2. `max_num_seqs` 与 `max_num_batched_tokens` 各限制什么？为什么不加 chunked prefill 时，
#    把后者调到比最长 prompt 还小，吞吐会断崖式下跌？
# 3. 静态批处理的三笔浪费里，哪一笔能靠 03 章的分页注意力消掉，哪一笔只能靠调度消掉？
#
# **真实工作场景题**：你接手一个推理服务，监控里 GPU 利用率一直很高，用户却在抱怨「十几秒才出第一个字」，
# 而每步 batch 里真正在产出 token 的 slot 不到一半。给出你的排查顺序与判据：
# 先看等待队列还是先看 batch 组成？动调度策略之前，哪两个参数必须一起看？
#
# **小结与下一章**
#
# 本章的连续批处理把吞吐拉起来，但它的 ITL p95（29.3ms）比静态批处理（7.6ms）更差：
# 一条 2048 token 的 prefill 一旦和 decode 同批，所有正在 decode 的请求都要多等约 170ms。
# 第 06 章用两种手段修这个问题：**chunked prefill**（把 prefill 切小、与 decode 交替）与**抢占**（KV 不够时让请求让位）。
#
# **延伸阅读**
#
# - Orca 论文（OSDI'22）：iteration-level scheduling 与 selective batching 的原始出处
# - vLLM V1 scheduler：`vllm/v1/core/sched/scheduler.py` 的 `schedule()` / `_schedule_running` / `_schedule_prefills`
# - vLLM 文档：`--max-num-seqs`、`--max-num-batched-tokens` 的调参建议
