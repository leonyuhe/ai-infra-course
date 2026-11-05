# %% [markdown]
# # 第 12 章 · 多卡并行：DP / TP / PP 切的是什么？
#
# > 单卡显存装不下、单卡吞吐喂不饱时，唯一的路是把计算切开；而切开的代价，全部记在通信账本上。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 75 分钟 |
# | 运行环境 | ☁️ 云上 NVIDIA GPU（本地 Mac 用线程模拟 all-reduce，无需 GPU） |
# | 前置章节 | 11（压测与调优） |
# | 本章接口 | `aiinfra.bench.bench`；`TinyGPT` 只作为显存/算力的参照物 |
# | vLLM 版本 | 0.28.x |
#
# **学完你能回答：**
# 1. DP、TP、PP 各自切的是什么，通信原语和通信量分别是什么？
# 2. 一个 decode token 的 TP 通信量怎么算？为什么同一套 TP=8 在 NVLink 上效率 89%、在以太网上只剩 50%？
# 3. 为什么 TP 通常不跨机、PP 可以跨机、DP 天然跨机？三者在一个集群里怎么组合？
#
# **本章的实验方式**：Mac 上没有多卡，也没有 NVLink。所以第二节「从零实现」用纯 PyTorch 在
# CPU 上把一个 Linear 切给 2/4/8 个「虚拟 rank」，用线程 + 队列模拟 ring all-reduce；
# 真实的通信时间**不靠实测**，而是用链路带宽 + 每 token 通信量算出来。云上实测脚本见
# `cloud/scripts/12_tp_bench.sh`。

# %% [markdown]
# ## 12.1 问题与直觉
#
# 三种并行切的东西完全不同，所以它们的通信账本也完全不同：
#
# ![](../assets/figures/ch12_tp_split.svg)
#
# - **DP（数据并行）**：模型**完整复制**到每张卡，每个 rank 吃不同的 batch 切片。反向传播后
#   每个 rank 都有一份「自己那份数据的梯度」，必须 all-reduce 求平均，才能保证各卡权重仍然一致。
#   切的是**数据**，通信量正比于**参数量**。
# - **TP（张量并行）**：把一个 Linear 的权重矩阵按列或按行切开，每张卡只算一部分输出。
#   切的是**单个算子内部**，每层都要 all-reduce 一次激活，通信量正比于**激活大小**。
# - **PP（流水并行）**：按层切，stage0 算前几层、stage1 算后几层，边界用点对点（P2P）传激活。
#   切的是**深度**，通信量小，但会引入流水空转（bubble）。
#
# 一个反例：把 8B 模型的 TP=8 直接铺在 25 GB/s 的以太网上。直觉上「8 张卡算力 ×8」，
# 实际上每层两次 all-reduce 的延迟远大于计算时间，扩展效率掉到 50% —— 花了 8 张卡的钱，
# 只拿到 4 张卡的吞吐。**这一章就是要把这笔账算清楚。**
#
# 扩展效率的定义（本章统一使用）：
#
# ```
# 扩展效率 = 单卡耗时 / (TP 卡数 × TP 后单 token 耗时)
#          = 计算时间 / (计算时间 + 卡数 × 通信时间)
# ```
#
# 其中 decode 阶段是**访存受限**（第 01 章的结论）：单卡计算时间 ≈ 权重字节数 ÷ HBM 带宽，
# 切给 TP 张卡后每卡只读 1/TP 的权重，所以计算时间 ÷ TP，而通信时间是额外开销。

# %% [markdown]
# ## 12.2 从零实现
#
# ### 1) 把一层的矩阵乘切给多个虚拟 rank
#
# 先约定数据和权重形状：`x: (B, in)`、`weight: (out, in)`、`y = x @ weight.T + bias`。
# 列并行（column parallel）按**输出维**切权重，每个 rank 算出一部分输出列，拼起来就是完整结果。

# %%
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path.cwd().parent))

from aiinfra.bench import bench
from aiinfra.plotting import save, setup_style

setup_style()
torch.manual_seed(0)
np.random.seed(0)
print("torch =", torch.__version__, "| 本章用 CPU 线程模拟通信，真实带宽靠模型推算")

# %%
def column_parallel_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, tp: int) -> torch.Tensor:
    """列并行：weight (out, in) 按 out 维切 tp 份，每个 rank 算一部分输出列。"""
    w_chunks = torch.chunk(weight, tp, dim=0)
    b_chunks = torch.chunk(bias, tp, dim=0)
    partials = [x @ w.T + b for w, b in zip(w_chunks, b_chunks)]
    return torch.cat(partials, dim=-1)


def row_parallel_linear(x: torch.Tensor, weight: torch.Tensor, tp: int) -> torch.Tensor:
    """行并行：weight (out, in) 按 in 维切 tp 份，每个 rank 算部分和，最后 all-reduce 求和。"""
    w_chunks = torch.chunk(weight, tp, dim=1)
    x_chunks = torch.chunk(x, tp, dim=-1)
    partials = [xi @ wi.T for wi, xi in zip(w_chunks, x_chunks)]
    return torch.stack(partials).sum(dim=0)  # 等价于 all-reduce(SUM)


B, IN, OUT = 8, 256, 512
x = torch.randn(B, IN)
w_col = torch.randn(OUT, IN)
bias = torch.randn(OUT)
ref = x @ w_col.T + bias
for tp in (2, 4):
    got = column_parallel_linear(x, w_col, bias, tp)
    assert torch.allclose(got, ref, atol=1e-5), f"TP={tp} 列并行结果不一致"
print(f"✅ 列并行 TP=4 与单卡一致，最大误差 = {(column_parallel_linear(x, w_col, bias, 4) - ref).abs().max().item():.2e}")

# %% [markdown]
# 行并行是它的镜像：按**输入维**切权重，每个 rank 只拿到一部分输入和一部分权重，算出的
# 是一个「部分和」，必须跨 rank 求和（all-reduce）才能得到正确结果。
#
# 为什么真实模型要把两者配对？因为 Transformer 的 MLP 是 `y = gelu(x W1ᵀ) W2ᵀ`：
# 用列并行切 W1（按输出维），得到完整的隐层 h；再对 W2 用行并行（按输入维），
# **中间不需要任何通信**，只在最后做一次 all-reduce。Megatron-LM 的 TP 就是这么设计的。

# %%
def dense_mlp(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    return F.gelu(x @ w1.T) @ w2.T


def tp_mlp(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, tp: int) -> torch.Tensor:
    """W1 列并行 + W2 行并行：中间无通信，只在最后做一次 all-reduce。"""
    h = F.gelu(column_parallel_linear(x, w1, torch.zeros(w1.shape[0]), tp))
    return row_parallel_linear(h, w2, tp)


w1 = torch.randn(512, IN)   # (hidden, in)
w2 = torch.randn(OUT, 512)  # (out, hidden)
ref_mlp = dense_mlp(x, w1, w2)
for tp in (2, 4):
    got = tp_mlp(x, w1, w2, tp)
    assert torch.allclose(got, ref_mlp, atol=1e-4, rtol=1e-4), f"TP={tp} MLP 结果不一致"
print(f"✅ TP MLP: 列并行 + gelu + 行并行 = 单卡结果，最大误差 = {(tp_mlp(x, w1, w2, 4) - ref_mlp).abs().max().item():.2e}")
print(f"   最后一次 all-reduce 的消息大小 = {tuple(ref_mlp.shape)} × 4B = {ref_mlp.numel() * 4} B（每个 token）")

# %% [markdown]
# ### 2) 用线程 + 队列模拟 ring all-reduce
#
# 真实的多卡通信是 NCCL 的 ring all-reduce：每个 rank 只和左右邻居交换数据，分两个阶段
# （scatter-reduce 求和、all-gather 广播），每个 rank 搬 `2(n-1)/n × 消息大小` 字节。
# 我们用一个线程代表一个 rank，用 `queue.Queue` 代表 rank 之间的链路——算法结构和 NCCL 一致，
# 只是「链路」变成了进程内的队列，所以**只能验证正确性，不能测真实带宽**。

# %%
def ring_allreduce(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """线程模拟 ring all-reduce：每个 rank 最终都拿到所有 rank 的逐元素和。"""
    n = len(tensors)
    chunks = [list(torch.chunk(t, n)) for t in tensors]
    results: list[torch.Tensor | None] = [None] * n
    qs = [[queue.Queue() for _ in range(n)] for _ in range(n)]  # qs[src][dst]
    barrier = threading.Barrier(n)

    def rank_worker(r: int) -> None:
        buf = [c.clone() for c in chunks[r]]
        right, left = (r + 1) % n, (r - 1) % n
        barrier.wait()  # 所有 rank 就绪后再开始
        for k in range(n - 1):  # scatter-reduce
            qs[r][right].put(buf[(r - k) % n])
            recv = qs[left][r].get()
            idx = (r - k - 1) % n
            buf[idx] = buf[idx] + recv
        for k in range(n - 1):  # all-gather
            qs[r][right].put(buf[(r - k + 1) % n])
            buf[(r - k) % n] = qs[left][r].get()
        results[r] = torch.cat(buf)

    threads = [threading.Thread(target=rank_worker, args=(r,)) for r in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results  # type: ignore[return-value]

# %%
for tp in (2, 4, 8):
    parts = [torch.full((8192,), float(r + 1)) for r in range(tp)]
    expect = sum(parts)
    outs = ring_allreduce(parts)
    err = max((o - expect).abs().max().item() for o in outs)
    assert all(torch.allclose(o, expect, atol=1e-4) for o in outs), f"TP={tp} ring all-reduce 结果错误"
    print(f"✅ TP={tp}: 8 个 rank 的 ring all-reduce 与串行求和一致，最大误差 {err:.2e}")

# %% [markdown]
# 正确性有了，但**别把线程耗时当成链路耗时**。下面测一下本机线程版的开销，再和带宽模型对比。

# %% [markdown]
# ## 12.3 实测：通信账本与扩展效率
#
# ### 实测 1：线程模拟的耗时（测的是 Python 开销，不是带宽）
#
# 消息 256 KB、TP=2/4/8，每个 rank 都跑一遍 ring。看着是毫秒级，但真实 NVLink 传 256 KB
# 只需要 0.3 µs 左右——差 4 个数量级。线程版的价值只在「算法正确」，通信时间必须另算。

# %%
def one_allreduce(tp: int) -> None:
    ring_allreduce([torch.randn(1 << 16) for _ in range(tp)])  # 每 rank 256 KB 消息


rows = []
for tp in (2, 4, 8):
    r = bench(lambda: one_allreduce(tp), name=f"ring tp={tp}", warmup=1, reps=3, device="cpu")
    rows.append({"TP 度": tp, "每 rank 消息(KB)": 256, "线程 all-reduce 中位数(ms)": round(r.median_ms, 3)})
thread_df = pd.DataFrame(rows)
thread_df

# %% [markdown]
# ### 实测 2：用带宽模型算真实通信时间
#
# 单个 decode token（batch=1）在 TP 下的通信量：
#
# ```
# 每层 2 次 all-reduce（attention 输出 + MLP 输出）
# 每次 ring all-reduce 搬运 2(n-1)/n × 消息大小 字节
# 消息大小 = hidden × dtype_bytes = 4096 × 2 = 8 KB
# 每 token 通信量 = 32 层 × 2 次 × 2×(n-1)/n × 8 KB
# ```
#
# 通信时间 = 传输时间 + 固定延迟 × collective 次数。固定延迟在万兆以太网下约 15 µs/次，
# 64 次 collective 就是 ~1 ms，比「8B 模型单卡 8 ms 的访存时间 ÷ 8 = 1 ms」还大——这就是
# TP=8 不能跨以太网的根本原因。参数口径：8B fp16，A100 80G（HBM ≈ 2 TB/s）。

# %%
@dataclass
class ModelShape:
    params: float = 8e9
    hidden: int = 4096
    n_layer: int = 32
    dtype_bytes: int = 2
    hbm_bw_gbps: float = 2000.0


@dataclass
class Link:
    name: str
    bw_gbps: float
    latency_us: float  # 每次 collective 的固定延迟


MODEL = ModelShape()
LINKS = [
    Link("NVLink 4", 900.0, 2.0),
    Link("PCIe Gen4", 64.0, 5.0),
    Link("Ethernet 25G", 25.0, 15.0),
]


def comm_bytes_per_token(tp: int, m: ModelShape) -> float:
    """decode（batch=1）单个 token 的 TP 通信量：每层 2 次 ring all-reduce。"""
    if tp == 1:
        return 0.0
    msg_bytes = m.hidden * m.dtype_bytes
    return m.n_layer * 2 * 2 * (tp - 1) / tp * msg_bytes


t1_ms = MODEL.params * MODEL.dtype_bytes / (MODEL.hbm_bw_gbps * 1e9) * 1e3
print(f"单卡 decode 一个 token ≈ {t1_ms:.1f} ms（8B × 2B ÷ 2TB/s）；TP=n 后每卡访存 ≈ {t1_ms:.1f}/n ms")

# %%
rows = []
for tp in (1, 2, 4, 8):
    for link in LINKS:
        comm_bytes = comm_bytes_per_token(tp, MODEL)
        n_coll = 0 if tp == 1 else MODEL.n_layer * 2
        comm_ms = comm_bytes / (link.bw_gbps * 1e9) * 1e3 + n_coll * link.latency_us / 1e3
        mem_ms = t1_ms / tp
        eff = t1_ms / (mem_ms + comm_ms) / tp
        rows.append(
            {
                "TP": tp,
                "互联": link.name,
                "每 token 通信(MB)": round(comm_bytes / 1e6, 3),
                "单卡访存(ms)": round(mem_ms, 2),
                "通信(ms)": round(comm_ms, 3),
                "计算/通信": "—" if comm_ms == 0 else f"{mem_ms / comm_ms:.1f}",
                "扩展效率": round(eff, 3),
            }
        )
tp_df = pd.DataFrame(rows)
eff = {(r["TP"], r["互联"]): r["扩展效率"] for r in rows}
assert eff[(8, "NVLink 4")] > eff[(8, "PCIe Gen4")] > eff[(8, "Ethernet 25G")]
assert eff[(1, "NVLink 4")] == 1.0
tp_df

# %% [markdown]
# 读表（数字来自上面的公式，绝对值和卡型有关，**结论看趋势**）：
#
# - TP=2 → 4 → 8，通信量只从 0.52 MB 涨到 0.92 MB（因为 ring 的 `2(n-1)/n` 收敛到 2），
#   但每卡的计算/访存时间在 **÷n**，所以通信占比越来越高。
# - TP=8：NVLink 扩展效率 0.886，PCIe 0.749，以太网只有 0.501。同一份代码、同一个模型，
#   换互联就掉一半效率。
# - 表中「通信(ms)」在 TP=8 时以太网是 0.997 ms，其中 0.96 ms 是 64 次 collective 的固定延迟，
#   **带宽只占 4%**。所以跨机 TP 的问题不是「带宽不够」，是「延迟太多、次数太多」。
#
# 这也解释了为什么 vLLM 的 TP 只在单机内开：单机 8 卡 NVLink 才是这套算法的合理载体。

# %%
fig, ax = plt.subplots(figsize=(7.2, 4.2))
colors = ["#2563EB", "#F59E0B", "#EF4444"]
tps = [1, 2, 4, 8]
for link, color in zip(LINKS, colors):
    effs = [eff[(tp, link.name)] for tp in tps]
    ax.plot(tps, effs, "o-", color=color, label=f"{link.name}（{link.bw_gbps:.0f} GB/s）")
ax.set_xticks(tps)
ax.set_ylim(0, 1.05)
ax.set_xlabel("TP degree (GPUs per node)")
ax.set_ylabel("Scaling efficiency (vs 1 GPU, 1.0 = linear)")
ax.axhline(0.9, color="#9CA3AF", ls=":", lw=1)
ax.annotate("TP=8 over Ethernet: half the throughput", xy=(8, 0.501), xytext=(4.2, 0.35),
            fontsize=10, color="#991B1B", arrowprops=dict(arrowstyle="->", color="#EF4444"))
ax.set_title("TP scaling efficiency: near-linear on NVLink, halved on Ethernet")
ax.legend(loc="lower left", fontsize=9)
save(fig, "ch12_tp_efficiency");

# %% [markdown]
# ### 实测 3：另外两本账 —— PP 的 bubble 与 DP 的梯度 all-reduce
#
# PP 切层不增加每层通信，但会「等」：一个 stage 在等上一个 stage 的激活时只能空转。
# 用 1F1B 调度时，空转比例近似 `(PP-1) / (microbatch + PP-1)`。microbatch 越大，bubble 占比越低。

# %%
rows = []
for pp in (2, 4):
    for mb in (4, 8, 16):
        rows.append({"PP 度": pp, "microbatch 数": mb, "bubble 比例": round((pp - 1) / (mb + pp - 1), 3)})
pp_df = pd.DataFrame(rows)
for pp in (2, 4):
    vals = [(pp - 1) / (mb + pp - 1) for mb in (4, 8, 16)]
    assert vals[0] > vals[1] > vals[2], "microbatch 越多，bubble 应该越小"
pp_df

# %% [markdown]
# DP 的通信量正比于**参数量**而不是激活：8B 模型 fp32 梯度 32 GB，每步 all-reduce 搬运
# `2(n-1)/n × 32 GB`。在 NVLink 上约 36–62 ms，在以太网上一秒以上——但 DP 的通信可以
# 和反向传播**重叠**（bucket 化 + 异步），而且它换来的是 N 倍的全局 batch。
# 所以「跨机 DP 可行、跨机 TP 不可行」不是带宽差 36 倍的问题，是**通信次数与重叠能力**的问题。

# %%
rows = []
for dp in (2, 4, 8):
    grad_bytes = 2 * (dp - 1) / dp * MODEL.params * 4  # fp32 梯度
    for link in LINKS:
        ms = grad_bytes / (link.bw_gbps * 1e9) * 1e3 + link.latency_us / 1e3
        rows.append({"DP 度": dp, "互联": link.name, "每步梯度 all-reduce(GB)": round(grad_bytes / 1e9, 1), "通信(ms)": round(ms, 1)})
dp_df = pd.DataFrame(rows)
assert dp_df["每步梯度 all-reduce(GB)"].nunique() == 3, "DP 度越高，ring all-reduce 的总搬运量应该越大"
dp_df

# %% [markdown]
# ## 12.4 对照工业实现（vLLM）
#
# | 我们的实现 | vLLM 中的对应物 | 文件/参数（vLLM 0.28.x） |
# | --- | --- | --- |
# | `column_parallel_linear` / `row_parallel_linear` | `ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear` | `vllm/model_executor/layers/linear.py` |
# | 每层 2 次 all-reduce | `tensor_model_parallel_all_reduce` | `vllm/distributed/communication_ops.py` |
# | `ring_allreduce` 的线程版 | NCCL / 自研 `custom_all_reduce`（单机小消息走 NVLink P2P） | `vllm/distributed/device_communicators/custom_all_reduce.py` |
# | PP 的 stage 划分与 P2P | `--pipeline-parallel-size`，worker 间传 `IntermediateTensors` | `vllm/v1/worker/gpu_worker.py` |
# | DP 的梯度 all-reduce | 训练侧概念；vLLM 推理侧对应 `--data-parallel-size`（复制模型、分摊请求） | 启动参数 `--data-parallel-size` |
# | 拓扑选择 | `--tensor-parallel-size` 一般 ≤ 单机 GPU 数；跨机用 PP/DP | `vllm serve --help` |
#
# 差距在哪：
#
# - **融合与重叠**：vLLM 把 QKV 三个投影合成一个 `QKVParallelLinear`，一次通信；我们的版本一层两次。
#   真实 TP 每层是 2 次 all-reduce（attention 输出、MLP 输出），工业实现用 CUDA Graph 把
#   这些通信和 kernel 固化成一张图，减少启动开销。
# - **通信后端**：单机内 NCCL 会选 NVLink P2P 或自研 all-reduce；跨机才走 IB/以太网。我们的
#   线程版只验证算法，不涉及任何真实链路。
# - **约束**：`--tensor-parallel-size` 必须整除 attention head 数（`n_head % tp == 0`），否则
#   无法均分 head；这就是我们 12.2 里「权重按 out 维切」的前提。
#
# 一句话：**TP 是「把一层拆开」，所以每层都要 synchronize；PP 是「把层拆开」，只需传边界激活；
# DP 是「把数据拆开」，只需同步梯度。** 三者的通信频率差了一个数量级，这才决定了它们
# 各自能跑在多快的互联上。

# %% [markdown]
# ## 12.5 练习与小结
#
# **练习**
#
# 1. ⭐ 把 `LINKS` 里的 PCIe 换成 PCIe Gen5（128 GB/s，延迟 4 µs），重跑 TP 表。TP=8 的扩展效率
#    提升多少？为什么提升幅度远小于带宽翻倍？
# 2. ⭐ 把 `dtype_bytes` 从 2 改成 1（FP8/int8 权重），通信量和扩展效率怎么变？为什么量化对
#    TP 的收益比对单卡更大？
# 3. ⭐⭐ 给 `ring_allreduce` 加一个「朴素版」：rank0 收齐所有 rank 的数据、求和、再广播回去。
#    验证结果一致，并用线程计时对比两种算法的开销，解释 ring 为什么在大 TP 下更优。
# 4. ⭐⭐ 用 DP 表算一笔账：8B 模型、全局 batch 64、每步计算量约 `6 × P × tokens`，在 25 GB/s
#    以太网上 DP=8 的通信/计算比是多少？结论是「必须重叠」还是「可以忽略」？
# 5. ⭐⭐⭐ 设计一个 64 卡集群（8 台 × 8 卡）训练 70B 模型的并行方案：TP/PP/DP 各取多少？
#    需要什么互联？预期 MFU 是多少？给出 3–5 条评分要点（显存、通信、bubble、故障域）。
#
# **你现在应该能回答：**
#
# - 为什么 TP 的通信量随 TP 度增长得很慢，但扩展效率还是随 TP 度下降？
# - 为什么「跨机 TP=8」在工程上几乎没人做，而「跨机 DP=64」很常见？
# - PP 的 bubble 和什么有关？microbatch 为什么会同时影响显存和吞吐？
#
# **小结与下一章**
#
# 这一章我们只算清了「多卡推理」的通信账本：TP 切一层、PP 切层、DP 切数据。
# 但真实系统里，模型能不能训出来，先取决于**显存够不够**：参数、梯度、优化器状态、
# 激活值四本账加起来，7B 模型在 DDP 下每卡要 112 GB。下一章我们手写 ZeRO 的账本与 LoRA，
# 把「训练侧 infra」这笔账算到底。
#
# **延伸阅读**
#
# - Megatron-LM 论文（Shoeybi et al., 2019）：tensor parallel 的列/行切分与 MLP 配对
# - ZeRO 论文（Rajbhandari et al., 2020）：显存账本与三阶段切分（第 13 章）
# - vLLM 文档：Tensor Parallelism / Pipeline Parallelism / Data Parallelism 部署章节
