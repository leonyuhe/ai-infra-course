# %% [markdown]
# # 第 01 章 · 一次前向：从矩阵乘法到 token 采样
#
# > 生成一个 token 到底花了多少时间？答案取决于你在**算**还是在**搬数据**。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 75 分钟 |
# | 运行环境 | 🏠 Mac（CPU/MPS）· 无需 GPU |
# | 前置章节 | 00 |
# | 本章接口 | `aiinfra.tinygpt.TinyGPT`、`sample_next` |
#
# **学完你能回答：**
# 1. prefill 和 decode 为什么是两种完全不同的负载（不只是快慢不同）？
# 2. 为什么大模型 decode 的速度上限由显存带宽决定，而不是算力？
# 3. 什么时候 decode 也会变成算力受限？
# 4. 你的 Mac 实测带宽是多少？用它推算 8B 模型的理论 tok/s 是多少？

# %% [markdown]
# ## 01.1 问题与直觉
#
# 生成一段回答 = **一次 prefill + T 次 decode**。这两个阶段不是"快慢不同"，而是**负载类型不同**。
#
# **prefill**：把 S 个 prompt token 排成矩阵，一次性喂给模型。S 个位置**并行**计算（一次大矩阵乘），
# 每个权重只读一次，所以算力能用满；代价是 FLOPs ≈ 2 × 参数量 × S。它同时把每层算出的 K/V 写进 cache。
#
# **decode**：自回归决定了必须**一个一个**生成。每次只有 1 个 token，形状从 (S, d) 变成 (1, d)：
# 矩阵乘退化成"向量 × 矩阵"，几乎不产生计算，却仍要把**全部权重**重新读一遍，还要读越来越长的 KV cache。
#
# 一句话记住差别：**prefill 是"并行地算很多"，decode 是"串行地搬很多"。**
# 所以 prefill 吃算力，decode 吃带宽。

# %% [markdown]
# ### 一张表看清两种负载
#
# | | prefill | decode |
# | --- | --- | --- |
# | 一次处理 | S 个 token（并行） | 1 个 token（串行） |
# | 每 token 读取的权重 | 权重总量 ÷ S（被摊薄） | 全部权重 |
# | 算术强度 AI | ≈ S | ≈ 1–2 |
# | 主要瓶颈 | 算力（FLOPS） | 带宽（bytes/s） |
# | 次要瓶颈 | 写 KV cache、attention 的 O(S²) | 读 KV cache、kernel 启动开销 |
# | 时间估算 | ≈ (2·P·S + 4·L·d·S²) / F | ≈ (权重字节 + KV 读取字节) / B |
# | 优化方向 | 更高 FLOPS、FlashAttention、量化、前缀缓存 | 更高带宽、量化、加大 batch、投机解码 |
#
# P = 参数量，L = 层数，d = 隐藏维度，S = prompt 长度，B = 有效带宽，F = 有效算力。
# 两个公式后面每章都会用：
#
# - **prefill 理论耗时 ≈ (2·P·S + 4·L·d·S²) ÷ 有效算力**（第二项是 attention，短 prompt 可先忽略）
# - **decode 理论 tok/s ≈ 显存带宽 ÷ (参数量 × 每参数字节数)**

# %% [markdown]
# ### 诊断口诀
#
# - **decode 慢 → 先查带宽。** 单步时间 ≈ (权重字节 + KV 读取字节) ÷ 有效带宽。
#   带宽不够时：换带宽更高的卡、量化（字节减半）、减小 KV（GQA / fp8 KV）、加大 batch（摊薄权重）。
# - **prefill 慢 → 先看算力和 S。** 长 prompt 时算力主导；短 prompt 时权重读取占比上升，带宽也有份。
#   要更快：提高有效 FLOPS（更好的 kernel、更低精度、更多 SM），或者少算（前缀缓存命中）。
#
# 由此可以推出一个很实用的判断：
#
# > 如果 **decode 稳定达标**，说明这台机器的**带宽**对你的模型/上下文够用；
# > 此时想让 **prefill 更快，主要得提升算力**（或减少要算的 token），而不是继续堆带宽。
#
# 两个必须知道的例外：
#
# 1. **batch 变大时，decode 会从带宽受限转向算力受限。** 权重只读一次、却被 batch 个请求摊薄，
#    算术强度从 ≈1 涨到 ≈batch，算力成为新瓶颈。这正是第 05 章连续批处理能提升吞吐的物理基础。
# 2. **模型极小时，decode 可能既不是带宽也不是算力受限。** 权重放得下 cache、单步只有几微秒时，
#    kernel 启动 / Python 调度 / 采样开销会占主导——就是第 00 章说的"测量地板"。
#
# roofline 图把上面的表画成了一张图：
#
# ![](../assets/figures/ch01_roofline.svg)
#
# 横轴是算术强度（每读 1 字节做多少次浮点运算），斜线是带宽上限，横线是算力上限。
# decode 落在最左端（AI ≈ 1），prefill 在右上（AI ≈ S）；01.3 会亲手测出这两个位置。

# %% [markdown]
# ## 01.2 从零实现
#
# ### 1) 先看全貌：一个 token 是怎么变成下一个 token 的
#
# 01.1 讲了"算"和"搬"，这一节把它落到一个具体模型上。实验对象 `TinyGPT` 与 Llama 同构（RMSNorm + RoPE + MHA + SwiGLU），
# 只是缩小到约 2.8M 参数。整条流水线是：
#
# ![](../assets/figures/ch01_pipeline.svg)
#
# 生成一个 token，就是把这条流水线跑一次：
#
# 1. **token id → 向量**：查 embedding 表（4096 × 256），把每个 id 变成 256 维向量；
# 2. **2 × Transformer Block**：每层先做 attention（让每个位置"看"前面的位置），再做 SwiGLU MLP（逐位置变换）；
# 3. **LM head**：把最后位置的 256 维隐藏向量乘成 4096 维 logits——词表里每个 token 一个原始分数；
# 4. **softmax → 概率 → 采样**：按概率抽一个 token，接到序列末尾，再跑下一次。
#
# 权重是随机的，所以它生成的"话"是乱码。**本章只关心形状、字节数和耗时**，不关心生成质量。

# %%
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path.cwd().parent))

from aiinfra.bench import bench, pick_device, sync
from aiinfra.plotting import save, setup_style
from aiinfra.tinygpt import GPTConfig, KVCache, TinyGPT, sample_next

setup_style()
device = pick_device()
torch.manual_seed(0)
np.random.seed(0)

cfg = GPTConfig()
model = TinyGPT(cfg).to(device).eval()
print("device =", device, "| params =", f"{model.num_params():,}")

# %% [markdown]
# ### 2) 参数账本：decode 每步要搬多少字节
#
# 先把参数按层拆开。decode 时每一步都要把**全部参数**从内存读进计算单元，所以这张表就是"每 token 的搬运量账本"。

# %%
rows = []
for name, p in model.named_parameters():
    rows.append({"参数": name, "数量": p.numel(), "fp32 字节": p.numel() * 4})
rows.append({"参数": "合计", "数量": model.num_params(), "fp32 字节": model.num_params() * 4})
params_df = pd.DataFrame(rows)
params_df

# %% [markdown]
# 合计 **2,753,792 个参数 × 4 字节 = 11,015,168 字节 ≈ 10.5 MiB**。
#
# 这个数字就是 01.1 里"decode 吃带宽"的具体化：每生成 1 个 token，都要把这 10.5 MiB 权重从内存搬到计算单元。
# 用 01.3 测出的带宽一除就是理论上限；而 TinyGPT 只有 10.5 MiB，固定开销会盖过带宽（01.3 会看到"理论 5 万、实测几百"）。
# 换成 8B 模型（fp16 约 16 GB）时，固定开销可以忽略，带宽才成为真正的硬上限。

# %% [markdown]
# ### 3) 跑一次前向，看清每一层的形状
#
# 把形状打印出来，比读代码更直观。**`(B, S, d_model)` 贯穿整个网络**，最后一步才变成 `(B, S, vocab)`。

# %%
ids = torch.randint(0, cfg.vocab_size, (1, 6), device=device)
x = model.tok_emb(ids)
print("token ids     ", tuple(ids.shape))
print("embedding     ", tuple(x.shape))
for i, block in enumerate(model.blocks):
    x = block(x)
    print(f"block {i} output", tuple(x.shape))
hidden = model.norm_f(x)
logits = model.lm_head(hidden)
print("hidden        ", tuple(hidden.shape))
print("logits        ", tuple(logits.shape))
assert logits.shape == (1, 6, cfg.vocab_size)

# %% [markdown]
# 生成时只取**最后一个位置**：`logits[:, -1, :]`，形状 `(1, 4096)`。
# 因为这个位置已经通过 attention"看过"前面的所有 token，它的 hidden 就是整段上下文的压缩表示；
# 乘上 LM head 后，它变成"下一个 token 是谁"的 4096 个分数。

# %% [markdown]
# ### 4) attention：每个位置如何"看"前面的位置
#
# attention 做三件事（Q、K、V 都是同一个输入乘三个不同矩阵得到的）：
#
# | 步骤 | 公式 | 直觉 |
# | --- | --- | --- |
# | 打分 | Q · Kᵀ / √d | 当前 token 对每个历史 token 的"相关度" |
# | 归一化 | softmax(·) + 因果掩码 | 只能看自己和之前，权重和为 1 |
# | 聚合 | 权重 · V | 按相关度把历史信息加权求和 |
#
# 手写一遍，是为了把公式和形状钉死，再和 PyTorch 的融合实现对照。

# %%
B, T, H, D = 2, 64, 4, 64
q, k, v = (torch.randn(B, H, T, D, device=device) for _ in range(3))


def manual_attention(q, k, v):
    scores = q @ k.transpose(-2, -1) / math.sqrt(D)       # (B,H,T,T) 两两相关度
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))      # 因果掩码：屏蔽未来位置
    weights = F.softmax(scores, dim=-1)                   # 每个 query 的权重和为 1
    return weights @ v                                    # 按权重聚合 value


ours = manual_attention(q, k, v)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
assert torch.allclose(ours, ref, atol=1e-5), "手写 attention 与框架实现不一致"
print("✅ 手写 attention 与 SDPA 一致，最大误差 =", (ours - ref).abs().max().item())

# %% [markdown]
# **SDPA 是什么？** `torch.nn.functional.scaled_dot_product_attention` 是 PyTorch 2.0 起的融合算子：
# 数学上就是上面那三行，实现上会按硬件自动选择 FlashAttention / memory-efficient / 朴素 kernel。
# vLLM 生产里用的就是这一类融合 attention kernel。
#
# 那为什么还要手写？三个原因：
#
# 1. **验证理解**：等价断言就是测试，证明公式没记错；
# 2. **留下参考实现**：第 03 章把连续 KV 换成 block table 分页后，用这个朴素版校验结果是否正确；
# 3. **看清成本**：打分矩阵是 S × S，attention 是 O(S²) 计算 + 每层 O(S) 的 KV 读写——这正是后面 chunked prefill、前缀缓存要优化的地方。

# %% [markdown]
# ### 5) 采样：从最后一维向量到下一个 token
#
# 最后一个位置的 hidden 是一个 256 维向量，它本身不是词，可以理解为"到目前为止整段上下文"的压缩表示。
# LM head 把它乘成 4096 维 **logits**：词表里每个 token 一个分数。softmax 之后就是概率分布。
#
# 三种最常见的采样策略：
#
# - **贪心（temperature = 0）**：直接取分数最高的 token。确定、可复现，适合评测。
# - **温度 T**：把 logits 除以 T。T < 1 分布更尖（更保守），T > 1 更平（更随机）。
# - **top-k**：只保留分数最高的 k 个，重新归一化后再采样，砍掉长尾乱码。
#   （top-p 是同一类思路：保留累计概率到 p 的最小集合。）
#
# 用同一组 logits 看温度怎么改变概率：

# %%
raw = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
rows = []
for T in [0.5, 1.0, 2.0]:
    probs = F.softmax(raw / T, dim=-1)
    rows.append({"温度 T": T, **{f"token {i}": round(float(probs[i]), 3) for i in range(4)}})
pd.DataFrame(rows)

# %% [markdown]
# 三个断言把采样行为钉死：贪心必须确定性、温度趋近 0 必须退化成贪心、top-k 必须屏蔽低概率 token。
# 再跑 4000 次，验证经验频率收敛到 softmax 概率：

# %%
logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device)
assert sample_next(logits, temperature=0).item() == 3          # 贪心选最大
assert sample_next(logits, temperature=1e-6).item() == 3       # 近似贪心
assert sample_next(logits, temperature=1.0, top_k=2).item() in (2, 3)  # 只在前 2 里选

torch.manual_seed(0)
draws = torch.tensor([sample_next(logits, temperature=1.0).item() for _ in range(4000)])
empirical = torch.bincount(draws, minlength=4).float() / len(draws)
pd.DataFrame(
    {
        "token": [0, 1, 2, 3],
        "softmax 理论概率": F.softmax(logits, dim=-1).squeeze().tolist(),
        "采样 4000 次频率": empirical.tolist(),
    }
).round(3)

# %% [markdown]
# ### 6) 回到 infra：这一章在算什么
#
# 采样本身几乎不耗算力（一次 argmax / multinomial），但它决定了输出风格，也是 OpenAI API 里 `temperature`、`top_p` 参数的来源。
#
# 一次 forward 的成本集中在两处：**attention**（O(S²) 分数矩阵 + KV 读写）和 **LM head**（d × V 的大矩阵乘）。
# 生成时每走一步，还要把全部权重读一遍——TinyGPT 是 10.5 MiB，8B 模型是 16 GB。
# 01.3 我们就把这三个数字亲手测出来：带宽、prefill 吞吐、decode tok/s。

# %% [markdown]
# ## 01.3 实测：算力、带宽与 roofline
#
# ### 实测 1：这台机器的有效带宽
#
# 用一次大 tensor 的 `clone()` 测：读 N 字节 + 写 N 字节。这是"理论带宽"最接近的可测替代。

# %%
def measure_bandwidth(size_mb: int = 256, reps: int = 5) -> float:
    n = size_mb * 1024 * 1024 // 4
    x = torch.empty(n, device=device)
    samples = []
    for _ in range(reps):
        sync(device)
        start = time.perf_counter()
        y = x.clone()
        sync(device)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return 2 * n * 4 / samples[len(samples) // 2] / 1e9  # GB/s


bw_measured = measure_bandwidth()
# 官方带宽规格只对 GPU 有意义：M3 Ultra 800 GB/s、A100 约 2039 GB/s；CPU 没有统一规格。
bw_spec = {"mps": 800.0, "cuda": 2039.0}.get(device.type)
rows = [{"指标": "实测 clone 带宽", "数值": f"{bw_measured:.1f} GB/s"}]
if bw_spec:
    rows += [
        {"指标": "本机官方规格", "数值": f"{bw_spec:.0f} GB/s"},
        {"指标": "达成率", "数值": f"{bw_measured / bw_spec:.0%}"},
    ]
else:
    rows += [{"指标": "官方规格", "数值": "CPU 无统一规格，只看实测"}]
pd.DataFrame(rows)

# %% [markdown]
# 在 GPU（MPS / CUDA）上，实测带宽通常是规格的 40%–70%（取决于 tensor 大小、是否命中缓存、后端效率）。
# CPU 上带宽由内存通道和单核效率决定，绝对值会低一个数量级——所以本页在不同机器上跑出的数字不同是正常的。
# **后面所有理论推算都用实测带宽**，不要用官网数字——这是性能估算最重要的纪律。

# %% [markdown]
# ### 实测 2：prefill 的计算量、上下文长度与"attention 反噬"
#
# 先把符号说清楚（后面会反复用）：
#
# | 符号 | 含义 | TinyGPT | wide 对照 |
# | --- | --- | --- | --- |
# | P | 参数量（权重个数） | 2,753,792 | 18.9M |
# | L | Transformer 层数 | 2 | 4 |
# | d | 隐藏维度 d_model | 256 | 512 |
# | T / S | 序列长度（prompt token 数） | 变量 | 变量 |
#
# **为什么每 token 是 2P**：前向时每个权重参与一次"乘 + 加"（1 MAC = 2 FLOPs），
# 所以每 token 的权重计算量 = 参数量 × 2。
#
# **attention 的 4·L·d·T 从哪来**：每层 attention 要算两张 T×T 的矩阵乘——
# `Q·Kᵀ`（2·T²·d FLOPs）和 `权重·V`（2·T²·d FLOPs），合计 4·T²·d；L 层就是 4·L·d·T²，
# 再摊到 T 个 token 上 = 4·L·d·T。
#
# 于是：
#
# - 总 FLOPs ≈ **2·P·T + 4·L·d·T²**
# - 每 token FLOPs ≈ **2·P + 4·L·d·T**
# - tokens/s 上限 ≈ 有效 FLOPS ÷ (2·P + 4·L·d·T)
#
# 第二项随 T 线性增长：**T 越大，每个 token 要干的活越多**。这就是"attention 反噬"。

# %% [markdown]
# ### 什么是"模型设计上下文"（max_seq_len）
#
# TinyGPT 的 `GPTConfig` 里有一个 `max_seq_len = 512`。它只在 `KVCache` 里用来**限制缓存长度**
# （预分配、防溢出），并不阻止你把更长的 T 喂给 forward。但它是"这个模型设计给多长上下文"的声明：
#
# - **真实模型**有训练时的位置编码范围，超过它 RoPE 外推会明显掉质量；
# - **vLLM 服务**里对应 `--max-model-len`，超过的请求直接被拒绝；
# - 做性能实验时，T 远超上下文没有意义：attention 的 O(T²) 会主导结果，测到的是"超长上下文的代价"。
#
# 所以下面只扫到 2048（对 512 上下文已是 4 倍），不再往 14800 这类数字上推。

# %%
def prefill_flops(P: int, L: int, d: int, T: int) -> int:
    """prefill 总 FLOPs ≈ 权重部分 2PT + attention 部分 4LdT²。"""
    return 2 * P * T + 4 * L * d * T * T


models = {
    "tiny (2.75M)": TinyGPT(GPTConfig()).to(device).eval(),
    "wide (18.9M)": TinyGPT(GPTConfig(d_model=512, n_layer=4, n_head=8, d_ff=2048)).to(device).eval(),
}
rows = []
for name, m in models.items():
    P, L, d = m.num_params(), m.cfg.n_layer, m.cfg.d_model
    for T in [64, 128, 256, 512, 1024, 2048]:
        ids = torch.randint(0, m.cfg.vocab_size, (1, T), device=device)
        r = bench(lambda: m(ids), name=f"{name} T={T}", warmup=3, reps=9, device=device)
        sec = r.median_ms / 1000
        flops = prefill_flops(P, L, d, T)
        rows.append(
            {
                "模型": name,
                "T": T,
                "耗时(ms)": round(r.median_ms, 2),
                "tokens/s": round(T / sec),
                "TFLOP/s(仅权重)": round(2 * P * T / sec / 1e12, 2),
                "TFLOP/s(含attn)": round(flops / sec / 1e12, 2),
                "attention占比": f"{4 * L * d * T * T / flops:.0%}",
                "attn分数矩阵(MiB)": round(m.cfg.n_head * T * T * 4 / 1024**2, 1),
            }
        )
prefill_df = pd.DataFrame(rows)
prefill_df

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
eff_col = "TFLOP/s(含attn)"  # 中文列名先取到英文变量，避免绘图行触发图内中文检查
for name, sub in prefill_df.groupby("模型", sort=False):
    axes[0].plot(sub["T"], sub["tokens/s"], "o-", label=name)
    axes[1].plot(sub["T"], sub[eff_col], "o-", label=name)
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xlabel("prompt length T (tokens)")
    ax.legend()
axes[0].set_ylabel("prefill throughput (tokens/s)")
axes[0].set_title("tokens/s: rises, then the attention tax bites")
axes[1].set_ylabel("effective TFLOP/s (incl. attention)")
axes[1].set_title("effective TFLOP/s keeps rising: hardware is not the issue")
save(fig, "ch01_prefill_scaling");

# %% [markdown]
# 读图要点：
#
# 1. **tokens/s 先升**：T 小时一步只有几毫秒，kernel 启动等固定开销占大头，T 越大摊得越薄。
# 2. **wide 在 1024 后下降**：每 token 的工作量 `2P + 4LdT` 随 T 线性增长，分母变大，tokens/s 自然掉——
#    **不是硬件变慢，是每个 token 的计算量变多了**。
# 3. **"TFLOP/s 下降"是统计口径的假象**：`TFLOP/s(仅权重)` 只数了 2PT，没数 attention。
#    看 `TFLOP/s(含attn)` 这一列——它**没有下降，仍在上升**，说明硬件效率没退化；
#    变的是每个 token 的工作量（看 `attention占比` 列，wide 从 1% 涨到 31%）。
# 4. tiny 为什么一直涨：2P = 5.5M，4LdT 到 2048 才 4.2M，两项相当还没到屋顶；
#    wide 的 attention 项更大，所以更早看到峰值和下滑。
# 5. 这也解释了 14800 的"崩"：每 token 工作量 ≈ 5.5M + 30.3M = 35.8 MFLOP，
#    按 4.3 TFLOP/s 算就是 ~120k tokens/s——和观测一致，不是硬件极限。
#
# 真实推理里压住 O(T²)：**FlashAttention**（不物化 S×S 矩阵，省访存）和
# **chunked prefill**（第 06 章，把长 prompt 切块，限制单步 S）。

# %% [markdown]
# ### 实测 3：decode 每步只喂 1 个 token
#
# 用 KV cache，每步输入形状是 (1, 1)。这一步几乎不"算"，只是在读权重和 cache。

# %%
prompt = torch.randint(0, cfg.vocab_size, (1, 64), device=device)


def decode_n_steps(n: int = 24) -> torch.Tensor:
    cache = KVCache(cfg, max_seq_len=64 + n, device=device)
    logits = model(prompt, start_pos=0, kv_cache=cache)
    cache.advance(prompt.shape[1])
    nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for _ in range(n):
        logits = model(nxt, start_pos=cache.seq_len, kv_cache=cache)
        cache.advance(1)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return nxt


r = bench(decode_n_steps, 24, name="decode 24 步", warmup=2, reps=5, device=device)
per_token_ms = r.median_ms / 24
achieved_bw = model.num_params() * 4 / (per_token_ms / 1000) / 1e9
print(f"每 token 延迟: {per_token_ms:.2f} ms  ->  {1000 / per_token_ms:.1f} tokens/s")
print(f"对应权重读取带宽: {achieved_bw:.1f} GB/s  (实测峰值 {bw_measured:.1f} GB/s)")

# %% [markdown]
# 注意：2.8M 参数 × fp32 = 11MB，理论上以 500GB/s 读一遍只要 0.02ms，也就是理论 5 万 tok/s。
# 实测通常只有几百 tok/s——**差了两个数量级**。原因不是模型算得慢，而是：
#
# 1. 小模型每步的 kernel 启动、Python 调度、MPS 命令缓冲开销占主导；
# 2. batch=1 时 GPU 根本喂不饱，带宽利用率极低。
#
# 这个反例说明：**小模型测不出带宽瓶颈**。真实 8B 模型时，权重 16GB，这些固定开销占比就很小了。
# 所以性能估算要"账本推算 + 大模型实测"结合，不能拿玩具模型的绝对值外推。

# %% [markdown]
# ### 实测 4：把两种负载放到 roofline 上
#
# 用实测带宽和实测算力画图，再把 8B 模型（fp16）的 decode/prefill 位置标上去。

# %%
a = torch.randn(4096, 4096, device=device)
r = bench(lambda: a @ a, name="4096 matmul", warmup=2, reps=5, device=device)
peak_tflops = 2 * 4096**3 / (r.median_ms / 1000) / 1e12

P = 8e9          # 8B 参数
bytes_per_param = 2  # fp16
ai = np.logspace(-1, 4, 200)
roof = np.minimum(bw_measured * 1e9 * ai, peak_tflops * 1e12) / 1e12

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.loglog(ai, roof, color="#2563EB", lw=2.5, label="Achievable performance")
ax.axhline(peak_tflops, color="#7C3AED", ls="--", lw=1.5, label=f"Compute roof ≈ {peak_tflops:.1f} TFLOP/s")
ax.axvline(1, color="#9CA3AF", ls=":", lw=1)

decode_ai = 2 * P / (P * bytes_per_param)   # ≈ 1 FLOP/byte
decode_y = min(bw_measured * 1e9 * decode_ai, peak_tflops * 1e12) / 1e12
ax.scatter([decode_ai], [decode_y], s=90, color="#EF4444", zorder=5)
ax.annotate("decode (8B, fp16)\nmemory-bound", (decode_ai, decode_y), xytext=(2, 6),
            fontsize=10, color="#991B1B", arrowprops=dict(arrowstyle="->", color="#EF4444"))

prefill_ai = 2048.0
prefill_y = min(bw_measured * 1e9 * prefill_ai, peak_tflops * 1e12) / 1e12
ax.scatter([prefill_ai], [prefill_y], s=90, color="#10B981", zorder=5)
ax.annotate("prefill (T=2048)\ncompute-bound", (prefill_ai, prefill_y), xytext=(60, prefill_y * 0.12),
            fontsize=10, color="#065F46", arrowprops=dict(arrowstyle="->", color="#10B981"))

ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
ax.set_ylabel("Achievable performance (TFLOP/s)")
ax.set_title("Roofline: prefill and decode hit different walls")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, which="both", alpha=0.3)
save(fig, "ch01_roofline");

# %% [markdown]
# 结论表：用实测参数推算 8B 模型，看看理论上限和"你实际会看到的数字"差多少。

# %%
theory = pd.DataFrame(
    [
        {
            "场景": "8B fp16 decode（batch=1）",
            "瓶颈": "带宽",
            "理论值": f"{bw_measured * 1e9 / (P * 2):.1f} tokens/s",
            "公式": "带宽 ÷ (参数量 × 2B)",
        },
        {
            "场景": "8B fp16 prefill（T=2048）",
            "瓶颈": "算力",
            "理论值": f"{2 * P * 2048 / (peak_tflops * 1e12) * 1000:.1f} ms",
            "公式": "2 × P × T ÷ 有效算力",
        },
        {
            "场景": "KV cache（8B, 32 层, 8 KV heads, 128 dim, 4K ctx, fp16）",
            "瓶颈": "容量",
            "理论值": f"{2 * 32 * 8 * 128 * 4096 * 2 / 1024**3:.1f} GiB / 条请求",
            "公式": "2 × L × H_kv × D × S × dtype",
        },
    ]
)
theory

# %% [markdown]
# ## 01.4 对照工业实现（vLLM）
#
# | 我们的概念 | vLLM 中的对应物 | 位置 |
# | --- | --- | --- |
# | prefill / decode 两阶段 | 由 scheduler 分别调度，V1 里是 `Prefill` / `Decode` 请求类型 | `vllm/v1/core/sched/scheduler.py` |
# | 显存账本 | `profile_run` + `gpu_memory_utilization` 决定 KV cache 块数 | `vllm/v1/worker/gpu_worker.py` |
# | 每步读全部权重 | `ModelRunner` 的 forward；decode 时 batch 内所有请求共享一次权重读取 | `vllm/v1/worker/gpu_model_runner.py` |
# | 算术强度 | 决定了 chunked prefill 的收益（把 prefill 切小塞进 decode 之间的空隙） | `--enable-chunked-prefill` |
#
# 差距在哪：
#
# - 我们的 `TinyGPT` 用 PyTorch eager 执行，每个算子一次 kernel launch；vLLM 用 **CUDA Graph** 把 decode 的固定形状调用固化，消除 launch 开销（这也是小模型在 Mac 上跑不快的原因之一）。
# - vLLM 的 decode 会把多个请求**拼成一个 batch**，把 batch=1 时浪费的带宽重新利用起来——这正是第 05 章连续批处理要解决的问题。
# - `gpu_memory_utilization` 的默认 0.9 不是拍脑袋：它决定了 KV cache 能开多少个块，进而决定 `max_num_seqs` 的实际上限。

# %% [markdown]
# ## 01.5 练习与小结
#
# **练习**
#
# 1. ⭐ 把 `size_mb` 从 256 改成 16，实测带宽会怎么变？解释 cache 与 kernel 开销的影响。
# 2. ⭐ 把 TinyGPT 的 `d_model` 从 256 改成 512、`n_layer` 从 2 改成 4，重跑 decode 实测。延迟变化符合"参数翻倍 → 时间翻倍"吗？为什么不是线性的？
# 3. ⭐⭐ 用实测带宽和算力，推算 Llama-3-8B（32 层、8 KV heads、head_dim=128）在 4K/32K 上下文、batch 8 时的 KV cache 总显存，并与公式对比。
# 4. ⭐⭐ 给 roofline 图加第三条线：int4 量化（bytes_per_param=0.5）后的 decode 位置，解释量化为什么能提速。
# 5. ⭐⭐⭐ 设计一个实验，分离「kernel launch 开销」和「真实计算时间」。提示：固定计算量，改变调用次数；或对比 eager 与 `torch.compile`。
#
# **你现在应该能回答：**
#
# - decode 的理论上限由什么决定？为什么 batch 能提高吞吐但不能降低单请求 TPOT？
# - 为什么小模型在 Mac 上"看起来很慢"，而公式说它应该很快？
# - 什么时候该怀疑"带宽不够"，什么时候该怀疑"算力不够"？
#
# **小结与下一章**
#
# 这一章我们发现了 decode 的本质：**每生成一个 token，都要把权重重新读一遍**。
# 下一章的问题是：每个 token 还要把历史的 K/V 重新算一遍吗？答案是 KV cache——以及它带来的显存灾难。
#
# **延伸阅读**
#
# - Roofline 模型原始论文（Williams et al., 2009）
# - vLLM 文档：Optimization and Tuning / Conserving Memory
