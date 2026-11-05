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
# 1. prefill 和 decode 为什么是两种完全不同的负载？
# 2. 为什么大模型 decode 的速度上限由显存带宽决定，而不是算力？
# 3. 你的 Mac 实测带宽是多少？用它推算 8B 模型的理论 tok/s 是多少？

# %% [markdown]
# ## 01.1 问题与直觉
#
# 一次生成分两个阶段，它们的瓶颈完全不同：
#
# - **prefill**：把整段 prompt 一次性喂进去。T 个 token 同时算，矩阵乘规模大，**算力受限（compute-bound）**。
# - **decode**：每次只生成 1 个 token。矩阵乘变成"向量 × 矩阵"，几乎不产生计算，却要把全部权重读一遍，**带宽受限（memory-bound）**。
#
# 用 roofline 模型看得很清楚：
#
# ![](../assets/figures/ch01_roofline.svg)
#
# 横轴是**算术强度**（每读 1 字节能算多少次浮点运算），纵轴是可达性能。斜线是带宽上限，横线是算力上限，实际性能取两者较小值。
# decode 落在最左边（AI ≈ 1），prefill 落在右边（AI ≈ T）。**这一章我们亲手测出这两个位置。**
#
# 先记住两个估算公式，后面每章都会用：
#
# - decode 理论 tok/s ≈ 显存带宽 ÷ (参数量 × 每参数字节数)
# - prefill 理论耗时 ≈ 2 × 参数量 × prompt 长度 ÷ 有效算力

# %% [markdown]
# ## 01.2 从零实现
#
# ### 1) 先看我们手里的实验对象
#
# `TinyGPT` 与 Llama 同构（RMSNorm + RoPE + MHA + SwiGLU），约 2.8M 参数，随机权重。我们只关心它的**形状和字节数**，不关心它生成什么。

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
# 把参数账本拆开：哪一层占多少字节，决定了 decode 每步要读多少内存。

# %%
rows = []
for name, p in model.named_parameters():
    rows.append({"参数": name, "数量": p.numel(), "fp32 字节": p.numel() * 4})
rows.append({"参数": "合计", "数量": model.num_params(), "fp32 字节": model.num_params() * 4})
params_df = pd.DataFrame(rows)
params_df

# %% [markdown]
# ### 2) 手写一次 attention，并与框架实现对照
#
# 所有"分页""批处理"的花活，底层都是这一个公式。先确认我们理解的和 PyTorch 做的是同一件事：

# %%
B, T, H, D = 2, 64, 4, 64
q, k, v = (torch.randn(B, H, T, D, device=device) for _ in range(3))


def manual_attention(q, k, v):
    scores = q @ k.transpose(-2, -1) / math.sqrt(D)  # (B,H,T,T)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


ours = manual_attention(q, k, v)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
assert torch.allclose(ours, ref, atol=1e-5), "手写 attention 与框架实现不一致"
print("✅ 手写 attention 与 SDPA 一致，最大误差 =", (ours - ref).abs().max().item())

# %% [markdown]
# ### 3) 采样：贪心、温度、top-k
#
# 模型输出 logits，采样策略决定下一个 token。三个断言把行为钉死：贪心必须确定性、温度趋近 0 必须退化成贪心、top-k 必须屏蔽掉低概率 token。

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
# ### 实测 2：prefill 随 prompt 变长，吞吐先升后平
#
# prefill 一次处理 T 个 token，浮点运算量 ≈ 2 × 参数量 × T，读取的权重是固定的。
# 所以 T 越大，算术强度越高，越接近算力上限。

# %%
rows = []
for T in [64, 128, 256, 512]:
    ids = torch.randint(0, cfg.vocab_size, (1, T), device=device)
    r = bench(lambda: model(ids), name=f"prefill T={T}", warmup=2, reps=5, device=device)
    rows.append(
        {
            "prompt 长度": T,
            "耗时(ms)": round(r.median_ms, 2),
            "tokens/s": round(T / (r.median_ms / 1000), 1),
            "TFLOP/s": round(2 * model.num_params() * T / (r.median_ms / 1000) / 1e12, 3),
        }
    )
prefill_df = pd.DataFrame(rows)
prefill_df

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
