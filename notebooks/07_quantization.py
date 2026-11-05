# %% [markdown]
# # 第 07 章 · 量化：把权重压小，把带宽省下来
#
# > decode 每生成一个 token 都要把全部权重读一遍（第 01 章）。权重小一半，理论上 token 就快一倍。
# > 这一章我们从零手写对称量化，量 TinyGPT 的真实权重，算出 7B/8B 的显存与 tok/s 账本。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 75 分钟 |
# | 运行环境 | 🏠 Mac（CPU/MPS）· 无需 GPU |
# | 前置章节 | 01（roofline 与 decode 带宽公式） |
# | 本章接口 | `aiinfra.tinygpt.TinyGPT` 的 qkv / mlp 权重；`aiinfra.bench` 的带宽与 decode 公式 |
# | 对照版本 | vLLM 0.28.x |
#
# **学完你能回答：**
# 1. 对称量化的一行公式是什么？误差由哪三件事决定？
# 2. per-tensor / per-channel / per-group 差多少？为什么工业界默认 `group_size=128`？
# 3. 7B/8B 模型在 fp16/int8/int4 下的权重显存、decode 理论 tok/s 各是多少？为什么量化主要收益在 decode？

# %% [markdown]
# ## 07.1 问题与直觉
#
# 量化就是**用更少的比特表示同一组浮点数**。对称量化的公式只有一行：
#
# ```text
# x_q = clamp(round(x / s), -qmax, qmax)，  qmax = 2^(b-1) - 1，  x̂ = s · x_q，  s = absmax / qmax
# ```
#
# `s` 是**步长**（相邻两个格点的距离），b 个比特最多表示 `2^b` 个格点。量化的全部误差来自三件事：
#
# ![](../assets/figures/ch07_quant_error.svg)
#
# 1. **舍入误差**：每个值最多偏离 `s/2`，这是不可避免的底噪；
# 2. **范围误差**：`absmax` 被一个离群值撑大，步长 `s` 跟着变大，所有正常值的噪声被同步放大；
# 3. **粒度**：per-tensor 全场共用一个 `s`；per-channel 每行一个；per-group 每 `g` 个元素一个。
#    越细，离群值越难污染别人，但额外存的 scale 也越多。
#
# **反例**：对真实 LLM 权重直接做 per-tensor int4，一个 20 倍离群值就能把步长撑大十几倍，
# 相对误差从 ~2% 掉到 >20%，输出分布直接崩（07.3 会在 TinyGPT 上把这个数字量出来）。
# 所以 AWQ / GPTQ 这些工业实现默认 `group_size=128`，而不是"一把梭 per-tensor"。
#
# 最后两个必须分清的概念：**对称量化**令零点 `z=0`，反量化只乘一次 scale，kernel 最简单；
# **非对称量化**多存一个 zero-point `z`，适合 ReLU 之后的偏斜分布。但 int4 的 `[-8,7]` 天生不对称，
# 工业实现通常仍走「对称 + group」的折中，本章主线也保持对称。

# %% [markdown]
# ## 07.2 从零实现
#
# ### 1) 接口：scale → 整数 → 反量化
#
# 先把契约钉死：`absmax_scale` 只负责算步长，`quantize` 返回整数值张量（int4 也先寄存在 `int8` 里，
# 真实存储时两个 int4 挤进一个字节），`dequantize` 乘回 scale。这样换粒度只改一个参数。

# %%
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path.cwd().parent))

from aiinfra.bench import bench, human_bytes, pick_device, sync
from aiinfra.plotting import save, setup_style
from aiinfra.tinygpt import GPTConfig, KVCache, TinyGPT

setup_style()
device = pick_device()
torch.manual_seed(0)
np.random.seed(0)

print("device =", device)

# %% [markdown]
# ### 2) 三个核心函数
#
# `absmax` 的求法决定了粒度：`tensor` 是全场一个标量；`channel` 沿输出维求（每行一个 scale）；
# `group` 先把最后一维切成 `in_features / group_size` 段，每段一个 scale。断言写在函数里，违反契约立刻报错。

# %%
def absmax_scale(x: torch.Tensor, bits: int, granularity: str = "tensor", group_size: int | None = None):
    """对称量化：s = absmax / qmax。返回与 x 同形状、可广播的 scale 和 qmax。"""
    assert x.dim() == 2, "权重形状是 (out_features, in_features)"
    assert granularity in ("tensor", "channel", "group")
    qmax = 2 ** (bits - 1) - 1
    if granularity == "tensor":
        amax = x.abs().amax()
    elif granularity == "channel":
        amax = x.abs().amax(dim=1, keepdim=True)
    else:
        assert group_size is not None and x.shape[1] % group_size == 0
        out_f, in_f = x.shape
        groups = x.reshape(out_f, in_f // group_size, group_size)
        amax = groups.abs().amax(dim=-1, keepdim=True)
        amax = amax.expand(out_f, in_f // group_size, group_size).reshape(out_f, in_f)
    return (amax / qmax).clamp_min(1e-8), qmax


def quantize(x: torch.Tensor, bits: int, granularity: str = "tensor", group_size: int | None = None):
    """返回整数值张量 q（int8 承载）与反量化用的 scale。"""
    scale, qmax = absmax_scale(x, bits, granularity, group_size)
    q = torch.clamp(torch.round(x / scale), -qmax, qmax).to(torch.int8)
    return q, scale


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale

# %% [markdown]
# ### 3) 误差指标与存储字节
#
# 三个误差指标各有用途：MSE 看整体噪声，最大绝对误差看最坏元素，
# **相对误差** `‖x−x̂‖/‖x‖` 最适合跨层、跨模型比较。存储字节必须把 scale 算进去：
# group 越细，权重越小但 scale 越多，存在一个最优点。

# %%
def quant_error(x: torch.Tensor, x_hat: torch.Tensor) -> dict:
    err = (x - x_hat).float()
    return {
        "MSE": err.pow(2).mean().item(),
        "最大绝对误差": err.abs().max().item(),
        "相对误差": (err.norm() / x.float().norm()).item(),
    }


def quant_bytes(shape: tuple, bits: int, granularity: str = "tensor", group_size: int | None = None) -> float:
    """权重 + scale 的真实存储字节（scale 用 fp16）。"""
    n = shape[0] * shape[1]
    if granularity == "tensor":
        n_scales = 1
    elif granularity == "channel":
        n_scales = shape[0]
    else:
        n_scales = shape[0] * (shape[1] // group_size)
    return n * bits / 8 + n_scales * 2


torch.manual_seed(0)
w = torch.randn(64, 256) * 0.03
w[:, 0] *= 30.0                                   # 单通道离群值：真实权重里存在
q8, s8 = quantize(w, 8, "tensor")
q4, s4 = quantize(w, 4, "tensor")
assert q8.dtype == torch.int8 and q8.shape == w.shape
assert int(q4.max()) <= 7 and int(q4.min()) >= -7  # 对称 int4 只用 [-7, 7]，不碰 -8
assert quant_error(w, dequantize(q8, s8))["最大绝对误差"] <= float(s8) / 2 + 1e-6
assert quant_error(w, dequantize(*quantize(w, 8, "group", 32)))["相对误差"] < \
       quant_error(w, dequantize(*quantize(w, 8, "tensor")))["相对误差"]
pd.DataFrame([{"位宽": 8, **quant_error(w, dequantize(q8, s8))},
              {"位宽": 4, **quant_error(w, dequantize(q4, s4))}])

# %% [markdown]
# 断言全部通过说明：舍入误差不超过 `s/2`（这是量化理论上界）、int4 被正确截断在 `[-7,7]`、
# 而 group 粒度在同一份权重上确实优于 per-tensor。下面开始跑真实实验。

# %% [markdown]
# ## 07.3 实测
#
# ### 实测 1：粒度对误差的影响（合成权重，固定 seed）
#
# 真实 LLM 权重近似高斯，但每个模型都有少量**离群值**。先造一个带离群值的权重，
# 把 per-tensor / per-channel / per-group 的误差摊开看：这部分是我们可以控制的。

# %%
torch.manual_seed(0)
w_out = torch.randn(768, 256) * 0.03
w_out[:, 5] *= 25.0        # 单通道离群值：真实模型某些通道会大 20–100 倍
w_out[3, ::41] *= 18.0     # 再加几个孤立元素的离群值

rows = []
for bits in (8, 4):
    for name, gran, gs in [("per-tensor", "tensor", None), ("per-channel", "channel", None),
                           ("group-128", "group", 128), ("group-64", "group", 64),
                           ("group-32", "group", 32)]:
        x_hat = dequantize(*quantize(w_out, bits, gran, gs))
        err = quant_error(w_out, x_hat)
        rows.append({"位宽": bits, "粒度": name,
                     "每行字节": round(quant_bytes(w_out.shape, bits, gran, gs) / w_out.shape[0]),
                     "相对误差": round(err["相对误差"], 4),
                     "最大绝对误差": round(err["最大绝对误差"], 4)})
weight_df = pd.DataFrame(rows)
weight_df

# %% [markdown]
# 读表：per-tensor int8 的相对误差是 9.7%，而 group-32 只有 1.1%——差约 9 倍。
# int4 更夸张：per-tensor 53.5% 已经接近"把权重随机化"，group-32 才 13.6%。
# 注意**每行字节**：group-32 的 scale 开销比 per-channel 多 5%–11%（位宽越低越明显），这是精度的代价。
#
# 把粒度从粗到细画成曲线，更容易看清 int4 为什么"必须"分组：

# %%
fig, ax = plt.subplots()
tick_labels = ["per-channel\n(1 scale/row)", "g=128", "g=64", "g=32"]
for bits, color in [(8, "#2563EB"), (4, "#EF4444")]:
    ys = [weight_df[(weight_df["位宽"] == bits) & (weight_df["粒度"] == name)]["相对误差"].item()
          for name in ["per-channel", "group-128", "group-64", "group-32"]]
    ax.plot(range(len(ys)), ys, "o-", color=color, label=f"int{bits} (grouped)")
    base = weight_df[(weight_df["位宽"] == bits) & (weight_df["粒度"] == "per-tensor")]["相对误差"].item()
    ax.axhline(base, color=color, ls="--", lw=1.5, alpha=0.55,
               label=f"int{bits} per-tensor = {base:.1%}")
ax.set_xticks(range(len(tick_labels)))
ax.set_xticklabels(tick_labels)
ax.set_yscale("log")
ax.set_ylabel("relative weight error  ||x-x_hat|| / ||x||  (log)")
ax.set_xlabel("granularity (finer to the right)")
ax.set_title("Outliers inflate the per-tensor step; grouping quarantines them")
ax.legend(fontsize=9)
save(fig, "ch07_quant_error_curve")

# %% [markdown]
# 读图：**现象**是两条线都随分组变细而下降，但 int4 的下降斜率远大于 int8；
# **原因**是 `s ∝ absmax / 2^b`，离群值固定时 int4 的绝对步长是 int8 的 16 倍，
# 一次舍入就能吃掉大量有效位；分组把 absmax 限定在小范围内，才把步长压回来。
# **对 vLLM 参数的启示**：`--quantization gptq/awq` 的 `group_size` 默认 128 不是随便定的——
# 它刚好让 int4 的误差落回可接受区，同时 scale 开销只有 ~3%。
#
# ### 实测 2：量化 TinyGPT 的真实 qkv / mlp 权重，看 logits 偏差
#
# 权重误差只是中间量，用户关心的是**输出变了多少**。我们把 TinyGPT 的 qkv、mlp 权重换成反量化后的值，
# 用同一批输入比较 logits。随机初始化的权重近似高斯，所以这里 per-tensor 也"够用"——
# 正好可以拿来和后面注入离群值的版本对照。

# %%
torch.manual_seed(0)
model = TinyGPT(GPTConfig()).to(device).eval()
torch.manual_seed(3)
prompt = torch.randint(0, model.cfg.vocab_size, (4, 64), device=device)
with torch.no_grad():
    ref_logits = model(prompt)


def quantized_copy(base: TinyGPT, bits: int, granularity: str, group_size=None,
                   scope: tuple = ("qkv", "mlp")) -> TinyGPT:
    """复制一份模型，把名字里含 scope 关键字的二维权重换成反量化值（模拟 W8/W4A16）。"""
    mq = TinyGPT(GPTConfig()).to(device).eval()
    mq.load_state_dict(base.state_dict())
    with torch.no_grad():
        for name, p in mq.named_parameters():
            if p.dim() != 2 or "lm_head" in name:
                continue
            if not any(tag in name for tag in scope):
                continue
            p.copy_(dequantize(*quantize(p.data.clone(), bits, granularity, group_size)))
    return mq


rows = []
for scope_name, scope in [("仅 qkv", ("qkv",)), ("仅 mlp", ("mlp",)), ("qkv + mlp", ("qkv", "mlp"))]:
    for bits in (8, 4):
        with torch.no_grad():
            out = quantized_copy(model, bits, "group", 32, scope=scope)(prompt)
        rows.append({"量化范围": scope_name, "位宽": bits,
                     "logits 相对误差": round(((out - ref_logits).norm() / ref_logits.norm()).item(), 5),
                     "top-1 一致率": round((out.argmax(-1) == ref_logits.argmax(-1)).float().mean().item(), 3)})
scope_df = pd.DataFrame(rows)
scope_df

# %% [markdown]
# 两个发现：**mlp 比 qkv 更敏感**（同样 int4，mlp 单独量化后的 logits 误差更大，因为参数量占 2/3），
# 并且 **top-1 一致率全是 1.0**——别被这个指标骗了。随机权重模型的 logits 由单一 token 主导，
# argmax 极其稳定；真实模型要看 perplexity / KL / 任务指标，不能只看 top-1。
#
# 现在把真实权重里的"离群值"注入进去：模拟某些输入通道大 20 倍的情况，再量化：

# %%
torch.manual_seed(0)
base = TinyGPT(GPTConfig()).to(device).eval()
torch.manual_seed(3)
prompt = torch.randint(0, base.cfg.vocab_size, (4, 64), device=device)


def with_outlier(model_: TinyGPT, col: int = 5, scale: float = 20.0) -> TinyGPT:
    """把 qkv 的某一列（一个输入通道）放大 scale 倍——AWQ/SmoothQuant 论文里的 outlier channel。"""
    m = TinyGPT(GPTConfig()).to(device).eval()
    m.load_state_dict(model_.state_dict())
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "qkv" in name and p.dim() == 2:
                p.data[:, col] *= scale
    return m


outlier_model = with_outlier(base)
with torch.no_grad():
    ref_outlier = outlier_model(prompt)

rows = []
for bits in (8, 4, 2):
    for name, gran, gs in [("per-tensor", "tensor", None), ("per-channel", "channel", None),
                           ("group-128", "group", 128), ("group-32", "group", 32)]:
        mq = quantized_copy(outlier_model, bits, gran, gs)
        with torch.no_grad():
            out = mq(prompt)
        rows.append({"位宽": bits, "粒度": name,
                     "logits 相对误差": round(((out - ref_outlier).norm() / ref_outlier.norm()).item(), 4),
                     "top-1 一致率": round((out.argmax(-1) == ref_outlier.argmax(-1)).float().mean().item(), 3)})
outlier_df = pd.DataFrame(rows)
outlier_df

# %% [markdown]
# 这张表就是本章最重要的数字：**注入一个 20 倍通道离群值之后**，int4 per-tensor 的 logits 相对误差
# 从 2.0% 涨到 20.8%，per-channel 也救不了（16.5%）——因为离群值在列上，每一行都躲不开；
# 只有 group-32 把它限制在 1/32 的列里，误差回到 6.4%。这就是 GPTQ / AWQ 必须分组量化的原因。
# 下面把"位宽"这条轴单独画出来：

# %%
rows = []
for bits in (2, 3, 4, 8):
    for name, gran, gs in [("per-tensor", "tensor", None), ("per-channel", "channel", None),
                           ("group-32", "group", 32)]:
        mq = quantized_copy(outlier_model, bits, gran, gs)
        with torch.no_grad():
            out = mq(prompt)
        rows.append({"位宽": bits, "粒度": name,
                     "logits 相对误差": ((out - ref_outlier).norm() / ref_outlier.norm()).item()})
bits_df = pd.DataFrame(rows)

fig, ax = plt.subplots()
for name, color, marker in [("per-tensor", "#EF4444", "s"), ("per-channel", "#F59E0B", "^"),
                            ("group-32", "#10B981", "o")]:
    sub = bits_df[bits_df["粒度"] == name]
    xs, ys = sub["位宽"].tolist(), sub["logits 相对误差"].tolist()
    ax.plot(xs, ys, marker=marker, color=color, label=name)
ax.set_yscale("log")
ax.set_xticks([2, 3, 4, 8])
ax.set_xlabel("weight bit-width (bits)")
ax.set_ylabel("relative logits error (log)")
ax.set_title("After a 20x channel outlier: int4 stays usable only with grouping")
ax.legend()
save(fig, "ch07_bits_vs_error")

# %% [markdown]
# 读图：int8 三条线挤在 0.4%–1.8% 之间，说明**位宽够用时粒度只是优化项**；
# int4 per-tensor 是 20.8%，比 int8 per-tensor 大一个数量级，group-32 才把它拉回 6.4%。
# 一句话：**int8 可以粗，int4 必须细。**

# %% [markdown]
# ### 实测 3：7B/8B 的显存与 decode 理论 tok/s 账本
#
# 先测这台机器实际能达到的带宽（第 01 章的 `clone` 法：读 N 字节 + 写 N 字节），
# 再用第 01 章的公式 `decode tok/s = 带宽 ÷ 每 token 读取字节` 推账本。

# %%
def measure_bandwidth(size_mb: int = 256, reps: int = 5) -> float:
    n = size_mb * 1024 * 1024 // 4
    x = torch.empty(n, device=device)
    samples = []
    for _ in range(reps):
        sync(device)
        t0 = time.perf_counter()
        y = x.clone()
        sync(device)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return 2 * n * 4 / samples[len(samples) // 2] / 1e9   # 读+写


bw_gbps = measure_bandwidth()
print(f"实测带宽 ≈ {bw_gbps:.1f} GB/s（M3 Ultra 规格 800 GB/s）")

rows = []
for model_name, P in [("7B", 7e9), ("8B", 8e9)]:
    fps16 = bw_gbps * 1e9 / (P * 2)
    for dtype, bytes_per_param in [("fp16", 2.0), ("int8", 1.0), ("int4", 0.5)]:
        weight_bytes = P * bytes_per_param
        tok_s = bw_gbps * 1e9 / weight_bytes
        rows.append({"模型": model_name, "精度": dtype,
                     "权重 GiB": round(weight_bytes / 1024**3, 2),
                     "每 token 权重字节": human_bytes(weight_bytes),
                     "理论 decode tok/s": round(tok_s, 1),
                     "相对 fp16 加速": round(tok_s / fps16, 2)})
ledger_df = pd.DataFrame(rows)
ledger_df

# %% [markdown]
# 读表（带宽随机器与当次实测浮动，绝对值以表格为准）：7B fp16 权重 13GiB，int8 减半到 6.5GiB，
# int4 只剩 3.3GiB——**精度每降一档，每 token 读取字节减半，理论 tok/s 就翻一倍**（表里"相对 fp16 加速"列恒为 2.0/4.0）。
# **注意前提**：这些数字假设 batch=1、权重带宽利用率 100%，实际还要扣掉 KV cache、kernel 效率、
# 采样开销（第 01 章实测 TinyGPT 只跑到理论值的 0.5%）。真实 7B/8B 在 M3 Ultra 上的 decode
# 通常在 20–60 tok/s 量级，但**相对关系**（int8≈2×、int4≈4×）成立。
#
# 长上下文时，KV cache 的读取会追上权重。按 Llama-3-8B 的形状（32 层、8 个 KV head、head_dim=128）：

# %%
rows = []
for ctx in (4096, 32768, 131072):
    for kv_dtype, kv_bytes in [("fp16", 2), ("fp8", 1)]:
        kv_total = 2 * 32 * 8 * 128 * ctx * kv_bytes          # K+V
        total = 8e9 * 2 + kv_total                             # 8B fp16 权重 + KV
        rows.append({"上下文": f"{ctx // 1024}K", "KV dtype": kv_dtype,
                     "每请求 KV GiB": round(kv_total / 1024**3, 2),
                     "权重+KV 读取 GiB": round(total / 1024**3, 2),
                     "8B decode 理论 tok/s": round(bw_gbps * 1e9 / total, 1)})
kv_df = pd.DataFrame(rows)
kv_df

# %% [markdown]
# 读表：4K 上下文时 KV 只有 0.5GiB，`--kv-cache-dtype fp8` 几乎看不出区别；
# 到 128K 上下文，fp16 KV 要 16GiB，和 fp16 权重一样大，理论 tok/s 直接腰斩；
# 换成 fp8 KV 后损失补回大半——**KV cache 量化是长上下文场景的刚需**，短上下文只是省显存。
#
# 最后对照一下 TinyGPT：它的 `parameters × 4B = 11MB`，理论 5 万 tok/s，实测只有几百。

# %%
def decode_steps(n: int = 24) -> torch.Tensor:
    """带 KV cache 的 24 步 decode（贪心），复用第 01 章的写法。"""
    cache = KVCache(model.cfg, batch_size=prompt.shape[0], max_seq_len=prompt.shape[1] + n + 2,
                    device=device)
    logits = model(prompt, start_pos=0, kv_cache=cache)
    cache.advance(prompt.shape[1])
    nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for _ in range(n):
        logits = model(nxt, start_pos=cache.seq_len, kv_cache=cache)
        cache.advance(1)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return nxt


r = bench(decode_steps, 24, name="tinygpt decode 24 步", warmup=2, reps=5, device=device)
measured_tok_s = 24 / (r.median_ms / 1000)
rows = []
for dtype, bytes_per_param in [("fp32", 4.0), ("int8", 1.0), ("int4", 0.5)]:
    theory = bw_gbps * 1e9 / (model.num_params() * bytes_per_param)
    note = f"实测 {measured_tok_s:.0f} tok/s" if dtype == "fp32" else "PyTorch 无 int 内核，实测不变"
    rows.append({"精度": dtype, "每 token 权重字节": human_bytes(model.num_params() * bytes_per_param),
                 "理论 tok/s": round(theory, 0), "备注": note})
tiny_df = pd.DataFrame(rows)
print(f"实测 TinyGPT decode：{r.median_ms:.1f} ms / 24 步 → {measured_tok_s:.0f} tok/s")
tiny_df

# %% [markdown]
# **理论 vs 实测差了 100 倍**：11MB 的权重读一遍理论上 0.02ms，实测每步 ~2ms。
# 原因不是带宽不够，而是固定开销（Python 调度、MPS 命令缓冲、kernel 启动）占绝对主导。
# 这解释了为什么本章不能"实测 int8 加速"：PyTorch 里把权重量化再反量化只会**多一次读放大**，
# 真正的收益必须靠 Marlin 这类融合 kernel（int4 权重直接进 tensor core 的寄存器），
# 也就是 07.4 里 vLLM 干的事。把账本画成图：

# %%
fig, ax = plt.subplots()
labels = [f"{m}\n{d}" for m, d in zip(ledger_df["模型"], ledger_df["精度"])]
colors = {"fp16": "#7C3AED", "int8": "#2563EB", "int4": "#10B981"}
bars = ax.bar(range(len(ledger_df)), ledger_df["理论 decode tok/s"],
              color=[colors[d] for d in ledger_df["精度"]])
for bar, v in zip(bars, ledger_df["理论 decode tok/s"]):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 3, f"{v:.0f}", ha="center", fontsize=9)
ax.set_xticks(range(len(ledger_df)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("theoretical decode tok/s (batch=1, measured BW)")
ax.set_title(f"Half the weights, double the tokens: decode ceiling at {bw_gbps:.0f} GB/s")
save(fig, "ch07_decode_budget")

# %% [markdown]
# 读图：int8 恰好是 fp16 的 2 倍，int4 是 4 倍——**因为 decode 每 token 只读一遍权重，字节数与 tok/s 严格成反比**。
# 这就是「量化主要收益在 decode」的完整推导：prefill 是 compute-bound，量化省的是带宽不是算力，
# 在 prefill 上收益有限（还可能因反量化变慢）；decode 的算术强度 ≈ 1 FLOP/byte，带宽减半 = 吞吐翻倍。
# 下一节看工业实现怎么把「账本上的加速」变成真加速。

# %% [markdown]
# ## 07.4 对照工业实现（vLLM 0.28.x）
#
# 我们手写的 40 行是"教学版最小闭环"，vLLM 把它拆成了 **配置（Config）+ 方法（LinearMethod）+ kernel** 三层：
#
# | 我们的实现 | vLLM 0.28.x 对应物 | 文件 |
# | --- | --- | --- |
# | `absmax_scale`（per-tensor） | `Fp8Config` / `Fp8LinearMethod` 的 per-tensor 缩放 | `vllm/model_executor/layers/quantization/fp8.py` |
# | per-channel / per-group int4 | `AWQConfig` / `GPTQConfig` + Marlin kernel | `.../quantization/awq.py`、`.../quantization/gptq.py`、`.../quantization/marlin_utils.py` |
# | int4/FP8/NVFP4 统一描述 | compressed-tensors 格式 | `.../quantization/compressed_tensors/compressed_tensors.py` |
# | 方法注册表 | `QUANTIZATION_METHODS` / `get_quantization_config` | `.../quantization/__init__.py` |
# | 反量化后 matmul | `LinearBase.quant_method.apply` / `process_weights_after_loading` | `.../layers/linear.py` |
# | 本章的"量化误差"指标 | 模型侧用 perplexity / KL，靠 `lm_eval` 与 `vllm bench` 对比 | — |
#
# 怎么在 CLI 里用（0.28.x）：
#
# ```bash
# # W4A16：权重 int4 + 激活 fp16，Marlin kernel（NVIDIA 上最快的一档）
# vllm serve Qwen/Qwen3-8B --quantization awq_marlin --max-model-len 32768
# vllm serve Qwen/Qwen3-8B --quantization gptq_marlin
#
# # W8A8：FP8 权重 + FP8 激活（Hopper/Blackwell 原生支持，Mac 上不可用）
# vllm serve Qwen/Qwen3-8B --quantization fp8
#
# # KV cache 降到 fp8，长上下文 / 大 batch 的显存与带宽双赢
# vllm serve Qwen/Qwen3-8B --kv-cache-dtype fp8 --calculate-kv-scales
# ```
#
# 我们 vs 它的差距：
#
# - **误差补偿**：我们只做 round-to-nearest；GPTQ 用 Hessian 逆把舍入误差逐列补偿，AWQ 用**激活幅值**
#   给显著通道乘缩放（等价变换，把离群值从激活搬到权重），两者都能把 int4 的精度损失再压低一截。
# - **kernel**：我们反量化后走普通 matmul（在 Mac 上反而更慢）；vLLM 的 Marlin 把 int4 权重
#   在寄存器里 dequant 后直接喂 tensor core，decode 时才真的吃到带宽减半的收益。
# - **粒度与例外**：我们整层统一 group；工业实现会保留 `lm_head`、embedding、部分显著通道为高精度，
#   并且不同层用不同 group/精度（mixed precision）。
# - **KV cache**：我们只算账本；vLLM 在 attention backend 里真的把 KV 存成 fp8，
#   由 `vllm/config/cache.py` 的 `CacheConfig.cache_dtype` 控制，attention kernel 按需反量化。
#
# 一句话总结：**量化不是"省显存"这么简单，它把 decode 的瓶颈从带宽端挪走，代价是精度和 kernel 复杂度。**
# 下一章我们换一条完全不改精度的路：用一个小模型猜、用大模型验——投机解码。

# %% [markdown]
# ## 07.5 练习与小结
#
# **练习**（参考答案见 `solutions/07_exercises.py`）
#
# 1. ⭐ 把 `absmax_scale` 的 `bits` 从 8 改成 4、2，只用 per-tensor，记录 `weight_df` 里相对误差的变化。
#    为什么 int4 的误差不是 int8 的 2 倍而是十几倍？（提示：`s ∝ 1/2^(b-1)` 与离群值共同作用）
# 2. ⭐ 把 group_size 从 128 改到 16 和 512，在同一份离群权重上跑一遍，画「相对误差 vs 每行字节」曲线。
#    最优点在哪？为什么工业界选 128 而不是 16？
# 3. ⭐⭐ 实现**非对称量化**：`q = clamp(round(x/s) + z, 0, 2^b-1)`，`z = -round(min/s)`，
#    在 ReLU 之后的偏斜分布（比如 `torch.relu(w)`）上比较对称 vs 非对称的误差，给出数字。
# 4. ⭐⭐ 实现简版 **GPTQ 误差补偿**：按输入二阶矩 `H = 2·XᵀX` 的对角从大到小逐列量化，
#    每量化一列就把误差按 `H` 传播给剩余列。对比 naive round 的 logits 相对误差。
# 5. ⭐⭐⭐ 设计 **AWQ 式激活感知缩放**：给每个输入通道乘 `s_j`，等价变换 `W·diag(s)⁻¹` 与 `diag(s)·x`，
#    目标是压低权重侧的离群值。写出搜索 `s` 的策略，给出评分要点：缩放后 per-tensor int4 误差下降多少、
#    是否真的不需要分组、对 kernel 的额外要求。
#
# **你现在应该能回答：**
#
# - 对称量化的一行公式、三个误差来源，以及 `group_size` 在其中的作用。
# - 为什么 int8 可以 per-tensor、int4 必须 per-group；为什么 `group_size=128` 是默认值。
# - 7B/8B 在 fp16/int8/int4 下的权重显存与 decode 理论 tok/s，以及为什么量化收益集中在 decode。
# - vLLM 的 `--quantization` / `--kv-cache-dtype fp8` 各自对应本章的哪个概念、源码在哪个目录。
#
# **延伸阅读**
#
# - GPTQ（Frantar et al., 2022）、AWQ（Lin et al., 2023）、SmoothQuant（Xiao et al., 2022）
# - vLLM 文档：Quantization 与 `--kv-cache-dtype`；源码 `vllm/model_executor/layers/quantization/`
# - 下一章：投机解码——不降低精度，用"猜 + 验"把 decode 的串行步数降下来。
