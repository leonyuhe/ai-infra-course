# %% [markdown]
# # 第 07 章 · 量化：把权重压小，把带宽省下来
#
# > 同一个 8B 模型、同一张卡、同一段 prompt：fp16 权重下 decode 只能跑到约 31 tok/s，
# > 换成 int8 就约 62 tok/s（07.4 实验 C 的账本）。模型没变、采样没变、算子也没变，
# > 变的只有一件事：每生成一个 token 要从显存搬多少字节。
#
# > 一句话：decode 每生成一个 token 都要把全部权重读一遍（01 章）。这一章把权重从 fp16 压到
# > int8/int4，用「字节减半 → token 翻倍」的账本解释收益，再用 TinyGPT 的真实权重把误差量给你看。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 90 分钟 |
# | 运行环境 | 🏠 Mac（MPS / CPU）· 无需 GPU · torch 2.15 |
# | 前置章节 | 01（decode 带宽、roofline、算术强度） |
# | 本章产物 | 对称量化器（tensor / channel / group）、误差指标与字节账本、TinyGPT 权重实验、7B/8B decode 账本 |
# | 本章接口 | `absmax_scale / quantize / dequantize / quant_error / quant_bytes`（本章内联，08 章不依赖） |
# | 对照版本 | vLLM 0.28.x |
#
# **学完你能回答：**
# 1. 对称量化的一行公式是什么？误差由哪三件事决定？
# 2. per-tensor / per-channel / per-group 差多少？为什么工业界默认 `group_size=128`？
# 3. 7B/8B 在 fp16/int8/int4 下的权重显存、decode 理论 tok/s 各是多少？为什么量化主要收益在 decode？
# 4. vLLM 的 `--quantization` / `--kv-cache-dtype fp8` 分别对应本章哪个概念？

# %% [markdown]
# ## 07.0 本章目标与路线
#
# **一句话目标**：把线性层权重的存储从 fp16 换成 int8 / int4，让 decode 每生成一个 token
# 少读一半到 3/4 的字节，同时把量化误差控制在不破坏输出的范围内。
#
# **代价 / 边界**：
#
# - **精度**：int4 的步长是 int8 的 16 倍，per-tensor 时权重误差可达几十个百分点，必须靠分组和离群值处理；
# - **kernel**：量化省的是带宽不是算力。没有融合的 int4 GEMM（Marlin 一类）时，反量化 + 普通 matmul 只会更慢；
# - **收益场景**：只在 decode（batch=1、带宽受限）明显；prefill 是 compute-bound，量化收益有限，甚至可能变慢。
#
# 还有一个容易被忽略的边界：**误差「能不能接受」不是数学问题，是产品问题**。同一个 2% 的相对误差，
# 落在代码补全上可能是偶尔多一个括号，落在数学题上可能是末位数字变了——它取决于误差落在哪个指标上、
# 这个指标有没有签收人。07.4 实验 B 会把这条判断拆成「看哪个指标、谁拍板」。
#
# **本章交付四样东西**：
#
# | # | 交付物 | 在哪一节 | 怎么验收 |
# | --- | --- | --- | --- |
# | 1 | 对称量化器：per-tensor / per-channel / per-group | 07.3 Step 1 | int4 落在 `[-7,7]`；反量化误差 ≤ s/2 |
# | 2 | 误差指标 + 存储字节账本 | 07.3 Step 2 | 舍入上界成立；group-32 优于 per-tensor |
# | 3 | 量化 TinyGPT 的 qkv / mlp 权重并对比 logits | 07.3 Step 3 | int8 logits 相对误差 < 1% |
# | 4 | 三组实测：粒度、离群值、显存与带宽账本 | 07.4 实验 A/B/C | 趋势与公式一致，账本可手算 |
#
# **阅读路线**：07.1 现象 → 07.2 高阶设计 → 07.3 三步实现（每步有验收）→ 07.4 实验 → 07.5 对照 vLLM。

# %% [markdown]
# ## 07.1 问题与直觉：decode 的瓶颈是「读权重」
#
# **先猜一下**：把 8B 模型的权重从 fp16 换成 int4，显存占用降到 1/4，decode 吞吐会变成几倍？
# (a) 几乎不变　(b) 约 2 倍　(c) 约 4 倍 —— 先选一个，账本就在下面三行。
#
# 01 章的账本：decode 每生成 1 个 token，都要把全部权重从显存读一遍，算术强度 ≈ 1 FLOP/byte，
# 所以吞吐上限就是 `带宽 ÷ 每 token 字节数`。以 8B 模型、M3 Ultra 实测带宽 ~500 GB/s 为例：
#
# | 精度 | 每参数字节 | 8B 权重 | decode 理论 tok/s |
# | --- | --- | --- | --- |
# | fp16 | 2 | 16 GB | ~31 |
# | int8 | 1 | 8 GB | ~62 |
# | int4 | 0.5 | 4 GB | ~125 |
#
# **结论**：精度每降一档，每 token 读取的字节减半，decode 上限就翻倍。这就是量化最直接的收益。
#
# （回收本节开头那题）答案是 (c) 约 4 倍——**但这是在「理想账本」里**：batch=1、带宽利用率 100%、
# kernel 白送。三个前提任何一个不成立，(c) 就会往 (a) 退化：07.4 实验 C 的 TinyGPT 实测就是 (a)，
# 因为每步前向的固定开销占了绝对主导，权重搬运省下的那点字节根本显不出来。
# 选 (b) 的人通常在想「显存省 4 倍，但 kernel 有开销，打个对折」——那是 **prefill** 的直觉：
# prefill 里成本是算力，量化只省存储；decode 里成本是读字节，字节减半就是吞吐翻倍。
# 所以 int4 为什么能快，答案朴素得不像玄学：**快的那部分是「搬运字节数除以 4」**，
# 不是「算得更快」——decode 里多算几个乘法的成本可以忽略。
#
# 换个比方：decode 像「每写一个字，都要把整本书重读一遍」。量化做的不是把书变薄，
# 而是把字改小——fp16 每个参数占 2 字节、int4 只占 0.5 字节，同样一次搬运能多搬 4 倍的参数。
# 书还是那一本（8B 权重），只是从 16 GB 变成了 4 GB。
#
# 但量化不是免费的：把权重从连续浮点映射到有限格点必然产生误差，
# 而误差的大小取决于**粒度**和**离群值**。下图是本章的核心矛盾：
#
# ![](../assets/figures/ch07_quant_error.svg)
#
# **反例**：对真实 LLM 权重直接做 per-tensor int4，一个 20 倍离群值就能把步长撑大十几倍；
# 07.4 实验 B 会在 TinyGPT 上量出「logits 相对误差从 ~2% 涨到 ~20%」的过程。
# AWQ / GPTQ 这些工业实现默认 `group_size=128`，就是为了避免这种情况。

# %% [markdown]
# 把两种精度的差异摊开看：
#
# | | fp16（基线） | int8 | int4 |
# | --- | --- | --- | --- |
# | 每参数存储 | 2 B | 1 B | 0.5 B（实际还要存 scale） |
# | 反量化 | 不需要 | `q × s` | `q × s`，且必须分组 |
# | 主要误差来源 | 无 | 舍入噪声（很小） | 舍入 + 离群值污染 |
# | kernel | 普通 GEMM | int8 GEMM 或 W8A16 | W4A16 + Marlin 一类融合 kernel |
# | 典型用途 | 精度基线 | 显存换倍、几乎无损 | 极致省带宽、需要分组/校准 |
#
# 把「省下来的字节」换算成人能感知的东西：8B fp16 权重 16 GB，int4 只占 4 GB；
# 省下的约 11 GiB，按 07.4 里 32K 上下文的每请求 4 GiB KV 算，**够再塞下约 3 个 32K 会话**。
# 对推理服务来说，这不是「显存数字好看」——同一张卡多 3 个长上下文并发，就是吞吐和成本。
#
# 本章要解决两个**互相牵制**的子问题：
#
# 1. **正确性（精度）**：量化后的输出还能不能用？用什么指标衡量、怎么让它可控？
# 2. **成本（收益）**：量化到底省了多少字节、decode 理论 tok/s 提升多少？为什么在 Mac 上测不出来？

# %% [markdown]
# ## 07.2 高阶设计
#
# 动手前先把方案定下来。整章只有三个组件和一条数据流。
#
# **组件 1：量化 / 反量化（存什么）**
# 每个权重张量存两样东西：整数张量 `q` 和缩放因子 `scale`。
# `scale` 的粒度决定误差：per-tensor 全场一个；per-channel 每行一个；per-group 每 `g` 个元素一个。
# 反量化就是 `x̂ = q × scale`——这是 W4A16 的思路：权重存整数，计算时再变回浮点。
# 另一种选择是**非对称量化**（多存一个 zero-point，适合 ReLU 之后的偏斜分布）；
# 本章主线保持对称，练习 3 会实现非对称版本并给出误差对比。
#
# **组件 2：离群值与分组（误差从哪来）**
# 对称量化的步长 `s = absmax / qmax`，由**绝对值最大的元素**决定。
# 一个离群值就会把全场步长撑大，让所有正常值一起变粗。
# 分组的作用就是把 `absmax` 限制在小范围内，不让一个离群值污染整层。
#
# > **⚠️ 坑：一个 20 倍离群值，能把整组 32 个权重一起打成 0**
# > 正常权重是 0.03 量级，int4 的 qmax=7，正常组的步长约 0.01，0.03 还能落到第 3 格；
# > 组里混进一个 20 倍离群值后，absmax 至少被撑到 0.6，步长变成 0.09 量级——
# > 组里所有 0.03 量级的正常值除以步长只剩 0.3，四舍五入**全成 0**。
# > 这一组 32 个权重不是「误差大了一点」，是整组信息全部丢失（07.4 实验 B 会量到
# > int4 per-tensor 的 logits 误差从 2.0% 涨到 20.8%）。
# > 哪怕只有 8 倍离群值，步长也大 8 倍，0.03 量级的值从第 3 格退到第 1 格——整组只剩 1 bit 分辨率；
# > 20 倍时连这 1 bit 也没了。**精度不是随离群值线性下降的，是断崖式的。**
#
# **组件 3：误差度量与字节账本（怎么验收）**
# 权重误差用 `‖x−x̂‖/‖x‖`（跨层可比），存储用 `权重比特数 + scale 字节`，
# 收益用 `字节 ÷ 带宽` 换算成 decode tok/s。三者必须能互相印证。
#
# 数据流：
#
# ```text
# fp32 权重 → absmax 求 scale → q = round(x/s) 截断 → 存 q + scale
#                                                    ↓ 推理时
#                                     x̂ = q × scale → matmul / 融合 kernel
# ```

# %% [markdown]
# ### 接口契约
#
# 后面的实现、07.4 的实验、以及练习共用同一组函数签名；换粒度只改一个参数：
#
# | 函数 / 参数 | 语义 |
# | --- | --- |
# | `absmax_scale(x, bits, granularity, group_size)` | 求步长 `s` 与 `qmax`，不修改权重 |
# | `quantize(x, bits, granularity, group_size)` | 返回 `(q, scale)`，`q` 用 int8 承载（int4 也暂存 int8） |
# | `dequantize(q, scale)` | 反量化回浮点：`q × scale` |
# | `quant_error(x, x_hat)` | 返回 MSE、最大绝对误差、相对误差 |
# | `quant_bytes(shape, bits, granularity, group_size)` | 权重 + scale 的真实存储字节（scale 按 fp16 算） |
#
# 为什么 int4 也用 int8 承载？PyTorch 没有 int4 张量类型；教学上先存 int8、值域限制在 `[-7,7]`，
# 真实实现会把两个 int4 打进一个字节，本章的字节账本用 `bits/8` 折算。
#
# ### 验收标准
#
# 每个实现步骤都配一条可执行的验收，写不出来就说明没理解：
#
# | 步骤 | 验收 |
# | --- | --- |
# | Step 1 量化器 | int4 值域 `[-7,7]`；任何值反量化误差 ≤ `s/2` |
# | Step 2 度量与账本 | `quant_bytes` 与手算一致；同一权重 group-32 误差 < per-tensor |
# | Step 3 接进模型 | qkv+mlp int8 的 logits 相对误差 < 1%，int4 < 5% |
# | 实验 A/B/C | 误差随粒度单调下降；账本比例严格 2×/4× |

# %% [markdown]
# ## 07.3 从零实现：分三步
#
# | Step | 做什么 | 代码 |
# | --- | --- | --- |
# | 1 | 对称量化三函数：scale → 整数 → 反量化 | `absmax_scale` / `quantize` / `dequantize` |
# | 2 | 误差指标与存储字节：把「精度代价」量化 | `quant_error` / `quant_bytes` |
# | 3 | 把反量化接进 TinyGPT 的 qkv / mlp 权重 | `quantized_copy` |
#
# 每一步末尾都有 `assert` 验收；三步全部通过，再进入 07.4 的实验。
#
# ### Step 1 · 对称量化三函数
#
# **目标**：给定任意二维权重，按指定粒度算出步长，量化成整数，再反量化回来。
#
# **做法**：`absmax_scale` 只负责求 `s`：per-tensor 取全场最大值；per-channel 沿输出维求；
# per-group 先把最后一维切段再求。`qmax = 2^(b-1)-1`，`q = clamp(round(x/s), -qmax, qmax)`。
# 对称量化令零点 = 0，所以反量化只有一次乘法；int4 的 `[-8,7]` 天生不对称，这里对称地截在 `[-7,7]`。

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
# **验收**：int4 的值域必须落在 `[-7,7]`；对任意值，反量化误差不超过半个步长 `s/2`。

# %%
torch.manual_seed(0)
w_demo = torch.randn(64, 256) * 0.03
w_demo[:, 0] *= 30.0                                   # 单通道离群值：真实权重里存在
q8, s8 = quantize(w_demo, 8, "tensor")
q4, s4 = quantize(w_demo, 4, "tensor")
assert q8.dtype == torch.int8 and q8.shape == w_demo.shape
assert int(q4.max()) <= 7 and int(q4.min()) >= -7       # 对称 int4 只用 [-7, 7]，不碰 -8
assert (w_demo - dequantize(q8, s8)).abs().max() <= float(s8) / 2 + 1e-6
print(f"int8 最大误差 { (w_demo - dequantize(q8, s8)).abs().max():.5f} ≤ s/2 = {float(s8) / 2:.5f} ✓")

# %% [markdown]
# ### Step 2 · 误差指标与存储字节
#
# **目标**：把「量化代价」变成可比较的数字，并把 scale 的开销也算进存储账本。
#
# **做法**：三个误差指标各有用途——MSE 看整体噪声，最大绝对误差看最坏元素，
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

# %% [markdown]
# **验收**：`quant_bytes` 与手算一致；同时把 demo 权重的两种粒度误差摊开，
# 确认「分组更细 → 误差更小」这条不变量在这份权重上成立。

# %%
pd.DataFrame([{"位宽": 8, **quant_error(w_demo, dequantize(q8, s8))},
              {"位宽": 4, **quant_error(w_demo, dequantize(q4, s4))}]).round(5)

# %%
assert quant_bytes((768, 256), 8, "tensor") == 768 * 256 + 2
assert quant_bytes((768, 256), 4, "channel") == 768 * 256 * 4 / 8 + 768 * 2
assert quant_bytes((768, 256), 4, "group", 32) == 768 * 256 * 4 / 8 + 768 * 8 * 2
err_group = quant_error(w_demo, dequantize(*quantize(w_demo, 8, "group", 32)))["相对误差"]
err_tensor = quant_error(w_demo, dequantize(*quantize(w_demo, 8, "tensor")))["相对误差"]
assert err_group < err_tensor, "分组量化的误差应小于 per-tensor"
print(f"group-32 = {err_group:.4f} < per-tensor = {err_tensor:.4f} ✓")

# %% [markdown]
# ### Step 3 · 把反量化接进 TinyGPT
#
# **目标**：在真实模型上验证「量化权重 → 输出偏差可控」，而不是只看权重误差。
#
# **做法**：复制一份 TinyGPT，把名字里含 `qkv` 或 `mlp` 的二维权重换成反量化后的值，
# 其余部分（embedding、norm、lm_head、注意力输出投影）保持 fp32——这就是 W4A16 的模拟：
# 权重是整数，激活和计算仍是浮点。用同一批输入比较 logits。

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

# %% [markdown]
# **验收**：按量化范围（仅 qkv / 仅 mlp / 全部）与位宽列表，int8 的 logits 相对误差应远小于 1%，
# int4 应小于 5%；同时记录 top-1 一致率（注意它在这里没有区分度，原因接下来说明）。

# %%
rows = []
for scope_name, scope in [("仅 qkv", ("qkv",)), ("仅 mlp", ("mlp",)), ("qkv + mlp", ("qkv", "mlp"))]:
    for bits in (8, 4):
        with torch.no_grad():
            out = quantized_copy(model, bits, "group", 32, scope=scope)(prompt)
        rows.append({"量化范围": scope_name, "位宽": bits,
                     "logits 相对误差": round(((out - ref_logits).norm() / ref_logits.norm()).item(), 5),
                     "top-1 一致率": round((out.argmax(-1) == ref_logits.argmax(-1)).float().mean().item(), 3)})
scope_df = pd.DataFrame(rows)
assert scope_df[scope_df["位宽"] == 8]["logits 相对误差"].max() < 0.01
assert scope_df[scope_df["位宽"] == 4]["logits 相对误差"].max() < 0.05
scope_df

# %% [markdown]
# 两个发现：**mlp 比 qkv 更敏感**（同样 int4，mlp 单独量化后的 logits 误差更大，因为参数量占 2/3），
# 并且 **top-1 一致率全是 1.0**——这个指标在这里没有区分度。随机权重模型的 logits 由单一 token 主导，
# argmax 极其稳定；真实模型要看 perplexity / KL / 任务指标，不能只看 top-1。
#
# 到这里，量化器的正确性验收完成——但这只证明了「能跑对」。真正决定要不要用量化的问题还没量：
# 粒度选粗一点到底差多少倍？一个离群值能把整层毁成什么样？省下的字节能不能真的变成 tok/s？
# 下面是三组实测，分别回答这三问。

# %% [markdown]
# ## 07.4 实测
#
# 三组实验，分别回答三个问题：
#
# | 实验 | 问题 | 产出 |
# | --- | --- | --- |
# | A | 粒度对权重误差的影响有多大？ | 误差/字节表 + 粒度曲线 |
# | B | 真实权重 + 离群值下，logits 偏差多大？ | 模型级误差表 + 位宽曲线 |
# | C | 7B/8B 的显存与 decode tok/s 账本是多少？ | 精度账本、KV 账本、图 |
#
# 测量协议：M3 Ultra / 96GB / torch 2.15 / MPS；量化误差是确定性计算（固定 seed，与机器无关），
# 带宽与延迟用 `warmup=2–3, reps=5–9` 取中位数。绝对值随机器变化，**趋势才是结论**。
#
# ### 实验 A · 粒度对权重误差的影响
#
# **先猜一下**：同一份带离群值的权重，per-tensor int8 和 group-32 int8 的相对误差差几倍？
# (a) 差不多　(b) 约 4 倍　(c) 一个数量级 —— 先选一个，表格就在下面。
#
# **问题**：per-tensor / per-channel / per-group 的权重误差差多少？scale 开销多大？
#
# **协议**：造一个 768×256 的合成权重（近似高斯），注入一个 20 倍单通道离群值和几个孤立元素离群值，
# 固定 seed，扫 5 种粒度 × 2 种位宽，报相对误差与每行字节。

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
# **读数**：per-tensor int8 的相对误差是 9.7%，而 group-32 只有 1.1%——差约 9 倍。
# int4 更夸张：per-tensor 53.5% 已经接近"把权重随机化"，group-32 才 13.6%。
# 注意**每行字节**：group-32 的 scale 开销比 per-channel 多 5%–11%（位宽越低越明显），这是精度的代价。
#
# （回收实验 A 开头那题）答案是 (c) 一个数量级。但仔细看会发现一个反直觉的比例：int8 上分组把误差从
# 9.7% 压到 1.1%（9 倍），int4 上只从 53.5% 压到 13.6%（约 4 倍）。所以选 (b) 不算错——
# 只是把 int8 的比例直接搬到了 int4 上。真正的原因是 **int4 per-tensor 已经差到没有多少下降空间**：
# 53.5% 意味着输出几乎与权重不相关，再粗也粗不到哪里去。这提醒我们：比较「分组带来多少收益」
# 必须先把位宽钉死，否则比例会骗人。
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
# **读图**：两条线都随分组变细而下降，但 int4 的绝对误差始终比 int8 高一个数量级（5–13 倍），
# 所以 int4 更依赖分组。原因是 `s ∝ absmax / 2^b`：离群值固定时 int4 的步长是 int8 的 16 倍，
# 一次舍入就能吃掉大量有效位；分组把 `absmax` 限定在小范围内，才把步长压回来。
# **对 vLLM 参数的启示**：`--quantization gptq/awq` 的 `group_size` 默认 128 不是随便定的——
# 它让 int4 的误差落回可接受区，同时 scale 开销只有 ~3%。
#
# ### 实验 B · 离群值下，logits 偏差有多大
#
# **问题**：权重误差只是中间量，用户关心的是**输出变了多少**。注入一个 20 倍的通道离群值后，
# 不同粒度/位宽的 logits 偏差分别是多少？
#
# **协议**：在 TinyGPT 的 qkv 权重里把第 5 个输入通道放大 20 倍（模拟 AWQ/SmoothQuant 论文里的
# outlier channel），以这个注入后的 fp32 模型为参考，量化 qkv+mlp 后比较 logits。

# %%
def with_outlier(model_: TinyGPT, col: int = 5, scale: float = 20.0) -> TinyGPT:
    """把 qkv 的某一列（一个输入通道）放大 scale 倍——AWQ/SmoothQuant 论文里的 outlier channel。"""
    m = TinyGPT(GPTConfig()).to(device).eval()
    m.load_state_dict(model_.state_dict())
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "qkv" in name and p.dim() == 2:
                p.data[:, col] *= scale
    return m


outlier_model = with_outlier(model)
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
# **读数**：注入一个 20 倍通道离群值之后，int4 per-tensor 的 logits 相对误差从 2.0% 涨到 20.8%，
# per-channel 也救不了（16.5%）——因为离群值在列上，每一行都躲不开；
# 只有 group-32 把它限制在 1/32 的列里，误差回到 6.4%。这就是 GPTQ / AWQ 必须分组量化的原因。
#
# 但这里有个常被跳过的问题：**「2.0% 能不能接受」不是数学题，是产品题。** 同一个 2.0%，
# 落在闲聊摘要上可能完全无感，落在代码补全上可能是偶尔多一个括号，落在数学或 JSON 这类
# 结构化输出上可能是末位数字、引号变了。工程上要能回答三件事：误差落在哪个指标上
# （权重误差 / logits 误差 / 任务准确率）、这个指标有没有签收人（谁拍板「可以上线」）、
# 量化范围能不能收窄（例如只量化 mlp、`lm_head` 保留 fp16）。
# 这也是 Step 2 只看权重误差、Step 3 还要看 logits、到 07.5 再落到任务指标（perplexity / KL）的原因：
# 它们对应的恰好是三个不同的签收方——kernel 工程师看权重误差，算法工程师看 logits，产品看任务指标。
#
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
# **读图**：int8 三条线挤在 0.4%–1.8% 之间，说明**位宽够用时粒度只是优化项**；
# int4 per-tensor 是 20.8%，比 int8 per-tensor 大一个数量级，group-32 才把它拉回 6.4%。
# 一句话：**int8 可以粗，int4 必须细。**
#
# ### 实验 C · 7B/8B 的显存与 decode 理论 tok/s 账本
#
# **问题**：7B/8B 在三种精度下的权重显存、decode 上限各是多少？长上下文时 KV cache 会怎么改变结论？
#
# **协议**：先用 `clone()` 测本机有效带宽（第 01 章的方法：读 N 字节 + 写 N 字节），
# 再用 `decode tok/s = 带宽 ÷ 每 token 读取字节` 推账本；假设 batch=1、带宽利用率 100%。

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
# **读数**（带宽随机器与当次实测浮动，绝对值以表格为准）：7B fp16 权重 13GiB，int8 减半到 6.5GiB，
# int4 只剩 3.3GiB——精度每降一档，每 token 读取字节减半，理论 tok/s 就翻一倍
# （表里"相对 fp16 加速"列恒为 2.0/4.0）。
# **注意前提**：这些数字假设 batch=1、带宽利用率 100%，实际还要扣掉 KV cache、kernel 效率、
# 采样开销（第 01 章实测 TinyGPT 只跑到理论值的约 1%）。真实 7B/8B 在 M3 Ultra 上的 decode
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
# **读数**：4K 上下文时 KV 只有 0.5GiB，`--kv-cache-dtype fp8` 几乎看不出区别；
# 到 128K 上下文，fp16 KV 要 16GiB，和 fp16 权重一样大，理论 tok/s 直接腰斩；
# 换成 fp8 KV 后损失补回大半——**KV cache 量化是长上下文场景的刚需**，短上下文只是省显存。
#
# 最后对照一下 TinyGPT：它的 `parameters × 4B = 11MB`，按带宽算理论数万 tok/s，实测只有几百。

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
#
# **反直觉的点在这里**：量化省掉的是 3/4 的权重字节，但 TinyGPT 每步 2ms 里，理论上权重搬运
# 只占 0.02ms（1%）——把 0.02ms 除以 4，省下 0.015ms，落进 2ms 的测量里连噪声都算不上。
# 所以「Mac 上测不出 int8 加速」不是量化没用，而是**测量地板比收益高两个数量级**；
# 要看见这笔账，必须换掉那个「固定开销占绝对主导」的测量环境——这就是 07.5 里融合 kernel 的意义。
#
# 这解释了为什么本章不能"实测 int8 加速"：PyTorch 里把权重量化再反量化只会**多一次读放大**，
# 真正的收益必须靠 Marlin 这类融合 kernel（int4 权重直接进 tensor core 的寄存器），
# 也就是 07.5 里 vLLM 干的事。把账本画成图：

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
# **读图**：int8 恰好是 fp16 的 2 倍，int4 是 4 倍——因为 decode 每 token 只读一遍权重，
# 字节数与 tok/s 严格成反比。这就是「量化主要收益在 decode」的完整推导：
# prefill 是 compute-bound，量化省的是带宽不是算力，在 prefill 上收益有限（还可能因反量化变慢）；
# decode 的算术强度 ≈ 1 FLOP/byte，带宽减半 = 吞吐翻倍。
# 下一节看工业实现怎么把「账本上的加速」变成真加速。

# %% [markdown]
# ## 07.5 对照工业实现（vLLM）
#
# 本课程锁定 **vLLM 0.28.x**。本章手写的 40 行是"教学版最小闭环"，
# vLLM 把它拆成了 **配置（Config）+ 方法（LinearMethod）+ kernel** 三层：
#
# | 我们的实现 | vLLM 中的对应物 | 真实文件（0.28.x） |
# | --- | --- | --- |
# | `absmax_scale`（per-tensor） | `Fp8Config` / `Fp8LinearMethod` 的 per-tensor 缩放 | `vllm/model_executor/layers/quantization/fp8.py` |
# | per-channel / per-group int4 | `AWQConfig` / `GPTQConfig` + Marlin kernel | `.../quantization/awq.py`、`.../quantization/gptq.py`、`.../quantization/marlin_utils.py` |
# | int4/FP8/NVFP4 统一描述 | compressed-tensors 格式 | `.../quantization/compressed_tensors/compressed_tensors.py` |
# | 方法注册表 | `QUANTIZATION_METHODS` / `get_quantization_config` | `.../quantization/__init__.py` |
# | 反量化后 matmul | `LinearBase.quant_method.apply` / `process_weights_after_loading` | `.../layers/linear.py` |
# | 本章的"量化误差"指标 | 模型侧用 perplexity / KL，靠 `lm_eval` 与 `vllm bench` 对比 | — |
#
# 对应的 CLI（0.28.x）：
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

# %% [markdown]
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
# 下一章换一条完全不改精度的路：用一个小模型猜、用大模型验——投机解码。

# %% [markdown]
# ## 07.6 练习与小结
#
# > **一句话记住**：量化换来的加速，来自「每 token 少搬多少字节」；
# > 它换来的误差，来自「一个离群值把整组步长撑大多少倍」。
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
# %% [markdown]
# **自检 3 问**（不看笔记能答出来才算过）：
#
# 1. 8B 模型从 fp16 换 int4，decode 理论 tok/s 为什么正好是 4 倍？哪三个前提一旦不成立，这个 4 倍就消失？
# 2. 同一份权重、同一位宽，per-tensor 和 group-32 的步长差在哪？为什么 int4 比 int8 更依赖分组？
# 3. 权重误差 2% 和 logits 误差 2%，哪个更能说明「这次量化能不能上线」？还缺哪个指标才能拍板？
#
# **工作场景题**（不是练习题，是你上线时会遇到的那种问题）：
#
# 你负责一个 8B 模型的推理服务，产品要求把上下文从 8K 提到 32K，显存不够。同事提议「把权重换成 int4
# 就省下 4 倍显存」。写出你的评估顺序：先算什么账（权重 / KV / 激活分别占多少），
# 先验证哪条指标（权重相对误差还是任务准确率），以及「上下文达标」和「任务指标回退」两个判据
# 哪个优先、为什么。三行 memo 就够，不要写代码。
#
# **你现在应该能回答：**
#
# - 对称量化的一行公式、三个误差来源，以及 `group_size` 在其中的作用。
# - 为什么 int8 可以 per-tensor、int4 必须 per-group；为什么 `group_size=128` 是默认值。
# - 7B/8B 在 fp16/int8/int4 下的权重显存与 decode 理论 tok/s，以及为什么量化收益集中在 decode。
# - vLLM 的 `--quantization` / `--kv-cache-dtype fp8` 各自对应本章的哪个概念、源码在哪个目录。
#
# **小结与下一章**
#
# 量化的收益在 decode 的带宽账本上，代价在精度和 kernel 复杂度上；它把「每 token 读多少字节」
# 压到原来的 1/2–1/4。下一章换一条不动物理字节、而是减少串行前向次数的路：投机解码。
#
# **延伸阅读**
#
# - GPTQ（Frantar et al., 2022）、AWQ（Lin et al., 2023）、SmoothQuant（Xiao et al., 2022）
# - vLLM 文档：Quantization 与 `--kv-cache-dtype`；源码 `vllm/model_executor/layers/quantization/`
