# %% [markdown]
# # 第 02 章 · KV Cache：把 O(T²) 重复计算变成 O(T)
#
# > 同一段 prompt，为什么有人生成快一个数量级？差别在于每一步少算了什么。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 90 分钟 |
# | 运行环境 | 🏠 Mac（MPS / CPU）· 无需 GPU · torch 2.15 |
# | 前置章节 | 01（TinyGPT 前向、采样、prefill/decode 两阶段） |
# | 本章产物 | 手写 `KVCache`、缓存版 attention、缓存生成循环、三组实测、KV 显存公式 |
# | 本章接口 | `KVCache.append / get / advance / seq_len`（03、04 章直接复用） |
#
# **学完你能回答：**
# 1. 为什么生成必须分 prefill 和 decode 两段？它们的瓶颈为什么不同？
# 2. 缓存版 attention 怎么写才对？怎么证明它和全量前向等价？
# 3. KV cache 占多少显存？给定模型和上下文能否心算？
# 4. 不用缓存时总时间为什么是 O(T²)，用了为什么是 O(T)？

# %% [markdown]
# ## 02.0 本章目标与路线
#
# **一句话目标**：把自回归生成从「每步重算整段 prompt」改成「prefill 一次、decode 复用 K/V」，
# 在不改变输出结果的前提下，把每步计算量从 O(S+t) 降到 O(1)，把总时间从二次降到一次。
#
# **代价**：显存。KV cache 随上下文线性增长，长上下文时它比权重还大；这决定了后面的分页（03 章）。
#
# **本章交付四样东西**：
#
# | # | 交付物 | 在哪一节 | 怎么验收 |
# | --- | --- | --- | --- |
# | 1 | `KVCache` 数据结构（连续预分配） | 02.3 Step 1 | 显存公式 == 实际分配字节数 |
# | 2 | 缓存版 attention + 模型前向 | 02.3 Step 2–3 | 与全量前向 logits 一致（atol 1e-3） |
# | 3 | 缓存生成循环（naive vs cached） | 02.3 Step 4–5 | 贪心 token 逐个相等 |
# | 4 | 三组实验：两阶段吞吐、显存账本、总时间 | 02.4 实验 A/B/C | 趋势与公式吻合 |
#
# **阅读路线**：02.1 问题 → 02.2 高阶设计 → 02.3 分步实现（5 步，每步有验收）→ 02.4 实验 → 02.5 对照 vLLM。

# %% [markdown]
# ## 02.1 问题与直觉：naive 生成为什么慢
#
# 因果注意力有一个性质：**生成第 t 个 token 时，前 t−1 个 token 的 K/V 只由它们自己决定**，
# 不会因为后面生成了什么而改变。
#
# 但最直的实现（naive）每步都把「prompt + 已生成」整段重新喂进模型：
#
# ![](../assets/figures/ch02_prefill_decode.svg)
#
# - 生成 T 个 token 要做 T 次前向；
# - 第 t 次前向的长度是 S+t，attention 计算量随长度平方增长；
# - 总计算量 ≈ O(T·(S+T))，**二次**。其中绝大部分是在重算已经算过的 K/V。
#
# KV cache 的做法：第一次前向（prefill）把 prompt 的 K/V 存下来，之后每步（decode）
# 只喂 1 个新 token，复用缓存——每步从 O(S+t) 降到 O(1)，总时间变成 O(S+T)。
#
# 两个必须分清的概念：
#
# | | prefill | decode |
# | --- | --- | --- |
# | 输入 | `(B, S)` 整段 prompt | `(B, 1)` 新 token |
# | 做什么 | 并行算所有位置、所有层的 K/V，写入缓存 | 读缓存 K/V，算 1 个 token，追加缓存 |
# | 瓶颈 | 算力（S 个位置并行） | 带宽（每步读全部权重 + 整个缓存） |
#
# 02.4 实验 A 会量出两者差两个数量级。先不要记数字，先记住**它们为什么不同**。

# %% [markdown]
# ## 02.2 高阶设计
#
# 动手前先把方案定下来。整章只有三个组件和一条数据流。
#
# **组件 1：KVCache（存什么）**
# 模型有 L 层，每层产生 K/V。对每条序列、每一层缓存两份张量，形状 `(B, H_kv, S, D)`；
# S 是已处理的 token 数，每多一个 token 就往 S 维追加一列。
# 本章选**连续预分配**：启动时按 `max_seq_len` 开好整块显存。优点是 attention 取 K/V 零拷贝；
# 缺点是短请求按最大长度占坑，浪费显存（第 03 章用分页解决）。
#
# **组件 2：缓存版 attention（怎么用）**
# 与普通 attention 的差别只有三处：
# 1. 新 token 的 K/V 先 `append` 进 cache；
# 2. 取回 `[0, prefix+T)` 的 K/V（历史 + 刚写入的）；
# 3. mask 规则：**历史前缀全可见，当前块内因果**。
#
# **组件 3：生成循环（谁在调度）**
# - prefill：整段 prompt 一次前向，所有层写入 cache，然后 `advance(S)`；
# - decode：每步喂 1 个 token，所有层 append，再 `advance(1)`。
#
# 数据流：
#
# ```text
# token id → embedding → [ 每层: attention(读 cache) + MLP ] → logits → sample → 下一个 token
# ```

# %% [markdown]
# ### 接口契约
#
# 02.3 的实现、第 03 章的分页、第 04 章的前缀共享，内部结构完全不同，
# 但上层 attention 的调用方式应该不变。所以先约定 4 个方法：
#
# | 方法 | 语义 |
# | --- | --- |
# | `append(layer, k, v)` | 把本步新算的 K/V 写到当前长度之后 |
# | `get(layer, upto=None)` | 取回 `[0, upto)` 的 K/V（`upto` 默认到当前长度） |
# | `advance(n)` | 推进逻辑长度 |
# | `seq_len` | 当前已缓存的 token 数 |
#
# 为什么 `append` 和 `advance` 要分开？看一次 decode step：
#
# ```text
# 当前 seq_len = n
# layer 0: append(k0, v0)   -> 写到位置 n
# layer 1: append(k1, v1)   -> 写到位置 n
# ...
# layer L-1: append(...)    -> 写到位置 n
# advance(1)                -> seq_len: n -> n+1
# ```
#
# 所有层必须写到同一个位置 n，所以长度不能在某一层写完后立刻推进；要等 L 层都写完再 `advance`。
# prefill 同理：每层 `append` 写入 T 个 token，全部层写完后 `advance(T)`。
# `seq_len` 同时给 RoPE 的 `start_pos` 和 causal mask 的边界使用。

# %% [markdown]
# ### 验收标准
#
# 每个实现步骤都配一条可执行的验收，写不出来就说明没理解：
#
# | 步骤 | 验收 |
# | --- | --- |
# | Step 1 数据结构 | 公式字节数 == `tensor.nbytes` 之和；反解 S == 配置值 |
# | Step 2 缓存 attention | 与全量前向的 logits 最大误差 < 1e-3 |
# | Step 3 接入模型 | 同上（整模型输出一致） |
# | Step 4 生成循环 | 贪心生成的 token 逐个相等 |
# | Step 5 实验 | 两阶段吞吐、显存、总时间三条曲线与公式趋势一致 |

# %% [markdown]
# ## 02.3 从零实现：分五步
#
# 分成 5 步，每步先讲目标和做法，再给代码，最后给验收：
#
# | Step | 做什么 | 代码 |
# | --- | --- | --- |
# | 1 | 定义 `KVCache`：怎么存、怎么追加、怎么取 | `KVCache` 类 |
# | 2 | 写缓存版 attention：mask 与位置怎么处理 | `attention_with_cache` |
# | 3 | 把 cache 接进整个模型的前向 | `forward_with_cache` |
# | 4 | 写两条生成路径：naive 与 cached | `generate_naive` / `generate_cached` |
# | 5 | 用断言证明两条路径结果相同 | logits / token 对比 |

# %%
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path.cwd().parent))  # 从 notebooks/ 回到仓库根目录

from aiinfra.bench import bench, kv_cache_bytes, pick_device
from aiinfra.plotting import COLORS, save, setup_style
from aiinfra.tinygpt import GPTConfig, TinyGPT, apply_rope, rope_cos_sin, sample_next

setup_style()
device = pick_device()
torch.manual_seed(0)
np.random.seed(0)

cfg = GPTConfig(max_seq_len=4096)  # 权重仍是 2.8M，只是把最长上下文放宽到 2048+
model = TinyGPT(cfg).to(device).eval()
print(f"device = {device} | params = {model.num_params() / 1e6:.2f}M")

# %% [markdown]
# ### Step 1 · 存储布局：KVCache
#
# 目标：把「每层 K/V、形状 `(B,H,S,D)`、按 token 追加」封装成 4 个方法。
# 本章用连续预分配：`__init__` 按 `max_seq_len` 一次开好全部显存；`append` 是切片赋值
# （不额外分配），`get` 返回**视图**（零拷贝）。

# %%
class KVCache:
    """第 02 章的接口契约：03/04 章复用 append / get / advance / seq_len。"""

    def __init__(self, cfg, batch_size=1, max_seq_len=4096, device="cpu", dtype=torch.float32):
        self.cfg = cfg
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self._len = 0
        shape = (batch_size, cfg.n_head, max_seq_len, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]

    def append(self, layer, k, v):
        """k/v: (B, H, T, D)，写到当前长度之后。"""
        t = k.shape[2]
        end = self._len + t
        assert end <= self.max_seq_len, f"KV cache 溢出：{end} > {self.max_seq_len}"
        self.k[layer][:, :, self._len : end] = k
        self.v[layer][:, :, self._len : end] = v

    def get(self, layer, upto=None):
        end = self._len if upto is None else min(upto, self.max_seq_len)
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, n=1):
        self._len += n

    @property
    def seq_len(self):
        return self._len

# %% [markdown]
# ### Step 2 · 缓存版 attention
#
# 目标：在普通 attention 上只改三处——写入新 K/V、取回前缀+当前块、修正 mask。
# mask 的正确语义：
#
# ```text
# mask[i, j] = 0      if j < prefix                 历史全可见
#            = 0      if prefix <= j <= prefix + i  块内因果
#            = -inf   if j > prefix + i
# ```
#
# 这里故意显式构造二维 mask 让语义可见；真实 kernel 用 `causal=True` + 分块偏移，
# 不会真的建 (T,S) 矩阵。

# %%
def attention_with_cache(attn, x, cache, layer, start_pos):
    """与 aiinfra.tinygpt 同构的 attention，把 KV cache 接进来并显式写对因果 mask。"""
    B, T, C = x.shape
    H, D = attn.cfg.n_head, attn.cfg.head_dim
    q, k, v = attn.qkv(x).split(C, dim=-1)
    q = q.view(B, T, H, D).transpose(1, 2)
    k = k.view(B, T, H, D).transpose(1, 2)
    v = v.view(B, T, H, D).transpose(1, 2)
    cos, sin = rope_cos_sin(torch.arange(start_pos, start_pos + T, device=x.device), D, attn.cfg.rope_theta, x.dtype)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    prefix = cache.seq_len  # 当前块的 query 在全局序列里的起始位置
    cache.append(layer, k, v)  # 写入新 K/V（每层写自己的缓存）
    k_all, v_all = cache.get(layer, upto=prefix + T)  # 取回 [0, prefix+T) 的 K/V

    mask = torch.zeros((T, k_all.shape[2]), device=x.device)
    mask[:, prefix:] = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
    att = F.softmax(((q @ k_all.transpose(-2, -1)) / math.sqrt(D) + mask).float(), dim=-1).to(q.dtype)
    return attn.proj((att @ v_all).transpose(1, 2).reshape(B, T, C))

# %% [markdown]
# ### Step 3 · 把 cache 接进模型
#
# 目标：embedding / MLP / norm 与 `TinyGPT.forward` 完全一致，只把每层 attention 换成缓存版。
# 这样可以在同一套权重上对比「有缓存」和「无缓存」的输出。

# %%
def forward_with_cache(model, idx, cache, start_pos=0):
    """把 cache 注入每一层；embedding / MLP / norm 与 TinyGPT.forward 完全一致。"""
    x = model.tok_emb(idx)
    for layer, block in enumerate(model.blocks):
        x = x + attention_with_cache(block.attn, block.norm1(x), cache, layer, start_pos)
        x = x + block.mlp(block.norm2(x))
    return model.lm_head(model.norm_f(x))

# %% [markdown]
# ### Step 4 · 两条生成路径
#
# - `generate_naive`：每步把整段重跑（反例，用来对照）；
# - `generate_cached`：prefill 一次，之后每步 1 个 token。
#
# 两者都用贪心采样（temperature=0），所以生成结果必须逐个 token 相等。

# %%
@torch.no_grad()
def generate_naive(model, prompt, max_new_tokens):
    """反例：每一步都把「prompt + 已生成」整段重跑。"""
    out = prompt
    for _ in range(max_new_tokens):
        out = torch.cat([out, sample_next(model(out)[:, -1, :], 0.0)], dim=1)
    return out


@torch.no_grad()
def generate_cached(model, prompt, max_new_tokens):
    """正解：prefill 一次，之后每步只喂 1 个 token。"""
    c = KVCache(model.cfg, prompt.shape[0], prompt.shape[1] + max_new_tokens, prompt.device, torch.float32)
    logits = forward_with_cache(model, prompt, c)
    c.advance(prompt.shape[1])
    out, per_step = prompt, [logits]
    for _ in range(max_new_tokens):
        nxt = sample_next(per_step[-1][:, -1, :], 0.0)
        out = torch.cat([out, nxt], dim=1)
        per_step.append(forward_with_cache(model, nxt, c, start_pos=c.seq_len))
        c.advance(1)
    return out, torch.cat(per_step, dim=1)

# %% [markdown]
# ### Step 5 · 验收：证明缓存没改变结果
#
# 把 cached 路径每一步的 logits 收集起来，与「一次全量前向」的 logits 逐位置比较。
# `atol=1e-3` 是给浮点累加顺序差异留的余量（一次算 120 个位置 vs 分 1+24 次算）。

# %%
prompt = torch.randint(0, cfg.vocab_size, (2, 96), device=device)
naive_out = generate_naive(model, prompt, 24)
cached_out, cached_logits = generate_cached(model, prompt, 24)
with torch.no_grad():
    full_logits = model(cached_out)

assert torch.equal(naive_out, cached_out), "贪心解码下两条路径必须生成完全相同的 token"
logits_gap = (full_logits - cached_logits).abs().max().item()
assert torch.allclose(full_logits, cached_logits, atol=1e-3, rtol=1e-3), f"logits 不一致：{logits_gap}"
assert naive_out.shape == (2, 120) and cached_out.shape == (2, 120)
print(f"token 完全一致 ✓  |  logits 最大误差 = {logits_gap:.2e}（阈值 1e-3）")

# %% [markdown]
# 两条路径输出一致，但计算量差一个数量级。下面用 00 章的 `bench()` 量出来。

# %% [markdown]
# ## 02.4 实测
#
# 三个实验，分别回答三个问题：
#
# | 实验 | 问题 | 产出 |
# | --- | --- | --- |
# | A | prefill 和 decode 差多少？为什么？ | 两阶段吞吐表 |
# | B | KV cache 占多少显存？公式对不对？ | 对账表 + 8B 模型显存表 |
# | C | 省下来的时间有多少？ | 每 token 延迟与累计时间曲线 |
#
# 测量协议：M3 Ultra / 96GB / torch 2.15 / MPS，`warmup=2, reps=5` 取中位数；
# 固定随机权重，绝对值随机器变化，**趋势才是结论**。

# %% [markdown]
# ### 实验 A · prefill 与 decode 的吞吐差
#
# batch=8，上下文 256–2048。prefill 吞吐 = `S ÷ 耗时`；decode 吞吐 = `1 ÷ 单步耗时`；
# 表里都是**每序列**数字，聚合吞吐 ×8。

# %%
B = 8
rows = []
for L in [256, 512, 1024, 2048]:
    idx = torch.randint(0, cfg.vocab_size, (B, L), device=device)

    def prefill(L=L, idx=idx):
        c = KVCache(cfg, B, L + 1, device, torch.float32)
        with torch.no_grad():
            return forward_with_cache(model, idx, c)

    cache = KVCache(cfg, B, L + 1, device, torch.float32)
    with torch.no_grad():
        forward_with_cache(model, idx, cache)
    cache.advance(L)

    def decode(cache=cache, idx=idx, L=L):
        with torch.no_grad():
            return forward_with_cache(model, idx[:, :1], cache, start_pos=L)

    b_pre = bench(prefill, warmup=2, reps=5, device=device)
    b_dec = bench(decode, warmup=2, reps=5, device=device)
    rows.append(
        {
            "上下文 tokens": L,
            "prefill（ms）": round(b_pre.median_ms, 2),
            "decode 单步（ms）": round(b_dec.median_ms, 3),
            "prefill 吞吐（tok/s）": round(L / b_pre.median_ms * 1000),
            "decode 吞吐（tok/s）": round(1 / b_dec.median_ms * 1000),
        }
    )

df_phase = pd.DataFrame(rows)
df_phase

# %% [markdown]
# **读表**：prefill 一次并行处理 S 个 token，吞吐 6.6–9.9 万 tok/s；decode 每步只出 1 个 token，
# 吞吐 455–750 tok/s，**差两个数量级**。原因不是 decode 算得多，而是它「读得多、算得少」：
# 每生成 1 个 token，至少要把全部权重和整个 KV cache 读一遍，算术强度 ≈ 1 FLOP/byte，
# 完全受带宽约束。这也解释了为什么 batch 越大 decode 越划算（同一份权重服务更多请求），
# 第 05 章的连续批处理就是围绕这一点做文章。

# %% [markdown]
# ### 实验 B · KV 显存账本
#
# 显存布局很朴素：每一层存两份张量 K / V，形状 `(B, H_kv, S, D)`；序列增长就是往 S 维追加：
#
# ![](../assets/figures/ch02_memory_layout.svg)
#
# 把图换算成公式（全是乘法）：
#
# ```text
# KV 字节 = 2 × L × H × D × S × B × dtype_bytes
# ```
#
# - `2`：K 和 V 两份张量；
# - `L`：层数；
# - `H`：**KV 头数**（GQA 下小于注意力头数，这是最容易算错的地方）；
# - `D`：每个头的维度 `head_dim`；
# - `S`：序列长度；`B`：batch；`dtype_bytes`：每个元素几字节（fp16=2，fp32=4）。
#
# 用本章 TinyGPT 代进去（L=2, H=4, D=64, S=4096, B=1, fp32）：
#
# ```text
# 2 × 2 × 4 × 64 × 4096 × 1 × 4 = 16,777,216 字节 = 16 MiB
# ```

# %% [markdown]
# #### 用两个方向对账
#
# 公式写出来容易，但很容易漏项。所以用一个**能精确测量**的小缓存来对账：
# 先用具体张量把公式推一遍，再从两个方向各核对一次。
#
# 先看**一个** K 张量（TinyGPT，B=1, S=4096, fp32）：
#
# - 形状：`(B, H, S, D) = (1, 4, 4096, 64)`
# - 元素数：`1 × 4 × 4096 × 64 = 1,048,576`
# - 字节数：`1,048,576 × 4 B = 4,194,304 B = 4 MiB`
#
# 每层有 K、V 两份 → 8 MiB；2 层 → **16 MiB**。这就是
# `2 × L × H × D × S × B × dtype_bytes` 的来源。

# %% [markdown]
# **对账 A（配置 → 字节）**：把配置代进公式，算出一个预测值，再和 PyTorch 实际分配的
# `t.nbytes` 之和比。相等，说明公式没有漏掉 K/V、层数、dtype 中的任何一项。
#
# **对账 B（字节 → 配置）**：反过来，只用总字节数和 L/H/D/B/dtype，解出
# `S = 字节数 ÷ (2·L·H·D·B·dtype_bytes)`。解出来必须等于构造缓存时传入的 `seq_len`（4096）。
#
# 为什么要两个方向都做？因为两种用法在工程里都会出现：
#
# - 上线前做**容量规划**：已知模型和上下文，用公式算要多少显存（方向 A）；
# - 线上做**故障诊断**：看到显存涨了，反推"现在缓存了多少 token"（方向 B）。
#
# 如果公式错一项，两个方向会同时暴露：忘了乘 2（K+V）→ 只剩 8 MiB；
# 忘了乘 dtype（fp32=4 字节）→ 差 4 倍；GQA 把 H 当成 32 而不是 8 → 高估 4 倍。

# %%
def measured_bytes(cache):
    return sum(t.nbytes for t in cache.k + cache.v)


demo = KVCache(cfg, batch_size=1, max_seq_len=4096, device="cpu", dtype=torch.float32)
k0 = demo.k[0]
print("K 张量:", tuple(k0.shape), "| 元素数:", f"{k0.numel():,}", "| 字节数:", f"{k0.nbytes:,}")
print("对账 A：公式 =", f"{kv_cache_bytes(cfg.n_layer, cfg.n_head, cfg.head_dim, 4096, 1, 4):,}",
      "| 实际 =", f"{measured_bytes(demo):,}")
per_token_bytes = 2 * cfg.n_layer * cfg.n_head * cfg.head_dim * 1 * 4
print("对账 B：反解 S =", measured_bytes(demo) // per_token_bytes, "（配置值 4096）")

rows = []
for dtype, name in [(torch.float32, "float32"), (torch.bfloat16, "bfloat16")]:
    for batch, seq in [(1, 1024), (3, 1024), (8, 2048)]:
        cache = KVCache(cfg, batch_size=batch, max_seq_len=seq, device="cpu", dtype=dtype)
        predicted = kv_cache_bytes(cfg.n_layer, cfg.n_head, cfg.head_dim, seq, batch, dtype.itemsize)
        measured = measured_bytes(cache)
        per_token = 2 * cfg.n_layer * cfg.n_head * cfg.head_dim * batch * dtype.itemsize
        implied_seq = measured // per_token
        assert measured == predicted, f"公式与 tensor.nbytes 不一致：{predicted} vs {measured}"
        assert implied_seq == seq, "从字节数反解出的序列长度必须等于配置值"
        rows.append(
            {
                "dtype": name,
                "batch": batch,
                "seq_len": seq,
                "公式（MiB）": round(predicted / 2**20, 2),
                "tensor.nbytes（MiB）": round(measured / 2**20, 2),
                "反解 seq_len": implied_seq,
            }
        )
df_mem = pd.DataFrame(rows)
df_mem

# %% [markdown]
# 双向验证都通过：公式 → 字节数、字节数 → 序列长度。顺便看 dtype 的影响：
# 同样配置，bfloat16 的显存正好是 float32 的一半。
#
# 现在把公式用到真实模型上。Qwen3-8B / Llama-3-8B 都是 GQA（32 个注意力头、8 个 KV 头），
# 所以公式里的 H 用 8，不是 32——这一点算错，显存会高估 4 倍。

# %%
def kv_gib(n_layer, n_kv_head, head_dim, seq_len, batch, dtype_bytes=2):
    return kv_cache_bytes(n_layer, n_kv_head, head_dim, seq_len, batch, dtype_bytes) / 2**30


models = {
    "Qwen3-8B（GQA，8 KV heads）": (36, 8, 128),
    "Llama-3-8B（GQA，8 KV heads）": (32, 8, 128),
}
rows = []
for name, (n_layer, n_kv_head, head_dim) in models.items():
    per_token_kib = 2 * n_layer * n_kv_head * head_dim * 2 / 1024
    for ctx in [4096, 32768]:
        for batch in [1, 8, 32]:
            gib = kv_gib(n_layer, n_kv_head, head_dim, ctx, batch)
            rows.append(
                {
                    "模型": name,
                    "每 token KV（KiB）": round(per_token_kib, 1),
                    "上下文": f"{ctx // 1024}K",
                    "batch": batch,
                    "KV 显存（GiB，fp16）": round(gib, 2),
                    "占 24GB 显存": f"{gib / 24:.0%}",
                }
            )
df_kv = pd.DataFrame(rows)
df_kv

# %% [markdown]
# **读表**：Qwen3-8B 每 token 的 KV 是 144 KiB；4K 上下文、batch=8 就要 4.5 GiB，看着还行；
# 但 32K 上下文、batch=32 是 **144 GiB**——单张 80GB 卡都放不下。这就是长上下文服务
# 「显存先于算力成为瓶颈」的原因。GQA 已经替我们省掉 4 倍；如果没有 GQA（32 个 KV 头），
# 32K / batch=32 会是 576 GiB。

# %% [markdown]
# ### 实验 C · 生成总时间：O(T²) vs O(T)
#
# 有了单步延迟，就能回答「生成长度翻倍，时间为什么变 4 倍」。
# prompt=512，测「已生成 t 个 token 时，下一个 token 要花多久」：
# naive 每步重算整个上下文，cached 每步只算 1 个 token。

# %%
P = 512
L_GRID = [512, 768, 1024, 1280, 1536, 1792, 2048]
rows = []
for L in L_GRID:
    idx = torch.randint(0, cfg.vocab_size, (B, L), device=device)

    def naive_step(idx=idx):
        with torch.no_grad():
            return model(idx)

    cache = KVCache(cfg, B, L + 1, device, torch.float32)
    with torch.no_grad():
        forward_with_cache(model, idx, cache)
    cache.advance(L)

    def cached_step(cache=cache, idx=idx, L=L):
        with torch.no_grad():
            return forward_with_cache(model, idx[:, :1], cache, start_pos=L)

    r_naive = bench(naive_step, warmup=2, reps=5, device=device)
    r_cached = bench(cached_step, warmup=2, reps=5, device=device)
    rows.append(
        {
            "已生成 tokens": L - P,
            "naive（ms/token）": round(r_naive.median_ms, 2),
            "cached（ms/token）": round(r_cached.median_ms, 2),
        }
    )
df_lat = pd.DataFrame(rows)
df_lat

# %%
generated = df_lat["已生成 tokens"]
naive_ms = df_lat["naive（ms/token）"]
cached_ms = df_lat["cached（ms/token）"]

fig, ax = plt.subplots()
ax.plot(generated, naive_ms, "o-", color=COLORS["danger"], label="naive: recompute full sequence")
ax.plot(generated, cached_ms, "s-", color=COLORS["success"], label="KV cache: one token per step")
ax.set_xlabel("Generated tokens (prompt = 512)")
ax.set_ylabel("Latency per new token (ms)")
ax.set_title("Per-token decode latency vs generated length")
ax.legend()
save(fig, "ch02_step_latency");

# %% [markdown]
# **读图**：naive 的每 token 延迟从 5.2ms 涨到 28.4ms（5.4 倍），cached 稳定在 1.5–1.7ms。
# 把两条曲线沿 x 轴累加，就得到「生成 T 个 token 的总时间」——naive 是二次，cached 是一次：

# %%
x = df_lat["已生成 tokens"].to_numpy()
naive_curve = np.interp(np.arange(0, 1537), x, df_lat["naive（ms/token）"])
cached_curve = np.interp(np.arange(0, 1537), x, df_lat["cached（ms/token）"])
gen = np.arange(1, 1538)
total_naive = np.cumsum(naive_curve)
total_cached = np.cumsum(cached_curve)

fig, ax = plt.subplots()
ax.plot(gen, total_naive, color=COLORS["danger"], label="naive: ~O(T^2)")
ax.plot(gen, total_cached, color=COLORS["success"], label="KV cache: ~O(T)")
ax.set_xlabel("Generated tokens")
ax.set_ylabel("Cumulative generation time (ms)")
ax.set_title("Cumulative time from measured step latency")
ax.legend()
save(fig, "ch02_total_time")
print(f"生成 1536 个 token：naive 约 {total_naive[-1] / 1000:.1f}s vs cached 约 {total_cached[-1] / 1000:.1f}s")

# %% [markdown]
# **读图**：生成 1536 个 token，naive 累计约 24s，cached 累计约 2.5s，差约 10 倍；
# 差距随生成长度继续拉大（二次 vs 一次）。
#
# 为什么 cached 不是「1 个 token 的算力」而是一个 ~1.5ms 的地板？因为 MPS 上是小模型，
# 每次前向的固定开销（Python 调用 + kernel 启动 + 同步）占了大头。真实 8B 模型上这部分
# 相对可忽略，decode 的单步时间会更接近「权重字节数 ÷ 显存带宽」。

# %% [markdown]
# ## 02.5 对照工业实现（vLLM）
#
# 本课程锁定 **vLLM 0.28.x**（v1 engine 默认）。本章手写的 60 行在 vLLM 里被拆成
# 「配置 → 显存池 → block 管理 → attention backend」四层：
#
# | 我们的实现 | vLLM 中的对应物 | 真实文件（0.28.x） |
# | --- | --- | --- |
# | `KVCache.__init__` 预分配 (B,H,S,D) | `CacheConfig`（`block_size`、`cache_dtype`、`gpu_memory_utilization`） | `vllm/config/cache.py` |
# | 显存总量估算 | `get_kv_cache_config()` 按 `gpu_memory_utilization` 反推 block 数 | `vllm/v1/core/kv_cache_utils.py` |
# | `KVCache.append / get` | `KVCacheManager.allocate_slots()` / `BlockPool.get_new_blocks()` | `vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/block_pool.py` |
# | 连续 `(B,H,S,D)` 布局 | 分页 block + `BlockTable` 逻辑→物理映射 | `vllm/v1/worker/block_table.py` |
# | `attention_with_cache()` | paged attention kernel（FlashAttention / FlashInfer 后端） | `vllm/v1/attention/backends/flash_attn.py` |
#
# 对应的 CLI / 配置参数（全部是真实参数名）：
#
# | 参数 | 本章的哪条公式 | 说明 |
# | --- | --- | --- |
# | `--block-size` | 03 章：分页粒度 | 默认 16；本章的连续预分配相当于 block_size = max_model_len |
# | `--gpu-memory-utilization` | KV 显存预算 | 默认 0.9，留 10% 给激活值/临时 buffer |
# | `--max-model-len` | 公式里的 S | 直接决定单请求 KV 上限 |
# | `--kv-cache-dtype` | 公式里的 dtype_bytes | `auto` / `fp8`；fp8 把 KV 显存再砍半 |
# | `--max-num-seqs` | 公式里的 B | 同时在跑的请求数上限 |
#
# **差距在哪**：vLLM 不预分配 `(B,H,S,D)`，而是先按 `gpu_memory_utilization` 算出能开多少
# 个 KV block（`num_gpu_blocks`），再按 block 粒度分给请求。这样不同长度的请求共享显存池，
# 且能实现 03 章的分页、04 章的前缀共享。本章的连续实现要求每个请求独占 `max_seq_len`——
# 这就是下一章要动手解决的问题。

# %% [markdown]
# ## 02.6 练习与小结
#
# **练习**
#
# 1. ⭐ 把 `KVCache` 的 `dtype` 从 `float32` 换成 `bfloat16`，重跑显存表与延迟表。显存应该减半，
#    延迟会减半吗？为什么（提示：decode 是带宽受限还是算力受限？MPS 对 dtype 的支持如何）？
# 2. ⭐ 在 prefill/decode 表里补上 batch=1 与 batch=32 两行，画出「decode 吞吐 vs batch」。
#    用「每步要读的字节数 ÷ 显存带宽」解释曲线形状。
# 3. ⭐⭐ 给 `KVCache` 加上 `reset()`，复用同一块显存连续跑两轮生成，验证第二轮不需要重新分配
#    显存（提示：记录 `torch.mps.current_allocated_memory()` 或 CPU 上的 `id()`）。
# 4. ⭐⭐ 把 `attention_with_cache` 里的显式 mask 换成两段式 `F.scaled_dot_product_attention`
#    （前缀 `is_causal=False`，当前块 `is_causal=True`），验证 logits 一致（`atol=1e-3`），
#    并对比 MPS 上的延迟。
# 5. ⭐⭐⭐ 容量规划：给定 SLO（TTFT < 300ms、TPOT < 30ms、并发 32），用本章公式推导
#    Qwen3-8B fp16 在 80GB 卡上的最大上下文长度。列出 3 条让你不敢直接采信这个数字的
#    不确定性来源（评分要点：GQA 的 KV 头数、权重/激活值预留、prefill 与 decode 的预算、
#    PagedAttention 碎片）。
#
# **你现在应该能回答：**
#
# - KV 显存公式的每一项分别对应什么？为什么 GQA 会让公式里的 H 变小？
# - prefill 和 decode 的瓶颈分别是什么？为什么 batch 增大对 decode 特别有利？
# - 不用 KV cache 时总时间为什么是 O(T²)？用了之后为什么是 O(T)？
#
# **小结与下一章**
#
# 本章的 KV cache 是**连续预分配**：每个请求按 `max_seq_len` 占一块显存。它简单、attention
# 零拷贝，但 200 个长短不一的请求会让 74% 的显存变成碎片（下一章实测），相同前缀也无法共享。
# 03 章引入分页（PagedAttention）：把 KV 切成固定大小的 block，用 block table 把逻辑位置
# 映射到物理 block——本章的 `append/get/advance/seq_len` 接口保持不变。
#
# **延伸阅读**
#
# - Kwon et al., *PagedAttention / vLLM*（SOSP 2023）：第 3 节的 KV cache 显存浪费分析
# - vLLM 文档：`gpu_memory_utilization`、`max_model_len` 与 KV cache 容量计算
# - 下一章：03 PagedAttention，把本章的「预分配 + 连续布局」换成「按需 + 分页」

