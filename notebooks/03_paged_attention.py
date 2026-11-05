# %% [markdown]
# # 第 03 章 · PagedAttention：用 block table 消灭 KV 显存碎片
#
# > 上一章的连续预分配能用，但 200 个长短不一的请求会让 74% 的显存躺在碎片里。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 90 分钟 |
# | 运行环境 | 🏠 Mac（MPS / CPU）· 无需 GPU · torch 2.15 |
# | 前置章节 | 02（KV cache 接口、连续布局的代价） |
# | 本章产物 | `BlockManager`、`PagedKVCache`、分页 attention、三组实测（碎片率 / block_size / gather 代价） |
# | 本章接口 | `PagedKVCache` 与 02 章 `KVCache` 同接口，04 章直接复用 |
#
# **学完你能回答：**
# 1. 连续预分配浪费的 74% 显存具体浪费在哪？分页为什么只浪费最后一个 block 的尾部？
# 2. block table 是怎么把「逻辑位置」映射到「物理 block」的？gather 版 attention 为什么慢？
# 3. block_size 越大越好还是越小越好？vLLM 为什么默认 16？
# 4. 同样一份显存，分页为什么能多服务 3.8 倍请求？

# %% [markdown]
# ## 03.0 本章目标与路线
#
# **一句话目标**：把 KV 显存从「每个请求按 `max_seq_len` 预留一整块连续区域」改成「按固定大小
# block 按需申请、用 block table 还原逻辑顺序」，在不改变 attention 数学结果的前提下，把 200 个
# 真实长度分布的请求的显存碎片率从 73.9% 压到 1.3%。
#
# **代价 / 边界**：分页引入两次额外开销 —— block table 的显存与查询、`get()` 时的 gather 拷贝
# （把散落的物理 block 拼成逻辑连续张量）。本章分别量化，并说明真实 kernel 如何消掉 gather。
#
# **本章交付四样东西**：
#
# | # | 交付物 | 在哪一节 | 怎么验收 |
# | --- | --- | --- | --- |
# | 1 | `BlockManager`：物理 block 的分配/追加/回收 | 03.3 Step 1 | 分配-释放账本断言（含双重释放） |
# | 2 | `PagedKVCache`：分页存储、跨 block 写入、gather 取回 | 03.3 Step 2 | 写入/取回逐元素一致，池字节数符合公式 |
# | 3 | 分页 attention 接入模型前向 | 03.3 Step 3 | logits 形状正确、无 NaN/Inf |
# | 4 | 等价性断言 + 三组实验 | 03.3 Step 4、03.4 实验 A/B/C | 与连续布局 logits 一致（atol 1e-5）；趋势与容量公式吻合 |
#
# **阅读路线**：03.1 问题 → 03.2 高阶设计 → 03.3 分步实现（4 步，每步有验收）→
# 03.4 实验（碎片率 / block_size / gather）→ 03.5 对照 vLLM → 03.6 练习。

# %% [markdown]
# ## 03.1 问题与直觉：一块「预留」的显存，大部分在睡觉
#
# 02 章的 KV cache 是**连续预分配**：请求一进来就按 `max_seq_len` 占一整块显存。它简单，
# attention 取 K/V 零拷贝，但有两个浪费来源：
#
# - **内部碎片**：请求只用了 400 个 token，却占着 2048 个 token 的位置 —— 浪费 80%。
# - **外部碎片**：不同长度的请求交错到达、释放后留下大小不一的空洞，新请求即使总量够也可能放不下。
#
# 分页（PagedAttention）借了操作系统虚拟内存的思路：把 KV 切成固定大小的 **block**（默认 16 个
# token），请求按需申请；物理 block 不必连续，用一张 **block table** 记录「逻辑第 i 块 → 物理第几号块」：
#
# ![](../assets/figures/ch03_contiguous_vs_paged.svg)
#
# 左边连续布局为了一个长请求预留整块；右边分页布局的 block 散落在池子里，逻辑顺序由 block table
# 还原。`get()` 时按 block table 把物理 block **gather** 成逻辑连续的 K/V 再送进 attention ——
# 这是教学版实现；真实 kernel 直接按 block 遍历，不做这次拷贝。03.4 实验 A 会量出碎片率的差别，
# 实验 C 会量出 gather 的代价。

# %% [markdown]
# **关键概念对比与两个子问题**
#
# | 维度 | 连续预留（02 章） | 分页（本章） |
# | --- | --- | --- |
# | 分配单位 | 一整个 `max_seq_len` | 一个 block（16 token） |
# | 内部碎片 | 最大 `max_seq_len - len` | 最多 `block_size - 1` 个 token |
# | 外部碎片 | 有（空洞无法复用） | 无（任何空闲 block 都能用） |
# | 请求增长 | 不需要（已按上限预留） | 追加新 block |
# | 物理连续性 | 必须连续 | 不要求 |
# | 取 K/V | 零拷贝切片 | gather 拷贝（可被 kernel 消除） |
# | 前缀共享 | 不可能 | 同一 block 可被多请求共享（04 章） |
#
# 把上一章的问题变成两个可验收的子问题：
#
# 1. **正确性**：分页布局只是「物理地址变了」，attention 的数学结果必须与连续布局**逐元素一致**。
#    这是本章所有断言里最硬的一条（Step 4）。
# 2. **成本**：分页换来显存利用率，代价是 block table 与 gather。哪一项更大、什么时候划算，
#    必须用数字回答（实验 A/B/C）。

# %% [markdown]
# ## 03.2 高阶设计
#
# 动手前先把方案定下来。整章只有三个组件和一条数据流。
#
# **组件 1：存什么（物理池 + block table）**
#
# 物理显存按层预分配成一个池：每层 `(num_blocks, H_kv, block_size, D)` 的 K 张量和 V 张量，
# 整个池子被所有请求共享。每个请求只持有一张小的 **block table**（`list[int]`），长度为
# `ceil(当前长度 / block_size)`。逻辑位置 `p` 对应的物理位置是：
#
# ```text
# block_id = block_table[p // block_size]
# offset   = p % block_size
# ```
#
# 这层间接寻址是分页的全部魔法：**逻辑连续，物理离散**。`BlockManager` 只回答「下一个可用物理块
# 是几号」，不碰显存内容，所以显存池可以跨请求共享，也方便 04 章加引用计数。

# %% [markdown]
# **组件 2：怎么用（写入与取用）**
#
# - 写入：新 token 的 K/V 按逻辑位置 `p` 拆到「已拥有的最后一个 block 的空位」或「新申请的
#   block」里；一个 token 可能跨 block 边界，所以要按段写。
# - 取用：`get()` 用 `block_table[:ceil(end/bs)]` 做 `index_select`，把物理 block 拼成
#   `(1, H, nb*bs, D)` 的连续张量，再切到 `end`。attention 代码**一行都不用改**。
# - 代价：这次拼接是 O(长度) 的拷贝；前缀越长，单次 decode 的额外成本越明显（实验 C 量化）。
#
# **组件 3：谁调度（生命周期）**
#
# - 请求创建：`block_table = []`；
# - prefill：`append` 按需向 manager 申请 block，写入 prompt 的 K/V；
# - decode：每步 `append` 1 个 token，所有层写完后 `advance(1)`；
# - 请求结束：`free(block_table)`，所有物理 block 回到池子。

# %% [markdown]
# 数据流：
#
# ```text
# 请求 tokens
#    │
#    ├─ BlockManager.append(block_table, length)   按需申请物理 block
#    ├─ 写 K/V 到 block_table[p // bs][:, p % bs]  按段写入
#    ├─ PagedKVCache.get(layer, upto)              gather 成逻辑连续张量
#    └─ attention(q, k_all, v_all)                 与 02 章完全相同
# 请求结束
#    └─ BlockManager.free(block_table)             归还物理 block
# ```

# %% [markdown]
# ### 接口契约
#
# 上层 attention 只认 02 章的四个方法；分页把「物理怎么摆」藏在下面。本章新增的
# `BlockManager` 与 `PagedKVCache` 分工如下：
#
# | 组件 | 方法 | 语义 |
# | --- | --- | --- |
# | `BlockManager` | `allocate()` | 取一个空闲物理 block，返回 id |
# | `BlockManager` | `append(block_table, length)` | 保证能容纳 `length` 个 token，返回本次新分配的 block |
# | `BlockManager` | `free(block_table)` | 归还整张表并清空它 |
# | `PagedKVCache` | `append(layer, k, v)` | 与 02 章同名同义，内部跨 block 写 |
# | `PagedKVCache` | `get(layer, upto=None)` | 返回**逻辑连续**的 K/V（gather 后） |
# | `PagedKVCache` | `advance(n)` / `seq_len` | 逻辑长度，语义与 02 章完全一致 |
#
# 关键时间线：一次 decode step 里，`seq_len = n` 时每层 `append` 把 K/V 写到逻辑位置 n，
# L 层都写完后才 `advance(1)`；`get` 的 `upto = seq_len + T` 决定要 gather 到哪个 block。

# %% [markdown]
# ### 验收标准
#
# | 步骤 | 验收 |
# | --- | --- |
# | Step 1 BlockManager | 分配/释放账本正确；双重释放被断言拦住 |
# | Step 2 PagedKVCache | 130 token = 9 个 block；写入/取回逐元素一致；池字节数符合公式 |
# | Step 3 接入 attention | 整模型前向 logits 形状正确、无 NaN/Inf |
# | Step 4 等价性 | 分页 vs 连续 logits 最大误差 < 1e-5；block 账本无泄漏 |

# %% [markdown]
# ## 03.3 从零实现：分四步
#
# 按「存 → 用 → 接 → 证」四步走，每步先讲目标和做法，再给代码，最后给验收：
#
# | Step | 做什么 | 代码 |
# | --- | --- | --- |
# | 1 | 管理物理 block 的分配/追加/回收 | `BlockManager` |
# | 2 | 分页存储 + 跨 block 写入 + gather 取回 | `PagedKVCache` |
# | 3 | 把分页缓存接进 attention 与模型前向 | `attention_with_cache` / `forward_with_cache` |
# | 4 | 证明分页与连续布局等价、账本无泄漏 | `assert` 断言 |

# %% [markdown]
# ### 准备：模型、设备与 02 章的参考实现
#
# 本章仍然用 2.8M 的 TinyGPT（随机权重，只关心形状与带宽）。为了让 Step 4 能证明「分页没改变
# 数学结果」，先把 02 章的连续 `KVCache` 内联一份最小实现作为参考 —— 它也是实验里的对照对象。
#
# 代码从仓库根目录导入 `aiinfra` 的模型与测量工具：

# %%
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path.cwd().parent))

from aiinfra.bench import bench, kv_cache_bytes, pick_device
from aiinfra.plotting import COLORS, save, setup_style
from aiinfra.tinygpt import GPTConfig, TinyGPT, apply_rope, rope_cos_sin

setup_style()
device = pick_device()
torch.manual_seed(0)
np.random.seed(0)

cfg = GPTConfig(max_seq_len=8192)  # 本章测到 4096 上下文
model = TinyGPT(cfg).to(device).eval()
H, D, NL = cfg.n_head, cfg.head_dim, cfg.n_layer
print(f"device = {device} | params = {model.num_params() / 1e6:.2f}M | H={H} D={D} L={NL}")

# %% [markdown]
# ### 参考实现：02 章的连续 KVCache
#
# 只保留对比需要的四个方法。`get` 返回**视图**（切片），没有拷贝 —— 这正是分页版要付代价的地方。

# %%
class KVCache:
    """02 章的连续布局实现，本章作为 allclose 的参考。"""

    def __init__(self, cfg, batch_size=1, max_seq_len=4096, device="cpu", dtype=torch.float32):
        self.cfg, self.batch_size, self.max_seq_len = cfg, batch_size, max_seq_len
        self._len = 0
        shape = (batch_size, cfg.n_head, max_seq_len, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]

    def append(self, layer, k, v):
        end = self._len + k.shape[2]
        assert end <= self.max_seq_len
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
# ### Step 1 · BlockManager：只做 id 级分配
#
# **目标**：把「哪些物理 block 空闲、下一个用几号」封装起来，完全不碰显存内容。
#
# **做法**：维护一个空闲 id 列表 + 一个集合（O(1) 判重复释放）；`append(table, length)` 把
# 「长度 → 需要几个 block」的换算收敛到一个地方，对应 vLLM 的 `allocate_slots()` 语义；
# `free` 清空传入的 block table，并用断言防双重释放 —— 账本错了在推理服务里极难排查。

# %%
class BlockManager:
    """固定大小的物理 block 池；只管理 id，不管理显存内容。"""

    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free = list(range(num_blocks))[::-1]  # 从尾部 pop，先分配 0 号 block
        self._free_set = set(range(num_blocks))
        self.num_allocated = 0

    def allocate(self):
        if not self._free:
            raise MemoryError("KV block pool exhausted")
        block_id = self._free.pop()
        self._free_set.discard(block_id)
        self.num_allocated += 1
        return block_id

    def append(self, block_table, length):
        """确保 block_table 能容纳 length 个 token，返回本次新分配的 block。"""
        need = -(-length // self.block_size)
        new_blocks = []
        while len(block_table) < need:
            block_id = self.allocate()
            block_table.append(block_id)
            new_blocks.append(block_id)
        return new_blocks

    def free(self, block_table):
        for block_id in block_table:
            assert block_id not in self._free_set, f"double free: block {block_id}"
            self._free.append(block_id)
            self._free_set.add(block_id)
            self.num_allocated -= 1
        block_table.clear()

    @property
    def num_free_blocks(self):
        return len(self._free)

# %%
# 验收：8 个 block 的池子里申请 3 块、释放后回到 8 块；重复释放被断言拦住
bm_demo = BlockManager(num_blocks=8, block_size=4)
table = []
new_blocks = bm_demo.append(table, length=10)  # ceil(10/4) = 3 块
assert table == [0, 1, 2] and new_blocks == [0, 1, 2]
assert bm_demo.num_allocated == 3 and bm_demo.num_free_blocks == 5
assert bm_demo.append(table, length=10) == []  # 容量够，不再分配
bm_demo.free(table)
assert bm_demo.num_allocated == 0 and bm_demo.num_free_blocks == 8
try:
    bm_demo.free([0])
    raise AssertionError("双重释放没有被拦住")
except AssertionError as exc:
    assert "double free" in str(exc)
print("BlockManager 账本 ✓：申请 3 块 → 释放 → 空闲 8 块；重复释放被断言拦截")

# %% [markdown]
# **验收**：申请 3 块、释放后回到 8 块；重复释放触发断言 —— 分配/回收账本不变量成立。

# %% [markdown]
# ### Step 2 · PagedKVCache：物理分页，逻辑连续
#
# **目标**：在分页布局上实现与 02 章相同的四个方法（`append / get / advance / seq_len`）。
#
# **做法**：`append` 先让 manager 保证容量，再按段写入；最后一段往往只写半个 block（内部碎片就
# 来源在这里）。`get` 用 `block_table` 做 `index_select` 拼出逻辑连续张量，再切到 `end`。
# 注意 `get` 返回的是**拷贝**而不是视图 —— 实验 C 会量出它的成本。

# %%
class PagedKVCache:
    """与 02 章 KVCache 同接口；物理存储是 (num_blocks, H, block_size, D) 的分页池。"""

    def __init__(self, cfg, block_manager, k_blocks, v_blocks, device="cpu"):
        self.cfg, self.k, self.v = cfg, k_blocks, v_blocks
        self.bm, self.block_size, self.device = block_manager, block_manager.block_size, device
        self.block_table, self._len = [], 0

    def append(self, layer, k, v):
        start, end = self._len, self._len + k.shape[2]
        self.bm.append(self.block_table, end)
        pos = start
        while pos < end:
            block_id = self.block_table[pos // self.block_size]
            offset = pos % self.block_size
            n = min(self.block_size - offset, end - pos)
            self.k[layer][block_id, :, offset : offset + n] = k[0, :, pos - start : pos - start + n]
            self.v[layer][block_id, :, offset : offset + n] = v[0, :, pos - start : pos - start + n]
            pos += n

    def get(self, layer, upto=None):
        capacity = len(self.block_table) * self.block_size
        end = self._len if upto is None else min(upto, capacity)
        nb = -(-end // self.block_size)
        ids = torch.tensor(self.block_table[:nb], device=self.device, dtype=torch.long)
        k = self.k[layer].index_select(0, ids).permute(1, 0, 2, 3).reshape(1, H, nb * self.block_size, D)
        v = self.v[layer].index_select(0, ids).permute(1, 0, 2, 3).reshape(1, H, nb * self.block_size, D)
        return k[:, :, :end], v[:, :, :end]

    def advance(self, n=1):
        self._len += n

    @property
    def seq_len(self):
        return self._len

# %%
# 验收：130 个 token = 9 个 block；写入/取回逐元素一致；池字节数符合公式
BS_DEMO, N_DEMO = 16, 64
bm_demo = BlockManager(N_DEMO, BS_DEMO)
k_demo = [torch.zeros(N_DEMO, H, BS_DEMO, D, device=device) for _ in range(NL)]
v_demo = [torch.zeros(N_DEMO, H, BS_DEMO, D, device=device) for _ in range(NL)]
cache_demo = PagedKVCache(cfg, bm_demo, k_demo, v_demo, device)
k_in = torch.randn(1, H, 130, D, device=device)
v_in = torch.randn(1, H, 130, D, device=device)
with torch.no_grad():
    cache_demo.append(0, k_in, v_in)
cache_demo.advance(130)
k_out, v_out = cache_demo.get(0)
assert len(cache_demo.block_table) == 9 and bm_demo.num_allocated == 9
assert torch.equal(k_out, k_in) and torch.equal(v_out, v_in)
pool_bytes_demo = sum(t.nbytes for t in k_demo + v_demo)
assert pool_bytes_demo == kv_cache_bytes(NL, H, D, BS_DEMO, N_DEMO, 4)
print(f"PagedKVCache ✓：130 token → {len(cache_demo.block_table)} 个 block，取回逐元素一致")

# %% [markdown]
# **验收**：130 个 token 用 block_size=16 正好占 9 个 block（8 满 + 1 半）；写入的 K/V 取回后
# 逐元素相等；池字节数 == `2 × L × H × D × num_blocks × block_size × dtype`。

# %% [markdown]
# ### Step 3 · 接进 attention 与模型前向
#
# **目标**：让分页缓存对上层完全透明 —— 换掉物理布局，不改 attention。
#
# **做法**：分页缓存已经把 K/V 拼成逻辑连续张量，所以 attention 本身一行都不用改（含 mask 语义）；
# `forward_with_cache` 只负责把每层 attention 换成缓存版。真实 vLLM 的 kernel 不做拼接，而是
# 直接在 block 上算（见 03.5 节）。

# %%
def attention_with_cache(attn, x, cache, layer, start_pos):
    B, T, C = x.shape
    q, k, v = attn.qkv(x).split(C, dim=-1)
    q = q.view(B, T, H, D).transpose(1, 2)
    k = k.view(B, T, H, D).transpose(1, 2)
    v = v.view(B, T, H, D).transpose(1, 2)
    cos, sin = rope_cos_sin(torch.arange(start_pos, start_pos + T, device=x.device), D, attn.cfg.rope_theta, x.dtype)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    prefix = cache.seq_len
    cache.append(layer, k, v)
    k_all, v_all = cache.get(layer, upto=prefix + T)
    mask = torch.zeros((T, k_all.shape[2]), device=x.device)
    mask[:, prefix:] = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
    att = F.softmax(((q @ k_all.transpose(-2, -1)) / math.sqrt(D) + mask).float(), dim=-1).to(q.dtype)
    return attn.proj((att @ v_all).transpose(1, 2).reshape(B, T, C))


def forward_with_cache(model, idx, cache, start_pos=0):
    x = model.tok_emb(idx)
    for layer, block in enumerate(model.blocks):
        x = x + attention_with_cache(block.attn, block.norm1(x), cache, layer, start_pos)
        x = x + block.mlp(block.norm2(x))
    return model.lm_head(model.norm_f(x))

# %%
# 验收：分页缓存下整模型前向 logits 形状正确、全部有限
bm_smoke = BlockManager(32, 16)
k_smoke = [torch.zeros(32, H, 16, D, device=device) for _ in range(NL)]
v_smoke = [torch.zeros(32, H, 16, D, device=device) for _ in range(NL)]
paged_smoke = PagedKVCache(cfg, bm_smoke, k_smoke, v_smoke, device)
tokens_smoke = torch.randint(0, cfg.vocab_size, (1, 48), device=device)
with torch.no_grad():
    logits_smoke = forward_with_cache(model, tokens_smoke, paged_smoke)
assert logits_smoke.shape == (1, 48, cfg.vocab_size)
assert torch.isfinite(logits_smoke).all()
print(f"分页前向 ✓：logits 形状 {tuple(logits_smoke.shape)}，全部有限")

# %% [markdown]
# **验收**：分页缓存下整模型前向的 logits 形状为 `(1, 48, vocab)`，全部有限（无 NaN/Inf）。

# %% [markdown]
# ### Step 4 · 验收：分页必须与连续布局逐元素一致
#
# **目标**：用断言钉死「只是地址变了，数学没变」。
#
# **做法**：同一个 prompt 分别走连续布局与分页布局，比较 logits；再断言 block 账本：
# 130 个 token 应占 9 个 block，释放后 `num_allocated` 回到 0，双重释放必须被拦住。

# %%
BS = 16
NUM_BLOCKS = 256
bm = BlockManager(NUM_BLOCKS, BS)
k_blocks = [torch.zeros(NUM_BLOCKS, H, BS, D, device=device) for _ in range(NL)]
v_blocks = [torch.zeros(NUM_BLOCKS, H, BS, D, device=device) for _ in range(NL)]
paged = PagedKVCache(cfg, bm, k_blocks, v_blocks, device)
contig = KVCache(cfg, 1, 4096, device)
prompt = torch.randint(0, cfg.vocab_size, (1, 130), device=device)
with torch.no_grad():
    logits_paged = forward_with_cache(model, prompt, paged)
    logits_contig = forward_with_cache(model, prompt, contig)
    logits_ref = model(prompt)

gap = (logits_paged - logits_contig).abs().max().item()
assert torch.allclose(logits_paged, logits_contig, atol=1e-5, rtol=1e-5), f"分页 vs 连续不一致：{gap}"
assert torch.allclose(logits_paged, logits_ref, atol=1e-5, rtol=1e-5)
assert len(paged.block_table) == 9 and bm.num_allocated == 9
assert bm.num_free_blocks == NUM_BLOCKS - 9
pool_bytes = sum(t.nbytes for t in k_blocks + v_blocks)
assert pool_bytes == kv_cache_bytes(NL, H, D, BS, NUM_BLOCKS, 4)
bm.free(paged.block_table)
assert bm.num_allocated == 0 and len(paged.block_table) == 0
try:
    bm.free([0])
    raise AssertionError("双重释放没有被拦住")
except AssertionError as exc:
    assert "double free" in str(exc)
print(f"分页 vs 连续：logits 最大误差 = {gap:.2e} | block 账本 ✓ | 池 {pool_bytes / 2**20:.1f} MiB")

# %% [markdown]
# **结论**：分页与连续布局 logits 最大误差为 0（阈值 1e-5），block 账本断言全部通过。
# 正确性有了，03.4 用三组实验回答「分页到底省了多少显存、代价有多大」。

# %% [markdown]
# ## 03.4 实测
#
# 三个实验，分别回答三个问题：
#
# | 实验 | 问题 | 产出 |
# | --- | --- | --- |
# | A | 200 个真实长度请求，连续预留 vs 分页各浪费多少显存？ | 碎片率/利用率表 + 容量内请求数 |
# | B | block_size 16/32/64 怎么权衡？ | block table 开销表 |
# | C | 分页的 gather 代价有多大？ | 每次 `get()` 的耗时表 + 曲线 |
#
# 测量协议（实验 C）：M3 Ultra / 96GB / torch 2.15 / MPS，`warmup=3, reps=9` 取中位数；
# 实验 A/B 是账本模拟，不跑模型。请求长度用固定 seed 的对数正态分布生成。
# 绝对值随机器变化，**趋势才是结论**。

# %% [markdown]
# ### 实验 A · 200 个请求的碎片率与容量
#
# **问题**：02 章的连续预留，在真实长度分布下到底浪费多少？分页能省多少？
#
# **协议**：200 个请求，长度服从对数正态分布（均值约 534、中位数 416、最长 1985）；
# 对比「连续预留 max_seq_len=2048」与 block_size=16/32/64 的分页；再用同一个 65536-token 池
# （32K 上下文 × 2 请求的显存量）模拟「能同时容纳多少请求」。注意：这是**账本模拟**，
# 只分配 block id，不分配真实 KV 张量。
#
# **公式**：设请求长度为 `len_i`、分配粒度为 `bs`，则
#
# ```text
# 连续预留：allocated = N × max_seq_len
# 分页：    allocated = Σ ceil(len_i / bs) × bs
# 利用率   = Σ len_i / allocated
# 碎片率   = 1 - 利用率
# ```
#
# 分页的内部碎片每请求最多 `bs-1` 个 token，平均约 `(bs-1)/2`；连续预留每请求最多
# `max_seq_len - len`。这就是两者差一个数量级的原因。

# %%
rng = np.random.default_rng(0)
MAX_SEQ_LEN = 2048
POOL_TOKENS = 65536
lengths = np.clip(rng.lognormal(np.log(400), 0.8, 200), 16, MAX_SEQ_LEN).astype(int)
actual_tokens = int(lengths.sum())
print(f"请求数 = {len(lengths)} | 总 token = {actual_tokens:,} | 均值 = {lengths.mean():.0f} | 最长 = {lengths.max()}")

rows = []
rows.append(
    {
        "方案": "连续预留（max_seq_len=2048）",
        "分配 tokens": len(lengths) * MAX_SEQ_LEN,
        "利用率": actual_tokens / (len(lengths) * MAX_SEQ_LEN),
        "碎片率": 1 - actual_tokens / (len(lengths) * MAX_SEQ_LEN),
        "block table 条目": 0,
        "池内可容纳请求数": POOL_TOKENS // MAX_SEQ_LEN,
    }
)

# %% [markdown]
# 连续预留这一行直接算；分页方案用 `BlockManager` 真实分配一遍：同一个池子谁能装下更多请求。

# %%
for bs in [16, 32, 64]:
    blocks = -(-lengths // bs)
    allocated = int(blocks.sum() * bs)
    bm_sim = BlockManager(POOL_TOKENS // bs, bs)
    n_fit = 0
    for n in lengths:
        table = []
        try:
            bm_sim.append(table, int(n))
        except MemoryError:
            break
        n_fit += 1
    assert n_fit >= 1 and bm_sim.num_allocated <= POOL_TOKENS // bs
    rows.append(
        {
            "方案": f"分页（block_size={bs}）",
            "分配 tokens": allocated,
            "利用率": actual_tokens / allocated,
            "碎片率": 1 - actual_tokens / allocated,
            "block table 条目": int(blocks.sum()),
            "池内可容纳请求数": n_fit,
        }
    )
df_frag = pd.DataFrame(rows)
df_frag["分配 KV @Qwen3-8B（GiB）"] = (df_frag["分配 tokens"] * 147456 / 2**30).round(1)
df_frag

# %% [markdown]
# **读表**：连续预留的利用率只有 26.1%，**74% 的显存是碎片**；分页把碎片率压到 1.3%（bs=16）。
# 换成 Qwen3-8B 的 KV 口径：同样 200 个请求，连续预留要 56.2 GiB，分页只要 14.8 GiB——**3.8 倍**。
# 同一个 65536-token 池，连续预留只能同时放 32 个请求，分页能放 116 个。
#
# 注意 block_size 的权衡已经显现：bs 从 16 → 64，碎片率 1.3% → 5.4%，可容纳请求数 116 → 111，
# 但 block table 条目少了 3.8 倍。下一张表把「block table 开销」单独拉出来看。

# %% [markdown]
# ### 实验 B · block_size 的权衡
#
# **问题**：block 越小碎片越少，但 block table 条目越多；到底谁主导？
#
# **协议**：同一批 200 个请求，只改 block_size = 16/32/64；换算成 Qwen3-8B fp16 的 KV 口径
# （每 token 144 KiB），看 block table 显存占 KV 的比例。
#
# **公式**：`block table 字节数 = Σ ceil(len_i / bs) × 4`（每个条目一个 int32）；
# `table / KV 比例 = table 字节数 / (allocated_tokens × 每 token 字节数)`。

# %%
rows = []
for bs in [16, 32, 64]:
    blocks = int((-(-lengths // bs)).sum())
    kv_bytes = int((-(-lengths // bs)).sum() * bs * 147456)  # Qwen3-8B fp16 每 token
    rows.append(
        {
            "block_size": bs,
            "block table 条目": blocks,
            "block table 显存（KiB）": round(blocks * 4 / 1024, 1),
            "对应 KV（GiB）": round(kv_bytes / 2**30, 1),
            "table / KV 开销": f"{blocks * 4 / kv_bytes:.2e}",
            "平均每请求尾部浪费（token）": round((lengths % bs).mean(), 1),
        }
    )
df_table = pd.DataFrame(rows)
df_table

# %% [markdown]
# **读表**：block table 的开销小到可以忽略（bs=16 时 26.4 KiB vs 14.8 GiB 的 KV，约 1.7e-6），
# 所以**显存不是选 block_size 的决定因素**。真正的权衡是：
#
# - block 越小 → 碎片越少、可容纳请求越多；但 block table 条目越多，调度器每次遍历/更新的
#   Python 开销越大，attention kernel 的 block 迭代次数也越多。
# - block 越大 → 调度开销小，但尾部浪费大（bs=64 时平均每请求浪费 31 个 token）。
#
# 这就是 vLLM 默认 `block_size=16` 的原因：在碎片率和 kernel 效率之间取平衡。

# %% [markdown]
# ### 实验 C · gather 的代价
#
# **问题**：分页把连续切片换成了 gather 拷贝，每次 decode 多付多少？
#
# **协议**：同一模型、同一上下文，分别测「取一次全部层的 K/V」的耗时（9 次中位数）。
# 连续布局的 `get` 是零拷贝切片，分页布局的 `get` 要物化前缀。这比测端到端 decode 更干净：
# 端到端会被每次前向的固定开销淹没（00 章讲过）。

# %%
rows = []
for L in [512, 1024, 2048, 4096, 8192]:
    idx = torch.randint(0, cfg.vocab_size, (1, L), device=device)
    num_blocks = (L + BS - 1) // BS + 8
    bm_run = BlockManager(num_blocks, BS)
    k_run = [torch.zeros(num_blocks, H, BS, D, device=device) for _ in range(NL)]
    v_run = [torch.zeros(num_blocks, H, BS, D, device=device) for _ in range(NL)]
    paged_run = PagedKVCache(cfg, bm_run, k_run, v_run, device)
    contig = KVCache(cfg, 1, L + 8, device, torch.float32)

    with torch.no_grad():
        forward_with_cache(model, idx, paged_run)
        paged_run.advance(L)
        forward_with_cache(model, idx, contig)
        contig.advance(L)

    def gather_all(paged_run=paged_run, L=L):
        for layer in range(NL):
            paged_run.get(layer, upto=L)

    def view_all(contig=contig, L=L):
        for layer in range(NL):
            contig.get(layer, upto=L)

    r_paged = bench(gather_all, warmup=3, reps=9, device=device)
    r_contig = bench(view_all, warmup=3, reps=9, device=device)
    rows.append(
        {
            "上下文 tokens": L,
            "分页 get（ms）": round(r_paged.median_ms, 3),
            "连续 get（ms）": round(r_contig.median_ms, 3),
            "拷贝字节（MiB）": round(2 * NL * L * H * D * 4 / 2**20, 2),
        }
    )
df_gather = pd.DataFrame(rows)
df_gather

# %%
context = df_gather["上下文 tokens"]
paged_get_ms = df_gather["分页 get（ms）"]
contig_get_ms = df_gather["连续 get（ms）"]

fig, ax = plt.subplots()
ax.plot(context, paged_get_ms, "s-", color=COLORS["accent"], label="Paged layout: gather materializes prefix")
ax.plot(context, contig_get_ms, "o-", color=COLORS["primary"], label="Contiguous layout: zero-copy slice")
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("KV get latency (ms, median of 9)")
ax.set_title("Cost of paged KV: one gather per decode step")
ax.legend()
save(fig, "ch03_gather_overhead");

# %% [markdown]
# **读表/读图**：连续布局的 `get` 是零拷贝切片（约 0.005ms，基本不随长度变）；分页布局每次
# 要 gather 0.73–0.87ms（512–8192 token）。注意它在本机上**没有明显随长度增长**——成本被
# `index_select + permute + reshape` 这串通用算子的固定开销主导，而不是拷贝带宽（8192 token
# 的 K/V 是 32 MiB，0.87ms 只相当于约 38 GB/s，远低于 MPS 峰值带宽）。结论：没有 fused kernel
# 时，每生成一个 token 都要固定多付约 0.8ms 的 gather 成本。
#
# 这就是 100 行教学版和 10000 行工业实现的核心差距：**真实 PagedAttention kernel 在 block 上
# 直接做 attention**（block 内连续、block 间用 block table 索引），根本没有这次物化拷贝，
# 分页的显存收益（3.8 倍并发）不需要用延迟来换。
#
# 顺带一提：绝对延迟随机器变化，趋势（分页 O(L) vs 连续 O(1)）才是可迁移的结论。

# %% [markdown]
# ## 03.5 对照工业实现（vLLM）
#
# vLLM 0.28.x（v1 engine）把本章的三件事拆成了独立模块：
#
# | 我们的实现 | vLLM 中的对应物 | 真实文件（0.28.x） |
# | --- | --- | --- |
# | `BlockManager._free` 列表 | `BlockPool` + `FreeKVCacheBlockQueue`（双向链表，O(1) 取/还） | `vllm/v1/core/block_pool.py`、`vllm/v1/core/kv_cache_utils.py` |
# | `BlockManager.append()` | `KVCacheManager.allocate_slots()` / `BlockPool.get_new_blocks()` | `vllm/v1/core/kv_cache_manager.py` |
# | `PagedKVCache.block_table` | `BlockTable`（GPU 上的 int32 张量） | `vllm/v1/worker/block_table.py` |
# | `PagedKVCache.get()` 的 gather | paged attention kernel 直接按 block 读 KV | `vllm/v1/attention/backends/flash_attn.py` |
# | 每请求一个 `PagedKVCache` 对象 | `KVCacheManager` 统一管理所有请求的 block table | `vllm/v1/core/kv_cache_manager.py` |
#
# 关键参数（真实名字）：
#
# | 参数 | 本章的哪个量 | 取值/说明 |
# | --- | --- | --- |
# | `--block-size`（`block_size`） | 实验 A/C 的 block 粒度 | 默认 16；kernel 会按 16/32/64 特化 |
# | `--gpu-memory-utilization` | 物理池总大小 `num_blocks × block_size` | 默认 0.9，决定 `num_gpu_blocks` |
# | `--max-num-seqs`（`max_num_seqs`） | 「池内可容纳请求数」 | 同时调度的请求上限 |
# | `--max-num-batched-tokens` | 一次 prefill/decode 的 token 预算 | 影响 block 分配节奏 |
#
# **差距在哪**：
#
# 1. **数据结构**：我们的空闲池是 Python list + set；vLLM 用双向链表 + 引用计数，取/还 O(1)，
#    且天然支持 04 章的共享 block（ref_count > 1 时不能回收）。
# 2. **计算**：我们的 `get()` 物化整个前缀；vLLM 的 attention backend 接收 block_table，
#    kernel 内按 block 迭代，只有 block 内是连续访存。
# 3. **一致性**：vLLM 把 block 分配、prefix caching、抢占（preemption）统一在
#    `KVCacheManager` 里；我们只做了「够用就好」的教学版。
#
# 另外注意 vLLM 没有「连续布局」这个阶段——v0 时代就已是分页，v1 进一步把 block 管理
# 从 model runner 里拆出来。这也是为什么 02 章的连续 `KVCache` 在 vLLM 里找不到直接对应物，
# 只有 `CacheConfig` 和 `BlockPool`。

# %% [markdown]
# ## 03.6 练习与小结
#
# **练习**
#
# 1. ⭐ 把 block_size 改成 8 / 128，重跑实验 A 和实验 C。碎片率、block table 条目、gather
#    延迟分别怎么变？为什么 vLLM 不选 8 或 128？
# 2. ⭐ 把请求长度分布换成均匀分布 U(16, 2048) 和重尾分布（lognormal σ=1.5），重复碎片实验，
#    解释「碎片率对流量分布有多敏感」。
# 3. ⭐⭐ 给 `PagedKVCache` 加 `free()`（释放 block_table 并清空长度），写测试断言：
#    分配 N 个 block 再全部释放后，`num_free_blocks` 回到初始值；连续两轮请求不会泄漏。
# 4. ⭐⭐ 把 `get()` 的 gather 去掉：让它返回 block 列表，改写 attention 为「逐 block 累加
#    softmax 分子/分母」（提示：flash decoding 的 split-K 思路），对比 logits 与延迟。
# 5. ⭐⭐⭐ 开放设计：分页布局下，如果需要在序列**中间**插入 token（beam search、并行采样、
#    多轮对话拼接），block table 该怎么更新？给出方案与代价分析（评分要点：是否复用了
#    最后一个 block 的空位、是否需要 copy-on-write、是否考虑了 04 章的共享 block）。
#
# **你现在应该能回答：**
#
# - 连续预分配的 74% 碎片具体浪费在哪？分页为什么只浪费最后一个 block 的尾部？
# - block table 存的是什么？为什么它只占 KV 显存的百万分之一，却决定了 kernel 的效率？
# - block_size 越大越好还是越小越好？vLLM 为什么默认 16？
#
# **小结与下一章**
#
# 分页把「按 max_seq_len 预留」变成「按需追加」，显存利用率从 26% 提到 98%。但 200 个请求里
# 如果有大量请求共享同一段 system prompt，每个请求仍会把这段前缀的 K/V 重算一遍、各存一份。
# 04 章在前缀层面做缓存：按 block 内容 hash，命中就共享物理 block，把 TTFT 直接打下来。
#
# **延伸阅读**
#
# - Kwon et al., *Efficient Memory Management for Large Language Model Serving with
#   PagedAttention*（SOSP 2023）：block table 与分页 attention 的原始设计
# - vLLM 源码：`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/block_pool.py`
# - 下一章：04 前缀缓存，把 page 共享从「同一次请求内」推广到「跨请求」
