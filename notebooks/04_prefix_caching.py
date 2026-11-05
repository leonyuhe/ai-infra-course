# %% [markdown]
# # 第 04 章 · 前缀缓存：让共享的 system prompt 只算一次
#
# > 同一个 system prompt 被 10 万个请求共享，凭什么每个请求都要重新 prefill 一遍？
#
# 早高峰，同一个客服机器人收到 10 万个请求：2K token 的 system prompt 一个月没改过，GPU 却
# 把它从零算了 10 万遍——**同一页纸，全公司轮流重念了 10 万次**。更荒唐的是，每一遍算出来
# 的 K/V 都一模一样，只是没人把它留下来。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 90 分钟 |
# | 运行环境 | 🏠 Mac（MPS / CPU）· 无需 GPU · torch 2.15 |
# | 前置章节 | 03（`PagedKVCache` + block table）、02（KV cache 接口） |
# | 本章产物 | 引用计数 block 池、链式 hash 前缀表、命中版 prefill、三组实测（TTFT / CPU 对照 / 命中率） |
# | 本章接口 | 复用 03 章 `PagedKVCache`，增加 `attach / release` 管理共享 block |
#
# **学完你能回答：**
# 1. 为什么前缀缓存必须按「父 block hash + 本块 token 内容」做链式 hash，而不是只 hash 本块？
# 2. 命中一段 P token 的前缀，TTFT 能降多少？为什么实测降幅远小于理论值？
# 3. 引用计数如何保证共享 block 不会被某个请求释放掉？
# 4. 缓存容量、block_size、驱逐策略（LRU）分别如何影响命中率？

# %% [markdown]
# ## 04.0 本章目标与路线
#
# **一句话目标**：在 03 章分页 KV 的基础上，让「前缀完全相同的 block」跨请求复用物理块，
# 命中时跳过对应的 prefill 计算，在不改变输出结果的前提下把共享前缀请求的 TTFT 打下来。
#
# **代价 / 边界**：要维护 hash 表与引用计数；缓存容量有限，驱逐策略决定命中率；共享块只读，
# 写入要走 copy-on-write（本章不展开）；命中收益还取决于 prefill 是否 compute-bound。
#
# **本章交付四样东西**：
#
# | # | 交付物 | 在哪一节 | 怎么验收 |
# | --- | --- | --- | --- |
# | 1 | `BlockPool`：带引用计数的物理块池 | 04.3 Step 1 | ref_count 增减与回收断言 |
# | 2 | `PrefixAwareKVCache`：挂载/释放共享块 | 04.3 Step 3 | 引用 +1/-1，共享内容不被清空 |
# | 3 | `PrefixCache`：链式 hash 前缀表 | 04.3 Step 4 | 命中前 k 块；改一块后只命中更短前缀 |
# | 4 | 命中版 prefill + 三组实验 | 04.3 Step 5–6、04.4 实验 A/B/C | 命中 vs 全量 logits 一致（atol 1e-3）；TTFT/命中率趋势 |
#
# **阅读路线**：04.1 问题 → 04.2 高阶设计 → 04.3 分步实现（6 步，每步有验收）→
# 04.4 实验（TTFT / CPU 对照 / 命中率）→ 04.5 对照 vLLM → 04.6 练习。

# %% [markdown]
# ## 04.1 问题与直觉：每个请求都在重算同一段 system prompt
#
# 真实场景里前缀重复极其普遍：客服机器人共享 2K token 的 system prompt；Agent 共享工具说明与
# few-shot 示例；多轮对话每一轮都复用之前的全部上下文。
#
# **反例**：没有前缀缓存时，N 个共享 P token 前缀的请求，每个都做一次完整 prefill ——
# 浪费 N × P 个 token 的 prefill 计算，而 TTFT 恰好被这段前缀的长度支配（02 章：prefill 是
# compute-bound）。把 N=10 万、P=2048 代进去：**约 2 亿个 token 的重复计算**，全是同一段前缀。
# 前缀缓存放下的第一笔账，就是把这个乘数从 10 万变成 1。
#
# 前缀缓存的做法：以 03 章的 block 为单位，如果两个请求的某个 block 内容相同、且它之前的所有
# block 也相同，这一块的 KV 就一定相同，可以直接复用物理 block：
#
# ![](../assets/figures/ch04_prefix_tree.svg)
#
# 图里请求 A 和请求 B 共享前 3 个 block（system prompt），只有最后一个 block（用户问题）不同。
# 命中 3 个 block = 跳过 3 × block_size = 48 个 token 的 prefill。
#
# 换个说法：prefill 像「把一篇文章读进脑子背下来」，前缀缓存让第一个客人背完，后面拿着同一篇
# 文章的客人**直接共用这份记忆**。按 04.4 实验 A 的口径，P=2048 的 system prompt 命中后只需算
# 剩下的 8 个 token——**理论上快 257 倍**（(2048+8)/8），省掉的正是那 99.6% 的 prefill 计算。

# %% [markdown]
# **关键概念对比与两个子问题**
#
# | 维度 | 无前缀缓存 | 有前缀缓存（命中） |
# | --- | --- | --- |
# | prefill token 数 | P + Q | Q |
# | 理论 TTFT 加速比 | 1 | (P + Q) / Q |
# | KV 显存 | 每请求各存一份前缀 | 前缀 block 共享一份 |
# | 命中条件 | 无 | 连续 block 内容完全相同 |
# | 失效方式 | 无 | 容量不足被驱逐 |
# | 额外开销 | 无 | hash/查表、引用计数、共享块只读约束 |
#
# 两个子问题：
#
# 1. **正确性**：KV 不是 token 的纯函数（依赖前面的全部上下文和 RoPE 位置），所以「复用」的
#    前提是**前缀完全一致**。怎么用 hash 表达这件事、怎么证明命中路径与全量重算等价，
#    是 04.2 与 Step 6 的核心。
# 2. **成本与命中率**：hash/查表几乎免费，但 gather 与固定开销会吃掉部分收益；容量和驱逐策略
#    决定命中率。这些都要用数字回答（实验 A/B/C）。
#
# 先记住一个基准：2048 token 的前缀理论上值 257 倍 TTFT。04.4 会量出它在真机上还剩多少——那个
# 落差本身就是这一章最重要的一课。

# %% [markdown]
# ## 04.2 高阶设计
#
# 动手前先定方案。整章在这三个组件上做文章。
#
# **组件 1：存什么（缓存表 + 引用计数池）**
#
# - `PrefixCache`：一张 `block hash → 物理 block id` 的表；另存 `hash → token 内容` 用于教学版
#   防碰撞。hash 是**链式**的：`h_i = hash(h_{i-1}, tokens_i)`。
# - `BlockPool`：与 03 章 `BlockManager` 相同的空闲池，多一个 `ref_count`。一个 block 可能同时
#   被「缓存表」和「若干请求」引用，只有 `ref_count == 0` 才能回收。
# - 共享粒度是 block（默认 16 token），因此命中长度必须是 block_size 的整数倍。

# %% [markdown]
# **组件 2：怎么用（链式 hash + 命中/写回）**
#
# 为什么必须链式：第 i 个 token 的 K/V 依赖它之前所有 token（attention）和它自己的位置（RoPE）。
# 只 hash 本块 16 个 token，会把「内容相同但前缀不同」的 block 误判为命中，结果直接算错。
#
# ```text
# h_0 = hash(0,   tokens[0:bs])
# h_1 = hash(h_0, tokens[bs:2bs])
# h_i = hash(h_{i-1}, tokens[i*bs:(i+1)*bs])
# ```
#
# 链式 hash 像给档案编页码：**每一页的编号里嵌着上一页的编号**，所以一个块的 hash 就代表了它
# 之前的全部历史。这带来一个必须记住的副作用——
#
# > **⚠️ 坑：改一个字符，后面所有 block 全废**
# > 链式 hash 下 `h_i` 依赖 `h_{i-1}`：第 2 个 block 里改一个 token，从第 2 块起所有 hash 全变。
# > 实验里改第 2 块的一个 token，命中数从 3 块掉到 1 块——**只剩改动点之前的那一块**。
# > 放大到生产更狠：把 user id、时间戳、随机 session 这类「每个请求都不同」的内容放在 prompt
# > 最前面，第一个 block 就 miss，而 `lookup` 遇到 miss 即停 → **整段前缀命中率归零，理论该拿的
# > 257 倍收益变成 1 倍**。修法只有一个：把所有动态内容挪到静态前缀之后。
#
# - 命中：从 block 0 逐块查，遇到第一个 miss 就停（前缀必须连续命中）；命中 n 块 → 跳过
#   n × block_size 个 token 的 prefill，只对剩余 token 做前向。
# - 写回：请求 prefill 时新写满的 block 注册进缓存（`incref`）；请求结束只释放自己的引用。

# %% [markdown]
# **组件 3：谁调度（请求生命周期）**
#
# - lookup：算链式 hash，命中 n 块；
# - attach：把命中 block 挂到请求的 block table 上（`incref`），`seq_len = n*bs`；
# - prefill：只对剩余 token 做前向，新写满的 block `insert` 进缓存（缓存表 `incref`）；
# - release：请求结束，按引用计数归还；缓存持有的那份继续存活，等下一次命中或被驱逐。
#
# 数据流：
#
# ```text
# 请求 tokens
#    ├─ PrefixCache.lookup()       逐块 hash，命中 n 块
#    ├─ KVCache.attach(ids, n*bs)  ref_count + 1，跳过 n*bs 个 token 的 prefill
#    ├─ prefill 剩余 token          写入新 block，insert 进缓存（ref_count + 1）
#    └─ KVCache.release()           请求引用 -1；ref_count == 0 的块回收到池子
# ```

# %% [markdown]
# ### 接口契约
#
# | 组件 | 方法 | 语义 |
# | --- | --- | --- |
# | `BlockPool` | `allocate()` / `incref(id)` / `release(id)` | 分配；引用 +1；引用 -1，归零回池 |
# | `PrefixCache` | `lookup(tokens_np, max_blocks)` | 返回连续命中的物理 block id 列表 |
# | `PrefixCache` | `insert(tokens_np, block_ids)` | 注册新写满的完整 block（已存在则跳过） |
# | `PrefixAwareKVCache` | `attach(block_ids, length)` | 挂载共享 block，长度必须是 block_size 整数倍 |
# | `PrefixAwareKVCache` | `release()` | 归还本请求持有的全部引用 |
# | `PrefixAwareKVCache` | `append / get / advance / seq_len` | 与 02/03 章完全一致 |
#
# 时间线上的引用计数变化（请求 A 写入、请求 B 命中）：
#
# | 时刻 | 共享 block 的 ref_count | 动作 |
# | --- | --- | --- |
# | A prefill 前 | 0 | 申请 block |
# | A prefill 后 | 2 | 请求 A +1，缓存表 +1 |
# | A release | 1 | 只剩缓存表 |
# | B attach | 2 | 请求 B +1 |
# | B release | 1 | 回到只剩缓存表 |
#
# 账本对上了，但「表正确」不等于「请求快」：从引用计数到「一次请求只算 8 个 token」，中间还差一条
# 把 lookup → attach → prefill → release 串起来的流水线。04.3 从最底层的池子开始搭。

# %% [markdown]
# ### 验收标准
#
# | 步骤 | 验收 |
# | --- | --- |
# | Step 1 BlockPool | alloc/incref/release 账本正确；重复 release 被断言拦住 |
# | Step 2 PagedKVCache | 40 token → 3 个 block；append/get 逐元素一致 |
# | Step 3 attach/release | 挂载后 ref_count +1，释放后 -1，共享内容不被清空 |
# | Step 4 PrefixCache | 相同前缀命中 3 块；改中间一个 token 后只命中 1 块 |
# | Step 5 命中 prefill | 命中 token 数 == 前缀长度 |
# | Step 6 等价性 | 命中路径 logits 与全量重算误差 < 1e-3；缓存条目/空闲块符合预期 |

# %% [markdown]
# ## 04.3 从零实现：分六步
#
# 按「池 → 分页 → 共享 → hash → 请求流程 → 验收」六步走。六步的依赖是单向的，每一步都能单独
# 跑断言，坏了立刻知道坏在哪一层：
#
# | Step | 做什么 | 代码 |
# | --- | --- | --- |
# | 1 | 带引用计数的物理块池 | `BlockPool` |
# | 2 | 03 章分页缓存 + 物理池工厂 | `PagedKVCache` / `make_pool` |
# | 3 | 挂载与释放共享 block | `PrefixAwareKVCache` |
# | 4 | 链式 hash 前缀表 | `PrefixCache` |
# | 5 | 一次请求的命中流程 | `prefill` |
# | 6 | 证明命中路径与全量重算等价 | `assert` 断言 |

# %% [markdown]
# ### 准备：模型、设备与复用的 attention
#
# 本章仍用 2.8M 的 TinyGPT。前缀缓存改变的是「K/V 从哪来」，不是「attention 怎么算」，
# 所以缓存版 attention 与 02/03 章完全一致，这里只内联一次、不再重复讲解。`BS = 16` 与 vLLM
# 默认 `block_size` 一致。

# %%
import hashlib
import math
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path.cwd().parent))

from aiinfra.bench import bench, pick_device
from aiinfra.plotting import COLORS, save, setup_style
from aiinfra.tinygpt import GPTConfig, TinyGPT, apply_rope, rope_cos_sin

setup_style()
device = pick_device()
torch.manual_seed(0)
np.random.seed(0)

cfg = GPTConfig(max_seq_len=8192)
model = TinyGPT(cfg).to(device).eval()
H, D, NL = cfg.n_head, cfg.head_dim, cfg.n_layer
BS = 16  # block_size，与 vLLM 默认值一致
print(f"device = {device} | params = {model.num_params() / 1e6:.2f}M | block_size = {BS}")

# %% [markdown]
# ### 复用 02 章的缓存版 attention
#
# 与 02/03 章逐行相同：写入新 K/V → 取回「前缀 + 当前块」→ 前缀全可见、块内三角 mask。

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

# %% [markdown]
# ### Step 1 · BlockPool：给每个 block 加引用计数
#
# **目标**：在 03 章 `BlockManager` 的空闲池上，增加「一个 block 能被多方引用」的能力。
#
# **做法**：分配时 `ref_count = 1`；缓存表与请求各自 `incref`；归还时 `release`，减到 0 才回到
# 空闲池。共享能成立的唯一机制就是它——只要还有人引用，block 就不能被回收、内容也不能被改写。

# %%
class BlockPool:
    """带引用计数的物理 block 池：ref_count > 0 的 block 不会被回收。"""

    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))[::-1]
        self.ref_count = [0] * num_blocks
        self.peak_used = 0

    def allocate(self):
        if not self.free_blocks:
            raise MemoryError("KV block pool exhausted")
        block_id = self.free_blocks.pop()
        self.ref_count[block_id] = 1
        self.peak_used = max(self.peak_used, self.num_blocks - len(self.free_blocks))
        return block_id

    def incref(self, block_id):
        self.ref_count[block_id] += 1

    def release(self, block_id):
        self.ref_count[block_id] -= 1
        assert self.ref_count[block_id] >= 0, "release 次数多于 incref"
        if self.ref_count[block_id] == 0:
            self.free_blocks.append(block_id)

    @property
    def num_free(self):
        return len(self.free_blocks)

# %%
# 验收：alloc/incref/release 账本正确，重复 release 被断言拦住
pool_demo = BlockPool(8, 16)
block = pool_demo.allocate()
assert pool_demo.num_free == 7 and pool_demo.ref_count[block] == 1
pool_demo.incref(block)
assert pool_demo.ref_count[block] == 2
pool_demo.release(block)
assert pool_demo.num_free == 7 and pool_demo.ref_count[block] == 1
pool_demo.release(block)
assert pool_demo.num_free == 8 and pool_demo.ref_count[block] == 0
try:
    pool_demo.release(block)
    raise AssertionError("重复 release 没有被拦住")
except AssertionError as exc:
    assert "incref" in str(exc)
print("BlockPool ✓：allocate/incref/release 账本正确，重复 release 被断言拦截")

# %% [markdown]
# **验收**：分配后 ref=1；`incref` 后 ref=2；两次 `release` 后回到空闲池；第三次 `release`
# 被断言拦住 —— 引用计数不会减成负数。

# %% [markdown]
# ### Step 2 · PagedKVCache：03 章的实现原样保留
#
# **目标**：物理存储与 block table 的取用方式不变，只是池子换成带引用计数的版本。
#
# **做法**：每层 `(num_blocks, H, bs, D)` 的物理池 + 每请求一张 block table；`append` 按段写、
# `get` 用 `index_select` 拼成逻辑连续张量。`make_pool` 负责一次性申请物理池（所有请求共享）。

# %%
class PagedKVCache:
    """03 章的分页缓存：物理 block 池 + 每请求 block table。"""

    def __init__(self, cfg, pool, k_blocks, v_blocks, device="cpu"):
        self.cfg, self.k, self.v = cfg, k_blocks, v_blocks
        self.pool, self.block_size, self.device = pool, pool.block_size, device
        self.block_table, self.owned, self._len = [], [], 0

    def append(self, layer, k, v):
        start, end = self._len, self._len + k.shape[2]
        while len(self.block_table) * self.block_size < end:
            block_id = self.pool.allocate()
            self.block_table.append(block_id)
            self.owned.append(block_id)
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
def make_pool(num_blocks=1024, dev="cpu"):
    pool = BlockPool(num_blocks, BS)
    k_blocks = [torch.zeros(num_blocks, H, BS, D, device=dev) for _ in range(NL)]
    v_blocks = [torch.zeros(num_blocks, H, BS, D, device=dev) for _ in range(NL)]
    return pool, k_blocks, v_blocks

# %%
# 验收：40 token → 3 个 block；append/get 逐元素一致
pool_demo, k_demo, v_demo = make_pool(64, device)
cache_demo = PagedKVCache(cfg, pool_demo, k_demo, v_demo, device)
k_in = torch.randn(1, H, 40, D, device=device)
v_in = torch.randn(1, H, 40, D, device=device)
with torch.no_grad():
    cache_demo.append(0, k_in, v_in)
cache_demo.advance(40)
k_out, v_out = cache_demo.get(0)
assert len(cache_demo.block_table) == 3
assert torch.equal(k_out, k_in) and torch.equal(v_out, v_in)
for block_id in cache_demo.block_table:  # 归还本请求持有的块
    pool_demo.release(block_id)
assert pool_demo.num_free == 64
print("PagedKVCache ✓：40 token → 3 个 block，append/get 逐元素一致")

# %% [markdown]
# **验收**：40 token 用 block_size=16 正好 3 个 block；写入的 K/V 取回后逐元素相等；
# 归还后池子回到初始状态。

# %% [markdown]
# ### Step 3 · PrefixAwareKVCache：挂载与释放共享块
#
# **目标**：在 03 章 `PagedKVCache` 上加两个动作，让一个请求「借用」别的请求（或缓存表）已经
# 算好的前缀 block。
#
# **做法**：`attach(block_ids, length)` 把共享 block 挂到本请求的 block table 上，引用 +1，
# 并把逻辑长度直接设成命中长度——**没有任何计算**；`release()` 归还本请求持有的全部引用。
# 共享 block 因为缓存表也持有一份引用，不会被清空。

# %%
class PrefixAwareKVCache(PagedKVCache):
    """在 03 章基础上加两个动作：挂载共享 block、按引用计数释放。"""

    def attach(self, block_ids, length):
        assert length == len(block_ids) * self.block_size, "命中长度必须是 block_size 的整数倍"
        for block_id in block_ids:
            self.pool.incref(block_id)
        self.block_table = list(block_ids)
        self.owned.extend(block_ids)
        self._len = length

    def release(self):
        for block_id in self.owned:
            self.pool.release(block_id)
        self.owned, self.block_table, self._len = [], [], 0

# %%
# 验收：attach 后引用 +1、release 后 -1，共享内容不被清空
pool_demo, k_demo, v_demo = make_pool(64, device)
shared = PrefixAwareKVCache(cfg, pool_demo, k_demo, v_demo, device)
tokens = torch.randint(0, cfg.vocab_size, (1, 32), device=device)
with torch.no_grad():
    forward_with_cache(model, tokens, shared)
shared.advance(32)
blocks = list(shared.block_table)
assert len(blocks) == 2 and pool_demo.num_free == 62
for block_id in blocks:
    pool_demo.incref(block_id)  # 模拟「缓存表也持有」
shared.release()
assert pool_demo.ref_count[blocks[0]] == 1
other = PrefixAwareKVCache(cfg, pool_demo, k_demo, v_demo, device)
other.attach(blocks, 32)
assert other.seq_len == 32 and pool_demo.ref_count[blocks[0]] == 2
with torch.no_grad():
    forward_with_cache(model, tokens[:, :8], other, start_pos=32)
other.release()
assert pool_demo.ref_count[blocks[0]] == 1
for block_id in blocks:  # 收尾：缓存表的引用也释放
    pool_demo.release(block_id)
assert pool_demo.num_free == 64
print("attach/release ✓：共享块引用 1→2→1，内容未被清空")

# %% [markdown]
# **验收**：挂载命中块后引用计数 +1、逻辑长度直接等于命中长度；请求释放后引用回到缓存表持有的
# 那一份；池子最终复位。

# %% [markdown]
# ### Step 4 · PrefixCache：链式 hash 前缀表
#
# **目标**：把「哪些前缀的 KV 已经算过」变成一张可 O(1) 查的表。
#
# **做法**：对每个**完整** block 计算 `h_i = hash(h_{i-1}, tokens_i)`，存 `hash → 物理 block id`；
# `lookup` 从 block 0 逐块查、遇到 miss 停；`insert` 注册新写满的 block（已存在则跳过，只把引用
# +1）。教学版额外存 token 内容做碰撞二次校验；工业实现靠 64-bit hash 的概率保证。

# %%
class PrefixCache:
    """block 级前缀缓存：链式 hash 保证「完整前缀相同」才命中。"""

    def __init__(self, pool):
        self.pool, self.table, self.contents = pool, {}, {}
        self.hit_blocks, self.lookup_blocks = 0, 0

    @staticmethod
    def _hash(parent, block_tokens):
        digest = hashlib.blake2b(digest_size=8)
        digest.update(parent.to_bytes(8, "big"))
        digest.update(block_tokens.astype(np.int32).tobytes())
        return int.from_bytes(digest.digest(), "big")

    def lookup(self, tokens_np, max_blocks):
        """从 block 0 开始逐块查，遇到第一个 miss 就停。"""
        hit, parent = [], 0
        for i in range(max_blocks):
            block = tokens_np[i * BS : (i + 1) * BS]
            h = self._hash(parent, block)
            self.lookup_blocks += 1
            block_id = self.table.get(h)
            if block_id is None:
                break
            assert np.array_equal(self.contents[h], block), "hash 碰撞！用 token 内容兜底"
            hit.append(block_id)
            parent = h
        self.hit_blocks += len(hit)
        return hit

    def insert(self, tokens_np, block_ids):
        """注册请求新写入的完整 block；已存在的直接跳过。"""
        parent = 0
        for i, block_id in enumerate(block_ids):
            block = tokens_np[i * BS : (i + 1) * BS]
            h = self._hash(parent, block)
            parent = h
            if h in self.table:
                continue
            self.table[h] = block_id
            self.contents[h] = block.copy()
            self.pool.incref(block_id)

# %%
# 验收：相同前缀命中 3 块；改中间一个 token 后只命中前 1 块
pool_demo, k_demo, v_demo = make_pool(64, device)
prefix_demo = PrefixCache(pool_demo)
system = torch.randint(0, cfg.vocab_size, (1, 48), device=device)
cache_demo = PrefixAwareKVCache(cfg, pool_demo, k_demo, v_demo, device)
with torch.no_grad():
    forward_with_cache(model, system, cache_demo)
cache_demo.advance(48)
tokens_np = system[0].cpu().numpy()
prefix_demo.insert(tokens_np, cache_demo.block_table[:3])
assert len(prefix_demo.table) == 3
assert len(prefix_demo.lookup(tokens_np, 3)) == 3
changed = tokens_np.copy()
changed[20] += 1  # 改第 2 个 block 里的一个 token
assert len(prefix_demo.lookup(changed, 3)) == 1
cache_demo.release()
for block_id in list(prefix_demo.table.values()):
    pool_demo.release(block_id)
assert pool_demo.num_free == 64
print("PrefixCache ✓：命中 3 块；改中间 token 后只命中 1 块；引用归还后池子复位")

# %% [markdown]
# **验收**：相同前缀命中 3 块；只改第 2 个 block 里的一个 token，命中数立刻降到 1 块 ——
# 链式 hash 把「前缀必须完全一致」编码进了 key 里。

# %% [markdown]
# ### Step 5 · 一次请求：查缓存 → 挂载 → 只算剩下的 → 注册新块
#
# **目标**：把前四步串成一条请求流水线。
#
# **做法**：`prefix_cache=None` 表示关闭前缀缓存（等价于 vLLM 关闭 `--enable-prefix-caching`），
# 走全量 prefill 作为基线；否则先 lookup、attach 命中块，只对剩余 token 做前向，新写满的 block
# insert 回缓存，最后 release 请求引用。`make_pool` 已在 Step 2 定义。
#
# 画成时间轴，一次命中请求长这样（按 P=2048、Q=8、block_size=16）：
#
# ```text
# t=0ms     ┃ 请求到达：tokens = 2048（system）+ 8（问题）
# t≈0.1ms   ┃ PrefixCache.lookup()：算 128 个链式 hash，全部命中
# t≈0.1ms   ┃ attach()：block table 直接指向这 128 个物理块，seq_len = 2048
# t≈0.1ms   ┃ prefill：只把最后 8 个 token 送进前向，start_pos = 2048
# t≈3ms     ┃ 第一个 token 出来——未命中时要 ~11ms（04.4 实验 A 量到）
# ```

# %%
@torch.no_grad()
def prefill(model, tokens, tokens_np, pool, k_blocks, v_blocks, prefix_cache=None):
    """返回 logits 与命中 token 数；prefix_cache=None 表示不做前缀缓存。"""
    full_blocks = tokens.shape[1] // BS
    hit = prefix_cache.lookup(tokens_np, full_blocks) if prefix_cache is not None else []
    n_hit = len(hit) * BS
    cache = PrefixAwareKVCache(model.cfg, pool, k_blocks, v_blocks, device=tokens.device)
    if n_hit:
        cache.attach(hit, n_hit)
    logits = forward_with_cache(model, tokens[:, n_hit:], cache, start_pos=n_hit)
    if prefix_cache is not None:
        prefix_cache.insert(tokens_np, cache.block_table[:full_blocks])
    cache.release()
    return logits, n_hit

# %% [markdown]
# **结论**：命中时 prefill 的 token 数从 `P+Q` 降到 `Q`，其余流程与普通分页请求完全相同。
# 下一步用断言证明「跳过的计算确实没有改变结果」。

# %% [markdown]
# ### Step 6 · 验收：命中路径必须与全量重算逐元素一致
#
# **目标**：这是本章最重要的测试——命中时跳过了 512 个 token 的计算，结果必须一点都不差。
#
# **做法**：同一个请求跑三遍——关缓存、首次写入缓存、再次命中；断言首次不命中、再次命中
# 512 个 token；命中路径的 logits 与全量重算一致（`atol=1e-3`）；缓存条目数 = 32（512/16），
# 空闲池数量正确（引用计数没有泄漏）。
#
# **先猜一下**：跳过 512 个 token 的计算、只算 8 个，logits 和全量重算比会差多少？
# (a) 恰好 0　(b) 1e-4 量级　(c) 1e-2 量级。提示：两次前向的矩阵乘形状不同（一次 520 行、
# 一次 8 行）。

# %%
pool, k_blocks, v_blocks = make_pool(1024, device)
prefix_cache = PrefixCache(pool)
P, Q = 512, 8
tokens = torch.cat(
    [torch.randint(0, cfg.vocab_size, (1, P), device=device), torch.randint(0, cfg.vocab_size, (1, Q), device=device)],
    dim=1,
)
tokens_np = tokens[0].cpu().numpy()
with torch.no_grad():
    ref = model(tokens)
logits_cold, hit_cold = prefill(model, tokens, tokens_np, pool, k_blocks, v_blocks)
logits_warm, hit_warm = prefill(model, tokens, tokens_np, pool, k_blocks, v_blocks, prefix_cache)
logits_hit, n_hit = prefill(model, tokens, tokens_np, pool, k_blocks, v_blocks, prefix_cache)
assert hit_cold == 0 and hit_warm == 0, "缓存未热身时不应命中"
assert n_hit == P, f"应命中 {P} 个 token，实际 {n_hit}"
gap = (logits_hit - ref[:, n_hit:]).abs().max().item()
assert torch.allclose(logits_hit, ref[:, n_hit:], atol=1e-3, rtol=1e-3), f"命中路径 logits 不一致：{gap}"
assert len(prefix_cache.table) == P // BS, "缓存条目数应等于完整 system block 数"
assert pool.num_free == 1024 - P // BS, "引用计数泄漏：空闲 block 数不对"
print(f"命中 {n_hit} tokens ✓ | logits 最大误差 = {gap:.2e} | 缓存条目 = {len(prefix_cache.table)} | 空闲 block = {pool.num_free}")

# %% [markdown]
# **结论**：答案是 (b)——最大误差在 1e-4 量级（阈值 1e-3）。不是 0 也不该是 0：同一组 K/V，但一次
# 矩阵乘 520 行、一次乘 8 行，浮点累加顺序不同，差个 1e-4 很正常。缓存 32 个条目、空闲 block 992
# 个——引用计数与命中逻辑都正确。

# %% [markdown]
# ## 04.4 实测
#
# | 实验 | 问题 | 产出 |
# | --- | --- | --- |
# | A | 512/1024/2048 token 共享前缀，TTFT 命中 vs 未命中差多少？ | TTFT 表 + 柱状图 |
# | B | 为什么 MPS 上收益远小于理论值？ | CPU 对照表 |
# | C | 容量与驱逐策略如何影响命中率？ | 命中率表 + 容量曲线 |
#
# 测量协议（实验 A/B）：M3 Ultra / 96GB / torch 2.15 / MPS，`warmup=3, reps=9` 取中位数；
# system prompt 512/1024/2048 token，用户问题固定 8 token（不足一个 block，因此只有 system
# prompt 的整块会被缓存）。实验 C 是纯账本模拟，不跑模型。绝对值随机器变化，**趋势才是结论**。

# %% [markdown]
# ### 实验 A · 共享 system prompt 的 TTFT：省下的计算去哪了
#
# **问题**：命中一段 P token 的前缀，TTFT 能降多少？
#
# **协议**：每个请求执行完整流程（查表 → 分配 → 前向 → 注册 → 释放）；「未命中」= 关闭前缀缓存，
# 走全量 prefill。P = 512 / 1024 / 2048，Q = 8。
#
# **公式**：命中时 prefill 的 token 数从 P+Q 降到 Q，所以
#
# ```text
# 理论 TTFT 加速比 = (P + Q) / Q
# P=512  → 65x
# P=1024 → 129x
# P=2048 → 257x
# ```
#
# **先猜一下**：按上面的上界，P=2048 时实测 TTFT 能快多少？(a) 约 257 倍——99.6% 的计算量都省了，
# 端到端该同比例下降　(b) 约 60 倍——固定开销吃掉一大截　(c) 只有 2–4 倍。
#
# 这是「纯计算」的上界；实测还要加上固定开销、gather 与 hash 的成本，实验 B 用 CPU 对照解释差距。

# %%
def measure_ttft(dev, prefix_lens):
    mdl = TinyGPT(cfg).to(dev).eval()
    rows = []
    for P in prefix_lens:
        pool, k_blocks, v_blocks = make_pool(1024, dev)
        prefix_cache = PrefixCache(pool)
        system = torch.randint(0, cfg.vocab_size, (1, P), device=dev)
        question = torch.randint(0, cfg.vocab_size, (1, 8), device=dev)
        tokens = torch.cat([system, question], dim=1)
        tokens_np = tokens[0].cpu().numpy()
        prefill(mdl, tokens, tokens_np, pool, k_blocks, v_blocks, prefix_cache)  # 热身：写入缓存
        r_miss = bench(lambda: prefill(mdl, tokens, tokens_np, pool, k_blocks, v_blocks, None), warmup=3, reps=9, device=dev)
        r_hit = bench(lambda: prefill(mdl, tokens, tokens_np, pool, k_blocks, v_blocks, prefix_cache), warmup=3, reps=9, device=dev)
        rows.append(
            {
                "设备": str(dev),
                "system prompt": P,
                "TTFT 未命中（ms）": round(r_miss.median_ms, 2),
                "TTFT 命中（ms）": round(r_hit.median_ms, 2),
                "实测加速比": round(r_miss.median_ms / r_hit.median_ms, 2),
                "理论加速比": (P + 8) // 8,
                "命中 tokens": P,
            }
        )
    return rows


df_ttft = pd.DataFrame(measure_ttft(device, [512, 1024, 2048]))
df_ttft

# %%
prefix_lens = df_ttft["system prompt"].to_numpy()
miss_ms = df_ttft["TTFT 未命中（ms）"].to_numpy()
hit_ms = df_ttft["TTFT 命中（ms）"].to_numpy()
speeds = df_ttft["实测加速比"].to_numpy()

fig, ax = plt.subplots()
x = np.arange(len(df_ttft))
width = 0.36
ax.bar(x - width / 2, miss_ms, width, color=COLORS["danger"], label="No prefix cache: full prefill")
ax.bar(x + width / 2, hit_ms, width, color=COLORS["success"], label="Prefix cache hit: suffix only")
for i in range(len(df_ttft)):
    ax.text(i, max(miss_ms[i], hit_ms[i]) + 0.4, f"{speeds[i]:.2f}x", ha="center", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([f"{P} tokens" for P in prefix_lens])
ax.set_xlabel("Shared system prompt length")
ax.set_ylabel("TTFT (ms, median of 9)")
ax.set_title("Prefix caching cuts TTFT on a shared system prompt")
ax.legend()
save(fig, "ch04_ttft");

# %% [markdown]
# **读表/读图**：答案是 (c)。(a) 之所以迷人，是因为它把 TTFT 当成了 FLOPs——99.6% 的 prefill
# **计算**确实被跳过了，但端到端还压着固定开销。system prompt 从 512 涨到 2048，未命中 TTFT
# 从约 5ms 涨到约 11ms，命中始终在 2.5–3.5ms，实测加速比约 2–4x（前缀越长收益越大；MPS 上
# 多次运行在 ±0.5 内波动）。
# 注意**实测远小于理论值**（65x / 129x / 257x），原因有三层，值得逐个说清：
#
# 1. **固定开销**：2.8M 小模型在 MPS 上每次 forward 有约 1.3–1.5ms 的 Python + kernel 启动
#    开销，命中路径再短也省不掉它（00 章的「测量污染源」）。
# 2. **gather 成本**：命中路径仍要把 512–2048 token 的 K/V gather 成连续张量（03 章实测约
#    0.8ms），这抵消了一部分收益。
# 3. **hash 与分配**：每个请求要算 32–128 个 block hash、做表查询和 block 分配（约 0.1ms 级）。
#
# 换句话说，TTFT 的**计算部分**确实降了 98% 以上，但端到端被固定开销封底。真实 8B 模型
# prefill 是 compute-bound，2048 token 前缀命中的 TTFT 降幅通常是 5–20 倍。
#
# 再做一个对照：同样的实验在 CPU 上跑（CPU 的前向是 compute-bound，没有 MPS 的 kernel 启动
# 开销），观察 2048 token 前缀的命中收益：

# %% [markdown]
# ### 实验 B · 为什么 MPS 上的收益被压缩
#
# **问题**：理论 65–257x，实测只有 2–4x，差在哪？
#
# **协议**：把同一个 `measure_ttft` 放到 CPU 上跑 P=2048。CPU 的前向是 compute-bound，
# 没有 MPS 的 kernel 启动地板，更接近真实 GPU 上 prefill 的处境。

# %%
df_cpu = pd.DataFrame(measure_ttft(torch.device("cpu"), [2048]))
df_cpu

# %% [markdown]
# CPU 上同一实验的实测加速比约 11x——更接近「跳过 99.6% prefill token」的理论预期。
# 这组对照说明：**前缀缓存的收益取决于 prefill 有多 compute-bound**，而不是取决于缓存的实现
# 有多聪明。在 GPU 上它会是 5–20x 的量级；在小模型的 MPS 上会被固定开销吃掉大半。
#
# 反直觉的地方在这里：换到快得多的 GPU 上，前缀缓存也不会线性地多给你收益——它只能拿掉
# prefill 里「算」的那部分，「必须启动 kernel、必须 gather、必须查表」的部分原样留着。

# %% [markdown]
# ### 实验 C · 缓存容量与驱逐策略：命中率的悬崖
#
# **问题**：缓存容量有限，满了要驱逐；容量与驱逐策略如何影响命中率？
#
# **协议**：工作集 = 4 个 system prompt（256/512/1024/2048 token）合计 16+32+64+128 = 240 个
# block。200 个请求按 10%/30%/30%/30% 抽取 prompt，纯账本模拟「block 命中率」与「整段前缀
# 命中率」；容量从 64 到 512 blocks，对比 LRU 与 FIFO。
#
# **先猜一下**：工作集是 240 个 block，缓存容量从 128 加到 256（刚好装下），整段前缀命中率会涨
# 多少？(a) 慢慢爬十几个点　(b) 从 20%–40% 一步跳到 98%　(c) 不变——命中率主要看 hash 算法够不够好。

# %%
rng = np.random.default_rng(0)
PROMPT_LENS = [256, 512, 1024, 2048]
PROBS = [0.1, 0.3, 0.3, 0.3]
sys_prompts = {P: rng.integers(0, 100000, size=P) for P in PROMPT_LENS}


def chain(P):
    """一个 system prompt 的链式 block hash 序列。"""
    tokens = sys_prompts[P]
    parent, out = 0, []
    for i in range(P // BS):
        h = hashlib.blake2b(digest_size=8)
        h.update(parent.to_bytes(8, "big"))
        h.update(tokens[i * BS : (i + 1) * BS].astype(np.int32).tobytes())
        parent = int.from_bytes(h.digest(), "big")
        out.append(parent)
    return out


chains = {P: chain(P) for P in PROMPT_LENS}
requests = [int(rng.choice(PROMPT_LENS, p=PROBS)) for _ in range(200)]
hotspot = [512 if i % 3 else (1024 if i % 2 else 2048) for i in range(200)]
print(f"工作集 = {sum(P // BS for P in PROMPT_LENS)} blocks | 请求序列长度 = {len(requests)}")

# %%
def simulate(requests, capacity, policy="lru"):
    """返回 (block 命中率, 整段前缀命中率)。policy ∈ {'lru', 'fifo'}。"""
    order, cached = OrderedDict(), set()
    hit_blocks = total_blocks = full_hits = 0
    for P in requests:
        ch = chains[P]
        n_hit = 0
        for h in ch:
            if h not in cached:
                break
            n_hit += 1
        hit_blocks += n_hit
        total_blocks += len(ch)
        full_hits += n_hit == len(ch)
        if policy == "lru":
            for h in ch[:n_hit]:
                order.move_to_end(h)
        for h in ch[n_hit:]:
            if h not in cached:
                cached.add(h)
                order[h] = None
        while len(order) > capacity:
            oldest = next(iter(order))
            del order[oldest]
            cached.discard(oldest)
    return hit_blocks / total_blocks, full_hits / len(requests)


rows = []
for capacity in [64, 128, 192, 256, 512]:
    block_rate, full_rate = simulate(requests, capacity)
    rows.append({"缓存容量（blocks）": capacity, "缓存容量（tokens）": capacity * BS,
                 "block 命中率": round(block_rate, 3), "整段前缀命中率": round(full_rate, 3)})
df_lru = pd.DataFrame(rows)
df_lru

# %%
rows = []
for policy in ["lru", "fifo"]:
    block_rate, full_rate = simulate(hotspot, 128, policy)
    rows.append({"驱逐策略": policy.upper(), "block 命中率": round(block_rate, 3),
                 "整段前缀命中率": round(full_rate, 3), "容量（blocks）": 128})
df_policy = pd.DataFrame(rows)
df_policy

# %%
capacity = df_lru["缓存容量（blocks）"].to_numpy()
block_rate = df_lru["block 命中率"].to_numpy()
full_rate = df_lru["整段前缀命中率"].to_numpy()

fig, ax = plt.subplots()
ax.plot(capacity, block_rate, "o-", color=COLORS["primary"], label="Block hit rate")
ax.plot(capacity, full_rate, "s--", color=COLORS["secondary"], label="Full-prefix hit rate")
ax.axvline(sum(P // BS for P in PROMPT_LENS), color=COLORS["accent"], ls=":", lw=2, label="Working set (240 blocks)")
ax.set_xscale("log", base=2)
ax.set_xlabel("Prefix cache capacity (blocks)")
ax.set_ylabel("Hit rate")
ax.set_title("Hit rate vs cache capacity: the working set is the cliff")
ax.legend()
save(fig, "ch04_hit_rate");

# %% [markdown]
# **读表/读图**：答案是 (b)——容量是命中率的**悬崖**而不是斜坡。(c) 错在把命中率当成了 hash
# 质量问题：容量不足时，大 prompt 的 block 链被拦腰挤掉一段，链式命中让它后面全部作废。
#
# - 容量 < 工作集的 1/2（≤128 blocks）时，整段前缀命中率只有 20%–40%：大 prompt 的 block 链
#   一旦被挤掉一段，后续 block 全部失效（链式命中的脆弱性）。
# - 容量刚过工作集（256 blocks）时命中率跳到 98%，再加容量几乎无收益。
# - 在存在「热点前缀」的序列上，LRU 明显优于 FIFO：128 blocks 容量下整段命中率 66% vs 50%。
#   因为 FIFO 会把刚被反复使用的热点前缀按插入顺序淘汰，而 LRU 会保留它。
#
# 工程含义：**前缀缓存的命中率主要取决于「容量 vs 工作集」和「请求的局部性」，而不是 hash 算法**。
# vLLM 的默认前缀缓存用引用计数 + 空闲队列（近似 LRU）实现驱逐，正是这个道理；生产上调
# `gpu_memory_utilization` 与上下文长度，本质上就是在调这个「容量 vs 工作集」的比值。

# %% [markdown]
# ## 04.5 对照工业实现（vLLM）
#
# vLLM 0.28.x（v1 engine）的前缀缓存就在 `KVCacheManager` 里，与 block 分配共用同一套数据结构：
#
# | 我们的实现 | vLLM 中的对应物 | 真实文件（0.28.x） |
# | --- | --- | --- |
# | `PrefixCache.lookup` | `KVCacheManager.get_computed_blocks()`（返回命中 block 与命中长度） | `vllm/v1/core/kv_cache_manager.py` |
# | `PrefixCache.insert` | `KVCacheManager.cache_blocks()` / `BlockPool.cache_full_blocks()` | `vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/block_pool.py` |
# | `_hash(parent, tokens)` | `hash_block_tokens()` + `BlockHash`（64-bit，含 `is_first_block`） | `vllm/v1/core/kv_cache_utils.py` |
# | `BlockPool.ref_count` | `KVCacheBlock.ref_cnt` + `FreeKVCacheBlockQueue` | `vllm/v1/core/block_pool.py` |
# | 每请求 `block_table` | `Request.block_hashes` + `BlockTable` | `vllm/v1/request.py`、`vllm/v1/worker/block_table.py` |
# | 我们的 `simulate()` 驱逐 | `FreeKVCacheBlockQueue`（双向链表，近似 LRU） | `vllm/v1/core/kv_cache_utils.py` |
#
# 关键参数（真实名字）：
#
# | 参数 | 本章对应物 | 说明 |
# | --- | --- | --- |
# | `--enable-prefix-caching` | `prefix_cache=None` 的开关 | v1 engine 默认开启；老版本默认关闭 |
# | `--block-size`（`block_size`） | `BS = 16` | 命中粒度；默认 16 |
# | `--prefix-caching-hash-algo` | `PrefixCache._hash` | 可选 `builtin` / `sha256` 等；影响碰撞概率与成本 |
# | `--gpu-memory-utilization` | `num_blocks` / 容量 | 直接决定缓存容量，对应本章的「悬崖」 |
#
# **差距在哪**：
#
# 1. **hash 实现**：vLLM 用 64-bit 整数 hash（`sha256` 可选），并把「是否第一个 block」纳入
#    hash，保证不同 prefix 的相同 block 不误命中；我们用 blake2b 8 字节 + token 内容二次校验。
# 2. **驱逐**：vLLM 用双向链表 + 引用计数做近似 LRU，且 block 在 `ref_cnt == 0` 时才可回收；
#    我们的模拟为了讲清命中率没有真正实现驱逐。
# 3. **与调度耦合**：vLLM 的调度器用「命中长度」估算 TTFT、决定是否抢占；本章只测了单请求。
# 4. **copy-on-write**：vLLM 对部分命中的 block 做 CoW（共享块只读、写时复制）；我们在
#    `attach` 后只追加新 block，不做 CoW（练习 5 让你设计）。
#
# 一句话：前缀缓存本身只有 100 行，难点在于**和分页、调度、内存池的交互**——这正好是 05 章
# 调度器要处理的事。

# %% [markdown]
# ## 04.6 练习与小结
#
# **练习**
#
# 1. ⭐ 把 block_size 从 16 改成 32 / 64，重跑实验 A。命中 token 数、TTFT 加速比怎么变？
#    为什么 block 太大时「不足一个 block 的尾部」会让缓存收益下降？
# 2. ⭐ 把用户问题从 8 token 改成 128 token，重跑 TTFT 表。用公式 `(P+Q)/Q` 解释为什么
#    命中收益变小。
# 3. ⭐⭐ 给 `PrefixCache` 实现真正的 LRU 驱逐（用 `OrderedDict` 或双向链表），在固定容量下
#    复现实验 C 的命中率表，并加断言：驱逐后总 block 数不超过容量、被驱逐的 block 引用计数为 0。
# 4. ⭐⭐ 把 hash 从 64-bit 改成 32-bit，构造（或估算）一次碰撞，你会观察到 `assert
#    np.array_equal(...)` 触发。解释 vLLM 为什么敢只用 64-bit hash，而你的教学版为什么要
#    内容兜底（提示：生日攻击的规模、以及「碰撞 = 返回错误结果」的代价）。
# 5. ⭐⭐⭐ 开放设计：多轮对话 + 工具调用场景下，同一会话的请求前缀不断增长，且 system prompt
#    可能被不同的工具调用打断。设计一个前缀缓存命中率模型，并给出 3 条优化建议
#    （评分要点：是否考虑了 block 粒度与部分命中、是否讨论了驱逐对长会话的影响、
#    是否提到 copy-on-write、是否把命中率换算成了 TTFT/成本收益）。

# %% [markdown]
# **一句话记住**
#
# > 前缀缓存卖的不是更快的 hash，而是「命中长度」——命中多长，TTFT 就省多长。
#
# **自检 3 问**（不看笔记能答出来才算过）
#
# 1. 为什么只 hash 本块内容会算错？链式 hash 到底把什么编进了 key？
# 2. 引用计数减到 0 之前，block 为什么不能被回收？谁在持有那些引用？
# 3. 同一套实现，MPS 上 2–4×、CPU 上 11×、真实 GPU 上 5–20×——三个数字差在哪？
#
# **真实工作场景题**
#
# 客服机器人上线一周后，产品在 prompt 最前面加了「当前时间 + 用户 ID」。命中率监控直接归零——
# 每个请求第一个 block 的 hash 都不一样。判据：**前缀缓存要求从第 0 个 block 起连续命中**，
# 任何「每个请求都不同」的内容只要出现在前缀里，它之后的所有 block 都会一起失效。
# 修法：把动态字段挪到用户问题之后，或由服务端在拼接时放进最后一段。
#
# **你现在应该能回答：**
#
# - 为什么只 hash 本 block 内容会算错？链式 hash 保证了什么？
# - 命中 P token 前缀时，理论 TTFT 加速比是多少？实测为什么差那么远？
# - 缓存容量、block_size、驱逐策略分别如何影响命中率？
#
# **小结与下一章**
#
# 前三章我们造出了 KV cache（02）、分页管理（03）、跨请求共享（04）。但一个请求什么时候被
# 调度、batch 里同时放几个请求、谁先被抢占——这些决定吞吐与延迟的关键决策还没人做。
# 05 章写调度器：静态批处理 vs 连续批处理，把本章的「命中长度」变成调度决策的输入。
#
# **延伸阅读**
#
# - vLLM 源码：`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/kv_cache_utils.py`
# - vLLM 文档：`--enable-prefix-caching`、`--prefix-caching-hash-algo` 与 block hash 说明
# - 下一章：05 调度器，决定「下一批跑谁」
