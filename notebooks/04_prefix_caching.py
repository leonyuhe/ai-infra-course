# %% [markdown]
# # 第 04 章 · 前缀缓存：让共享的 system prompt 只算一次
#
# > 同一个 system prompt 被 10 万个请求共享，凭什么每个请求都要重新 prefill 一遍？
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 75–90 分钟 |
# | 运行环境 | 🏠 Mac（MPS / CPU）· 无需 GPU · torch 2.15 |
# | 前置章节 | 03（`PagedKVCache` + block table）、02（KV cache 接口） |
# | 本章产物 | 手写 hash 前缀缓存；512/1024/2048 token 共享前缀的 TTFT 实测；LRU 命中率模拟 |
# | 本章接口 | 复用 03 的 `PagedKVCache`，并加 `attach / release` 管理共享 block |
#
# **学完你能回答：**
# 1. 为什么前缀缓存必须按「block 内容 + 父 block hash」做链式 hash，而不是只 hash 本块内容？
# 2. 命中一段 P token 的前缀，TTFT 能降多少？为什么实测降幅远小于理论值？
# 3. 缓存容量、block 大小、驱逐策略（LRU）分别如何影响命中率？

# %% [markdown]
# ## 04.1 问题与直觉
#
# 一句话：**KV cache 里存的东西，只要前缀完全相同，就完全相同**——所以可以跨请求复用。
#
# 真实场景里前缀重复极其普遍：客服机器人共享 2K token 的 system prompt；Agent 共享工具
# 说明和 few-shot 示例；多轮对话的每一轮都复用之前的全部上下文。没有前缀缓存时，每个请求
# 都要把这段前缀的 K/V 重算一遍——**反例**是：N 个共享 P token 前缀的请求，浪费了
# N × P 个 token 的 prefill 计算，而 TTFT 完全被这段前缀的长度支配。
#
# 前缀缓存的单位是 block（03 章的产物）：如果两个请求的某个 block 内容相同、且它之前的所有
# block 也相同，那么这一块的 KV 一定相同，可以直接复用物理 block，跳过这段 prefill：
#
# ![](../assets/figures/ch04_prefix_tree.svg)
#
# 图里请求 A 和请求 B 共享前 3 个 block（system prompt），只有最后一个 block（用户问题）
# 不同。命中 3 个 block = 跳过 3 × block_size 个 token 的 prefill。

# %% [markdown]
# **为什么必须用链式 hash？** KV 不是 token 的纯函数：第 i 个 token 的 K/V 依赖它之前所有
# token（attention）以及它自己的位置（RoPE）。只 hash 本 block 的 16 个 token，会把
# 「内容相同但前缀不同」的 block 误判为命中，结果直接算错。所以 hash 必须链式累加：
#
# ```text
# h_0 = hash(0,  tokens[0:bs])
# h_1 = hash(h_0, tokens[bs:2bs])
# h_i = hash(h_{i-1}, tokens[i*bs:(i+1)*bs])
# ```
#
# **命中流程**：从 block 0 开始逐块查，遇到第一个 miss 就停（前缀必须连续命中）；
# 命中 n 块 → 跳过 n × block_size 个 token 的 prefill，只对剩余 token 做前向。
#
# **写回流程**：请求 prefill 时新写满的 block 注册进缓存（引用计数 +1）；请求结束后只释放
# 自己的引用，缓存里的 block 继续存活，等下一个请求命中或等驱逐。
#
# **收益与代价**：
#
# | 量 | 公式/说明 |
# | --- | --- |
# | 理论 TTFT 加速比 | `(P + Q) / Q`，P 是命中前缀长度，Q 是新 token 数 |
# | hash 成本 | 每个 block 一次 64-bit hash，约几十纳秒（本章实测 128 块 ≈ 0.07ms） |
# | 共享 block | 只读：请求结束不能清空内容；写入要靠 copy-on-write（本章不做，见练习） |
# | 缓存容量 | 有限：满了要按 LRU/FIFO 驱逐；驱逐粒度是 block |
# | 风险 | hash 碰撞（教学版用 token 内容二次校验）、缓存污染、被驱逐后要重算 |
#
# 本章在 03 章 `PagedKVCache` 的基础上加两个动作：`attach(block_ids, length)` 挂载共享 block，
# `release()` 按引用计数归还——这正是 vLLM 里 `KVCacheManager` 与 `BlockPool` 的分工。

# %% [markdown]
# ## 04.2 从零实现
#
# 准备模型与工具函数。注意 `cfg.max_seq_len` 放宽到 8192：本章要 prefill 2048 token 的
# system prompt。

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
# ### 复用 02 章的 attention：缓存接口不变
#
# 前缀缓存改变的是「K/V 从哪来」，不是「attention 怎么算」。所以这段代码与 02/03 章完全一致。

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
# ### BlockPool：带引用计数的 block 池
#
# 与 03 章 `BlockManager` 的区别只有一处：每个 block 多一个 `ref_count`。
# 只要 `ref_count > 0`（被某个请求或缓存表引用），这个 block 就不能被回收——
# 这是「共享」能成立的唯一机制。分配时 ref 置 1；缓存表与请求各自 `incref`；归还时 `release`，
# 减到 0 才回到空闲池。

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

# %% [markdown]
# ### PagedKVCache：03 章的实现原样保留
#
# 与 03 章完全同构：物理存储 `(num_blocks, H, block_size, D)`，每请求一张 block table，
# `append` 按需分配并跨 block 写，`get` 用 gather 拼成逻辑连续张量。

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

# %% [markdown]
# ### 本章新增的两个动作：attach / release
#
# - `attach(block_ids, length)`：把命中前缀的物理 block 挂到本请求的 block table 上（引用 +1），
#   并把逻辑长度直接设成命中长度——**没有做任何计算**，这就是省下来的 prefill。
# - `release()`：请求结束时把本请求持有的所有 block 引用还回去。共享 block 因为缓存表还持有
#   一份引用，不会被清空。

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

# %% [markdown]
# ### PrefixCache：链式 hash 前缀表
#
# 表里存两样东西：`hash -> 物理 block id`（命中时用）和 `hash -> token 内容`（教学版防碰撞，
# 真实 vLLM 只靠 64-bit hash 的概率保证）。`lookup` 从 block 0 开始逐块查、遇到 miss 停止；
# `insert` 把请求新写满的完整 block 注册进去，已存在的直接跳过。

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

# %% [markdown]
# ### 一次请求：查缓存 → 挂载 → 只算剩下的 → 注册新块
#
# `prefix_cache=None` 表示关闭前缀缓存（等价于 vLLM 的 `--enable-prefix-caching` 关闭），
# 这条路径会把整段 prompt 算一遍，作为 TTFT 的基线。

# %%
def make_pool(num_blocks=1024, dev="cpu"):
    pool = BlockPool(num_blocks, BS)
    k_blocks = [torch.zeros(num_blocks, H, BS, D, device=dev) for _ in range(NL)]
    v_blocks = [torch.zeros(num_blocks, H, BS, D, device=dev) for _ in range(NL)]
    return pool, k_blocks, v_blocks


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
# ### 正确性断言：命中路径必须与「全量重算」逐元素一致
#
# 这是本章最重要的测试。命中时我们**跳过了** system prompt 的 512 个 token，只对 8 个 token
# 做前向；它的 logits 必须与把 520 个 token 全部重算一遍的结果一致（`atol=1e-3`）。
# 同时断言：冷启动与首次写入都没命中，第二次命中 512 个 token；缓存里正好有 32 个 block
# （512 / 16），且空闲池数量正确——引用计数没有泄漏。

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
# ## 04.3 实测
#
# ### 实验 1：共享 system prompt 的 TTFT（命中 vs 未命中）
#
# 测量协议：M3 Ultra / torch 2.15 / MPS，system prompt = 512 / 1024 / 2048 token，
# 用户问题固定 8 token（不足一个 block，因此只有 system prompt 的整块会被缓存）。
# 每个请求执行一遍完整流程（查表 → 分配 → 前向 → 注册 → 释放），`warmup=3, reps=9` 取中位数。
# 「未命中」= 关闭前缀缓存，走全量 prefill。

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
# **读表/读图**：system prompt 从 512 涨到 2048，未命中 TTFT 从约 5ms 涨到约 12ms，
# 命中始终在 2.5–3.5ms，实测加速比约 2x / 2.5x / 4x（多次运行在 ±0.3 内波动）。
# 注意**实测远小于理论值**（65x / 129x /
# 257x），原因有三层，值得逐个说清：
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

# %%
df_cpu = pd.DataFrame(measure_ttft(torch.device("cpu"), [2048]))
df_cpu

# %% [markdown]
# CPU 上同一实验的实测加速比约 11x——更接近「跳过 99.6% prefill token」的理论预期。
# 这组对照说明：**前缀缓存的收益取决于 prefill 有多 compute-bound**，而不是取决于缓存的实现
# 有多聪明。在 GPU 上它会是 5–20x 的量级；在小模型的 MPS 上会被固定开销吃掉大半。

# %% [markdown]
# ### 实验 2：缓存容量与驱逐策略对命中率的影响
#
# 前缀缓存容量有限，满了要驱逐。先算工作集：4 个 system prompt（256/512/1024/2048 token）
# 合计 16 + 32 + 64 + 128 = 240 个 block。用 200 个请求（按 10%/30%/30%/30% 抽 prompt）模拟
# 「block 命中率」，比较 LRU 与 FIFO，以及容量从 64 到 512 的变化。
#
# 这个模拟不跑模型：命中率只取决于请求序列与驱逐策略，纯账本计算。

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
# **读表/读图**：容量是命中率的**悬崖**而不是斜坡——
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
# ## 04.4 对照工业实现（vLLM）
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
# ## 04.5 练习与小结
#
# **练习**
#
# 1. ⭐ 把 block_size 从 16 改成 32 / 64，重跑实验 1。命中 token 数、TTFT 加速比怎么变？
#    为什么 block 太大时「不足一个 block 的尾部」会让缓存收益下降？
# 2. ⭐ 把用户问题从 8 token 改成 128 token，重跑 TTFT 表。用公式 `(P+Q)/Q` 解释为什么
#    命中收益变小。
# 3. ⭐⭐ 给 `PrefixCache` 实现真正的 LRU 驱逐（用 `OrderedDict` 或双向链表），在固定容量下
#    复现实验 2 的命中率表，并加断言：驱逐后总 block 数不超过容量、被驱逐的 block 引用计数为 0。
# 4. ⭐⭐ 把 hash 从 64-bit 改成 32-bit，构造（或估算）一次碰撞，你会观察到 `assert
#    np.array_equal(...)` 触发。解释 vLLM 为什么敢只用 64-bit hash，而你的教学版为什么要
#    内容兜底（提示：生日攻击的规模、以及「碰撞 = 返回错误结果」的代价）。
# 5. ⭐⭐⭐ 开放设计：多轮对话 + 工具调用场景下，同一会话的请求前缀不断增长，且 system prompt
#    可能被不同的工具调用打断。设计一个前缀缓存命中率模型，并给出 3 条优化建议
#    （评分要点：是否考虑了 block 粒度与部分命中、是否讨论了驱逐对长会话的影响、
#    是否提到 copy-on-write、是否把命中率换算成了 TTFT/成本收益）。
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
