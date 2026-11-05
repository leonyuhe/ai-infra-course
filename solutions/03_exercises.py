#!/usr/bin/env python
"""第 03 章 · 练习参考答案与解题提示。

对应 `notebooks/03_paged_attention.py` 的五道练习。脚本可以直接运行：

    .venv/bin/python solutions/03_exercises.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aiinfra.bench import bench, pick_device  # noqa: E402
from aiinfra.tinygpt import GPTConfig, TinyGPT, apply_rope, rope_cos_sin  # noqa: E402

cfg = GPTConfig(max_seq_len=8192)
device = pick_device()
torch.manual_seed(0)
H, D, NL = cfg.n_head, cfg.head_dim, cfg.n_layer


class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.num_blocks, self.block_size = num_blocks, block_size
        self._free = list(range(num_blocks))[::-1]
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
        need = -(-length // self.block_size)
        new_blocks = []
        while len(block_table) < need:
            block_id = self.allocate()
            block_table.append(block_id)
            new_blocks.append(block_id)
        return new_blocks

    def free(self, block_table):
        """释放整张 block table（练习 3 的前置：PagedKVCache.free 会调用它）。"""
        for block_id in block_table:
            assert block_id not in self._free_set, f"double free: block {block_id}"
            self._free.append(block_id)
            self._free_set.add(block_id)
            self.num_allocated -= 1
        block_table.clear()

    @property
    def num_free_blocks(self):
        return len(self._free)


class PagedKVCache:
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

    def free(self):  # 练习 3 的答案
        self.bm.free(self.block_table)
        self._len = 0

    @property
    def seq_len(self):
        return self._len


def make_paged(block_size, num_blocks=512):
    bm = BlockManager(num_blocks, block_size)
    k = [torch.zeros(num_blocks, H, block_size, D, device=device) for _ in range(NL)]
    v = [torch.zeros(num_blocks, H, block_size, D, device=device) for _ in range(NL)]
    return PagedKVCache(cfg, bm, k, v, device)


# --------------------------------------------------------------------------- #
# 练习 1（⭐）：block_size 扫描
# --------------------------------------------------------------------------- #
def exercise_1():
    """block 越小碎片越少、表越大；8 太小（kernel/调度开销），128 太大（尾部浪费）。"""
    rng = np.random.default_rng(0)
    lengths = np.clip(rng.lognormal(np.log(400), 0.8, 200), 16, 2048).astype(int)
    actual = int(lengths.sum())
    print(f"  {'block_size':>10} {'分配 tokens':>12} {'碎片率':>8} {'表条目':>8} {'平均尾部浪费':>12}")
    for bs in [8, 16, 32, 64, 128]:
        blocks = -(-lengths // bs)
        allocated = int(blocks.sum() * bs)
        print(f"  {bs:10d} {allocated:12d} {1 - actual / allocated:8.3%} {int(blocks.sum()):8d} {float((lengths % bs).mean()):12.1f}")
    print("  结论：碎片率 ≈ (block_size/2) / 平均长度；block table 条目 ≈ 2 × 请求 token 数 / block_size。")


# --------------------------------------------------------------------------- #
# 练习 2（⭐）：分布敏感性
# --------------------------------------------------------------------------- #
def exercise_2():
    """碎片率主要取决于「请求长度分布 vs block_size」，而不是 block 管理算法本身。"""
    rng = np.random.default_rng(0)
    distributions = {
        "uniform U(16,2048)": rng.integers(16, 2048, 200),
        "lognormal sigma=0.8": np.clip(rng.lognormal(np.log(400), 0.8, 200), 16, 2048).astype(int),
        "lognormal sigma=1.5": np.clip(rng.lognormal(np.log(400), 1.5, 200), 16, 2048).astype(int),
    }
    for name, lengths in distributions.items():
        actual = int(lengths.sum())
        for bs in [16, 64]:
            blocks = -(-lengths // bs)
            allocated = int(blocks.sum() * bs)
            print(f"  {name:22s} block_size={bs:2d}  碎片率={1 - actual / allocated:6.2%}  均值={lengths.mean():6.0f}")
    print("  结论：重尾分布的碎片率更高（长请求的尾部浪费被短请求的平均值掩盖），均匀分布最低。")


# --------------------------------------------------------------------------- #
# 练习 3（⭐⭐）：free() + 池复用
# --------------------------------------------------------------------------- #
def exercise_3():
    """PagedKVCache.free() 只是把 block 还回池子；连续两轮请求后空闲数必须回到初始值。"""
    paged = make_paged(16, 256)
    initial_free = paged.bm.num_free_blocks
    prompt = torch.randint(0, cfg.vocab_size, (1, 300), device=device)
    for _ in range(2):
        with torch.no_grad():
            k = torch.randn(1, H, 300, D, device=device)
            paged.append(0, k, k)
        assert paged.bm.num_allocated == 19  # ceil(300/16)
        paged.free()
        assert paged.bm.num_free_blocks == initial_free, "释放后空闲 block 数未回到初始值"
        assert paged.seq_len == 0
    print("  两轮分配/释放后空闲 block 回到初始值 ✓（无泄漏）")


# --------------------------------------------------------------------------- #
# 练习 4（⭐⭐）：逐 block 计算 attention（flash decoding 思路）
# --------------------------------------------------------------------------- #
def blockwise_decode_attention(q, paged, layer, scale):
    """不物化前缀：逐 block 做 online softmax，最后得到与全量 attention 等价的结果。

    q: (H, 1, D) 已带 RoPE；返回 (H, 1, D)。
    """
    # 注意形状：(H,1,1) 才能与 (H,1,D) 正确广播；写成 (H,1) 会按最后一维错配
    m = torch.full((H, 1, 1), float("-inf"), device=q.device)
    l = torch.zeros((H, 1, 1), device=q.device)
    acc = torch.zeros((H, 1, D), device=q.device)
    pos = 0
    for block_id in paged.block_table:
        kb = paged.k[layer][block_id]  # (H, bs, D)
        vb = paged.v[layer][block_id]
        valid = min(paged.block_size, paged.seq_len - pos)  # 最后一个 block 只取有效 token
        scores = (q @ kb.transpose(1, 2))[:, :, :valid] / scale  # (H, 1, valid)
        m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True).values)
        alpha = torch.exp(m - m_new)
        p = torch.exp(scores - m_new)
        acc = acc * alpha + p @ vb[:, :valid]
        l = l * alpha + p.sum(dim=-1, keepdim=True)
        m = m_new
        pos += paged.block_size
    return acc / l  # l 是 (H,1,1)，与 (H,1,D) 广播


def exercise_4():
    """去掉 gather 拷贝，但引入了 Python 级 block 循环；在本机上反而更慢。"""
    block_size = 16
    paged = make_paged(block_size, 256)
    L = 2048
    tokens = torch.randint(0, cfg.vocab_size, (1, L), device=device)
    model = TinyGPT(cfg).to(device).eval()

    # 构造真实 K/V：借第 0 层 attention 的 qkv 权重算一遍
    with torch.no_grad():
        x = model.blocks[0].norm1(model.tok_emb(tokens))
        q_raw, k_raw, v_raw = model.blocks[0].attn.qkv(x).split(cfg.d_model, dim=-1)
        q = q_raw.view(1, L, H, D).transpose(1, 2)
        k = k_raw.view(1, L, H, D).transpose(1, 2)
        v = v_raw.view(1, L, H, D).transpose(1, 2)
        cos, sin = rope_cos_sin(torch.arange(0, L, device=device), D, cfg.rope_theta, x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        paged.append(0, k, v)
        paged.advance(L)

    q_last = q[0, :, -1:, :]  # (H,1,D)
    scale = math.sqrt(D)
    ref = torch.softmax((q_last @ k[0].transpose(-2, -1)) / scale, dim=-1) @ v[0]  # (H,1,D)
    out = blockwise_decode_attention(q_last, paged, 0, scale)
    gap = (ref - out).abs().max().item()
    assert torch.allclose(ref, out, atol=1e-3, rtol=1e-3), gap
    print(f"  逐 block attention vs 全量 attention：最大误差 = {gap:.2e} ✓")
    print("  提示：去掉 gather 的收益要等 kernel 把 block 循环并行化后才能吃到；Python 循环反而更慢。")


# --------------------------------------------------------------------------- #
# 练习 5（⭐⭐⭐）：开放设计
# --------------------------------------------------------------------------- #
def exercise_5():
    """中间插入 token 的 block 管理设计（评分要点，不要求代码）。"""
    print("  参考方案：")
    print("  1) 优先复用最后一个 block 的空位（若剩余空间 >= 插入长度），O(1)。")
    print("  2) 空间不够时对最后一个 block 做 copy-on-write：申请新 block、复制前缀 K/V、再追加。")
    print("  3) 在中间插入（不是末尾）时，必须把插入点之后的逻辑 block 全部后移，最坏 O(序列长度)。")
    print("  4) 如果该 block 被前缀缓存共享（ch04），copy-on-write 是必须的，否则会改坏别的请求。")
    print("  5) 代价：额外一次 block 拷贝 + 可能触发重算（插入点之后 token 的 RoPE 位置变化）。")


def main():
    print("练习 1：block_size 扫描")
    exercise_1()
    print("\n练习 2：分布敏感性")
    exercise_2()
    print("\n练习 3：free() 与池复用")
    exercise_3()
    print("\n练习 4：逐 block attention")
    exercise_4()
    print("\n练习 5：中间插入 token 的 block 管理")
    exercise_5()


if __name__ == "__main__":
    main()
