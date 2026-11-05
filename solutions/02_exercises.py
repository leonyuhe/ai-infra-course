#!/usr/bin/env python
"""第 02 章 · 练习参考答案与解题提示。

对应 `notebooks/02_kv_cache.py` 的五道练习。脚本可以直接运行：

    .venv/bin/python solutions/02_exercises.py

为了保持自包含，这里内联了本章最小的 `KVCache` 与缓存版 attention。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aiinfra.bench import bench, pick_device  # noqa: E402
from aiinfra.tinygpt import GPTConfig, TinyGPT, apply_rope, rope_cos_sin  # noqa: E402

cfg = GPTConfig(max_seq_len=4096)
device = pick_device()
torch.manual_seed(0)


class KVCache:
    """与第 02 章一致的接口：append / get / advance / seq_len。"""

    def __init__(self, cfg, batch_size=1, max_seq_len=4096, device="cpu", dtype=torch.float32):
        self.cfg, self.batch_size, self.max_seq_len, self.dtype = cfg, batch_size, max_seq_len, dtype
        self._len = 0
        shape = (batch_size, cfg.n_head, max_seq_len, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]

    def append(self, layer, k, v):
        end = self._len + k.shape[2]
        assert end <= self.max_seq_len
        self.k[layer][:, :, self._len : end] = k.to(self.dtype)
        self.v[layer][:, :, self._len : end] = v.to(self.dtype)

    def get(self, layer, upto=None):
        end = self._len if upto is None else min(upto, self.max_seq_len)
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, n=1):
        self._len += n

    def reset(self):  # 练习 3 的答案
        self._len = 0

    @property
    def seq_len(self):
        return self._len


def attention_with_cache(attn, x, cache, layer, start_pos):
    B, T, C = x.shape
    H, D = attn.cfg.n_head, attn.cfg.head_dim
    q, k, v = attn.qkv(x).split(C, dim=-1)
    q = q.view(B, T, H, D).transpose(1, 2)
    k = k.view(B, T, H, D).transpose(1, 2)
    v = v.view(B, T, H, D).transpose(1, 2)
    cos, sin = rope_cos_sin(torch.arange(start_pos, start_pos + T, device=x.device), D, attn.cfg.rope_theta, x.dtype)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    if cache.dtype != x.dtype:  # 练习 1：让计算精度跟随缓存 dtype（模拟 bf16 模型）
        q, k, v = q.to(cache.dtype), k.to(cache.dtype), v.to(cache.dtype)
    prefix = cache.seq_len
    cache.append(layer, k, v)
    k_all, v_all = cache.get(layer, upto=prefix + T)
    mask = torch.zeros((T, k_all.shape[2]), device=x.device)
    mask[:, prefix:] = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
    att = F.softmax(((q @ k_all.transpose(-2, -1)) / math.sqrt(D) + mask).float(), dim=-1).to(q.dtype)
    out = (att @ v_all).transpose(1, 2).reshape(B, T, C).to(x.dtype)
    return attn.proj(out)


def forward_with_cache(model, idx, cache, start_pos=0):
    x = model.tok_emb(idx)
    for layer, block in enumerate(model.blocks):
        x = x + attention_with_cache(block.attn, block.norm1(x), cache, layer, start_pos)
        x = x + block.mlp(block.norm2(x))
    return model.lm_head(model.norm_f(x))


MODEL = TinyGPT(cfg).to(device).eval()


# --------------------------------------------------------------------------- #
# 练习 1（⭐）：dtype 从 fp32 换成 bf16
# --------------------------------------------------------------------------- #
def exercise_1():
    """显存严格减半；延迟不会减半。

    原因：decode 是 bandwidth-bound，但 BF16 在 MPS 上要走 dequant/精度转换，且小模型
    的每次前向被 kernel 启动开销主导；在 CPU 上 bf16 甚至可能更慢（无原生 bf16 加速）。
    """
    prompt = torch.randint(0, cfg.vocab_size, (8, 512), device=device)
    for dtype in [torch.float32, torch.bfloat16]:
        cache = KVCache(cfg, 8, 513, device, dtype)
        with torch.no_grad():
            forward_with_cache(MODEL, prompt, cache)
        cache.advance(512)
        one = prompt[:, :1]
        r = bench(lambda: forward_with_cache(MODEL, one, cache, start_pos=512), warmup=2, reps=5, device=device)
        nbytes = sum(t.nbytes for t in cache.k + cache.v)
        print(f"  dtype={str(dtype):15s} KV={nbytes / 2**20:7.2f} MiB  decode={r.median_ms:6.3f} ms/token")
    print("  提示：MPS 上 bf16 的 kernel 选择不同，延迟不一定随带宽线性下降——以实测为准。")


# --------------------------------------------------------------------------- #
# 练习 2（⭐）：decode 吞吐 vs batch
# --------------------------------------------------------------------------- #
def exercise_2():
    """decode 每步要读一遍权重 + KV，batch 越大同一份读取服务越多 token。

    吞吐 ≈ batch × tokens / 单步时间；单步时间随 batch 增长得很慢（memory-bound），
    所以在带宽饱和前吞吐近似线性上涨，之后拐弯。
    """
    print(f"  {'batch':>5} {'decode 单步(ms)':>16} {'吞吐(tok/s)':>12}")
    for B in [1, 2, 4, 8, 16]:
        prompt = torch.randint(0, cfg.vocab_size, (B, 512), device=device)
        cache = KVCache(cfg, B, 513, device, torch.float32)
        with torch.no_grad():
            forward_with_cache(MODEL, prompt, cache)
        cache.advance(512)
        one = prompt[:, :1]
        r = bench(lambda: forward_with_cache(MODEL, one, cache, start_pos=512), warmup=2, reps=5, device=device)
        print(f"  {B:5d} {r.median_ms:16.3f} {B / r.median_ms * 1000:12.0f}")


# --------------------------------------------------------------------------- #
# 练习 3（⭐⭐）：reset() 复用显存
# --------------------------------------------------------------------------- #
def exercise_3():
    """reset 只改逻辑长度，不重新分配张量；两轮生成复用同一块显存。"""
    cache = KVCache(cfg, 1, 600, device, torch.float32)
    ptrs = [t.data_ptr() for t in cache.k + cache.v]
    prompt = torch.randint(0, cfg.vocab_size, (1, 512), device=device)
    for _ in range(2):
        cache.reset()
        with torch.no_grad():
            forward_with_cache(MODEL, prompt, cache)
        cache.advance(512)
        assert cache.seq_len == 512
    assert [t.data_ptr() for t in cache.k + cache.v] == ptrs, "重置后不应重新分配"
    print("  reset 后两次生成复用同一显存 ✓")


# --------------------------------------------------------------------------- #
# 练习 4（⭐⭐）：用两段 SDPA 替代显式 mask
# --------------------------------------------------------------------------- #
def attention_sdpa(attn, x, cache, layer, start_pos):
    """用 SDPA 的 attn_mask 参数替代显式加法 mask。"""
    B, T, C = x.shape
    H, D = attn.cfg.n_head, attn.cfg.head_dim
    q, k, v = attn.qkv(x).split(C, dim=-1)
    q = q.view(B, T, H, D).transpose(1, 2)
    k = k.view(B, T, H, D).transpose(1, 2)
    v = v.view(B, T, H, D).transpose(1, 2)
    cos, sin = rope_cos_sin(torch.arange(start_pos, start_pos + T, device=x.device), D, attn.cfg.rope_theta, x.dtype)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    if cache.dtype != x.dtype:  # 练习 1：让计算精度跟随缓存 dtype（模拟 bf16 模型）
        q, k, v = q.to(cache.dtype), k.to(cache.dtype), v.to(cache.dtype)
    prefix = cache.seq_len
    cache.append(layer, k, v)
    k_all, v_all = cache.get(layer, upto=prefix + T)
    if prefix == 0:
        out = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=True)
    else:
        # bool mask：True 表示允许参与 attention（前缀全可见 + 当前块内三角）
        keep = torch.ones((T, k_all.shape[2]), dtype=torch.bool, device=x.device)
        keep[:, prefix:] = ~torch.triu(torch.ones((T, T), dtype=torch.bool, device=x.device), diagonal=1)
        out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=keep)
    return attn.proj(out.transpose(1, 2).reshape(B, T, C))


def forward_chunk(model, idx, cache, start_pos, attn_fn):
    """跑一段 token（chunked prefill），用于验证 prefix > 0 时 SDPA 的 mask 语义。"""
    x = model.tok_emb(idx)
    for layer, block in enumerate(model.blocks):
        x = x + attn_fn(block.attn, block.norm1(x), cache, layer, start_pos)
        x = x + block.mlp(block.norm2(x))
    return model.lm_head(model.norm_f(x))


def exercise_4():
    """核心不在 SDPA 本身，而在「前缀全可见 + 当前块三角」这个 mask 语义。

    写对之后 logits 必须与显式 mask 版本一致；在 GPU 上 SDPA 会直接调 flash attention
    kernel，这是工业实现里 mask / softmax 被 fused 掉的起点。
    """
    torch.manual_seed(0)
    model = TinyGPT(cfg).to(device).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 200), device=device)
    with torch.no_grad():
        ref = model(tokens)
        # 先 prefill 120，再 prefill 80（两段都走带缓存路径）
        for attn_fn in (attention_with_cache, attention_sdpa):
            cache = KVCache(cfg, 2, 201, device, torch.float32)
            out = forward_chunk(model, tokens[:, :120], cache, 0, attn_fn)
            cache.advance(120)
            out = torch.cat([out, forward_chunk(model, tokens[:, 120:], cache, 120, attn_fn)], dim=1)
            gap = (ref - out).abs().max().item()
            name = attn_fn.__name__
            assert torch.allclose(ref, out, atol=2e-3, rtol=2e-3), f"{name}: {gap}"
            print(f"  {name}: logits 最大误差 = {gap:.2e} ✓")

# --------------------------------------------------------------------------- #
# 练习 5（⭐⭐⭐）：容量规划
# --------------------------------------------------------------------------- #
def exercise_5():
    """Qwen3-8B fp16 在 80GB 卡上的最大上下文（粗算）。

    假设：权重 16 GiB、可用 KV 预算 = 80 × gpu_memory_utilization(0.9) - 权重 - 激活值余量(2 GiB)。
    每 token KV = 144 KiB。并发 32 时上下文 = 预算 / (32 × 144 KiB)。
    """
    kv_budget_gib = 80 * 0.9 - 16 - 2
    per_token = 2 * 36 * 8 * 128 * 2 / 1024**2  # MiB
    max_ctx = kv_budget_gib * 1024 / (32 * per_token)
    print(f"  KV 预算 ≈ {kv_budget_gib:.1f} GiB，每 token ≈ {per_token * 1024:.0f} KiB")
    print(f"  并发 32 时最大上下文 ≈ {max_ctx:.0f} tokens（理论值，需实测校验）")
    print("  三条不确定性：1) 权重是否真的 16 GiB（量化/激活值占用）；")
    print("               2) 分页碎片与 block 粒度；3) prefill 长度不同导致激活值峰值不同。")


def main():
    print("练习 1：dtype 对显存与延迟的影响")
    exercise_1()
    print("\n练习 2：decode 吞吐 vs batch")
    exercise_2()
    print("\n练习 3：reset() 复用显存")
    exercise_3()
    print("\n练习 4：两段 SDPA")
    exercise_4()
    print("\n练习 5：容量规划")
    exercise_5()


if __name__ == "__main__":
    main()
