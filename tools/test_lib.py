#!/usr/bin/env python
"""共享库回归测试（不依赖 Notebook，秒级）。

重点覆盖曾经出过 bug 的路径：
1. KV cache 分块 prefill 与整段 prefill 的 logits 必须一致；
2. generate_cached 与 generate_naive（贪心）必须生成相同 token；
3. KV cache 的字节账本与理论公式一致；
4. 采样函数的关键不变量。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aiinfra.tinygpt import GPTConfig, KVCache, TinyGPT, generate_cached, generate_naive, sample_next  # noqa: E402


def test_split_prefill_matches_full() -> None:
    torch.manual_seed(0)
    cfg = GPTConfig()
    model = TinyGPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 20))

    full = model(idx)
    cache = KVCache(cfg, max_seq_len=20)
    model(idx[:, :11], start_pos=0, kv_cache=cache)
    cache.advance(11)
    tail = model(idx[:, 11:], start_pos=11, kv_cache=cache)

    err = (full[:, -1, :] - tail[:, -1, :]).abs().max().item()
    assert err < 1e-4, f"分块 prefill 与整段 prefill 不一致，max err={err}"
    print(f"✅ 分块 prefill 一致性：max err={err:.2e}")


def test_cached_generation_matches_naive() -> None:
    torch.manual_seed(1)
    cfg = GPTConfig()
    model = TinyGPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 12))

    naive = generate_naive(model, prompt, max_new_tokens=8, temperature=0.0)
    cached = generate_cached(model, prompt, max_new_tokens=8, temperature=0.0)
    assert torch.equal(naive, cached), "generate_cached 与 generate_naive 的贪心结果不一致"
    print("✅ 贪心生成：cached == naive")


def test_kv_cache_bytes() -> None:
    cfg = GPTConfig()
    cache = KVCache(cfg, batch_size=3, max_seq_len=128)
    expected = 2 * cfg.n_layer * 3 * cfg.n_head * 128 * cfg.head_dim * 4
    assert cache.memory_bytes() == expected, "KV cache 字节账本与公式不一致"
    assert sum(t.numel() for t in cache.k) + sum(t.numel() for t in cache.v) == expected // 4
    print("✅ KV cache 字节账本 == 2·L·B·H·S·D·dtype")


def test_sampling_invariants() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert sample_next(logits, temperature=0).item() == 3
    assert sample_next(logits, temperature=1e-6).item() == 3
    for _ in range(50):
        assert sample_next(logits, temperature=1.0, top_k=2).item() in (2, 3)
    print("✅ 采样不变量：贪心 / 近似贪心 / top-k 屏蔽")


def test_cache_overflow() -> None:
    cfg = GPTConfig()
    cache = KVCache(cfg, max_seq_len=4)
    try:
        cache.append(0, torch.zeros(1, cfg.n_head, 5, cfg.head_dim), torch.zeros(1, cfg.n_head, 5, cfg.head_dim))
    except ValueError:
        print("✅ KV cache 溢出保护生效")
        return
    raise AssertionError("KV cache 溢出未报错")


if __name__ == "__main__":
    test_split_prefill_matches_full()
    test_cached_generation_matches_naive()
    test_kv_cache_bytes()
    test_sampling_invariants()
    test_cache_overflow()
    print("\n全部共享库测试通过")
