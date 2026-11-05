"""随机权重的微型 GPT —— 02–08 章共用的实验对象。

设计目标不是"能生成好文本"，而是：
1. 结构上足够真实：RMSNorm + RoPE + MHA + SwiGLU，与 Llama/Qwen 同构；
2. 足够小：~2.8M 参数，CPU 上也能秒级跑完一次实验；
3. 显式暴露 KV cache 接口（``KVCache``），后面的章节可以替换它。

随机权重意味着输出是乱码，这是刻意的：本章只关心形状、带宽、显存与调度，
不关心生成质量。真实模型的质量实验放在 vLLM 章节。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class GPTConfig:
    vocab_size: int = 4096
    n_layer: int = 2
    n_head: int = 4
    d_model: int = 256
    d_ff: int = 768
    max_seq_len: int = 512
    rope_theta: float = 10_000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """每个 token 在 KV cache 中占用的字节数（MHA，无 GQA）。"""
        return 2 * self.n_layer * self.n_head * self.head_dim * dtype_bytes


# --------------------------------------------------------------------------- #
# 基础模块
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def rope_cos_sin(
    positions: torch.Tensor,
    head_dim: int,
    theta: float = 10_000.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE 的 cos/sin 表，形状均为 (len(positions), head_dim//2)。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim))
    freqs = torch.outer(positions.float(), inv_freq)  # (T, D/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D)；旋转前一半/后一半维度。"""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# --------------------------------------------------------------------------- #
# 参考版 KV cache：连续分配，长度随解码增长（第 02 章会手写它，第 03 章替换为分页版）
# --------------------------------------------------------------------------- #
class KVCache:
    """最朴素的 KV cache：每个请求为每层预分配 (B, H, max_seq_len, D)。

    预分配 + 连续布局，正是第 03 章要解决的问题（内部碎片 + 无法共享）。
    """

    def __init__(
        self,
        cfg: GPTConfig,
        batch_size: int = 1,
        max_seq_len: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.cfg = cfg
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len or cfg.max_seq_len
        self.device = device
        self.dtype = dtype
        self._len = 0
        shape = (batch_size, cfg.n_head, self.max_seq_len, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """k/v: (B, H, T, D)，写入当前长度之后的位置。"""
        t = k.shape[2]
        end = self._len + t
        if end > self.max_seq_len:
            raise ValueError(f"KV cache 溢出：{end} > {self.max_seq_len}")
        self.k[layer][:, :, self._len : end] = k
        self.v[layer][:, :, self._len : end] = v

    def get(self, layer: int, upto: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        end = self._len if upto is None else min(upto, self.max_seq_len)
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, n: int = 1) -> None:
        self._len += n

    @property
    def seq_len(self) -> int:
        return self._len

    def memory_bytes(self) -> int:
        return 2 * self.cfg.n_layer * self.batch_size * self.cfg.n_head * self.max_seq_len * self.cfg.head_dim * self.dtype.itemsize


# --------------------------------------------------------------------------- #
# Transformer 组件
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.cfg.n_head, self.cfg.head_dim
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        cos, sin = rope_cos_sin(
            torch.arange(start_pos, start_pos + T, device=x.device), D, self.cfg.rope_theta, x.dtype
        )
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if kv_cache is not None:
            kv_cache.append(layer_idx, k, v)
            k_all, v_all = kv_cache.get(layer_idx, upto=kv_cache.seq_len + T)
            S = k_all.shape[2]  # 前缀长度 + 本次 T
            # 因果掩码：前缀部分全部可见，本次 T 个 query 内部再按 triu 遮挡未来
            causal = torch.full((T, S), float("-inf"), device=x.device)
            causal[:, : S - T] = 0.0
            causal[:, S - T :] = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device), diagonal=1
            )
            att = (q @ k_all.transpose(-2, -1)) / math.sqrt(D) + causal
        else:
            k_all, v_all = k, v
            causal = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
            att = (q @ k_all.transpose(-2, -1)) / math.sqrt(D) + causal

        att = F.softmax(att.float(), dim=-1).to(q.dtype)
        out = (att @ v_all).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, start_pos=0, kv_cache=None, layer_idx=0):
        x = x + self.attn(self.norm1(x), start_pos, kv_cache, layer_idx)
        return x + self.mlp(self.norm2(x))


class TinyGPT(nn.Module):
    """与 Llama 同构的极小 GPT（约 2.8M 参数，随机初始化）。"""

    def __init__(self, cfg: GPTConfig | None = None):
        super().__init__()
        self.cfg = cfg or GPTConfig()
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layer)])
        self.norm_f = RMSNorm(self.cfg.d_model)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # 权重共享，和 GPT-2/Llama 一致

    def forward(
        self,
        idx: torch.Tensor,
        start_pos: int = 0,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """idx: (B, T) token id；返回 (B, T, vocab) logits。"""
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, start_pos, kv_cache, i)
        return self.lm_head(self.norm_f(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# 采样与生成
# --------------------------------------------------------------------------- #
def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    """logits: (B, T, V) 或 (B, V)；返回 (B, 1)。temperature=0 表示贪心。"""
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.shape[-1])
        thresh = torch.topk(logits, k).values[:, [-1]]
        logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_naive(
    model: TinyGPT,
    prompt: torch.Tensor,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
) -> torch.Tensor:
    """每步都把整个序列重跑一遍 —— 第 02 章的反例。"""
    out = prompt
    for _ in range(max_new_tokens):
        logits = model(out)[:, -1, :]
        nxt = sample_next(logits, temperature)
        out = torch.cat([out, nxt], dim=1)
    return out


@torch.no_grad()
def generate_cached(
    model: TinyGPT,
    prompt: torch.Tensor,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
) -> torch.Tensor:
    """prefill 一次 + 每步只喂 1 个 token。"""
    cfg = model.cfg
    cache = KVCache(cfg, batch_size=prompt.shape[0], max_seq_len=prompt.shape[1] + max_new_tokens, device=prompt.device, dtype=model.tok_emb.weight.dtype)
    logits = model(prompt, start_pos=0, kv_cache=cache)
    cache.advance(prompt.shape[1])
    out = prompt
    for _ in range(max_new_tokens):
        nxt = sample_next(logits[:, -1, :], temperature)
        out = torch.cat([out, nxt], dim=1)
        logits = model(nxt, start_pos=cache.seq_len, kv_cache=cache)
        cache.advance(1)
    return out
