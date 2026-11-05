"""第 10 章练习参考答案（用 vLLM 提供服务）。

可独立运行的部分：练习 1、2 的显存估算；练习 3、4 的客户端/接口实现（用
`cloud/fixtures/` 回放，不需要真实服务）。

运行：.venv/bin/python solutions/10_exercises.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "cloud" / "fixtures"


# 与第 10 章一致的模型显存参数
MODELS = {
    "Qwen3-8B": {"params": 8.2e9, "n_layer": 36, "n_kv_head": 8, "head_dim": 128},
    "Qwen3-27B": {"params": 27.0e9, "n_layer": 64, "n_kv_head": 8, "head_dim": 128},
    "Qwen3-32B": {"params": 32.8e9, "n_layer": 64, "n_kv_head": 8, "head_dim": 128},
}


def kv_bytes_per_token(m: dict, dtype_bytes: int = 2) -> int:
    """每 token 的 KV 字节数：2 (K+V) × layers × kv_heads × head_dim × dtype。"""
    return 2 * m["n_layer"] * m["n_kv_head"] * m["head_dim"] * dtype_bytes


def memory_table(max_model_len: int = 8192, util: float = 0.90, card_gb: int = 80,
                 overhead_gb: float = 2.0) -> list[dict]:
    rows = []
    for name, m in MODELS.items():
        for dt, wb in (("fp16", 2), ("int8", 1), ("int4", 0.5)):
            weights = m["params"] * wb / 1e9
            kv_gb = card_gb * util - weights - overhead_gb
            per_tok = kv_bytes_per_token(m)
            tokens = max(kv_gb, 0) * 1e9 // per_tok
            rows.append({
                "model": name, "dtype": dt, "weights_gb": round(weights, 1),
                "kv_gb": round(kv_gb, 1), "kv_tokens": int(tokens),
                # 单请求最长 max_model_len，所以并发上限 = 总 token / max_model_len
                "max_concurrent": int(tokens // max_model_len),
            })
    return rows


# --------------------------------------------------------------------------- #
# 练习 1（⭐）：max-model-len 8192 -> 4096
# --------------------------------------------------------------------------- #
def ex1_max_model_len() -> None:
    """`--max-model-len` 不改变 KV 总容量，只改变「单个请求能占多长」。

    8B / fp16 / 80GB / util=0.90 时 KV 预算约 53.6GB，总容量约 36.3 万 token：
    - max-model-len=8192 → 单请求最长 8192，按满长算并发上限约 44；
    - max-model-len=4096 → 并发上限约 88（翻倍），但每个请求也只能到 4096。
    注意：并不是「并发翻倍」这么简单——真实并发由流量长度分布决定；
    短请求场景下把 max-model-len 调小不会增加容量，因为容量本来就是按 token 算的。
    """
    for mml in (8192, 4096):
        rows = [r for r in memory_table(max_model_len=mml) if r["model"] == "Qwen3-8B" and r["dtype"] == "fp16"]
        r = rows[0]
        print(f"  max_model_len={mml}: kv={r['kv_gb']}GB tokens={r['kv_tokens']} "
              f"max_concurrent(满长)={r['max_concurrent']}")


# --------------------------------------------------------------------------- #
# 练习 2（⭐）：gpu-memory-utilization 0.90 -> 0.95
# --------------------------------------------------------------------------- #
def ex2_util() -> None:
    """0.90 -> 0.95：8B fp16 的 KV 预算从 53.6GB 涨到 57.6GB（+7.5%），
    但「留给框架的余量」从 80-72=8GB 变成 80-76=4GB。

    危险点：CUDA graph 捕获、激活峰值、碎片、其他进程（NCCL/监控）都可能吃掉
    1-3GB；一旦瞬时超过余量就是 CUDA OOM，而 OOM 是启动失败或运行中崩，
    对在线服务是不可接受的。所以默认 0.90，只有确认峰值后才试 0.95。
    """
    for util in (0.90, 0.95):
        r = [x for x in memory_table(util=util) if x["model"] == "Qwen3-8B" and x["dtype"] == "fp16"][0]
        headroom = 80 - 80 * util
        print(f"  util={util}: kv={r['kv_gb']}GB tokens={r['kv_tokens']} "
              f"headroom_above_util={headroom:.0f}GB")


# --------------------------------------------------------------------------- #
# 练习 3（⭐⭐）：补 /v1/completions 与 /v1/models
# --------------------------------------------------------------------------- #
def ex3_client_methods() -> None:
    """在 VLLMClient 中追加两个方法（回放模式用 fixture 的字段演示）。

    def completion(self, prompt: str, max_tokens: int = 64) -> dict:
        payload = {"model": "Qwen/Qwen3-8B", "prompt": prompt, "max_tokens": max_tokens}
        if not self.online:
            return {"status_code": 200, "replay": True,
                    "body": {"choices": [{"text": "(replay) " + prompt[:16]}],
                             "usage": {"completion_tokens": 0}}}
        return _post_stream(self.base_url, "/v1/completions", payload | {"stream": True})

    def list_models(self) -> dict:
        if not self.online:
            fx = self.chat_fx["records"][0]["response"]
            return {"data": [{"id": fx["model"], "object": "model"}]}
        return httpx.get(self.base_url + "/v1/models", timeout=10).json()

    验证：回放模式断言 /v1/models 里含 Qwen3-8B；在线模式对同一 prompt 各打一条，
    比较 chat 与 completion 的 usage.completion_tokens 是否一致（应当接近）。
    """
    # 回放侧的可运行演示：从 fixture 读出模型名
    fx = json.loads((FIXTURES / "chat_completions.json").read_text(encoding="utf-8"))
    model = fx["records"][0]["response"]["model"]
    print("  /v1/models (replay) ->", {"data": [{"id": model, "object": "model"}]})


# --------------------------------------------------------------------------- #
# 练习 4（⭐⭐）：动态加载 LoRA
# --------------------------------------------------------------------------- #
def ex4_lora() -> None:
    """服务端以 `--enable-lora` 启动，运行中动态加载：

        POST /v1/load_lora_adapter
        {"lora_name": "legal-v2", "lora_path": "/workspace/loras/legal-v2"}

    然后在 chat 请求里把 `model` 写成 `legal-v2` 即可。

    对比实验：
    - 基座 vs 适配器：同一 prompt 各请求 20 次，记录 TTFT/TPOT；
      预期 LoRA 的额外开销主要在 decode 的 adapter matmul（小），TTFT 基本不变；
    - 同时加载 4 个适配器压测：观察 `--max-loras` 上限与显存增长；
    - 卸载：POST /v1/unload_lora_adapter（vLLM 扩展；路径与字段名待核对版本）。
    """
    print("  load_lora_adapter payload ->",
          {"lora_name": "legal-v2", "lora_path": "/workspace/loras/legal-v2"})


# --------------------------------------------------------------------------- #
# 练习 5（⭐⭐⭐）：chat + embeddings + LoRA 同机部署
# --------------------------------------------------------------------------- #
DEPLOY_PLAN = """\
评分要点（5 条）：
1) 资源隔离：chat 与 embeddings 是两种 task，必须两个进程（两个端口）：
   chat :8000（--task generate），embeddings :8001（--task embed）。
   若只有一张卡，用 gpu-memory-utilization 各分一半（例如 0.45 / 0.40），
   留 5%-10% 余量；不要两个进程都设 0.9。
2) 显存账本：embedding 模型（0.6B）权重约 1.2GB + KV 很小，可忽略；
   主要是 chat 的 KV。给出 8B/2048ctx 的并发数并以压测验证。
3) 探活与路由：/health 探活，/v1/models 做能力发现；网关按路径分流
   （/v1/embeddings → 8001，其余 → 8000），并设置超时与重试预算。
4) 失败降级：embeddings 挂掉不应影响 chat；LoRA 加载失败时回落到基座，
   并在响应头/日志里标记 fallback。
5) 成本与监控：两进程共享一张卡时，用 Prometheus 分别抓 :8000/:8001/metrics；
   设 p99 TTFT 与 KV 使用率告警；给出单实例 $/1M token 与扩容阈值。
"""


if __name__ == "__main__":
    print("== 练习 1：max-model-len 8192 vs 4096 ==")
    ex1_max_model_len()
    print("\n== 练习 2：gpu-memory-utilization 0.90 vs 0.95 ==")
    ex2_util()
    print("\n== 练习 3：completion() / list_models() ==")
    ex3_client_methods()
    print("\n== 练习 4：动态 LoRA ==")
    ex4_lora()
    print("\n== 练习 5：同机部署方案 ==")
    print(DEPLOY_PLAN)
