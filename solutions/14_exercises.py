"""第 14 章《Capstone 容量规划》练习参考答案。

运行：.venv/bin/python solutions/14_exercises.py
说明：下面复用了 notebook 14.2 的 plan_capacity()，只改输入参数；所有数字与正文同源。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GPU:
    name: str
    vram_gb: float
    hbm_bw_gbps: float
    tflops_fp16: float
    hourly_usd: float


@dataclass(frozen=True)
class Workload:
    qps: float = 100.0
    ttft_budget_ms: float = 500.0
    tpot_budget_ms: float = 50.0
    context: int = 2048
    output: int = 300
    peak_to_avg: float = 3.0
    budget_usd_month: float = 10_000.0


A100 = GPU("A100 80GB", 80.0, 2000.0, 150.0, 1.9)
PLANS = {
    "A 8B fp16 x1": dict(params=8e9, weight_bytes=2.0, n_layer=32, n_kv_head=8, head_dim=128, tp=1),
    "B 8B int4 x1": dict(params=8e9, weight_bytes=0.5, n_layer=32, n_kv_head=8, head_dim=128, tp=1),
    "C 27B int4 x2": dict(params=27e9, weight_bytes=0.5, n_layer=46, n_kv_head=8, head_dim=128, tp=2),
}


def plan(params, weight_bytes, n_layer, n_kv_head, head_dim, tp=1, gpu=A100,
         wl=Workload(), util=0.9, workspace_gb=2.0, prefix_hit=0.0) -> dict:
    """notebook 14.2 的精简版（新增 prefix_hit 用于练习 4）。"""
    weights_bytes = params * weight_bytes
    kv_per_token = 2 * n_layer * n_kv_head * head_dim * 2
    kv_per_req = kv_per_token * (wl.context + wl.output)
    kv_read = kv_per_token * (wl.context + wl.output / 2)
    kv_budget = gpu.vram_gb * util * 1e9 - weights_bytes / tp - workspace_gb * 1e9
    max_kv = max(0, int(kv_budget / kv_per_req))
    tpot_bytes = wl.tpot_budget_ms / 1000 * gpu.hbm_bw_gbps * 1e9
    max_tpot = max(0, int((tpot_bytes * tp - weights_bytes) / kv_read))
    concurrency = min(max_kv, max_tpot)
    step_ms = (weights_bytes + concurrency * kv_read) / tp / (gpu.hbm_bw_gbps * 1e9) * 1000
    prefill_ms = 2 * params * wl.context * (1 - prefix_hit) / (tp * gpu.tflops_fp16 * 1e12) * 1000 + 20
    qps_i = concurrency / (prefill_ms / 1000 + wl.output * step_ms / 1000)
    instances = max(1, int(np.ceil(wl.qps / (qps_i * 0.8))))
    monthly = instances * tp * gpu.hourly_usd * 730
    tokens_m = wl.qps / wl.peak_to_avg * 3600 * 730 * (wl.context + wl.output) / 1e6
    return dict(concurrency=concurrency, step_ms=step_ms, ttft_ms=prefill_ms,
                qps_per_instance=qps_i, instances=instances, gpus=instances * tp,
                monthly=monthly, usd_per_m=monthly / tokens_m,
                ttft_ok=prefill_ms <= wl.ttft_budget_ms,
                tpot_ok=step_ms <= wl.tpot_budget_ms,
                budget_ok=monthly <= wl.budget_usd_month)


def show(title: str, wl: Workload, prefix_hit: float = 0.0) -> None:
    print(f"\n{title}")
    for name, kw in PLANS.items():
        r = plan(**kw, wl=wl, prefix_hit=prefix_hit)
        print(f"  {name}: 并发 {r['concurrency']:>3d}, TPOT {r['step_ms']:5.1f}ms, "
              f"TTFT {r['ttft_ms']:6.1f}ms, 实例 {r['instances']:>2d}, "
              f"月 ${r['monthly']:,.0f}, SLO(ttft/tpot/budget)="
              f"{r['ttft_ok']}/{r['tpot_ok']}/{r['budget_ok']}")


# 练习 1 ⭐：QPS=60，谁进入预算？A 最多能撑多少 QPS？
def ex1() -> None:
    show("练习 1：QPS=60", Workload(qps=60))
    print("  结论：B 每月 $5,548 最便宜；A 也进预算（$6,935）；C 超预算（$11,096）。")
    best = 0
    for qps in range(1, 120):
        if plan(**PLANS["A 8B fp16 x1"], wl=Workload(qps=qps))["budget_ok"]:
            best = qps
    print(f"  A 在 7 台实例（$9,709）时最多撑到 QPS={best}；再高就进入 8 台（$11,096）超预算。")


# 练习 2 ⭐：TPOT 预算收紧到 30ms，哪些方案失败？
def ex2() -> None:
    show("练习 2：TPOT 预算 30ms", Workload(tpot_budget_ms=30))
    print("  结论：30ms 下三方案 TPOT 都达标，没有方案直接失败；")
    print("        但 A 的并发上限从 175 降到 152（TPOT 成为新约束），余量变薄。")
    show("  补充：若收紧到 20ms", Workload(tpot_budget_ms=20))
    print("  20ms 时 A 需要 10 台（$13,870）→ 预算失败；B 7 台（$9,709）仍可行；")
    print("  说明「TPOT 预算」本身很少直接杀死方案，它通过抬高实例数间接杀死预算。")


# 练习 3 ⭐⭐：上下文 2K → 8K
def ex3() -> None:
    show("练习 3：上下文 8K", Workload(context=8192))
    print("  结论：三个方案全部 TTFT 失败（894ms / 894ms / 1495ms > 500ms），")
    print("        而且实例数从 8/6/6 暴涨到 30/24/23 —— 每请求 KV 线性 ×4，")
    print("        并发上限下降，同时 prefill 计算量 ×4，单实例 QPS 崩掉。")
    print("  对策：chunked prefill + 前缀缓存（RAG 的 system prompt 复用）、")
    print("        或把 SLO 的上下文假设改成分层（80% 请求 <2K，长请求单独走大卡实例）。")


# 练习 4 ⭐⭐：前缀缓存命中率 80%
def ex4() -> None:
    show("练习 4：前缀缓存命中 80%", Workload(), prefix_hit=0.8)
    print("  结论：A/B 实例数不变（decode 占主导，prefill 从 238ms → 64ms 对 QPS 影响小）；")
    print("        C 从 6 台降到 5 台（$16,644 → $13,870），因为 27B 的 prefill 占比更高，")
    print("        前缀缓存对「长上下文 + 大模型」最值钱。")
    print("  实现：vLLM --enable-prefix-caching + RAG 把稳定 system prompt 放最前面。")


# 练习 5 ⭐⭐⭐：给老板的一页容量规划（开放题评分要点）
def ex5() -> None:
    print("\n练习 5：一页容量规划（参考答案要点）")
    print("""
  推荐结构（一页）：
    1) 结论先行：推荐方案 B（8B int4×1，6 台 A100），$8,322/月，SLO 全达标，余量 17%
    2) 假设表：QPS 100（峰值）、2K/300、A100 $1.9/h、利用率 80%、峰值/均值 3
    3) 三方案对比：实例数 / GPU 数 / $/M tokens / TTFT / TPOT / 预算判定 / 风险
    4) 风险与对策：int4 精度（评测集+AB 测试）、排队（压测 p95）、供应商锁（多云报价）
    5) 验证计划：第 1 周 vllm bench serve 校准，第 2 周灰度 10% 流量，SLO 不达标回滚到 A
  评分要点（每个 20 分）：
    a) SLO 是否量化到 p95，而不是「感觉快」；
    b) 成本是否含全部假设（利用率、峰值/均值、存储/网络）；
    c) 风险是否有对应动作（评测、灰度、回滚）；
    d) 验证是否可执行（脚本、指标、验收阈值、时间点）；
    e) 是否给出「不建议」的选项及原因（例如 27B 超预算 66%）。
  """)


if __name__ == "__main__":
    ex1()
    ex2()
    ex3()
    ex4()
    ex5()
