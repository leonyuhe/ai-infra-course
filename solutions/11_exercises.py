"""第 11 章练习参考答案（压测与调优）。

数据来自 `cloud/fixtures/benchmark_results.json`（示例录制，非本机实测）；
答案里的代码可直接运行，趋势可复现，绝对值需用自己的录制数据替换。

运行：.venv/bin/python solutions/11_exercises.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BENCH = json.loads((ROOT / "cloud" / "fixtures" / "benchmark_results.json").read_text(encoding="utf-8"))
SWEEP = pd.DataFrame(BENCH["sweep"])
EXT = pd.DataFrame(BENCH["extended"])
ALL = pd.concat([SWEEP, EXT], ignore_index=True)
OUT_LEN = BENCH["config"]["output_len"]


# --------------------------------------------------------------------------- #
# 练习 1（⭐）：SLO 改成 p99 TTFT <= 500ms
# --------------------------------------------------------------------------- #
def ex1_tighter_slo() -> None:
    """容量从 32 掉到 16（goodput 6.34 -> 4.63 req/s）。

    数据：p99 TTFT = 68 / 96 / 158 / 318 / 921 / 2600 ms（并发 1/4/8/16/32/64）。
    所以 500ms 的分界线落在 16（318ms 通过）与 32（921ms 不通过）之间。
    代价：同样 QPS 需要的实例数几乎翻倍（6.34/4.63 ≈ 1.37 倍），这就是
    「SLO 越严，单位 token 成本越高」的定量版本。
    """
    for slo in (1000.0, 500.0):
        ok = ALL[(ALL["ttft_p99_ms"] <= slo) & (ALL["tpot_p50_ms"] <= 40.0)]
        cap = int(ok["concurrency"].max())
        goodput = float(ALL.loc[ALL["concurrency"] == cap, "request_throughput_req_s"].iloc[0])
        print(f"  SLO p99 TTFT<={slo:.0f}ms -> capacity={cap:2d} req/s={goodput:.2f}")
    print("  实例数（QPS=20）：1000ms 需 ceil(20/6.34)=4；500ms 需 ceil(20/4.63)=5"
          "（按 100% 利用率算，生产建议再乘 1.3 冗余）")


# --------------------------------------------------------------------------- #
# 练习 2（⭐）：拐点判据 5% / 10% / 20%
# --------------------------------------------------------------------------- #
def ex2_thresholds(thresholds=(0.05, 0.10, 0.20)) -> None:
    """判据越松（阈值越大），拐点越早（容量越保守）。

    吞吐单步增益：1->4: +294%，4->8: +80%，8->16: +61%，16->32: +37%，32->64: +6.2%。
    - 阈值 5%：没有一段小于 5%（最小 6.2%）→ 数据范围内找不到拐点，
      只能报告「64 仍未到拐点」；
    - 阈值 10% / 20%：32->64 这一段命中 → 拐点 = 32。
    所以写容量报告时必须把「判据」一起写出来，否则别人无法复现你的结论。
    """
    df = ALL.copy()
    df["gain"] = df["output_throughput_tok_s"].pct_change()
    df["dttft"] = df["ttft_p99_ms"].pct_change()
    for th in thresholds:
        hit = df[(df["gain"] < th) & (df["dttft"] > 0.5)]
        if hit.empty:
            print(f"  threshold={th:.0%}: no knee within measured range (<= {int(df['concurrency'].max())})")
        else:
            idx = hit.index[0]
            print(f"  threshold={th:.0%}: knee={int(df.loc[idx - 1, 'concurrency'])} "
                  f"(next point gain={df.loc[idx, 'gain']:.1%}, p99 TTFT +{df.loc[idx, 'dttft']:.0%})")


# --------------------------------------------------------------------------- #
# 练习 3（⭐⭐）：p99 的 bootstrap 置信区间
# --------------------------------------------------------------------------- #
def bootstrap_p99(mean_latency_s: float, cv: float = 0.25, n: int = 200,
                  n_boot: int = 2000, seed: int = 0) -> dict:
    """用「均值 + 变异系数」合成一个对数正态样本，再 bootstrap p99。

    现实中应把每次请求的延迟列表传进来；这里用示例分布演示方法。
    结论：p99 对样本量和尾部形状都很敏感，bootstrap CI 能把它量化出来；
    n=50 与 n=200 时 CI 宽度接近（约 10%），n=1000 才明显收窄（约 6.5%）。
    所以报「p99=900ms」时必须同时报 n 与 CI；样本少于 200 时，p99 结论只能当参考。
    """
    rng = np.random.RandomState(seed)
    mu = np.log(mean_latency_s / np.sqrt(1 + cv ** 2))
    sigma = np.sqrt(np.log(1 + cv ** 2))
    samples = rng.lognormal(mu, sigma, n)
    boots = [np.percentile(rng.choice(samples, n, replace=True), 99) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": n, "p99_s": round(float(np.percentile(samples, 99)), 3),
            "ci95_s": (round(float(lo), 3), round(float(hi), 3)),
            "ci_width_pct": round(float((hi - lo) / np.percentile(samples, 99) * 100), 1)}


def ex3_bootstrap() -> None:
    for n in (50, 200, 1000):
        print("  ", bootstrap_p99(mean_latency_s=2.4, n=n))


# --------------------------------------------------------------------------- #
# 练习 4（⭐⭐）：开环压测（泊松到达）vs 闭环
# --------------------------------------------------------------------------- #
def ex4_open_loop(arrival_rate: float = 20.0, b_ms: float = 0.663,
                  prefill_ms: float = 20.0) -> dict:
    """开环（泊松到达）用 M/D/1 排队模型：容量 μ 由饱和态吞吐给出。

    闭环固定并发、等响应再发；开环固定到达率 λ。饱和态每 token 成本 = b ms，
    所以容量 μ = 1000/b token/s，单请求服务时间 S = out_len/μ。
    M/D/1 的平均排队时间 Wq = ρ·S / (2·(1-ρ))，ρ = λ·S；ρ >= 1 时系统不稳定。
    吞吐 = min(λ, 1/S)，TTFT ≈ Wq + prefill，端到端 ≈ Wq + S。
    """
    service_s = OUT_LEN * b_ms / 1000.0
    mu_rps = 1.0 / service_s
    rho = arrival_rate * service_s
    if rho >= 1.0:
        return {"arrival_rate": arrival_rate, "rho": round(rho, 2),
                "throughput_req_s": round(mu_rps, 2), "ttft_p50_ms": None,
                "note": "unstable: queue grows without bound"}
    wq_ms = rho * service_s / (2.0 * (1.0 - rho)) * 1000
    return {"arrival_rate": arrival_rate, "rho": round(rho, 2),
            "throughput_req_s": round(min(arrival_rate, mu_rps), 2),
            "ttft_p50_ms": round(wq_ms + prefill_ms, 1),
            "latency_p50_ms": round(wq_ms + service_s * 1000, 1)}


def ex4_compare() -> None:
    print("  开环（M/D/1 排队模型，μ = 1000/b ≈ 1508 tok/s）：")
    for rate in (2.0, 5.0, 8.0, 10.0, 11.0):
        print("   ", ex4_open_loop(rate))
    print("  读法：到达率翻倍时，吞吐先接近容量上界，随后 p99 延迟按排队论指数上升；")
    print("  闭环扫描给出的「拐点并发」在开环里对应「系统容量」，两者不能混用。")


# --------------------------------------------------------------------------- #
# 练习 5（⭐⭐⭐）：容量规划（p95 TTFT < 300ms、QPS=20、512/128）
# --------------------------------------------------------------------------- #
CAPACITY_PLAN = """\
评分要点（5 条 + 参考数字）：
1) SLO 拆解：p95 TTFT < 300ms。回放数据只有 p50/p90/p99，用 p99 做保守代理：
   并发 8 时 p99=158ms（安全），并发 16 时 p99=318ms（越线）→ 单实例并发上限取 8-12。
2) 单实例容量：并发 8 时 request_throughput = 2.88 req/s（128 输出 token）。
   要 QPS=20：ceil(20 / 2.88) = 7 个实例；乘 1.3 生产冗余 → 9 个实例。
3) 显存/卡型：8B fp16 在 24GB 卡上：权重 16.4GB + 8 并发 × 2048 token × 144KB
   ≈ 2.4GB KV + 2GB 预留 ≈ 20.8GB，可行；A100 80G 更宽裕（可同时跑 27B int8）。
4) 成本：4090 7×$0.69=4.83/h；A100 7×$1.99=13.93/h。按 QPS=20、128 tok 输出，
   每小时产出 20×3600×128 = 9.2M token → 4090 ≈ $0.52/1M，A100 ≈ $1.51/1M。
5) 验证与回滚：每实例先跑 bench_sweep（1/4/8/16/32）确认拐点；再做集群级
   开环压测（QPS 20）看 p95 TTFT 与 waiting；告警阈值 p95>250ms、
   waiting>10 时扩容；若 KV>90% 先处理 KV 而不是加并发。
注意：以上数字来自示例录制，上云后必须用自己的 4090/A100 数据重算。
"""


if __name__ == "__main__":
    print("== 练习 1：SLO 收紧到 500ms ==")
    ex1_tighter_slo()
    print("\n== 练习 2：拐点判据敏感性 ==")
    ex2_thresholds()
    print("\n== 练习 3：p99 的 bootstrap CI ==")
    ex3_bootstrap()
    print("\n== 练习 4：开环契约 ==")
    ex4_compare()
    print("\n== 练习 5：容量规划 ==")
    print(CAPACITY_PLAN)
