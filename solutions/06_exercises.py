"""第 06 章练习参考答案。

用法（在仓库根目录）：
    .venv/bin/python solutions/06_exercises.py

答案直接复用 notebooks/06_chunked_prefill.py 里的 Request / CostModel / run_chunked / 负载函数。
数字来自 M3 Ultra / torch 2.15 / 本课程成本模型；趋势比绝对值重要。
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AIINFRA_HEADLESS", "1")


def load_chapter(stem: str):
    spec = importlib.util.spec_from_file_location(stem, ROOT / "notebooks" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


ch06 = load_chapter("06_chunked_prefill")
cost = ch06.cost


def run(reqs=None, **kwargs):
    return ch06.run_chunked(reqs or ch06.chat_workload(), cost, max_num_seqs=16, **kwargs)["metrics"]


# --------------------------------------------------------------------------- #
# 练习 1 ⭐：chunk_size=256
# --------------------------------------------------------------------------- #
def exercise_1() -> None:
    base = run(chunk_size=512)
    tiny_chunk = run(chunk_size=256)
    print("[练习 1] chunk 从 512 缩到 256（长 prompt 2048）")
    print(f"  chunk=512: TPOT p95 {base['TPOT p95(ms)']:.1f} ms, 吞吐 {base['吞吐(tokens/s)']:.1f} tokens/s, "
          f"TTFT p95 {base['TTFT p95(ms)']:.0f} ms")
    print(f"  chunk=256: TPOT p95 {tiny_chunk['TPOT p95(ms)']:.1f} ms, 吞吐 {tiny_chunk['吞吐(tokens/s)']:.1f} tokens/s, "
          f"TTFT p95 {tiny_chunk['TTFT p95(ms)']:.0f} ms")
    print("  结论：TPOT 尾部继续下降（单步 prefill 更短），但吞吐也降——每步都多付一次固定发射成本；")
    print("        标定实验里 chunk=256 的 prefill 总耗时是 chunk=2048 的 7 倍左右，这就是代价。\n")


# --------------------------------------------------------------------------- #
# 练习 2 ⭐：max_num_batched_tokens 与 chunk 的关系
# --------------------------------------------------------------------------- #
def exercise_2() -> None:
    a = run(chunk_size=1024, max_num_batched_tokens=1024)
    b = run(chunk_size=1024, max_num_batched_tokens=4096)
    c = run(chunk_size=512, max_num_batched_tokens=512)
    print("[练习 2] token 预算 vs chunk 大小")
    print(f"  budget=1024, chunk=1024: TPOT p95 {a['TPOT p95(ms)']:.1f} ms, 吞吐 {a['吞吐(tokens/s)']:.1f}")
    print(f"  budget=4096, chunk=1024: TPOT p95 {b['TPOT p95(ms)']:.1f} ms, 吞吐 {b['吞吐(tokens/s)']:.1f}")
    print(f"  budget=512,  chunk=512 : TPOT p95 {c['TPOT p95(ms)']:.1f} ms, 吞吐 {c['吞吐(tokens/s)']:.1f}")
    print("  结论：budget 是「一步总 token 上限」，chunk 是「一步 prefill 上限」，实际 prefill 上限 =")
    print("        min(chunk, budget - decode 占用的 token)。所以 budget 小于 chunk 时，真正生效的是 budget；")
    print("        两者要一起看，单独调小任何一个都会把另一个变成瓶颈。\n")


# --------------------------------------------------------------------------- #
# 练习 3 ⭐⭐：抢占时优先选「剩余输出最短」的请求
# --------------------------------------------------------------------------- #
def exercise_3() -> None:
    original_preempt = ch06.preempt

    def shortest_first(st, r):
        victim = min(st["running"], key=lambda x: x.max_new_tokens - x.num_generated)
        original_preempt(st, victim)

    ch06.preempt = shortest_first
    sjf = ch06.run_chunked(ch06.preempt_workload(), cost, max_num_seqs=12, max_num_batched_tokens=2048,
                           chunk_size=1024, max_kv_tokens=4096)["metrics"]
    ch06.preempt = original_preempt  # 还原，避免影响后面的实验
    fcfs = ch06.run_chunked(ch06.preempt_workload(), cost, max_num_seqs=12, max_num_batched_tokens=2048,
                            chunk_size=1024, max_kv_tokens=4096)["metrics"]
    print("[练习 3] 抢占 victim 选择：队尾（FCFS） vs 剩余输出最短（SJF）")
    print(f"  FCFS: 抢占 {fcfs['preemptions']:.0f} 次, 重算 {fcfs['重算token']:.0f} token, "
          f"吞吐 {fcfs['吞吐(tokens/s)']:.1f}, E2E p95 {fcfs['E2E p95(ms)']:.0f} ms")
    print(f"  SJF : 抢占 {sjf['preemptions']:.0f} 次, 重算 {sjf['重算token']:.0f} token, "
          f"吞吐 {sjf['吞吐(tokens/s)']:.1f}, E2E p95 {sjf['E2E p95(ms)']:.0f} ms")
    print("  结论：SJF 减少总重算量、提高吞吐，但被抢占的短请求 E2E 尾延迟变差；")
    print("        而且「按剩余长度插队」会让长请求饥饿，vLLM 因此坚持 FCFS（等价于按到达顺序）。\n")


# --------------------------------------------------------------------------- #
# 练习 4 ⭐⭐：每 N 步才放一个 prefill chunk 的节奏控制
# --------------------------------------------------------------------------- #
def exercise_4() -> None:
    original_build_plan = ch06.build_plan
    rows = []
    for n in (1, 2, 4):
        def paced(st, n=n):
            plan, prefill_tokens, n_dec = original_build_plan(st)
            if n > 1 and len(st["steps"]) % n and prefill_tokens > 0:
                return [(r, 0, d) for r, n_pref, d in plan if n_pref == 0], 0, n_dec  # 本步只做 decode
            return plan, prefill_tokens, n_dec

        ch06.build_plan = paced
        m = run(chunk_size=512)
        ch06.build_plan = original_build_plan
        rows.append((n, m["TPOT p95(ms)"], m["吞吐(tokens/s)"], m["TTFT p95(ms)"]))
    print("[练习 4] prefill 节奏：每 N 步放一个 chunk")
    for n, tpot, tps, ttft in rows:
        print(f"  N={n}: TPOT p95 {tpot:.1f} ms, 吞吐 {tps:.1f} tokens/s, TTFT p95 {ttft:.0f} ms")
    print("  结论：N 越大，prefill 越稀疏 → 长请求 TTFT 变差、吞吐下降；TPOT p95 基本不变，")
    print("        因为每个 chunk 步本身的耗时已经由 chunk 大小决定。节奏控制只在「预算紧张、想保住 decode」时才有意义。\n")


# --------------------------------------------------------------------------- #
# 练习 5 ⭐⭐⭐：自适应 chunk 策略
# --------------------------------------------------------------------------- #
def exercise_5_hints() -> None:
    print("[练习 5] 开放设计题参考路径（评分要点）")
    print("  1) 规则：chunk = clamp(budget - decode_tokens, 256, 2048)，再按当前 TPOT 目标动态收缩；")
    print("  2) 依据：监控最近 N 步的 ITL p95，若超过 SLO 的 80%，下一轮把 chunk 减半；低于 40% 则加倍；")
    print("  3) 对比：与固定 chunk=512/1024 在低载/过载两种场景下比较吞吐与 TPOT p95，都不能退化；")
    print("  4) 落点：vLLM 的 Scheduler.schedule() 里决定每步 token 预算的那段（vllm/v1/core/sched/scheduler.py），")
    print("     以及 SchedulerConfig.max_num_batched_tokens 的动态调整；")
    print("  5) 风险：抖动（chunk 频繁变化会让 batch 形状不稳定）、可解释性、需要稳定的反馈信号。")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5_hints()
