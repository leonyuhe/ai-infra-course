"""第 05 章练习参考答案。

用法（在仓库根目录）：
    .venv/bin/python solutions/05_exercises.py

每个答案都直接调用 notebooks/05_scheduler.py 里定义的 Request / CostModel / run_static /
run_continuous / chat_workload，避免复制粘贴导致口径漂移。
参考答案里的数字来自 M3 Ultra / torch 2.15 / 本课程成本模型，趋势比绝对值重要。
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
    """把 percent 格式的章节 .py 当模块执行，拿到其中的类与函数。"""
    spec = importlib.util.spec_from_file_location(stem, ROOT / "notebooks" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)  # 会顺带执行章节里的标定实验（几秒钟），输出静音
    return mod


ch05 = load_chapter("05_scheduler")
np = ch05.np


def run(fn, reqs, cost=None, **kwargs):
    return fn(copy.deepcopy(reqs), cost or ch05.cost, **kwargs).metrics


# --------------------------------------------------------------------------- #
# 练习 1 ⭐：把输出长度改成全部 64，重跑对比表
# --------------------------------------------------------------------------- #
def exercise_1() -> None:
    reqs = copy.deepcopy(ch05.chat_workload())
    for r in reqs:
        r.max_new_tokens = 64
    static = run(ch05.run_static, reqs)
    cont = run(ch05.run_continuous, reqs)
    print("[练习 1] 输出长度全部 = 64")
    print(f"  静态: {static['吞吐(tokens/s)']:.1f} tokens/s, "
          f"decode 槽位浪费={static['decode槽位浪费']:.2%}, padding 浪费={static['prefill填充浪费']:.2%}")
    print(f"  连续: {cont['吞吐(tokens/s)']:.1f} tokens/s")
    print(f"  吞吐倍数: {cont['吞吐(tokens/s)'] / static['吞吐(tokens/s)']:.2f}x（原重尾负载是 2.55x）")
    print("  结论：decode 槽位浪费从 82% 掉到 0%，倍数从 2.55x 降到 2.15x；")
    print("        剩下的浪费主要是 padding（66%）与「批间空窗」，连续批处理只能消掉后者。\n")


# --------------------------------------------------------------------------- #
# 练习 2 ⭐：把 decode_per_seq_ms 从 0.12 调到 0.5
# --------------------------------------------------------------------------- #
def exercise_2() -> None:
    expensive = ch05.CostModel(decode_per_seq_ms=0.5)
    reqs = ch05.chat_workload()
    static = run(ch05.run_static, reqs, expensive)
    cont = run(ch05.run_continuous, reqs, expensive)
    print("[练习 2] decode 每序列成本 ×4（0.12 → 0.5 ms）")
    print(f"  静态 {static['吞吐(tokens/s)']:.1f} tokens/s；连续 {cont['吞吐(tokens/s)']:.1f} tokens/s；"
          f"倍数 {cont['吞吐(tokens/s)'] / static['吞吐(tokens/s)']:.2f}x（原 2.55x）")
    print("  结论：decode 越重，batch 的「省一次权重读」相对收益越小，连续批处理优势收窄。\n")


# --------------------------------------------------------------------------- #
# 练习 3 ⭐⭐：prefill 优先（不混批）的连续批处理
# --------------------------------------------------------------------------- #
def exercise_3() -> None:
    """一步里先排完 prefill、再排 decode 的变体。

    实现方式：拿到原调度器的 plan 后，如果本步有 prefill，就把 decode 全部推迟到下一步。
    这模拟了 V0「prefill batch 与 decode batch 分开跑」的行为。
    """

    def run_prefill_first(reqs, cost, max_num_seqs=16):
        """把 run_continuous 的步骤替换成「prefill step / decode step 交替」。"""
        reqs = copy.deepcopy(reqs)
        pending = sorted(reqs, key=lambda r: r.arrival)
        waiting, running = [], []
        now, ttft, done = 0.0, [], []
        lanes = list(range(max_num_seqs))
        lane_of = {}
        while pending or waiting or running:
            while pending and pending[0].arrival <= now:
                waiting.append(pending.pop(0))
            admits, budget = [], max_num_seqs - len(running)
            while waiting and len(admits) < budget:
                admits.append(waiting.pop(0))
            if not running and not admits:
                if not pending:
                    break
                now = pending[0].arrival
                continue
            # prefill-only step：先算新请求，decode 留到下一步
            if admits:
                dur = cost.step_ms(sum(r.prompt_len for r in admits), 0) / 1000
                for r in admits:
                    lane_of[r.id] = lanes.pop(0)
                    r.num_computed, r.num_generated = r.prompt_len, 1
                    r.t_first = now + dur
                    ttft.append(r.t_first - r.arrival)
                    running.append(r)
                now += dur
                continue
            dur = cost.step_ms(0, len(running)) / 1000
            for r in running:
                r.num_computed += 1
                r.num_generated += 1
            now += dur
            for r in list(running):
                if r.finished:
                    r.t_done = now
                    done.append(r)
                    lanes.append(lane_of.pop(r.id))
                    lanes.sort()
                    running.remove(r)
        e2e = np.array([r.t_done - r.arrival for r in reqs])
        return {"吞吐(tokens/s)": sum(r.max_new_tokens for r in reqs) / now,
                "TTFT p95(ms)": float(np.quantile(ttft, 0.95) * 1000),
                "E2E p95(ms)": float(np.quantile(e2e, 0.95) * 1000)}

    mixed = run(ch05.run_continuous, ch05.chat_workload())
    separated = run_prefill_first(ch05.chat_workload(), ch05.cost)
    print("[练习 3] prefill 优先（分步跑） vs 混批")
    print(f"  混批  : 吞吐 {mixed['吞吐(tokens/s)']:.1f} tokens/s, TTFT p95 {mixed['TTFT p95(ms)']:.0f} ms")
    print(f"  分步  : 吞吐 {separated['吞吐(tokens/s)']:.1f} tokens/s, TTFT p95 {separated['TTFT p95(ms)']:.0f} ms")
    print("  结论：分步让 prefill 与 decode 互相等待，吞吐下降；混批用更长的 ITL 换吞吐。\n")


# --------------------------------------------------------------------------- #
# 练习 4 ⭐⭐：SJF 静态批处理（批内按输出长度升序）
# --------------------------------------------------------------------------- #
def exercise_4() -> None:
    def run_static_sjf(reqs, cost, max_num_seqs=16):
        reqs = copy.deepcopy(reqs)
        pending = sorted(reqs, key=lambda r: r.arrival)
        waiting, running = [], []
        now = 0.0
        while pending or waiting or running:
            while pending and pending[0].arrival <= now:
                waiting.append(pending.pop(0))
            if not running:
                if not waiting:
                    if not pending:
                        break
                    now = pending[0].arrival
                    continue
                waiting.sort(key=lambda r: (r.max_new_tokens, r.arrival))  # SJF
                batch, waiting = waiting[:max_num_seqs], waiting[max_num_seqs:]
                dur = cost.step_ms(max(r.prompt_len for r in batch) * len(batch), 0) / 1000
                for r in batch:
                    r.num_computed, r.num_generated = r.prompt_len, 1
                    r.t_first = now + dur
                    r.token_times.append(r.t_first)
                now += dur
                running = [r for r in batch if not r.finished]
                for r in batch:
                    if r.finished:
                        r.t_done = now
                continue
            dur = cost.step_ms(0, len(running)) / 1000
            for r in running:
                r.num_computed += 1
                r.num_generated += 1
            now += dur
            for r in running:
                if r.finished:
                    r.t_done = now
            running = [r for r in running if not r.finished]
        e2e = np.array([r.t_done - r.arrival for r in reqs])
        return {"吞吐(tokens/s)": sum(r.max_new_tokens for r in reqs) / now,
                "E2E p95(ms)": float(np.quantile(e2e, 0.95) * 1000)}

    base = run(ch05.run_static, ch05.chat_workload())
    sjf = run_static_sjf(ch05.chat_workload(), ch05.cost)
    print("[练习 4] 静态批处理 + SJF 打包")
    print(f"  FCFS 静态: {base['吞吐(tokens/s)']:.1f} tokens/s, E2E p95 {base['E2E p95(ms)']:.0f} ms")
    print(f"  SJF 静态 : {sjf['吞吐(tokens/s)']:.1f} tokens/s, E2E p95 {sjf['E2E p95(ms)']:.0f} ms")
    print("  结论：同长度打包让静态吞吐从 310 提到 437 tokens/s（+41%），但仍只有连续批处理（792）的 55%；")
    print("        而且「凑同长度再发」会拉长排队，长输出请求可能被无限推迟——生产上不敢这么排。\n")


# --------------------------------------------------------------------------- #
# 练习 5 ⭐⭐⭐：设计一个同时优化吞吐与 p95 TTFT 的调度策略
# --------------------------------------------------------------------------- #
def exercise_5_hints() -> None:
    print("[练习 5] 开放设计题参考路径（评分要点）")
    print("  1) 准入：给 waiting 队列按 prompt_len / 输出长度估计的「服务时间」排序，短请求优先；")
    print("  2) 阈值：当 running 队列占用率 < 70% 时按 FCFS 接纳，> 90% 时暂停接纳并只做 decode；")
    print("  3) 指标：与 FCFS 对比吞吐、TTFT p50/p95、以及最长等待时间（饥饿检测）；")
    print("  4) 落点：vLLM 里对应 Scheduler.schedule() 的 waiting 遍历顺序（vllm/v1/core/sched/scheduler.py），")
    print("     并把策略做成一个 scheduling policy / priority 字段。")
    print("  5) 风险：短请求优先会让长 prompt 饿死，需要老化（aging）或配额。")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5_hints()
