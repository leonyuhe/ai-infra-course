"""第 09 章练习参考答案（架构与源码地图）。

对照 `notebooks/09_vllm_architecture.py` 的实验代码阅读；数字来自本机运行
`simulate(CFG_VLLM, make_workload(), ...)` 的结果（Apple Silicon / torch 2.15），
只反映模拟器的趋势，不代表真实 GPU。

运行：.venv/bin/python solutions/09_exercises.py
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# 与第 09 章一致的模拟器（内联一份，保证答案可独立运行）
# --------------------------------------------------------------------------- #
@dataclass
class Req:
    rid: int
    prompt_len: int
    max_new: int
    arrival: float
    prefill_left: int = 0
    num_generated: int = 0
    kv_tokens: int = 0
    blocks: int = 0
    ttft: float | None = None
    finish_t: float | None = None

    def __post_init__(self):
        self.prefill_left = self.prompt_len


@dataclass
class SimConfig:
    name: str
    max_num_seqs: int
    max_num_batched_tokens: int
    block_size: int = 16
    prefill_ms_per_token: float = 0.04
    decode_base_ms: float = 8.0
    decode_ms_per_seq: float = 0.12


class BlockPool:
    def __init__(self, total_blocks: int, block_size: int = 16):
        self.total, self.free, self.block_size = total_blocks, total_blocks, block_size
        self.peak_used = 0

    def ensure(self, req: Req, new_tokens: int) -> bool:
        need = math.ceil((req.kv_tokens + new_tokens) / self.block_size)
        delta = need - req.blocks
        if delta <= 0:
            return True
        if delta > self.free:
            return False
        self.free -= delta
        req.blocks += delta
        self.peak_used = max(self.peak_used, self.total - self.free)
        return True

    def release(self, req: Req) -> None:
        self.free += req.blocks
        req.blocks = 0
        req.kv_tokens = 0


def schedule_step(cfg: SimConfig, pool: BlockPool, active: list, waiting: deque):
    budget, scheduled, preempted = cfg.max_num_batched_tokens, [], set()
    for r in list(active):
        n = 1 if r.prefill_left == 0 else min(r.prefill_left, budget)
        if budget <= 0:
            break
        if not pool.ensure(r, n):
            pool.release(r)
            r.prefill_left, r.num_generated = r.prompt_len, 0
            active.remove(r)
            waiting.appendleft(r)
            preempted.add(r.rid)
            continue
        budget -= n
        scheduled.append((r, n))
    while waiting and len(active) < cfg.max_num_seqs and budget > 0:
        r = waiting[0]
        if r.rid in preempted:
            break
        n = min(r.prefill_left, budget)
        if not pool.ensure(r, n):
            break
        waiting.popleft()
        active.append(r)
        budget -= n
        scheduled.append((r, n))
    return scheduled, preempted


def step_cost(cfg: SimConfig, scheduled: list):
    n_prefill = sum(n for r, n in scheduled if r.prefill_left > 0)
    n_decode = sum(1 for r, n in scheduled if r.prefill_left == 0)
    cost = n_prefill * cfg.prefill_ms_per_token
    if n_decode:
        cost += cfg.decode_base_ms + cfg.decode_ms_per_seq * n_decode
    return n_prefill, n_decode, cost


def simulate(cfg: SimConfig, workload: list[Req], total_blocks: int = 24000) -> dict:
    pool = BlockPool(total_blocks, cfg.block_size)
    arrivals = deque(sorted(workload, key=lambda r: r.arrival))
    waiting, active, done = deque(), [], []
    t, preempts = 0.0, 0
    for _ in range(20000):
        if not active and not waiting and arrivals:
            t = max(t, arrivals[0].arrival)
        while arrivals and arrivals[0].arrival <= t:
            waiting.append(arrivals.popleft())
        if not (active or waiting):
            break
        scheduled, preempted = schedule_step(cfg, pool, active, waiting)
        preempts += len(preempted)
        if not scheduled:
            break
        _, _, step_ms = step_cost(cfg, scheduled)
        for r, n in scheduled:
            r.kv_tokens += n
            if r.prefill_left > 0:
                r.prefill_left -= n
                if r.prefill_left == 0:
                    r.num_generated, r.ttft = 1, t + step_ms
            else:
                r.num_generated += 1
        for r in list(active):
            if r.num_generated >= r.max_new:
                r.finish_t = t + step_ms
                pool.release(r)
                active.remove(r)
                done.append(r)
        t += step_ms
    return {"finished": done, "makespan": t, "preemptions": preempts,
            "peak_blocks": pool.peak_used}


def make_workload(n: int = 80, rate: float = 40.0, seed: int = 0) -> list[Req]:
    rng = np.random.RandomState(seed)
    prompt = rng.randint(128, 1025, n)
    max_new = rng.randint(32, 129, n)
    arrival = np.cumsum(rng.exponential(1000.0 / rate, n))
    return [Req(i, int(prompt[i]), int(max_new[i]), float(arrival[i])) for i in range(n)]


def summarize(result: dict, label: str) -> dict:
    done = result["finished"]
    lat = np.array([r.finish_t - r.arrival for r in done])
    ttft = np.array([r.ttft for r in done])
    tokens = sum(r.max_new for r in done)
    return {"方案": label, "makespan_s": round(result["makespan"] / 1000, 2),
            "吞吐(tok/s)": round(tokens / result["makespan"] * 1000, 1),
            "TTFT_p50_ms": round(float(np.percentile(ttft, 50)), 1),
            "延迟_p99_ms": round(float(np.percentile(lat, 99)), 1),
            "抢占次数": result["preemptions"]}


# --------------------------------------------------------------------------- #
# 练习 1（⭐）：max_num_batched_tokens 8192 -> 2048
# --------------------------------------------------------------------------- #
def ex1_budget() -> dict:
    """预算从 8192 降到 2048：吞吐 -1.5%、TTFT p50 +4.3%。

    机制：token 预算只限制「一步最多处理多少 token」，不改变总工作量。
    预算变小 → prefill 被切成更多步，每步的固定 decode 开销被重复支付；
    同时 prefill 完成时刻被推后，所以 TTFT 变大。
    在长 prompt + 低并发的负载下差距会明显得多。
    """
    return {bud: summarize(simulate(SimConfig(f"256/{bud}", 256, bud), make_workload()), f"256/{bud}")
            for bud in (8192, 2048)}


# --------------------------------------------------------------------------- #
# 练习 2（⭐）：decode_ms_per_seq 0.12 -> 0.02（更快的卡）
# --------------------------------------------------------------------------- #
def ex2_faster_card() -> None:
    """每请求 decode 成本降 6 倍（0.12 -> 0.02 ms）：峰值吞吐 +18.6%（1634 -> 1938 tok/s），
    但**表观饱和点不动**——因为这份负载只有 80 个请求、且 prefill 偏重，128 并发时
    所有请求都已并行，吞吐被「总工作量」卡住，换卡只抬高上界。

    若把负载换成 decode 偏重（prompt≈48 / output≈384），同样改 b：
    峰值吞吐 3398 -> 4796 tok/s（+41%），改善明显得多。
    结论：拐点会不会移动，取决于【瓶颈是 batching 摊薄还是 prefill/负载规模】，
    不能只看参数名。想看真实的硬件拐点，需要持续到达流 + 超过 max_num_seqs 的请求数
    （第 11 章的开环压测）。
    """
    for seqs in (32, 64, 128, 256):
        cfg = SimConfig(f"fast-{seqs}", seqs, 8192, decode_ms_per_seq=0.02)
        print(f"  max_num_seqs={seqs:3d} -> {summarize(simulate(cfg, make_workload()), 'fast')}")


# --------------------------------------------------------------------------- #
# 练习 3（⭐⭐）：优先级队列（短 prompt 优先）
# --------------------------------------------------------------------------- #
def schedule_step_priority(cfg: SimConfig, pool: BlockPool, active: list, waiting: list):
    """waiting 从 deque 换成按 prompt_len 升序的 list；其余逻辑与第 09 章一致。

    预期：p50 延迟下降（短请求先走），p99 可能上升（长请求被饿死），
    吞吐基本不变——因为每一步的 token 预算总量没变，只是排队顺序变了。
    """
    scheduled, preempted = [], set()
    budget = cfg.max_num_batched_tokens
    for r in list(active):
        n = 1 if r.prefill_left == 0 else min(r.prefill_left, budget)
        if budget <= 0:
            break
        if not pool.ensure(r, n):
            pool.release(r)
            r.prefill_left, r.num_generated = r.prompt_len, 0
            active.remove(r)
            waiting.append(r)
            preempted.add(r.rid)
            continue
        budget -= n
        scheduled.append((r, n))
    waiting.sort(key=lambda r: r.prompt_len)
    while waiting and len(active) < cfg.max_num_seqs and budget > 0:
        r = waiting[0]
        if r.rid in preempted:
            break
        n = min(r.prefill_left, budget)
        if not pool.ensure(r, n):
            break
        waiting.pop(0)
        active.append(r)
        budget -= n
        scheduled.append((r, n))
    return scheduled, preempted


# --------------------------------------------------------------------------- #
# 练习 4（⭐⭐）：swap 式抢占
# --------------------------------------------------------------------------- #
def swap_preempt_design() -> str:
    """把 recompute 换成 swap 需要改三处：

    1. `Req` 增加 `swapped_blocks: list[int]`，记录被换出的物理块号；
    2. 抢占时不再丢弃 KV，而是 `pool.release(r)` 后把块号写进 `swapped_blocks`，
       并把 `kv_tokens` 冻结（不要清零）；
    3. 重新调度时先检查 `swapped_blocks` 是否能换回（块池够就换回、否则继续等待）。

    代价对比（用第 09 章的数字估算）：
    - recompute：抢占一次重算整段 prefill，成本 = prompt_len × 0.04ms；
      在本 workload（均值 608 token）约 24ms/次，21 次抢占 ≈ 0.5s；
    - swap：成本 = 2 × KV 字节 / 显存带宽。8B fp16 一个 2048 token 的请求
      ≈ 288MB，A100 2TB/s 下换出+换入 ≈ 0.29ms（理想值，实际受碎片影响）。
    结论：本 workload 下 swap 便宜一个数量级；但 swap 把成本转移到显存与
    PCIe/NVLink 带宽上，且实现复杂（需要预留 swap 空间），vLLM 的不同版本
    对两种模式的默认选择不同（以 0.28.x 的 `PreemptionMode` 为准，待核对）。
    """
    return "见 docstring"


# --------------------------------------------------------------------------- #
# 练习 5（⭐⭐⭐）：首 token 慢、TPOT 正常 —— 读码路线
# --------------------------------------------------------------------------- #
READING_ROUTE = """\
评分要点（4 条）：
1) 先排除客户端/网络：用 curl -w 看 time_connect/time_starttransfer，对比服务端日志里的
   TTFT；如果服务端 TTFT 正常而客户端慢 → 不是引擎问题。
2) 定位到 prefill/调度层：现象是 TTFT 慢而 TPOT 正常，说明 decode 正常、prefill 或排队慢。
   在 `vllm/v1/core/sched/scheduler.py::Scheduler.schedule()` 里打印每一步的
   num_scheduled_tokens（prefill 占比）与 waiting/running 长度。
3) 区分三种原因：
   - KV 不足：看 `vllm:kv_cache_usage_perc` 是否 >0.9、日志是否有 Preemption；
   - 排队：`vllm:num_requests_waiting` 持续 >0 且 `running` 已达 max_num_seqs；
   - prefill 本身慢：`vllm:prompt_tokens_total` 的增速 vs GPU SM/DRAM 利用率，
     若 SM 高 → 算力 bound；DRAM 高 SM 低 → 带宽 bound。
4) 验证与回滚：改一个参数（如 max-num-batched-tokens 或 gpu-memory-utilization），
   用第 11 章的并发扫描复测 p50/p99 TTFT，并保留旧配置一键回滚。
预期证据：修复前后同一并发下的 TTFT p50/p99 对比 + metrics 截图 + 一句因果解释。
"""


if __name__ == "__main__":
    print("== 练习 1：token 预算 8192 vs 2048 ==")
    for row in ex1_budget().values():
        print(" ", row)
    print("\n== 练习 2：decode_ms_per_seq=0.02 ==")
    ex2_faster_card()
    print("\n== 练习 3：优先级队列 ==")
    print("  把 waiting 从 deque 换成按 prompt_len 排序的 list，见 schedule_step_priority()")
    print("  预期：p50 下降、p99 上升、吞吐不变（预算总量没变）")
    print("\n== 练习 4：swap 抢占 ==")
    print(swap_preempt_design())
    print("\n== 练习 5：读码路线 ==")
    print(READING_ROUTE)
