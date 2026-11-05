#!/usr/bin/env python
"""第 04 章 · 练习参考答案与解题提示。

对应 `notebooks/04_prefix_caching.py` 的五道练习。前四题给可运行的参考实现，
第五题给开放设计的评分要点。脚本可以直接运行：

    .venv/bin/python solutions/04_exercises.py
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict

BS = 16


# --------------------------------------------------------------------------- #
# 练习 1（⭐）：block_size 对命中长度的影响
# --------------------------------------------------------------------------- #
def exercise_1():
    """只有「完整 block」能命中；block 越大，前缀尾部丢掉的 token 越多。

    命中 token 数 = floor(P / block_size) * block_size；理论 TTFT 加速比 = (P + Q) / (P - hit + Q)。
    当 P 是 block_size 的整数倍时三者相同；否则大 block 会丢更多尾部 token。
    """
    print(f"  {'P':>6} {'block_size':>10} {'命中 tokens':>12} {'丢失 tokens':>12} {'理论加速比':>10}")
    for P in [512, 520, 530, 1024]:
        for block_size in [16, 32, 64]:
            hit = (P // block_size) * block_size
            q = 8
            speedup = (P + q) / (P - hit + q)
            print(f"  {P:6d} {block_size:10d} {hit:12d} {P - hit:12d} {speedup:10.2f}")
    print("  结论：P 越大、block_size 越小，尾部损失占比越小；工业默认 16 是权衡后的折中。")


# --------------------------------------------------------------------------- #
# 练习 2（⭐）：问题变长后，命中收益按 (P+Q)/Q 缩小
# --------------------------------------------------------------------------- #
def exercise_2():
    """命中只省下 P 个 token 的 prefill，Q 个新 token 无论如何都要算。"""
    print(f"  {'P':>6} {'Q':>5} {'理论加速比':>10}")
    for P in [512, 1024, 2048]:
        for q in [8, 32, 128, 512]:
            print(f"  {P:6d} {q:5d} {(P + q) / q:10.2f}")
    print("  结论：Q 接近 P 时前缀缓存几乎不赚；这也是为什么 system prompt 越长、用户问题越短，")
    print("        前缀缓存的 TTFT 收益越大。实测值总会低于理论值（固定开销 + gather + hash）。")


# --------------------------------------------------------------------------- #
# 练习 3（⭐⭐）：实现 LRU 驱逐并断言不变量
# --------------------------------------------------------------------------- #
def simulate_lru(requests, capacity):
    """requests: 每个元素是「块 hash 列表」（一个请求的前缀链）。

    返回 (block 命中率, 整段命中率)。LRU 顺序用 OrderedDict 维护；请求进行中持有引用
    （ref_count > 0）的 block 不允许被驱逐——这是「共享块不能被正在用的请求回收」的模型。
    """
    cached = set()
    order = OrderedDict()
    ref_count = {}
    hit_blocks = total_blocks = full_hits = 0
    for chain in requests:
        # 1) 命中：从 block 0 开始，遇到第一个 miss 停止
        n_hit = 0
        for h in chain:
            if h not in cached:
                break
            n_hit += 1
        hit_blocks += n_hit
        total_blocks += len(chain)
        full_hits += n_hit == len(chain)

        # 2) 请求持有命中块与新写入块（ref +1）
        for h in chain[:n_hit]:
            order.move_to_end(h)
        for h in chain:
            if h not in cached:
                cached.add(h)
                order[h] = None
                ref_count[h] = 0
            ref_count[h] += 1

        # 3) 请求结束，释放引用（此时所有块 ref_count == 0）
        for h in chain:
            ref_count[h] -= 1

        # 4) 按 LRU 驱逐到容量以内；断言：被驱逐的块没有活跃引用
        while len(cached) > capacity:
            victim = next(iter(order))
            assert ref_count[victim] == 0, "被驱逐的 block 仍有活跃引用"
            del order[victim]
            cached.discard(victim)
            ref_count.pop(victim)
        assert len(cached) <= capacity
    return hit_blocks / total_blocks, full_hits / len(requests)


def exercise_3():
    """LRU 的关键不变量：容量不超限、被驱逐 block 在当时没有活跃引用。"""
    chain_of = {
        P: [bytes([P % 251, j]) for j in range(P // BS)] for P in [256, 512, 1024, 2048]
    }
    requests = []
    for i in range(200):
        P = 512 if i % 3 else (1024 if i % 2 else 2048)
        requests.append(chain_of[P])
    print(f"  {'容量(block)':>10} {'block 命中率':>12} {'整段命中率':>10}")
    for cap in [64, 128, 192, 256, 512]:
        block_rate, full_rate = simulate_lru(requests, cap)
        print(f"  {cap:10d} {block_rate:12.3f} {full_rate:10.3f}")
    print("  LRU 在热点前缀上明显优于 FIFO（FIFO 会按插入顺序淘汰热点块）。")


# --------------------------------------------------------------------------- #
# 练习 4（⭐⭐）：32-bit hash 的碰撞概率
# --------------------------------------------------------------------------- #
def exercise_4():
    """生日界：n 个 hash 发生碰撞的概率 ≈ 1 - exp(-n^2 / (2 * 2^bits))。

    32-bit：约 7.7 万个 block 就有 50% 碰撞概率；64-bit：约 5.1e9 个。
    一个 block 16 token，32-bit 下 7.7 万 block ≈ 120 万 token 的缓存，生产上很容易撞上。
    """
    for bits in [32, 64]:
        n_50 = math.sqrt(2 * (2**bits) * math.log(2))
        print(f"  {bits:2d}-bit: 50% 碰撞概率需要约 {n_50:,.0f} 个不同 block")
    n_blocks = 100_000
    for bits in [32, 64]:
        p_collision = 1 - math.exp(-(n_blocks**2) / (2 * 2**bits))
        print(f"  {bits:2d}-bit, {n_blocks:,} blocks: 碰撞概率 ≈ {p_collision:.3e}")
    print("  结论：vLLM 用 64-bit hash（+ 可选 sha256）可以把碰撞压到可忽略；")
    print("        教学版仍保留 token 内容二次校验，因为碰撞的后果是「返回错误结果」，不可接受。")


# --------------------------------------------------------------------------- #
# 练习 5（⭐⭐⭐）：多轮对话 + 工具调用的前缀缓存命中率设计
# --------------------------------------------------------------------------- #
def exercise_5():
    """评分要点（开放设计，不要求代码）。"""
    print("  参考要点：")
    print("  1) 前缀 = system prompt + 历史轮次 + 工具描述；只有完全一致且连续的 block 能命中。")
    print("  2) 工具调用会插入新内容，导致后续 block hash 全部失效——命中率应按「回合」建模。")
    print("  3) 长会话的 KV 会被不断追加，缓存容量与驱逐策略决定「回滚重问」的成本。")
    print("  4) 多轮采样（同会话多个分支）需要 copy-on-write 或按分支保留 block。")
    print("  5) 评分看是否把命中率换算成 TTFT/成本收益，而不是只谈算法。")


def main():
    print("练习 1：block_size 对命中长度的影响")
    exercise_1()
    print("\n练习 2：问题长度对收益的稀释")
    exercise_2()
    print("\n练习 3：LRU 驱逐与不变量")
    exercise_3()
    print("\n练习 4：hash 碰撞概率")
    exercise_4()
    print("\n练习 5：开放设计")
    exercise_5()


if __name__ == "__main__":
    main()
