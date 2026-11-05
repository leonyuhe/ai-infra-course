"""第 12 章《多卡并行》练习参考答案。

运行：.venv/bin/python solutions/12_exercises.py
说明：答案里的数字与 notebook 12.4 实验 C 的模型同源（8B、fp16、A100 80G、decode batch=1），
      换机器/换模型时重算即可；结论（趋势）比绝对值重要。
"""

from __future__ import annotations

HIDDEN = 4096
N_LAYER = 32
PARAMS = 8e9
HBM_BW_GBPS = 2000.0
COMPUTE_TFLOPS = 150.0
LINKS = {
    "NVLink4": (900.0, 2.0),   # (带宽 GB/s, 每次 collective 固定延迟 µs)
    "PCIe4": (64.0, 5.0),
    "Ethernet25": (25.0, 15.0),
}


def comm_bytes_per_token(tp: int, dtype_bytes: int = 2) -> float:
    """每层 2 次 ring all-reduce，每次搬运 2(n-1)/n × 消息大小。"""
    if tp == 1:
        return 0.0
    return N_LAYER * 2 * 2 * (tp - 1) / tp * HIDDEN * dtype_bytes


def decode_efficiency(tp: int, link: str, dtype_bytes: int = 2) -> dict:
    """返回单 token 的扩展效率与耗时分解（decode 访存受限模型）。"""
    bw, latency = LINKS[link]
    t1_ms = PARAMS * dtype_bytes / (HBM_BW_GBPS * 1e9) * 1e3
    comm_bytes = comm_bytes_per_token(tp, dtype_bytes)
    n_coll = 0 if tp == 1 else N_LAYER * 2
    comm_ms = comm_bytes / (bw * 1e9) * 1e3 + n_coll * latency / 1e3
    mem_ms = t1_ms / tp
    return {"tp": tp, "link": link, "comm_MB": comm_bytes / 1e6,
            "memory_ms": mem_ms, "comm_ms": comm_ms,
            "scaling_eff": t1_ms / (mem_ms + comm_ms) / tp}


# --------------------------------------------------------------------------- #
# 练习 1 ⭐：把 PCIe4 换成 PCIe5（128 GB/s，4 µs），TP=8 的效率提升多少？
# --------------------------------------------------------------------------- #
def ex1_pcie_gen5() -> None:
    print("练习 1：PCIe Gen5")
    LINKS["PCIe5"] = (128.0, 4.0)
    for link in ("PCIe4", "PCIe5"):
        r = decode_efficiency(8, link)
        print(f"  {link}: comm={r['comm_ms']:.3f} ms, eff={r['scaling_eff']:.3f}")
    print("  答案：0.749 → 约 0.79，只提升约 4 个百分点。")
    print("  原因：TP=8 的通信时间里 64 次 collective 的固定延迟约占 96%，")
    print("        带宽从 64→128 GB/s 只压缩了那 4% 的传输部分。")
    LINKS.pop("PCIe5")


# --------------------------------------------------------------------------- #
# 练习 2 ⭐：dtype_bytes=1（FP8/int8 权重），通信量与效率怎么变？
# --------------------------------------------------------------------------- #
def ex2_dtype_bytes() -> None:
    print("\n练习 2：权重 fp16 → int8")
    for dtype_bytes in (2, 1):
        r = decode_efficiency(8, "NVLink4", dtype_bytes)
        print(f"  dtype={dtype_bytes}B: 每 token 通信 {r['comm_MB']:.3f} MB, "
              f"单卡访存 {r['memory_ms']:.2f} ms, eff={r['scaling_eff']:.3f}")
    print("  答案：绝对延迟从 8 ms 降到约 4.6 ms（访存减半），但 TP 扩展效率")
    print("        反而从 0.886 降到约 0.795 —— 因为计算/访存时间减半后，")
    print("        固定延迟在账本里的占比翻倍。量化买的是「绝对速度」，不是「并行效率」。")
    print("        例外：如果同时把 collective 次数减半（融合 QKV + MLP），效率会回升。")


# --------------------------------------------------------------------------- #
# 练习 3 ⭐⭐：朴素 all-reduce（rank0 收集 + 广播）与 ring 的对比
# --------------------------------------------------------------------------- #
def ex3_naive_allreduce() -> None:
    print("\n练习 3：朴素 all-reduce vs ring")
    print("  朴素版伪代码（线程模拟时把 qs 换成 queue.Queue 即可）：")
    print("""
    def naive_allreduce(tensors):
        n = len(tensors)
        # 阶段 1：所有 rank 把完整向量发给 rank0，rank0 累加
        for r in range(1, n):
            rank0_handle(qs[r][0].get())          # rank r -> rank0，消息大小 S
        # 阶段 2：rank0 把结果广播给所有 rank
        for r in range(1, n):
            qs[0][r].put(rank0_result)            # rank0 -> rank r，消息大小 S
    """)
    for n in (2, 4, 8):
        naive_per_rank = 2 * (n - 1) * 1.0   # 以消息大小 S 为 1
        ring_per_rank = 2 * (n - 1) / n
        print(f"  TP={n}: 朴素版每 rank 搬 {naive_per_rank:.2f}S, ring 搬 {ring_per_rank:.2f}S, "
              f"朴素版多 {naive_per_rank / ring_per_rank:.1f}×")
    print("  答案：朴素版每 rank 是 ring 的 n 倍（TP=8 时 8×）；而且 rank0 成为单点瓶颈，")
    print("        无法和计算重叠。ring 的每 rank 搬运量随 n 增大收敛到 2S，才是可扩展的。")
    print("        线程实测的绝对毫秒数只反映 Python/GIL 开销，不能当链路带宽用。")


# --------------------------------------------------------------------------- #
# 练习 4 ⭐⭐：25 GB/s 以太网上 DP=8 的通信/计算比
# --------------------------------------------------------------------------- #
def ex4_dp_ratio() -> None:
    print("\n练习 4：跨机 DP 的通信/计算比")
    global_batch, seq_len, dp = 64, 2048, 8
    tokens = global_batch * seq_len
    flops = 6 * PARAMS * tokens          # 训练一步约 6·P·tokens FLOPs
    compute_s = flops / (COMPUTE_TFLOPS * 1e12)
    grad_bytes = 2 * (dp - 1) / dp * PARAMS * 4   # fp32 梯度 ring all-reduce
    comm_s = grad_bytes / (25e9)
    print(f"  每步全局 token = {tokens:,}; 计算 = {flops / 1e15:.1f} PFLOPs → "
          f"{compute_s:.1f} s（按 {COMPUTE_TFLOPS:.0f} TFLOP/s）")
    print(f"  DP=8 梯度 all-reduce = {grad_bytes / 1e9:.1f} GB → {comm_s:.2f} s（25 GB/s）")
    print(f"  通信/计算 = {comm_s / compute_s:.1%}")
    print("  答案：约 5%，且 DDP 可以在反向传播中分桶重叠 → 跨机 DP 完全可行。")
    print("        对比 TP=8 的通信/计算比 > 1（必须同步等 all-reduce），所以 TP 不跨机。")


# --------------------------------------------------------------------------- #
# 练习 5 ⭐⭐⭐：64 卡训练 70B 的并行方案（开放题，给评分要点）
# --------------------------------------------------------------------------- #
def ex5_64gpu_design() -> None:
    print("\n练习 5：64 卡训练 70B（开放题参考答案要点）")
    params = 70e9
    total_state_gb = params * 16 / 1e9
    print(f"  模型状态总量 = 70B × 16B = {total_state_gb:.0f} GB；64 卡摊平 = "
          f"{total_state_gb / 64:.1f} GB/卡（不含激活）。")
    print("""
  方案（示例）：TP=8（机内 NVLink）× PP=2（跨机 IB）× DP=4，共 64 卡；
  优化器状态用 ZeRO-1/2 在 DP 组内再切（若用 FSDP/ZeRO-3 则全 64 卡切分）。
  - 显存：参数 140 GB / (TP×PP) = 4.4 GB/卡；梯度同量级；优化器 840 GB 若只在 DP=4
    组内切 → 210 GB/卡，放不下 —— 所以必须 ZeRO-3/FSDP 全切，或改成 DP=8、PP=1。
  - 通信：TP 每层 2 次 all-reduce（只走 NVLink）；PP 只在边界传激活（IB）；
    DP 每步一次梯度 all-reduce，必须与反向重叠。
  - bubble：PP=2、microbatch=16 → (2-1)/(16+2-1) ≈ 5.9%。
  - 预期 MFU：40%–50%（A100/H100 混合精度，受限于 TP 通信与 bubble）。
  - 故障域：PP 让单卡故障影响整个 stage；需要 checkpoint + 弹性训练。
  评分要点（每题 20 分）：
    1) 显存账本是否把参数/梯度/优化器/激活分开算，并给出可行卡数；
    2) TP/PP/DP 的取值是否与互联拓扑匹配（TP 不跨机、PP 通信小、DP 可跨机）；
    3) 是否给出 bubble 与 MFU 的量化预期；
    4) 是否考虑故障域、checkpoint 与弹性；
    5) 是否说明「64 卡训 70B 的性价比」是否值得（对比租 128–256 卡）。
  """)


if __name__ == "__main__":
    ex1_pcie_gen5()
    ex2_dtype_bytes()
    ex3_naive_allreduce()
    ex4_dp_ratio()
    ex5_64gpu_design()
