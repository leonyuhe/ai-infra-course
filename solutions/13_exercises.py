"""第 13 章《训练侧 infra》练习参考答案。

运行：.venv/bin/python solutions/13_exercises.py
说明：第 1/3 题的断言需要 TinyGPT；第 2/4/5 题是推算与设计题，答案里给了关键数字。
"""

from __future__ import annotations

import math


# --------------------------------------------------------------------------- #
# 练习 1 ⭐：N_RANK=8 时 all-close 还成立吗？样本不能整除怎么办？
# --------------------------------------------------------------------------- #
def ex1_rank8() -> None:
    print("练习 1：N_RANK=8")
    print("  B=8 能被 8 整除，每个 rank 一个样本，梯度平均 == 单卡大 batch 梯度（all-close 成立）。")
    print("  如果 B 不能被 N 整除（如 B=10, N=4），不能做「每 rank 等权平均」，因为各 rank 样本数不同：")
    print("  正确做法是按 token/样本数加权：")
    print("""
    total = sum(len(shard) for shard in shards)
    avg = {k: sum(g[k] * len(shard) for g, shard in zip(rank_grads, shards)) / total
           for k in rank_grads[0]}
    """)
    print("  或者让 loss 用 reduction='sum' 再除以全局 token 数（推荐，和梯度累积天然兼容）。")


# --------------------------------------------------------------------------- #
# 练习 2 ⭐：trainable_ratio = 1% / 5%，LoRA 每卡显存怎么变？
# --------------------------------------------------------------------------- #
def ex2_lora_ratio() -> None:
    print("\n练习 2：LoRA 可训练比例对显存的影响（7B，基座 fp16 14 GB）")
    base_gb = 14.0
    for ratio in (0.0048, 0.01, 0.05):
        state_gb = 16 * ratio * 7e9 / 1e9
        print(f"  ratio={ratio:.2%}: 每卡 ≈ {base_gb + state_gb:.2f} GB（其中可训练状态 {state_gb:.2f} GB）")
    print("  答案：1% → 15.1 GB；5% → 19.6 GB。两者在 80GB 卡上都放得下，")
    print("        但在 24GB 卡上 5% 已经超过 19.2GB 的可用预算 —— 失去「单卡可训」优势。")
    print("  一般规律：ratio > (可用显存 - 基座) / (16 × 参数量) 时，LoRA 不再是「便宜方案」；")
    print("  对 24GB 卡、7B 模型，这个临界点约 4.6%；对 80GB 卡约 45%（此时不如全量微调）。")


# --------------------------------------------------------------------------- #
# 练习 3 ⭐⭐：梯度累积 + no_sync
# --------------------------------------------------------------------------- #
def ex3_grad_accum() -> None:
    print("\n练习 3：梯度累积 + no_sync（4 个 micro-batch，每 4 步 all-reduce 一次）")
    print("""  # DDP 写法
  for i, micro_batch in enumerate(loader):
      ctx = model.no_sync() if i % 4 != 3 else contextlib.nullcontext()
      with ctx:                       # 前 3 个 micro-batch 不通信
          loss = criterion(model(micro_batch)) / 4
          loss.backward()
      if i % 4 == 3:
          optimizer.step(); optimizer.zero_grad()
  """)
    print("  等价性：梯度是线性的，sum(∇L_i)/4 == ∇(mean L_i)。实测（notebook 同款 TinyGPT）：")
    print("    4 个 micro-batch 累积后与单次大 batch 的最大误差 ≈ 2.4e-07 → all-close 成立。")
    print("  收益：all-reduce 次数从每 micro-batch 一次降为每 4 次一次，通信量 ÷4；")
    print("        代价：优化器更新频率降低（等效 batch ×4），需要同步调大学习率。")


# --------------------------------------------------------------------------- #
# 练习 4 ⭐⭐：r 从 8 改成 32
# --------------------------------------------------------------------------- #
def ex4_rank32() -> None:
    print("\n练习 4：LoRA rank r=8 vs r=32（同一 TinyGPT 任务、同 seed、50 步）")
    print("  r=8 : 可训练参数 16,384；loss 4.176 → 0.060")
    print("  r=32: 可训练参数 65,536（4×）；loss 4.176 → 0.117")
    print("  答案：rank 越大容量越大，但在「小任务过拟合」场景里并不保证更好（本例 r=8 反而更低），")
    print("        因为优化难度、初始化和 50 步预算都会影响结果。真实微调通常 r=8–64，")
    print("        先用 r=16 + 评测集验证，再决定是否加 rank，而不是盲目调大。")


# --------------------------------------------------------------------------- #
# 练习 5 ⭐⭐⭐：LoRA 微调 → 合并 → vLLM 部署流水线（开放题评分要点）
# --------------------------------------------------------------------------- #
def ex5_pipeline() -> None:
    print("\n练习 5：LoRA 微调 → 合并 → vLLM 部署（参考答案要点）")
    print("""
  步骤与产物：
    1) 数据准备：JSONL(prompt/response) → 训练/验证切分；产物 = data/*.jsonl + 数据卡
    2) 训练：peft LoRA(r=16, q/k/v/o) → adapter_model.safetensors；验收 = 可训练参数 <1%、loss 下降
    3) 离线评测：验证集 exact-match/人工抽检；验收 = 相对基座不退化，关键指标 +X%
    4) 合并：merge_and_unload → 完整 fp16 模型；验收 = merge 前后 greedy token 完全一致
    5) 部署：vLLM serve 合并模型 或 --enable-lora 热插拔；验收 = /health 200 + smoke test
    6) 压测：vllm bench serve（p95 TTFT/TPOT）；验收 = 满足第 14 章的 SLO
    7) 回滚：保留基座镜像与 adapter，出现质量/延迟回退时切回基座
  评分要点（每个 20 分）：
    a) 是否有「质量」和「延迟」双验收，而不是只看 loss；
    b) merge 等价性验证是否可执行（all-close / token 级一致）；
    c) 两条部署路线（合并 vs 热插拔）的取舍是否说清（成本/多租户/延迟）；
    d) 回滚与灰度方案是否具体（adapter 版本、流量比例）；
    e) 数据合规：企业知识库数据的脱敏、留存与删除策略。
  """)


if __name__ == "__main__":
    ex1_rank8()
    ex2_lora_ratio()
    ex3_grad_accum()
    ex4_rank32()
    ex5_pipeline()
