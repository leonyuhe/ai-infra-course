# %% [markdown]
# # 第 13 章 · 训练侧 infra：DDP、ZeRO 与 LoRA
#
# > 训练一个 7B 模型，每张卡要背的不只是 14 GB 权重，而是 112 GB 的「模型状态」；先把账算清，再谈并行。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 80 分钟 |
# | 运行环境 | ☁️ 云上 NVIDIA GPU（本地 Mac 用 TinyGPT + 线程模拟，全程可跑） |
# | 前置章节 | 12（多卡并行） |
# | 本章接口 | `aiinfra.tinygpt.TinyGPT`、`aiinfra.bench.pick_device` |
# | 参考实现 | PyTorch DDP / FSDP、DeepSpeed ZeRO、PEFT LoRA、vLLM 0.28.x |
#
# **学完你能回答：**
# 1. 15 亿参数的模型，为什么训练时每个参数要按 16 字节算？这 16 字节分别是什么？
# 2. ZeRO-1/2/3 各切走了账本里的哪几项？同样 7B 模型，每卡显存从 112 GB 降到多少？
# 3. LoRA 为什么能把可训练参数降到 0.5% 以下？训练完怎么合并、怎么给 vLLM 用？
#
# **本章的实验方式**：DDP 的梯度同步用线程模拟（4 个虚拟 rank 的数据切片 + all-reduce 平均），
# 并与「单卡大 batch」的梯度做 all-close 断言；ZeRO 只做纯 Python 的显存账本推算；
# LoRA 则在 TinyGPT 上真实训练 50 步、真实合并权重并验证前后输出一致。云上脚本见
# `cloud/scripts/13_lora_finetune.sh`。

# %% [markdown]
# ## 13.1 问题与直觉
#
# 推理时一个参数只占 2 字节（fp16），训练时却要按 16 字节准备预算：
#
# ![](../assets/figures/ch13_zero_ledger.svg)
#
# | 账本项 | 精度 | 字节/参数 | 7B 模型合计 |
# | --- | --- | --- | --- |
# | 参数（weights） | fp16 | 2 | 14 GB |
# | 梯度（gradients） | fp16 | 2 | 14 GB |
# | fp32 master weights | fp32 | 4 | 28 GB |
# | Adam 一阶矩 m | fp32 | 4 | 28 GB |
# | Adam 二阶矩 v | fp32 | 4 | 28 GB |
# | **合计** | | **16** | **112 GB** |
#
# 为什么需要 fp32 master weights？因为 fp16 的动态范围窄，直接累积小梯度会「加不动」（精度损失）；
# 混合精度训练的标准做法是：前向/反向用 fp16，更新时在 fp32 副本上做，再转回 fp16。
# 这 16 字节是**训练侧 infra 的第一性事实**，ZeRO、FSDP、LoRA 全都围绕它做文章。
#
# 一个反例：7B 模型 + 8 张 A100 80GB 直接上 DDP。表上 112 GB > 80 GB，**单卡第一步就 OOM**——
# 加卡也没用，因为 DDP 是复制模型，每张卡都要完整背 112 GB。这不是「卡不够多」，是「并行方式错了」。

# %% [markdown]
# ## 13.2 从零实现
#
# ### 1) DDP：手写梯度 all-reduce，并证明它等于大 batch 梯度
#
# DDP 的数学契约很简单：N 个 rank 各算自己那份数据的梯度，然后**逐元素求平均**，
# 结果必须等于「把 N 份数据拼成一个大 batch 算出的梯度」。我们先把这件事验证掉，
# 因为工业实现（梯度 bucket、通信重叠、fp16 压缩）都只是在这个契约上做优化。

# %%
import sys
import threading
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path.cwd().parent))

from aiinfra.bench import pick_device
from aiinfra.plotting import save, setup_style
from aiinfra.tinygpt import GPTConfig, TinyGPT

setup_style()
device = pick_device()
torch.manual_seed(0)
np.random.seed(0)
print("device =", device, "| tokenizer 与数据全部本地随机生成，不联网")

# %%
cfg = GPTConfig(vocab_size=64, n_layer=2, n_head=4, d_model=64, d_ff=128, max_seq_len=64)
model = TinyGPT(cfg).to(device)
B, T, N_RANK = 8, 24, 4
ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)
target = ids.roll(-1, dims=1)  # next-token 预测


def token_loss(m: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(m(x).reshape(-1, cfg.vocab_size), y.reshape(-1))


def grads(m: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    m.zero_grad(set_to_none=True)
    token_loss(m, x, y).backward()
    return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}


g_full = grads(model, ids, target)  # 单卡、大 batch 的“真值”
model.zero_grad(set_to_none=True)
print(f"模型参数 = {model.num_params():,}；全局 batch = {B}，切成 {N_RANK} 份")

# %%
def threaded_allreduce_average(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """线程模拟 DDP 的梯度 all-reduce(AVG)：每个 rank 把梯度加进共享缓冲区，最后除以 N。"""
    n = len(parts)
    out = {k: torch.zeros_like(v) for k, v in parts[0].items()}
    lock = threading.Lock()
    barrier = threading.Barrier(n)

    def worker(r: int) -> None:
        barrier.wait()
        with lock:
            for k, v in parts[r].items():
                out[k] += v

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {k: v / n for k, v in out.items()}


per_rank = [grads(model, x, y) for x, y in zip(torch.chunk(ids, N_RANK), torch.chunk(target, N_RANK))]
model.zero_grad(set_to_none=True)
avg = threaded_allreduce_average(per_rank)

err = max((avg[k] - g_full[k]).abs().max().item() for k in g_full)
unsynced = max((per_rank[0][k] - per_rank[1][k]).abs().max().item() for k in g_full)
assert all(torch.allclose(avg[k], g_full[k], atol=1e-5, rtol=1e-4) for k in g_full), "DDP 梯度与单卡大 batch 不一致"
assert unsynced > 1e-3, "不同 rank 的梯度本应不同，否则数据切片没生效"
print(f"✅ 4 个 rank 梯度平均 == 单卡大 batch 梯度，最大误差 {err:.2e}")
print(f"   不同步时 rank0 与 rank1 的梯度最大相差 {unsynced:.3f} —— 这就是必须 all-reduce 的原因")

# %% [markdown]
# 真 DDP 不会「等全部算完再一次性 all-reduce」，而是把梯度按 bucket（默认 25 MB 左右）分桶，
# 反向传播算完一个桶就立刻发一次，通信和下一个桶的计算重叠。数学不变，只是把
# 「一次大通信」拆成「多次小通信 + 流水」。这个设计在 13.4 会对照 `torch.nn.parallel.DistributedDataParallel`。

# %% [markdown]
# ### 2) ZeRO 显存账本：16 字节怎么被切走
#
# 先把账本写成一个函数。三阶段的区别只有一句话：
#
# - **ZeRO-1**：只切「优化器状态」（master + m + v，共 12 B/参数）；
# - **ZeRO-2**：ZeRO-1 + 再切梯度（2 B/参数）；
# - **ZeRO-3**：ZeRO-2 + 连参数也切（2 B/参数）。
#
# 每一项都是「总字节 ÷ N」，但**没被切的项仍然每卡一份**，这就是为什么 DDP 的 112 GB
# 在 ZeRO-1 下只降到 38.5 GB，而不是 112/8 = 14 GB。

# %%
def zero_ledger(params: float, n_gpu: int, stage: int, trainable_ratio: float = 1.0) -> dict[str, float]:
    """返回每卡显存（GB）。stage: 0=DDP, 1/2/3=ZeRO 对应阶段。"""
    train = params * trainable_ratio
    frozen = params - train
    param_b = 2 * frozen + 2 * train  # 冻结基座也占显存（LoRA 场景）
    grad_b = 2 * train
    opt_b = 12 * train  # fp32 master 4 + Adam m 4 + Adam v 4
    if stage == 0:  # DDP / LoRA：每卡一份完整状态
        pass
    elif stage == 1:
        opt_b /= n_gpu
    elif stage == 2:
        grad_b /= n_gpu
        opt_b /= n_gpu
    elif stage == 3:
        param_b = 2 * frozen + 2 * train / n_gpu
        grad_b /= n_gpu
        opt_b /= n_gpu
    else:
        raise ValueError(f"未知 stage: {stage}")
    return {
        "参数(GB)": param_b / 1e9,
        "梯度(GB)": grad_b / 1e9,
        "优化器状态(GB)": opt_b / 1e9,
        "每卡合计(GB)": (param_b + grad_b + opt_b) / 1e9,
    }


P_7B, N_GPU = 7e9, 8
rows = []
for name, stage in [("DDP", 0), ("ZeRO-1", 1), ("ZeRO-2", 2), ("ZeRO-3", 3)]:
    rows.append({"方案": name, **zero_ledger(P_7B, N_GPU, stage)})
rows.append({"方案": "LoRA(0.48%)", **zero_ledger(P_7B, N_GPU, 0, trainable_ratio=0.0048)})
ledger_df = pd.DataFrame(rows).round(2)
totals = dict(zip(ledger_df["方案"], ledger_df["每卡合计(GB)"]))
assert totals["DDP"] > totals["ZeRO-1"] > totals["ZeRO-2"] > totals["ZeRO-3"] == 112 / N_GPU
assert totals["LoRA(0.48%)"] < totals["DDP"] * 0.2
ledger_df

# %% [markdown]
# 注意 LoRA 这一行：**基座 14 GB 冻结不动**，只有约 0.48% 的参数（3360 万）需要梯度与优化器状态，
# 所以每卡只多 0.54 GB。下面顺手回答「7B 模型用 80 GB / 24 GB 卡最少要几张」。

# %%
def min_gpus(stage: int, params: float = 7e9, gpu_gb: float = 80.0, util: float = 0.8, trainable_ratio: float = 1.0) -> int | None:
    """仅按「模型状态」估算最少卡数；util 留 20% 给激活值等开销。None = 这种并行方式无解。"""
    budget = gpu_gb * util
    train = params * trainable_ratio
    frozen = params - train
    if stage == 0:  # DDP / LoRA：每卡需要完整状态，加卡不省
        return 1 if (2 * frozen + 16 * train) / 1e9 <= budget else None
    if frozen and 2 * frozen / 1e9 >= budget and stage == 3:
        return None
    if stage == 3:  # 参数(含冻结基座)与 16B/参数 的可训练状态都除以 N
        if train == 0:
            return 1
        return int(np.ceil(16 * train / 1e9 / (budget - 2 * frozen / 1e9)))
    if stage == 1:  # 参数 + 梯度固定，优化器状态（12B/参数）除以 N
        fixed, sharded = 2 * (frozen + train) / 1e9, 12 * train / 1e9
    else:  # stage 2：参数固定，梯度 + 优化器状态（14B/参数）除以 N
        fixed, sharded = 2 * (frozen + train) / 1e9, 14 * train / 1e9
    if fixed >= budget:
        return None
    return int(np.ceil(sharded / (budget - fixed)))


rows = []
for gpu_gb in (80, 24):
    for name, stage in [("DDP", 0), ("ZeRO-1", 1), ("ZeRO-2", 2), ("ZeRO-3", 3)]:
        rows.append({"卡型": f"{gpu_gb} GB", "方案": name, "最少卡数（仅模型状态）": min_gpus(stage, gpu_gb=gpu_gb) or "无解"})
    rows.append({"卡型": f"{gpu_gb} GB", "方案": "LoRA(0.48%)", "最少卡数（仅模型状态）": min_gpus(0, gpu_gb=gpu_gb, trainable_ratio=0.0048)})
min_df = pd.DataFrame(rows)
assert min_gpus(3, gpu_gb=24) < min_gpus(2, gpu_gb=24), "24GB 卡上 ZeRO-3 应比 ZeRO-2 省卡"
min_df

# %% [markdown]
# ### 3) 从零实现 LoRA：A/B 两个小矩阵
#
# LoRA 的假设是：微调带来的权重变化 `ΔW` 是低秩的。于是冻结原权重 `W`，只训练
# `ΔW = B @ A`，其中 `A: (r, in)`、`B: (out, r)`，`r << min(in, out)`。
# 前向变成 `y = x Wᵀ + (x Aᵀ) Bᵀ × scaling`；`B` 初始化为 0，所以训练开始时
# `ΔW = 0`，模型行为与基座完全一致——这是 LoRA 能安全热插拔的关键。
#
# 两个实现细节：
#
# - TinyGPT 的 `lm_head.weight` 与 `tok_emb.weight` 是**共享**的。如果对共享权重做 LoRA 再合并，
#   改 lm_head 会同时改 token embedding，merge 前后输出就不等价了。所以本实验先把它解绑成
#   两份独立权重（真实 Llama/Qwen 的输入 embedding 与输出投影本来就不共享）。
# - 为了让随机初始化模型的 logits 量级正常（不爆成 60 的 loss），把初始参数整体乘 0.1，
#   相当于一个「还没训练好的小模型」。

# %%
class LoRALinear(nn.Module):
    """y = base(x) + (x Aᵀ) Bᵀ × alpha/r；merge 后直接把 B@A×scaling 加进 base.weight。"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r, self.scaling, self.merged = r, alpha / r, False
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.merged:
            out = out + (x @ self.A.T @ self.B.T) * self.scaling
        return out

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight += (self.B @ self.A) * self.scaling
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight -= (self.B @ self.A) * self.scaling
            self.merged = False

# %%
def apply_lora(model: nn.Module, r: int = 8, alpha: int = 16) -> int:
    """冻结全部参数，把目标 Linear 替换成 LoRALinear；返回替换个数。"""
    for p in model.parameters():
        p.requires_grad = False
    targets = ("qkv", "proj", "w1", "w2", "w3", "lm_head")
    count = 0

    def rec(parent: nn.Module) -> None:
        nonlocal count
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(parent, name, LoRALinear(child, r, alpha))
                count += 1
            else:
                rec(child)

    rec(model)
    return count

# %%
torch.manual_seed(0)
lora_model = TinyGPT(cfg).to(device)
lora_model.lm_head.weight = nn.Parameter(lora_model.lm_head.weight.detach().clone())  # 解绑输出投影
with torch.no_grad():
    for p in lora_model.parameters():
        p.mul_(0.1)
n_replaced = apply_lora(lora_model, r=8, alpha=16)
lora_model = lora_model.to(device)

total_p = sum(p.numel() for p in lora_model.parameters())
train_p = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
trainable_names = [n for n, p in lora_model.named_parameters() if p.requires_grad]
assert len(trainable_names) == 2 * n_replaced, "每个 LoRA 层应只有 A、B 两个可训练参数"
assert all(n.endswith(".A") or n.endswith(".B") for n in trainable_names)
assert all(not p.requires_grad for n, p in lora_model.named_parameters() if n not in trainable_names)
print(f"替换 {n_replaced} 个 Linear；可训练参数 {train_p:,} / {total_p:,} = {train_p / total_p:.2%}")
pd.DataFrame(
    [{"项目": "总参数", "数量": total_p}, {"项目": "LoRA 可训练参数", "数量": train_p},
     {"项目": "占比", "数量": f"{train_p / total_p:.2%}"}]
)

# %% [markdown]
# ## 13.3 实测：训练 50 步、冻结验证与 merge 等价性
#
# 任务：固定 batch 的「随机 next-token 预测」——本质是让小任务过拟合，用来观察
# 「只更新 LoRA 参数」是否真的能让 loss 下降。这不是有意义的语言建模，但足够验证闭环。
# 每 10 步记录一次 loss。

# %%
torch.manual_seed(0)
ids2 = torch.randint(0, cfg.vocab_size, (8, 16), device=device)
target2 = torch.randint(0, cfg.vocab_size, (8, 16), device=device)
base_snapshot = {n: p.detach().clone() for n, p in lora_model.named_parameters() if not p.requires_grad}

opt = torch.optim.AdamW([p for p in lora_model.parameters() if p.requires_grad], lr=3e-2)
losses = []
for step in range(50):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(lora_model(ids2).reshape(-1, cfg.vocab_size), target2.reshape(-1))
    loss.backward()
    opt.step()
    losses.append(loss.item())
assert losses[-1] < losses[0] * 0.2, "50 步后 loss 应显著下降"

log_df = pd.DataFrame({"step": [0, 10, 20, 30, 40, 49], "loss": [round(losses[i], 3) for i in [0, 10, 20, 30, 40, 49]]})
log_df

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
ax = axes[0]
ax.plot(range(1, 51), losses, color="#2563EB", lw=2)
ax.scatter([1, 50], [losses[0], losses[-1]], color=["#EF4444", "#10B981"], zorder=5, s=40)
ax.set_xlabel("Training step")
ax.set_ylabel("Cross-entropy loss")
ax.set_title("LoRA loss: 50 steps")

ax = axes[1]
schemes = ledger_df["方案"].tolist()
bottom = np.zeros(len(schemes))
for col, color, label in [("参数(GB)", "#2563EB", "params (GB)"), ("梯度(GB)", "#F59E0B", "grads (GB)"), ("优化器状态(GB)", "#7C3AED", "optimizer (GB)")]:
    ax.bar(schemes, ledger_df[col], bottom=bottom, color=color, label=label)
    bottom += ledger_df[col].to_numpy()
ax.axhline(80, color="#EF4444", ls="--", lw=1.5, label="A100 80GB")
ax.set_ylabel("Per-GPU memory (GB)")
ax.set_title("7B per-GPU ledger (8 GPUs)")
ax.tick_params(axis="x", rotation=20)
ax.legend(fontsize=8, loc="upper right")
save(fig, "ch13_zero_and_lora");

# %% [markdown]
# 读图：左图 loss 从约 4.18（≈ ln 64，随机猜）降到 0.06，说明 LoRA 参数确实在学习；
# 右图 DDP 的 112 GB 超过 80 GB 红线，ZeRO-3 的 14 GB 与 LoRA 的 14.5 GB 稳稳在线下。
# 注意右图是**模型状态**，不含激活值——真实训练还要给激活留空间，所以 14 GB 不能当成「一张卡够」。
#
# 接下来验证两件事：**只更新 LoRA 参数**（基座冻结），以及 **merge 前后输出一致**。

# %%
for name, p in lora_model.named_parameters():
    if name in base_snapshot:
        assert torch.equal(p.detach(), base_snapshot[name]), f"基座参数被改动了：{name}"
print("✅ 50 步训练后，所有基座参数逐位不变；梯度只流经 LoRA 的 A/B")

eval_ids = torch.randint(0, cfg.vocab_size, (4, 16), device=device)
with torch.no_grad():
    before = lora_model(eval_ids).clone()
for m in lora_model.modules():
    if isinstance(m, LoRALinear):
        m.merge()
with torch.no_grad():
    after = lora_model(eval_ids).clone()
merge_err = (before - after).abs().max().item()
assert torch.allclose(before, after, atol=1e-4, rtol=1e-4), "merge 前后输出不等价"
for m in lora_model.modules():
    if isinstance(m, LoRALinear):
        m.unmerge()
with torch.no_grad():
    restored = lora_model(eval_ids).clone()
assert torch.allclose(before, restored, atol=1e-4, rtol=1e-4), "unmerge 后未恢复原始行为"
print(f"✅ merge 前后输出最大误差 {merge_err:.2e}；unmerge 后恢复（误差 {(before - restored).abs().max().item():.2e}）")

# %% [markdown]
# ### 实战换算：TinyGPT 的 15% vs 7B 的 0.48%
#
# TinyGPT 只有 8.6 万参数，LoRA rank=8 对这么小的模型来说占比很高（15%）；真实 7B 模型上
# LoRA 通常只占 0.1%–0.5%。这不是矛盾：**LoRA 省的是「可训练状态的字节数」，不是「层数占比」**。

# %%
lora_full_state_gb = 16 * 7e9 / 1e9
lora_state_gb = 16 * 0.0048 * 7e9 / 1e9
ratio_df = pd.DataFrame(
    [
        {"对象": "TinyGPT（8.6 万参数，r=8）", "可训练参数占比": f"{train_p / total_p:.2%}", "训练状态(GB)": round(16 * train_p / 1e9, 6)},
        {"对象": "7B 全量微调", "可训练参数占比": "100%", "训练状态(GB)": round(lora_full_state_gb, 1)},
        {"对象": "7B LoRA（r=32，0.48%）", "可训练参数占比": "0.48%", "训练状态(GB)": round(lora_state_gb, 2)},
    ]
)
assert lora_state_gb / lora_full_state_gb < 0.02
ratio_df

# %% [markdown]
# ## 13.4 对照工业实现
#
# | 我们的实现 | 工业实现中的对应物 | 文件/参数 |
# | --- | --- | --- |
# | `threaded_allreduce_average` | DDP 的 bucket all-reduce + 通信重叠 | `torch/nn/parallel/distributed.py`（`_DDPSink`、`Reducer`） |
# | `zero_ledger(stage=1/2/3)` | ZeRO-1/2/3 与 FSDP 的分片策略 | DeepSpeed `deepspeed/runtime/zero/stage1.py` / `stage2.py` / `stage3.py`；PyTorch `torch/distributed/fsdp` |
# | `LoRALinear.merge()` | PEFT 的 `merge_and_unload()` | `peft/tuners/lora/layer.py` |
# | `apply_lora` 的目标层 | LoRA target modules（q/k/v/o、gate/up/down） | `LoraConfig(target_modules=[...])` |
# | 合并后的模型给 vLLM | `--enable-lora --lora-modules name=path`，或直接加载合并权重 | `vllm/lora/models.py`、`vllm/lora/punica_wrapper/` |
#
# 差距在哪：
#
# - **DDP**：PyTorch 的 Reducer 会在反向传播过程中按 bucket 触发 all-reduce，用 `no_sync()`
#   控制梯度累积时的通信；还支持梯度压缩（fp16/bf16）。我们只验证了数学等价。
# - **ZeRO**：DeepSpeed 还要管理参数分片在 forward/backward 时的 gather、offload 到 CPU/NVMe、
#   以及 bucket 级的 all-gather。ZeRO-3 的代价是**每层前向都要 gather 一次参数**，所以它省显存
#   但增加通信；在 8 卡 NVLink 内通常划算，跨机要谨慎。
# - **LoRA**：PEFT 支持 dropout、rsLoRA、DoRA、多 adapter 热切换；vLLM 则用 Punica kernel
#   在推理时动态批处理多个 adapter，不必为每个 adapter 复制一份基座。
#
# **训练 → 推理闭环**（本章代码在本地跑通了「训练 → merge → 等价性」）：
#
# ```
# 1) 训练：LoRA 适配器保存为 adapter_model.safetensors（只有 A/B，几十 MB）
# 2) 路线 A（合并）：merge 进基座 → 导出完整 fp16 模型 → vLLM 直接加载（无额外开销）
# 3) 路线 B（热插拔）：基座 + adapter 分别加载 → vLLM --enable-lora，多租户共享基座
# 4) 验收：同一批 prompt，merge 前后 / 启用 adapter 前后的输出做 all-close（13.3 的做法）
# ```

# %% [markdown]
# ## 13.5 练习与小结
#
# **练习**
#
# 1. ⭐ 把 `N_RANK` 从 4 改成 8，DDP 的 all-close 还成立吗？如果每个 rank 的样本数不能整除，
#    应该用加权平均还是简单平均？
# 2. ⭐ 把 `zero_ledger` 的 `trainable_ratio` 从 0.48% 改成 1% 和 5%，LoRA 每卡显存怎么变？
#    什么时候 LoRA 会失去「单卡可训」的优势？
# 3. ⭐⭐ 给 DDP 模拟加「梯度累积 + no_sync」：每 4 个 micro-batch 才 all-reduce 一次，
#    验证结果与不做累积等价，并解释为什么这样能省通信。
# 4. ⭐⭐ 把 LoRA 的 r 从 8 改成 32，重跑 50 步，比较 loss 曲线与可训练参数量。rank 越大越好吗？
# 5. ⭐⭐⭐ 设计一条「LoRA 微调 → 合并 → vLLM 部署 → 压测」的完整流水线：列出每一步的
#    产物、失败回滚点和验收指标（至少 4 条评分要点）。
#
# **你现在应该能回答：**
#
# - 16 B/参数是怎么来的？为什么 fp16 推理只要 2 B/参数？
# - ZeRO-1/2/3 分别把哪一项除以 N？为什么 ZeRO-1 只省了 84 GB 而不是全部？
# - LoRA 的 B 为什么初始化为 0？merge 前后为什么必须用 all-close 验证？
#
# **小结与下一章**
#
# 到这里，「训练侧」的三件事就齐了：算清显存账本（ZeRO）、只训一小部分参数（LoRA）、
# 把更新同步到多卡（DDP）。但学完 12、13 章，你手里只有「一堆公式和模拟」——真实业务
# 会给你一个更硬的问题：**给定 QPS 和 SLO，到底该买几张卡、用什么精度、一个月花多少钱？**
# 下一章是 Capstone：把前面 13 章的公式串成一份能交给老板的容量规划。
#
# **延伸阅读**
#
# - ZeRO 论文（Rajbhandari et al., 2020）与 DeepSpeed 文档
# - LoRA 论文（Hu et al., 2021）；PEFT 文档的 `merge_and_unload`
# - PyTorch FSDP 文档；vLLM 的 LoRA 服务章节
