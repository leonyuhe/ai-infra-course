"""第 08 章练习参考答案（对应 notebooks/08_speculative_decoding.py 的 5 道练习）。

直接用 ``.venv/bin/python solutions/08_exercises.py`` 运行，全部是 TinyGPT 上的小实验，几秒内跑完。
第 4 题是设计题（给实现要点与预期数字），第 5 题是真实模型实验方案（给评分要点）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiinfra.bench import pick_device
from aiinfra.tinygpt import GPTConfig, TinyGPT, generate_cached

device = pick_device()
torch.manual_seed(0)
V = GPTConfig().vocab_size


def make_model(seed: int, embed_std: float = 0.02) -> TinyGPT:
    """与章节一致：把共享的 embedding/lm_head 缩到真实模型量级，得到有熵的分布。"""
    torch.manual_seed(seed)
    m = TinyGPT(GPTConfig())
    with torch.no_grad():
        m.tok_emb.weight.mul_(embed_std)
    return m.to(device).eval()


torch.manual_seed(11)
ctx = torch.randint(0, V, (64, 48), device=device)


def last_probs(model: TinyGPT, ids: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    with torch.no_grad():
        return F.softmax(model(ids)[:, -1, :] / temperature, dim=-1)


def acceptance(p: torch.Tensor, q: torch.Tensor) -> float:
    """α = Σ min(p,q) = 1 − TV(p,q)。"""
    return torch.minimum(p, q).sum(-1).mean().item()


def speedup(alpha: float, gamma: int, c: float) -> float:
    tokens = (1 - alpha ** (gamma + 1)) / (1 - alpha)
    return tokens / (1 + gamma * c)


# --------------------------------------------------------------------------- #
# 练习 1（⭐）：扫 γ 与温度，解释饱和
# --------------------------------------------------------------------------- #
def exercise_1() -> None:
    print("=" * 78)
    print("练习 1：γ 与温度的扫描（α 由分布精确计算，加速比按 c=0.03 的模型公式）")
    target, draft = make_model(0), make_model(1)
    rows = []
    for T in (1.0, 0.5, 0.2, 0.1):
        alpha = acceptance(last_probs(draft, ctx, T), last_probs(target, ctx, T))
        for gamma in (1, 2, 4, 8):
            rows.append({"温度 T": T, "α": round(alpha, 3), "γ": gamma,
                         "模型加速比": round(speedup(alpha, gamma, 0.03), 2),
                         "每轮期望 token 数": round((1 - alpha ** (gamma + 1)) / (1 - alpha), 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "答案：每轮期望 token 数随 γ 指数饱和到 1/(1−α)。α=0.82 时上限约 5.6，γ=4 已拿到 3.1，"
        "γ=8 只多 0.6；α=0.26 时上限 1.35，γ 再大也只是让分母 1+γc 变大，加速比反而下降。"
        "所以 num_speculative_tokens 要跟 α 走：高 α 用 4–8，低 α 用 1–2，很低就直接关。"
    )


# --------------------------------------------------------------------------- #
# 练习 2（⭐）：把 draft 换成「复制最后一个 token」
# --------------------------------------------------------------------------- #
def copy_drafter_alpha(target: TinyGPT, temperature: float = 1.0) -> float:
    """复制 draft 的分布是 one-hot(last token)，直接算 α = Σ min(p,q)。"""
    with torch.no_grad():
        q = F.softmax(target(ctx)[:, -1, :] / temperature, -1)
    last = ctx[:, -1]
    p = F.one_hot(last, V).float()
    return torch.minimum(p, q).sum(-1).mean().item()


def exercise_2() -> None:
    print("=" * 78)
    print("练习 2：copy-last-token draft 的 α（校准后的模型 vs 默认初始化）")
    scaled = make_model(0)
    torch.manual_seed(0)
    raw = TinyGPT(GPTConfig()).to(device).eval()   # 默认 init：embedding 是 N(0,1)
    rows = [
        {"target": "校准后（embed_std=0.02）", "温度 T": 1.0, "copy draft α": round(copy_drafter_alpha(scaled, 1.0), 3)},
        {"target": "校准后（embed_std=0.02）", "温度 T": 0.1, "copy draft α": round(copy_drafter_alpha(scaled, 0.1), 3)},
        {"target": "默认 init（embed_std=1.0）", "温度 T": 1.0, "copy draft α": round(copy_drafter_alpha(raw, 1.0), 3)},
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "答案：默认初始化的 TinyGPT 里，RMSNorm 之后的 token 向量与共享 embedding 做自点积，"
        "logits 被『复制最后一个 token』这一项支配（argmax 几乎总是 last token），"
        "所以 copy draft 的 α≈1——这是玩具模型的退化现象，不是真实 LLM 的性质。"
        "校准到真实量级后，copy draft 的 α 掉到很低；真实系统里最接近它的零成本 draft 是"
        "n-gram / prompt-lookup，只有在输出高度重复（代码、JSON）时 α 才高。"
    )


# --------------------------------------------------------------------------- #
# 练习 3（⭐⭐）：贪心版投机解码，输出必须与 generate_cached(temperature=0) 一致
# --------------------------------------------------------------------------- #
@torch.no_grad()
def greedy_spec_decode(model: TinyGPT, prompt: torch.Tensor, n: int = 12, gamma: int = 3) -> torch.Tensor:
    """贪心投机解码（无 cache，方便看清逻辑）：
    draft 连续 argmax 猜 γ 个；target 一次前向验证；第一个 argmax 不一致处截断并采用 target 的 argmax。
    """
    out = prompt
    while out.shape[1] - prompt.shape[1] < n:
        ctx_ids, draft = out, []
        for _ in range(gamma):                                  # draft：连续贪心猜 γ 个
            nxt = model(ctx_ids)[:, -1, :].argmax(-1, keepdim=True)
            draft.append(nxt)
            ctx_ids = torch.cat([ctx_ids, nxt], dim=1)
        drafts = torch.cat(draft, dim=1)
        logits = model(torch.cat([out, drafts], dim=1))          # target：一次前向验证
        new = []
        for i in range(gamma):
            q = logits[:, out.shape[1] - 1 + i, :]              # 验证第 i 个 draft 的分布
            if q.argmax(-1, keepdim=True).item() == drafts[:, i].item():
                new.append(drafts[:, i : i + 1])
            else:
                new.append(q.argmax(-1, keepdim=True))          # 拒绝：直接用 target 的 argmax
                break
        out = torch.cat([out] + new, dim=1)
    return out[:, : prompt.shape[1] + n]


def exercise_3() -> None:
    print("=" * 78)
    print("练习 3：贪心投机解码 vs 普通贪心 decode（必须逐 token 相等）")
    torch.manual_seed(0)
    model = TinyGPT(GPTConfig()).to(device).eval()
    torch.manual_seed(5)
    prompt = torch.randint(0, V, (1, 16), device=device)
    ref = generate_cached(model, prompt, max_new_tokens=12, temperature=0.0)
    for gamma in (1, 2, 4):
        got = greedy_spec_decode(model, prompt, n=12, gamma=gamma)
        same = torch.equal(got, ref)
        print(f"γ={gamma}: 与贪心 baseline 逐 token 相同 = {same}")
        assert same, "贪心投机解码改变了输出序列——拒绝分支的修正写错了"
    print(
        "答案：温度 0 时接受条件退化为『draft 的 argmax == target 的 argmax』，"
        "拒绝后直接输出 target 的 argmax，因此序列与普通贪心完全相同。"
        "这也是 vLLM 的 greedy rejection sampler 分支：不采样、不修正分布，只比较 argmax。"
    )


# --------------------------------------------------------------------------- #
# 练习 4（⭐⭐）：树状 draft（EAGLE 风格）
# --------------------------------------------------------------------------- #
def exercise_4() -> None:
    print("=" * 78)
    print("练习 4（实现要点）：树状 draft")
    print(
        """
    树状 draft 的核心：一次 draft 前向得到每个候选后续的分布，展开成深度 D 的树，
    把树里所有节点（例如 D=3、宽度 2 → 最多 14 个 token）拼成一条序列喂给 target 一次前向验证，
    再沿树走上最长的、每个位置都与 target argmax 一致的路径。

    实现步骤：
    1. 用 draft 在当前位置算 top-k（k=2~4）的候选分布，按 token 展开成节点，记录 parent 指针；
    2. 把树节点按 DFS 顺序拼成一条 token 序列，同时构造 attention mask 让每个节点只能看见自己的祖先；
       —— 这正是 EAGLE 里 `tree_attn_mask` 干的事，也是它比线性 draft 复杂的地方；
    3. target 一次前向，对每个节点算出 target 分布，与节点的 draft 分布做接受/拒绝；
    4. 沿树自根向下找最长合法路径，接受其全部 token，然后从该路径末端的 target 分布继续。

    预期收益（D=3、宽度 2、α≈0.8 时）：线性 γ=3 每轮期望 2.95 个 token，
    树状最多能覆盖 3 层 × 2 分支，期望接受长度可到 4–5 个 token；但验证序列更长、
    mask 更复杂，c 和每轮 target 成本都会上升，实际收益要实测。

    参考：EAGLE-2 的动态树、EAGLE-3 的多层 draft head；vLLM 里见 `vllm/v1/spec_decode/eagle.py`。
    """
    )


# --------------------------------------------------------------------------- #
# 练习 5（⭐⭐⭐）：真实模型实验方案（开放题，评分要点）
# --------------------------------------------------------------------------- #
def exercise_5() -> None:
    print("=" * 78)
    print("练习 5（开放题）参考方案：Qwen3-0.6B draft + 8B target")
    print(
        """
    测量方案：
    1. 基线：target 单独跑 `vllm bench serve`（固定 prompt 集、同温度、同 max_tokens、warmup 足够）；
    2. 投机：`--speculative-config '{"model":"Qwen3-0.6B","num_speculative_tokens":k}'`，k=1..5 扫描；
    3. 指标：TPOT、tokens/s、`vllm:spec_decode_num_accepted_tokens / num_draft_tokens`（得到 α）、
       每轮实际接受长度分布、以及不同任务类别（代码 / 摘要 / 闲聊）的 α。
    4. 控制变量：同一 batch、同一 KV cache 配置（`--kv-cache-dtype`）、同一 `gpu_memory_utilization`；
       关掉与投机无关的优化，避免混淆。

    判定规则（什么时候关掉投机）：
    - 先算 c = draft 单步成本 / target 单步成本（可以用小模型单独 benchmark）；
    - 用实测 α 套公式 speedup = (1−α^{γ+1}) / ((1−α)(1+γc))，若 <1.05 就不值得开；
    - 若 α 在目标任务上 <0.3（例如自由创作、低温采样），直接关掉；
    - 还要看显存：draft 模型本身占显存，会挤压 KV cache 的块数，长上下文时可能反而降低吞吐。

    评分要点：
    1. 基线是否公平（同 batch/温度/上下文）；
    2. 是否把 α 按任务拆分报告，而不是一个全局平均；
    3. 是否解释了 c 与 TPOT 的关系，并与理论公式对照；
    4. 是否讨论了 draft 模型显存对 KV cache 的挤压；
    5. 是否给出可执行的开关阈值（例如 α < 0.3 或 speedup < 1.05 关闭）。
    """
    )


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
