# %% [markdown]
# # 第 08 章 · 投机解码：小模型猜，大模型验
#
# > 同一套 8B 服务，挂上 0.6B 的 draft、`num_speculative_tokens: 5`，是线上最常见的一档配置；
# > 把 draft 换成同尺寸的 8B，配置只改一行 model 名，加速比立刻退回 1×。算法一模一样，
# > 差别只在两个数：**猜得准不准（α）**、**猜一次贵不贵（c）**——08.4 在 TinyGPT 上量到的是
# > α 高时 2–3 倍、α 低时连 1 倍都不到。
#
# > 一句话：decode 是串行的，每生成一个 token 就要跑一次大模型前向（01 章），token 间隔被带宽锁死。
# > 投机解码让一个便宜的 draft 先猜 γ 个 token，再由 target 一次前向并行验证，
# > 在不改变输出分布的前提下减少 target 的串行前向次数。这一章算清这笔赌局：α 是赔率、c 是成本。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 90 分钟 |
# | 运行环境 | 🏠 Mac（MPS / CPU）· 无需 GPU · torch 2.15 |
# | 前置章节 | 01（decode 带宽、采样）、07（decode 为什么是瓶颈） |
# | 本章产物 | 词表级接受/残差重采样、两种 draft proposer、带 KV cache 的 draft-verify 循环、正确性验收、三组实测 |
# | 本章接口 | `aiinfra.tinygpt.TinyGPT / KVCache`；本章内联 `spec_sample_once / spec_decode` |
# | 对照版本 | vLLM 0.28.x |
#
# **学完你能回答：**
# 1. draft-verify 的接受规则 `min(1, q/p)` 与拒绝后的 `(q−p)₊` 重采样，为什么能保证输出分布不变？
# 2. 接受率 α、猜测长度 γ、draft 成本 c 三者如何决定加速比？为什么同尺寸 draft 永远赚不到钱？
# 3. 实测 TinyGPT 的 α 随温度怎么变？为什么低 α 时投机反而比 baseline 慢？
# 4. vLLM 的 `--speculative-config`、EAGLE / MTP / n-gram 各自对应本章的哪个部件？

# %% [markdown]
# ## 08.0 本章目标与路线
#
# **一句话目标**：把 decode 从「每个 token 一次 target 前向」改成「每轮一次 target 前向、
# 一次验证 γ+1 个位置」，在**输出分布严格不变**的前提下减少串行前向次数。
#
# **代价 / 边界**：
#
# - **draft 成本 c**：每一轮要先跑 γ 步 draft。c 不低时，省下的 target 前向又会被 draft 花回去；
# - **接受率 α**：draft 猜得不准时，大部分 token 被拒绝，验证前向的成本收不回来，反而更慢；
# - **实现复杂度**：接受/拒绝、残差重采样、KV cache 回滚、批内多请求调度，工业实现比手写循环复杂得多；
# - **收益条件**：只有 α 高且 c 低时才有加速；同尺寸 draft（c≈1）的收益上限恒为 1×。
#
# **本章交付四样东西**：
#
# | # | 交付物 | 在哪一节 | 怎么验收 |
# | --- | --- | --- | --- |
# | 1 | 单步接受 + 残差重采样（词表级） | 08.3 Step 1 | 卡方检验通过；错误示范被明显拒绝 |
# | 2 | 两种 draft proposer（均匀 / 模型） | 08.3 Step 2 | 输出 `(γ,)` token 与 `(γ,V)` 概率 |
# | 3 | 带 KV cache 的 draft-verify 循环 | 08.3 Step 3 | 能生成 ≥n 个 token，cache 不溢出 |
# | 4 | 正确性与成本验收 + 三组实测 | 08.3 Step 4 / 08.4 | 同权重 α=1；加速比与 α、c 的模型吻合 |
#
# **阅读路线**：08.1 现象 → 08.2 高阶设计 → 08.3 四步实现（每步有验收）→ 08.4 实验 → 08.5 对照 vLLM。

# %% [markdown]
# ## 08.1 问题与直觉：每步一次大前向，只换来一个 token
#
# decode 的串行结构：第 t 个 token 必须等第 t−1 个 token 算完才能开始。
# 8B fp16 的 decode 上限只有几十 tok/s（07 章账本：实测带宽下约 20–30 tok/s），
# 瓶颈是**每个 token 都要把全部权重读一遍**。
#
# 更可惜的是：target 一次前向本来可以并行算很多个位置（prefill 就是一次算 T 个），
# 但自回归只取最后一个位置，**并行能力被浪费**。
# 投机解码把这份并行能力利用起来：让便宜的 draft 先猜 γ 个 token，
# target 一次前向把「这 γ 个 token + 1 个待验证位置」全部算完，再从第一个猜错的位置截断。
#
# **先猜一下**：draft 猜的 token 里总有一部分会被 target 拒绝。对拒绝的位置，最自然的处理是
# 「扔掉，再从 target 重新采一个」——这样输出不就回到 target 的分布了吗？
# (a) 可以，扔掉就行　(b) 不行，输出会偏向 draft 偏好的 token　(c) 不行，但只要多采几次就修回来了
# —— 先选一个，答案在本节末尾。
#
# 关键在于**猜错不能改变输出分布**。做法是逐位置接受/拒绝：
#
# ![](../assets/figures/ch08_verify.svg)
#
# 1. draft 分布 `p`、target 分布 `q`。draft 采出 token `d` 后，以概率 `min(1, q(d)/p(d))` 接受；
# 2. 一旦拒绝，从**残差分布** `(q − p)₊ / Σ(q − p)₊` 重采样一个 token，然后停止本轮；
# 3. 全部接受时，从 `q` 采样一个免费的 bonus token。

# %% [markdown]
# （回收本节开头那题）答案是 (b)：拒绝之后直接从 target 重采，输出会偏向 draft 偏好的 token。
# 原因是「被接受」那部分质量 `min(p,q)` 天然偏向 draft 猜得到的位置；如果拒绝的部分再从 `q`
# 原样抽一遍，等于让 p 偏爱的 token 白拿一份额外的质量，分布就被 draft 带偏了。
# 正确做法不是丢弃，而是把没用掉的残差 `(q−p)₊` 单独归一化后采样——它恰好补上「q 比 p 多出来」的部分。
# 这也是 (a) 最诱人的地方：贪心解码下这个错法完全正常，只有采样时才会露馅。
#
# **反例**：如果拒绝后直接从 `p` 重采样（或跳过拒绝直接拼接），分布同样会被改变，而且偏得更明显；
# 08.3 Step 1 会把这个错误示范也跑一遍，用卡方检验把它抓出来。
#
# 把普通 decode 和投机解码放在一起对比，差异一目了然：
#
# | | 普通 decode | 投机解码 |
# | --- | --- | --- |
# | 每轮 target 前向 | 1 次（T=1） | 1 次（T=γ+1） |
# | 每轮产出 | 1 token | 1 ~ γ+1 token |
# | 输出分布 | target 采样 | 必须与 target 完全相同（靠残差修正保证） |
# | 额外成本 | 无 | draft γ 步 + 接受/拒绝 |
# | 收益条件 | — | draft 便宜且猜得准 |
#
# **反直觉的地方**：投机解码每轮让 target 算 γ+1 个位置，比普通 decode 的 1 个位置「多算」了 γ 个，
# 但 target 的耗时可没有涨 γ 倍——因为 decode 是带宽受限的：一次前向只把权重读一遍，
# 算 1 个位置和算 γ+1 个位置，搬运费一样。**白送的那 γ 个位置，才是投机的利润来源。**
#
# 本章两个子问题：
#
# 1. **正确性**：接受/拒绝 + 残差重采样之后，输出分布还是不是 target？怎么用统计检验证明？
# 2. **成本**：接受率、猜测长度、draft 成本各自如何影响加速比？实测能拿到多少？

# %% [markdown]
# ## 08.2 高阶设计
#
# 动手前先把方案定下来。整章只有三个组件和一条数据流。
#
# **组件 1：draft proposer（猜什么）**
# 给定当前已生成的上下文，自回归地猜 γ 个 token，同时给出这 γ 步的分布 `p_1 … p_γ`。
# 本章实现两种：均匀随机（成本 c≈0，用来隔离 α 的影响）和模型自回归（与 target 同尺寸，c≈1，
# 用来验证正确性与"同尺寸 draft 不赚钱"这个负结果）。
#
# **组件 2：target 验证（怎么验）**
# target 一次前向要同时得到 γ 个待验证分布 + 1 个 bonus 分布。关键是**少算**：
# 上一轮已算过的位置不再重算。做法是保留一个 **pending token**（已输出、还没进 cache），
# 每轮把它和 γ 个 draft 一起喂进去，一次前向算 γ+1 个位置；拒绝时把 cache 的写入指针回滚。
#
# **组件 3：接受/拒绝与残差重采样（怎么保正确）**
# 逐位置按 `min(1, q/p)` 接受；第一个被拒绝的位置用 `(q−p)₊` 重采样一个 token 并结束本轮；
# 全部接受则从 `q` 采样一个 bonus token。这三条规则合起来保证「输出分布 = target 分布」。
#
# **为什么恰好是 `(q−p)₊`？** 接受的分支拿走了质量 `min(p,q)`，剩下的质量正好是 `(q−p)₊`，
# 两者相加恰好等于 `q`。所以**输出分布严格等于 target 的采样分布**——08.4 实验 C 会用卡方检验量这个不变量。
#
# %% [markdown]
# **成本模型（验收与调参的依据）**：单步接受率 `α = Σ min(p,q) = 1 − TV(p,q)`，
# 一轮期望产出 `1 + α + α² + … + α^γ = (1−α^{γ+1})/(1−α)` 个 token，而一轮成本是 `1 + γ·c`
# （c = draft 单步成本 / target 单步成本）。**推论**：α=1、c=1（同权重同尺寸的 draft）时加速比恒等于 1
# ——猜得再准也白猜，因为 draft 把省下的钱又花回去了。真正赚钱的是 EAGLE/MTP 这类 `c ≈ 0.03–0.1` 的 draft 头。
#
# 把这条公式当成一场赌局，三个量各就各位：**α 是赔率**（每猜一个位置，有多大比例能兑现），
# **c 是买彩票的成本**（猜一个位置要付 target 单步的几分之几），**γ 是下注次数**。
# 注意期望产出是个**几何级数**：第二张票必须第一张中、第三张必须前两张都中，所以收益不是 `γ·α`，
# 而是很快饱和的 `1/(1−α)`；成本却是实打实的 `1 + γ·c`，随 γ 线性涨。
# 一正一反，决定了「不是猜得越多越好」——08.4 实验 B 会把这句话量成一张表。
#
# 代一组实测数字进去（08.4 的温度 1.0 场景）：α≈0.87、γ=4、c≈0.03，
# 一轮期望产出 `1+0.87+0.87²+0.87³+0.87⁴ ≈ 3.86` 个 token，成本 `1+4×0.03 ≈ 1.12`，
# 加速比约 3.4×——正是 08.4 表格里「模型预测」那一列的量级。而如果 α 掉到 0.3 附近，
# 同样 γ=4、c≈0.1（工业 0.6B draft 的成本），期望产出只剩约 1.43，成本 1.4——**基本白干，再往下就是倒亏**。
#
# 数据流：
#
# ```text
# 已生成 token（context）
#   → draft 自回归猜 γ 个 token，记录 p_1..p_γ
#   → target 一次前向（pending + γ drafts）→ q_1..q_γ + bonus 分布
#   → 逐个比较 min(1, q_i/p_i)：接受前缀；第一个拒绝处用 (q_i − p_i)₊ 重采样
#   → 输出 1..γ+1 个 token，回滚未接受的 KV，进入下一轮
# ```

# %% [markdown]
# ### 接口契约
#
# 后面的实现、08.4 的实验、以及练习共用同一组签名；换 draft 策略只改一个参数：
#
# | 对象 / 函数 | 语义 |
# | --- | --- |
# | `drafter(ctx, gamma, gen)` | 返回 `(tokens (γ,), probs (γ, V))`：猜的 token 与对应分布 |
# | `spec_sample_once(q, p, n, gen)` | 词表级单步修正采样：输出分布 = `q`（Step 1 的核心） |
# | `spec_decode(drafter, n, gamma, temperature, gen)` | 完整循环：返回 `(out, accept_rate)` |
# | `target_only(n, temperature, gen)` | baseline：标准带 cache 的逐 token decode |
#
# 为什么接受率由 `spec_decode` 返回？因为它既是正确性指标（同权重必须 = 1），
# 也是调参依据（α 掉下来就该减小 γ）——这两个用途都要在循环里逐轮累计。
#
# ### 验收标准
#
# | 步骤 | 验收 |
# | --- | --- |
# | Step 1 词表级修正 | 卡方 < 临界值；错误示范的卡方 > 20 倍临界值 |
# | Step 2 draft proposer | token 形状 `(γ,)`、概率形状 `(γ,V)`，概率和为 1 |
# | Step 3 verify 循环 | 生成的 token 数 ≥ n；KV cache 不溢出 |
# | Step 4 正确性与成本 | 同权重接受率 = 1.0；draft 单步成本可测（c） |
# | 实验 A/B/C | α 与温度单调；高 α 加速 > 1；低 α 加速 < 1；分布检验通过 |
# | 成本模型 | 预测加速比与实测同量级（差值是固定开销） |

# %% [markdown]
# ## 08.3 从零实现：分四步
#
# | Step | 做什么 | 代码 |
# | --- | --- | --- |
# | 1 | 词表级接受 + 残差重采样，先在小词表上证明分布不变 | `spec_sample_once` / `naive_sample_once` |
# | 2 | 两种 draft proposer：均匀随机、模型自回归 | `uniform_drafter` / `model_drafter` |
# | 3 | target 验证循环：pending + γ drafts + KV 回滚 | `spec_decode` / `target_only` |
# | 4 | 端到端验收：同权重 α=1 + 成本口径 c | 断言 + `bench` |
#
# ### 准备实验对象
#
# TinyGPT 的默认初始化把 embedding 初始化为 `N(0,1)`，而 RMSNorm 让 logits 被"自点积"支配：
# 模型退化成复读最后一个 token，所有 draft 都会 100% 被接受——那样测不出任何 α 的差异。
# 真实的 LLM embedding 初始化是 `~1/√d` 量级，所以这里把 `tok_emb` 缩到 0.02（与 lm_head 共享权重），
# 得到有意义的、有熵的分布。这是刻意为之的**玩具校准**，不影响算法本身。

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path.cwd().parent))

from aiinfra.bench import bench, pick_device
from aiinfra.plotting import save, setup_style
from aiinfra.tinygpt import GPTConfig, KVCache, TinyGPT

setup_style()
device = pick_device()
torch.manual_seed(0)
np.random.seed(0)
V = GPTConfig().vocab_size


def make_model(seed: int, embed_std: float = 0.02) -> TinyGPT:
    """随机初始化 TinyGPT，并把共享的 embedding/lm_head 缩到真实模型的量级。"""
    torch.manual_seed(seed)
    m = TinyGPT(GPTConfig())
    with torch.no_grad():
        m.tok_emb.weight.mul_(embed_std)
    return m.to(device).eval()


target = make_model(0)          # 大模型（target）
alt = make_model(1)             # 另一个随机种子的"小模型"（draft 的替身，尺寸相同）
cfg = target.cfg
torch.manual_seed(11)
prompt = torch.randint(0, V, (1, 32), device=device)


def last_probs(model: TinyGPT, ids: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    with torch.no_grad():
        return F.softmax(model(ids)[:, -1, :] / temperature, dim=-1)

print(f"device = {device} | params = {target.num_params():,}")

# %% [markdown]
# ### Step 1 · 词表级接受 + 残差重采样
#
# **目标**：在最小的设定（4 个 token 的分布）上把正确性做对，再用卡方检验证明。
#
# **做法**：`spec_sample_once` 给定 target 分布 `q` 和 draft 分布 `p`，采一个 token：
# draft 采样 → 按 `min(1, q/p)` 接受 → 拒绝时从 `(q−p)₊` 重采样。
# 旁边放一个"错误示范" `naive_sample_once`：拒绝后仍然从 `p` 采样（不做修正）。

# %%
def spec_sample_once(q: torch.Tensor, p: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    """正确的投机采样：输出分布 = q。"""
    d = torch.multinomial(p, n, replacement=True, generator=gen)
    accept = torch.rand(n, generator=gen, device=p.device) < (q[d] / p[d]).clamp(max=1.0)
    resid = (q - p).clamp_min(0)
    cdf = (resid / resid.sum()).cumsum(0)
    r = torch.searchsorted(cdf, torch.rand(n, generator=gen, device=p.device)).clamp(max=len(q) - 1)
    return torch.where(accept, d, r)


def naive_sample_once(q: torch.Tensor, p: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    """错误示范：拒绝后继续从 draft 分布 p 采样 —— 不修正残差，分布会偏。"""
    d = torch.multinomial(p, n, replacement=True, generator=gen)
    accept = torch.rand(n, generator=gen, device=p.device) < (q[d] / p[d]).clamp(max=1.0)
    again = torch.multinomial(p, n, replacement=True, generator=gen)
    return torch.where(accept, d, again)


def chi2_stat(obs: torch.Tensor, exp: torch.Tensor) -> float:
    return float(((obs - exp).pow(2) / exp).sum())

# %% [markdown]
# **验收**：正确版本与 target `q` 的卡方检验通过；错误版本被显著拒绝。

# %%
p_toy = torch.tensor([0.60, 0.25, 0.10, 0.05])
q_toy = torch.tensor([0.15, 0.25, 0.45, 0.15])
N = 50_000
gen = torch.Generator().manual_seed(0)
direct = torch.multinomial(q_toy, N, replacement=True, generator=gen)
ours = spec_sample_once(q_toy, p_toy, N, gen)
bad = naive_sample_once(q_toy, p_toy, N, gen)
exp = q_toy * N
crit = 7.815   # 卡方临界值，df=3，α=0.05

chi2_direct, chi2_ours, chi2_bad = (chi2_stat(torch.bincount(x, minlength=4).float(), exp)
                                    for x in (direct, ours, bad))
assert chi2_ours < crit, f"修正后的分布不对：chi2={chi2_ours:.2f} > {crit}"
assert chi2_bad > 20 * crit, "错误示范居然也通过检验？请检查 naive_sample_once"
pd.DataFrame({"token": [0, 1, 2, 3], "target q": q_toy.tolist(), "draft p": p_toy.tolist(),
              "正确（修正）的频率": (torch.bincount(ours, minlength=4).float() / N).tolist(),
              "错误（不修正）的频率": (torch.bincount(bad, minlength=4).float() / N).tolist(),
              "直接采样频率": (torch.bincount(direct, minlength=4).float() / N).tolist()}).round(3)

# %% [markdown]
# **读数**：**正确（修正）的频率**和 target `q` 对得上，卡方远小于临界值；
# **错误（不修正）的频率**明显偏向 draft `p` 的形状（token 0 被抬高、token 2 被压低），
# 卡方大得离谱。一句话：**残差修正不是可选项**——跳过它，投机解码就在偷偷换分布。
#
# ### Step 2 · 两种 draft proposer
#
# **目标**：把「谁来猜」抽象成统一接口，后面换 draft 策略只改一个参数。
#
# **做法**：`uniform_drafter` 在词表上均匀乱猜，成本 c≈0（只有采样开销），用来隔离 α 的影响；
# `model_drafter` 用模型自回归猜 γ 步，每步从头 forward（不接 cache），只用于正确性实验与负结果，
# 不用于性能对比——它的成本天然接近 target。

# %%
def uniform_drafter(ctx: torch.Tensor, gamma: int, gen: torch.Generator):
    """最便宜的 draft：在词表上均匀乱猜，成本 c ≈ 0（只有采样开销）。"""
    tokens = torch.randint(0, V, (gamma,), device=device, generator=gen)
    return tokens, torch.full((gamma, V), 1.0 / V, device=device)


def model_drafter(model: TinyGPT, temperature: float = 1.0):
    """用模型自回归猜 γ 个 token（每步从头 forward，只用于正确性实验，不用于性能对比）。"""
    @torch.no_grad()
    def draft(ctx: torch.Tensor, gamma: int, gen: torch.Generator):
        tokens, probs = [], []
        for _ in range(gamma):
            p = F.softmax(model(ctx)[:, -1, :] / temperature, dim=-1)
            d = torch.multinomial(p, 1, generator=gen)
            tokens.append(d)
            probs.append(p)
            ctx = torch.cat([ctx, d], dim=1)
        return torch.cat(tokens).view(-1), torch.cat(probs, dim=0)
    return draft

# %% [markdown]
# **验收**：两种 drafter 都返回形状为 `(γ,)` 的 token 与 `(γ,V)` 的概率，且每行概率和为 1。

# %%
gen = torch.Generator(device=device).manual_seed(0)
u_tokens, u_probs = uniform_drafter(prompt, 3, gen)
m_tokens, m_probs = model_drafter(target)(prompt, 3, gen)
for tokens, probs in [(u_tokens, u_probs), (m_tokens, m_probs)]:
    assert tokens.shape == (3,) and probs.shape == (3, V)
    assert torch.allclose(probs.sum(-1), torch.ones(3, device=device), atol=1e-4)
print("uniform / model drafter：形状与归一化检查通过 ✓")

# %% [markdown]
# ### Step 3 · draft-verify 主循环（带 KV cache）
#
# **目标**：把 Step 1 的接受规则放进真实解码循环，并且**不重复计算** target 已经算过的位置。
#
# **做法**：保留一个 **pending token**（已输出但还没进 cache），每轮把它和 γ 个 draft 一起喂给 target：
# 一次前向算 γ+1 个位置 → 得到 γ 个待验证分布 + 1 个 bonus 分布。
# 拒绝时把 cache 的写入指针回滚（工业实现里是 block manager 释放 block，第 03 章）。
#
# 一轮长这样（时间为示意，取 08.4 的实测口径：target 单步 ~2ms、c≈0.03–0.08，γ 取 5）：
#
# ```text
# t=0.0ms   ┃ 上下文就绪，pending = 最后一个已输出的 token（还没进 cache）
# t=0.5ms   ┃ draft 自回归猜 5 个 token —— 5 步加起来只花 target 一步的 15%–40%
# t=2.5ms   ┃ target 一次前向算 6 个位置（pending + 5 个 draft），权重只读了一遍
# t=2.6ms   ┃ 逐位置比 min(1, q/p)：第 1、2、3 个接受，第 4 个拒绝
# t=2.7ms   ┃ 回滚 cache 里未接受的位置，用 (q−p)₊ 在第 4 位重采一个 token
# t=2.8ms   ┃ 本轮产出 4 个 token（3 个 draft + 1 个残差），进入下一轮
# ```
#
# 盯住 `t=2.6ms` 这一行：5 个 draft 只能**从第一个拒绝处往前**兑现，第 4 位一错，第 5 位猜得再对也作废。
# 接受是一段**前缀**，不是一串散点——这就是为什么 α 掉一点点，平均每轮产出会成段地掉。

# %%
@torch.no_grad()
def spec_decode(drafter, n: int = 24, gamma: int = 4, temperature: float = 1.0, gen=None):
    """draft γ 个 → target 一次前向验证 γ+1 个位置 → 接受前缀 / 拒绝后 (q−p)₊ 重采样。"""
    Lp = prompt.shape[1]
    cache = KVCache(cfg, max_seq_len=Lp + n + gamma + 4, device=device)
    target(prompt[:, :-1], start_pos=0, kv_cache=cache)
    cache.advance(Lp - 1)                 # 最后一个 prompt token 留作 pending
    pending, out, n_new, n_acc, n_draft = prompt[:, -1:], prompt, 0, 0, 0
    while n_new < n:
        d_tokens, d_probs = drafter(out, gamma, gen)
        inp = torch.cat([pending, d_tokens.view(1, gamma)], dim=1)
        logits = target(inp, start_pos=cache.seq_len, kv_cache=cache)
        cache.advance(gamma + 1)                        # 先提交全部验证过的位置
        q = F.softmax(logits[0] / temperature, dim=-1)  # q_1..q_γ + bonus 分布
        idx = d_tokens.view(-1, 1)
        p_d, q_d = d_probs.gather(1, idx).squeeze(1), q[:gamma].gather(1, idx).squeeze(1)
        reject = torch.rand(gamma, device=device, generator=gen) >= (q_d / p_d).clamp(max=1.0)
        first = int(torch.argmax(reject.float()).item()) if reject.any() else gamma
        n_draft, n_acc = n_draft + gamma, n_acc + first
        if first < gamma:                               # 拒绝：回滚 cache + 残差重采样
            resid = (q[first] - d_probs[first]).clamp_min(0)
            token = torch.multinomial(resid / resid.sum().clamp_min(1e-12), 1, generator=gen)
            cache.advance(-(gamma - first))
            new = torch.cat([d_tokens[:first], token])
        else:                                           # 全接受：白拿一个 bonus token
            new = torch.cat([d_tokens, torch.multinomial(q[gamma], 1, generator=gen)])
        new = new.view(1, -1)
        out, n_new, pending = torch.cat([out, new], dim=1), n_new + new.shape[1], new[:, -1:]
    return out, n_acc / max(n_draft, 1)


# %%
@torch.no_grad()
def target_only(n: int = 24, temperature: float = 1.0, gen=None):
    """baseline：标准的带 cache 逐 token decode。"""
    cache = KVCache(cfg, max_seq_len=prompt.shape[1] + n + 2, device=device)
    logits = target(prompt, start_pos=0, kv_cache=cache)
    cache.advance(prompt.shape[1])
    out = prompt
    for _ in range(n):
        nxt = torch.multinomial(F.softmax(logits[:, -1, :] / temperature, dim=-1), 1, generator=gen)
        out = torch.cat([out, nxt], dim=1)
        logits = target(nxt, start_pos=cache.seq_len, kv_cache=cache)
        cache.advance(1)
    return out

# %% [markdown]
# **验收**：循环能跑通、生成数量不少于请求的 n（一轮可能一次产出多个 token），
# 且 cache 不溢出（`KVCache.append` 自带溢出断言）。

# %%
gen = torch.Generator(device=device).manual_seed(1)
out_spec, ar = spec_decode(uniform_drafter, n=4, gamma=2, temperature=1.0, gen=gen)
out_base = target_only(n=4, temperature=1.0, gen=gen)
assert out_spec.shape[1] - prompt.shape[1] >= 4, "投机解码生成数量应 ≥ n（可能一轮产出多个）"
assert out_base.shape[1] - prompt.shape[1] == 4
print(f"spec 生成 {out_spec.shape[1] - prompt.shape[1]} 个 token（请求 4），接受率 {ar:.2f}；baseline 生成 4 个 ✓")

# %% [markdown]
# ### Step 4 · 端到端验收：同权重 α=1 与成本口径 c
#
# **目标**：验证两个不变量——「draft 完美时接受率必须为 1」和「draft 成本 c 可测量」。
#
# **做法**：把 target 自己当 draft（同权重同尺寸），每个位置的 `p_i ≡ q_i`，`q/p = 1`，
# 所有 draft 都应被接受；再用 `bench` 量一次 target 单步与一次 draft 采样的耗时，得到 c。
#
# **验收**：同权重接受率 = 1.0；c 与实测的 target 单步耗时可比。

# %%
gen = torch.Generator(device=device).manual_seed(0)
same_out, accept_same = spec_decode(model_drafter(target, 1.0), n=8, gamma=2, temperature=1.0, gen=gen)
assert accept_same == 1.0, f"同权重接受率应恒为 1，实际 {accept_same}"
print(f"同权重：接受率 = {accept_same:.3f}，生成长度 = {same_out.shape[1] - prompt.shape[1]}")
same_out[0, -4:]

# %%
cache_probe = KVCache(cfg, max_seq_len=128, device=device)
target(prompt, start_pos=0, kv_cache=cache_probe)
cache_probe.advance(prompt.shape[1])
tok = prompt[:, :1]
r_step = bench(lambda: target(tok, start_pos=cache_probe.seq_len, kv_cache=cache_probe),
               name="target 单步", warmup=3, reps=9, device=device)


def draft_cost(gamma: int = 4) -> None:
    uniform_drafter(None, gamma, torch.Generator(device=device).manual_seed(0))


r_draft = bench(draft_cost, 4, name="uniform draft", warmup=3, reps=20, device=device)
c_measured = (r_draft.median_ms / 4) / r_step.median_ms
print(f"target 单步 {r_step.median_ms:.2f} ms | draft 每 token {r_draft.median_ms / 4:.2f} ms | c ≈ {c_measured:.3f}")

# %% [markdown]
# `accept_same == 1.0` 是一个免费的不变量测试：算法在"draft 完美"时必须退化成 target 自己。
# 成本口径 c 由纯采样开销算出（实测 c ≈ 0.03–0.08，随负载浮动），和 target 单步 ~2ms 相比几乎可以忽略——
# 这正是 08.2 成本模型里 `1 + γ·c ≈ 1` 的来源。
#
# **容易被误读的一点**：α=1 只是「理想上界」的一半，不是「一定赚」。加速比公式里 α 只决定分子
# （产出封顶 `1/(1−α)`），分母的 `1 + γ·c` 才是账单；同权重 draft 的 α 恒为 1，但它同时带来 c≈1，
# 分子分母一起涨，除完正好是 1×——08.4 的热力图里那个橙色方块就落在这条对角线上。
# 所以 EAGLE 这类工业 draft 的努力方向不是「只把 α 做高」，而是**同时把 c 做低**：
# 只把 α 拉满、c 不动，是这场赌局里最贵的错误。
#
# 到这里，正确性与成本口径都验收完成。但真正决定要不要开投机的两个数还没量：
# **α 到底由什么决定**？**端到端能快多少、什么时候反而更慢**？下面三组实测分别回答它们，
# 再加一组「输出分布真的没变吗」的统计检验。

# %% [markdown]
# ## 08.4 实测
#
# 三组实验，分别回答三个问题：
#
# | 实验 | 问题 | 产出 |
# | --- | --- | --- |
# | A | 接受率 α 由什么决定？ | α vs 温度的精确表 + 循环交叉验证 |
# | B | 端到端能快多少？α、γ、c 谁说了算？ | tokens/s 表 + 实测/预测曲线 + 热力图 |
# | C | 接受/拒绝真的不改变输出分布吗？ | 卡方检验 + 直方图 |
#
# 测量协议：M3 Ultra / 96GB / torch 2.15 / MPS；α 用闭式解精确计算（无采样噪声），
# 速度用 `warmup=3, reps=7` 取中位数，生成 24 个 token。绝对值随机器负载变化，**趋势才是结论**。
#
# ### 实验 A · 接受率 α = 1 − TV
#
# **问题**：α 随 draft 质量和温度怎么变？
#
# **协议**：给定 64 个上下文，对每个上下文精确算 `α = Σ min(p,q) = 1 − TV(p,q)`（不是采样估计）。
# 三种 draft：同权重、不同随机种子、均匀随机；温度从 1.0 降到 0.1。

# %%
torch.manual_seed(11)
ctxs = torch.randint(0, V, (64, 48), device=device)


def acceptance(p: torch.Tensor, q: torch.Tensor) -> float:
    return torch.minimum(p, q).sum(-1).mean().item()   # α = Σ min(p,q) = 1 - TV


rows = []
for T in (1.0, 0.5, 0.2, 0.1):
    q = last_probs(target, ctxs, T)
    for name, p in [("同权重", q), ("不同种子", last_probs(alt, ctxs, T)),
                    ("均匀随机", torch.full_like(q, 1.0 / V))]:
        rows.append({"draft": name, "温度 T": T, "α": round(acceptance(p, q), 4)})
alpha_df = pd.DataFrame(rows).pivot(index="draft", columns="温度 T", values="α")
assert alpha_df.loc["同权重"].min() > 0.999
alpha_df

# %% [markdown]
# **读数**：温度 1.0 时 target 分布接近均匀，任意 draft 的 α 都在 0.8 以上（TV 距离小）；
# 温度降到 0.1，target 变尖，不同种子 draft 的 α 直接掉到 0.02，均匀 draft 掉到 0.09。
# **α 不是 draft 的固定属性，而是"draft 与 target 在给定上下文上的分布重叠度"。**
# 真实系统里温度、任务类型、draft 训练数据都会改变它，所以 vLLM 必须把接受率当运行时指标暴露出来。
#
# > **⚠️ 坑：拿 T=1 的 α 去估线上低温场景，加速比能从 2–3 倍变成 <1 倍**
# > 同一个均匀 draft，α 在 T=1.0 是 0.87，在 T=0.1 只剩 0.12——**差 7 倍**。
# > 下面实验 B 里同一份代码：T=1.0 加速 2–3 倍，T=0.1 四行全慢于 baseline。
# > 如果只在开发环境的默认温度上调 γ，上线后遇到低温 / 结构化输出任务，就会踩到「越开投机越慢」。
# > 判据很简单：**α 是运行时指标，不是上线前的配置参数。**
#
# 用不同随机种子的两个 TinyGPT 实际跑一段，验证"α 表"与"循环里数出来的接受率"一致：

# %%
rows = []
for name, drafter in [("同权重", model_drafter(target)), ("不同种子", model_drafter(alt))]:
    gen = torch.Generator(device=device).manual_seed(0)
    _, ar = spec_decode(drafter, n=24, gamma=1, temperature=1.0, gen=gen)
    rows.append({"draft": name, "循环实测接受率": round(ar, 4),
                 "公式 α（T=1.0）": round(alpha_df.loc[name, 1.0], 4)})
loop_df = pd.DataFrame(rows)
loop_df

# %% [markdown]
# **读数**：同权重必然是 1.0；不同种子实测在公式值 0.82 附近抖动（24 个 draft token，标准误约 0.08），
# 量级一致就够——这正说明 α 可以事先用分布算出来，不必真跑一遍。draft 循环用的是"从头 forward"的
# 近似实现，数值误差也会贡献一点偏差。在真实系统里，这个数字由 vLLM 的 `spec_decode` 指标直接暴露，
# 见 08.5。
#
# ### 实验 B · 端到端 tokens/s 与加速比热力图
#
# **问题**：完整解码能快多少？α、γ、c 三个参数谁说了算？
#
# **先猜一下**：温度降到 0.1（α≈0.12）后，把 γ 从 1 调到 4，加速比会怎么变？
# (a) 变快，猜得多总能中几个　(b) 基本不变，draft 成本可以忽略　(c) 变慢，每轮验证的钱收不回来
# —— 先选一个，表在下面。
#
# **协议**：用成本几乎为零的均匀 draft 跑 24 个 token，与 baseline 对比。
# 温度 1.0 是"高 α"场景，温度 0.1 是"低 α"场景——同一份代码、同一个 target，只换温度。
# γ 取 1..4；预测值用 08.2 的成本模型代入当次实测的 α 与 c。

# %%
rows = []
for T in (1.0, 0.1):
    gen = torch.Generator(device=device).manual_seed(0)
    base_ms = bench(lambda: target_only(24, T, gen), name="target", warmup=3, reps=7, device=device).median_ms
    base_tps = 24 / (base_ms / 1000)
    alpha = float(alpha_df.loc["均匀随机", T])
    for gamma in (1, 2, 3, 4):
        gen = torch.Generator(device=device).manual_seed(0)
        r = bench(lambda: spec_decode(uniform_drafter, 24, gamma, T, gen),
                  name=f"spec γ={gamma} T={T}", warmup=3, reps=7, device=device)
        spec_tps = 24 / (r.median_ms / 1000)
        tokens_per_round = (1 - alpha ** (gamma + 1)) / (1 - alpha)
        predicted = tokens_per_round / (1 + gamma * c_measured)
        rows.append({"温度": T, "γ": gamma, "baseline tok/s": round(base_tps, 1),
                     "投机 tok/s": round(spec_tps, 1), "实测加速比": round(spec_tps / base_tps, 2),
                     "模型预测加速比": round(predicted, 2)})
speed_df = pd.DataFrame(rows)
assert speed_df[(speed_df["温度"] == 1.0) & (speed_df["γ"] == 4)]["实测加速比"].item() > 1.3
assert speed_df[speed_df["温度"] == 0.1]["实测加速比"].max() < 1.0
speed_df

# %% [markdown]
# **读数**：温度 1.0（α≈0.87）时 γ 越大越快，γ=4 实测 2–3 倍（随机器负载浮动；模型预测 3.4，差值是
# 玩具模型的每轮固定开销）；温度 0.1（α≈0.12）时**所有 γ 都慢于 baseline**——每轮大部分 draft 都被拒绝，
# 验证前向的成本收不回来。（回收实验 B 开头那题）答案是 (c)：α=0.12 时期望产出从 γ=1 的 1.12 到 γ=4 的 1.14
# 几乎没涨，可每轮成本随 γ 一直涨——分母涨得比分子快，所以越猜越慢。
# 预测偏高还有一个原因：模型把每轮成本记作 1 个 target 单步，
# 而实测里 Python 循环、`.item()` 同步、残差采样都要额外花时间。
# 另外注意 baseline 的 tok/s 只有 ~250–400（随机器负载），而单步前向只要 ~2ms——采样和 Python 开销占了大部分，
# 说明**这个加速比里有一部分是"摊薄了玩具模型的固定开销"，不能直接外推到 7B 模型**。
#
# 把模型预测与实测画在一张图上：

# %%
fig, ax = plt.subplots()
for T, color in [(1.0, "#2563EB"), (0.1, "#EF4444")]:
    sub = speed_df[speed_df["温度"] == T]
    gammas_used = sub["γ"].tolist()
    measured = sub["实测加速比"].tolist()
    predicted = sub["模型预测加速比"].tolist()
    alpha_val = alpha_df.loc["均匀随机", T]
    ax.plot(gammas_used, measured, "o-", color=color, label=f"measured (T={T}, alpha={alpha_val:.2f})")
    ax.plot(gammas_used, predicted, "--", color=color, alpha=0.55, label=f"model (T={T})")
ax.axhline(1.0, color="#6B7280", ls=":", lw=1)
ax.set_xticks([1, 2, 3, 4])
ax.set_xlabel("draft tokens per round (gamma)")
ax.set_ylabel("speedup vs baseline (tokens/s)")
ax.set_title("alpha sets the ceiling: larger gamma helps at high alpha, hurts at low alpha")
ax.legend(fontsize=9)
save(fig, "ch08_measured_speedup")

# %% [markdown]
# **读图**：两条线的开口方向完全相反——`E[tokens/round]` 随 α 指数饱和，而每轮至少付一次 target 前向的钱。
# **对 vLLM 参数的启示**：`num_speculative_tokens` 要跟着 α 调：α 高时 γ=4–8 划算，
# α 低时 γ=1–2、甚至该关掉投机。
#
# **反直觉结论：你没法靠调大 γ 变快。** 第一反应是「多猜几个、多中几个」，但 γ 只决定下注次数，
# 决定不了 α 和 c 这两个分母——猜得越多，`1 + γ·c` 越贵，而期望产出只饱和到 `1/(1−α)`。
# 所以在 c>0 时存在一个**最优 γ**：超过它继续加注，每一注都是贴钱（分子趋近常数、分母继续线性涨，
# 加速比必然掉头向下）。08.4 的表里 T=0.1 时四个 γ 全慢于 baseline，γ=4 还是最低的一档——
# 不是实现有 bug，是数学不允许。
# 真正能变的只有两件事：把 draft 换小（c↓）、把 draft 练准（α↑）。γ 只是把这两者的收益**兑现**出来。
#
# 单看两组实测还不够，把 08.2 的成本模型画成热力图，才能看清 α、γ、c 三轴的设计空间：

# %%
alphas = np.linspace(0.05, 0.99, 80)
gammas = np.arange(1, 9)
C_MODEL = 0.05


def speedup(alpha, gamma, c):
    tokens = (1 - alpha ** (gamma + 1)) / (1 - alpha)
    return tokens / (1 + gamma * c)


A, G = np.meshgrid(alphas, gammas)
S_left = speedup(A, G, C_MODEL)
cs = np.logspace(-2, 0, 60)
A2, C2 = np.meshgrid(alphas, cs)
S_right = speedup(A2, 4, C2)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
im = axes[0].imshow(S_left, origin="lower", aspect="auto", cmap="viridis",
                    extent=[alphas[0], alphas[-1], gammas[0] - 0.5, gammas[-1] + 0.5])
alpha_meas = float(alpha_df.loc["均匀随机", 1.0])
axes[0].plot([alpha_meas], [4], "o", color="#EF4444", ms=8, label="measured: uniform draft, gamma=4")
axes[0].plot([1.0], [1], "s", color="#F59E0B", ms=8, label="same-weight draft (alpha=1, c=1)",
             clip_on=False)
axes[0].set_xlabel("acceptance rate alpha"); axes[0].set_ylabel("draft tokens per round (gamma)")
axes[0].set_title(f"Speedup: alpha vs gamma (c={C_MODEL})")
axes[0].legend(fontsize=8, loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
plt.colorbar(im, ax=axes[0], label="speedup")

# %%
im2 = axes[1].imshow(S_right, origin="lower", aspect="auto", cmap="viridis",
                     extent=[alphas[0], alphas[-1], cs[0], cs[-1]])
axes[1].set_yscale("log")
axes[1].plot([alpha_meas], [c_measured], "o", color="#EF4444", ms=8, label="measured: uniform draft")
axes[1].plot([0.8], [0.05], "D", color="#F59E0B", ms=7, label="EAGLE-like c=0.05 (illustrative)")
axes[1].set_xlabel("acceptance rate alpha"); axes[1].set_ylabel("draft cost ratio c (log)")
axes[1].set_title("Speedup: alpha vs c (gamma=4)")
axes[1].legend(fontsize=8, loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
plt.colorbar(im2, ax=axes[1], label="speedup")
fig.suptitle("Speculative decoding pays only when alpha is high and c is low", fontsize=12)
save(fig, "ch08_speedup_heatmap")

# %% [markdown]
# **读图**：左下角（α 低、c 高）一片深色——加速比 < 1，这就是"用同尺寸模型当 draft"的死区；
# 右上角（α→1、c→0）趋近 γ+1，即 08.2 成本模型的理论上界。两个标记点说明了本章的实测位置：
# 均匀 draft 的 c 几乎为 0，所以即使 α 只有 0.87，γ=4 也能拿到 2–3 倍；
# 同权重 draft 的 α=1 却因为 c=1 落在深色区（加速比恒为 1）。**这解释了为什么 EAGLE/MTP 要拼命把 draft 做小。**
#
# ### 实验 C · 接受/拒绝真的不改变输出分布吗？
#
# **问题**：修正后的输出分布还是不是 target？用统计检验回答。
#
# **协议**：用温度 1.0 的 target（分布接近均匀，卡方检验才有足够平稳的期望频数），
# 在同一个上下文上采 50000 个 token：一组直接从 target 采样，一组走"均匀 draft + 接受/拒绝 + 残差修正"。
# 把词表里概率最高的 20 个 token 单独成组、其余合并成"其他"，做卡方拟合优度检验。
# df=20 时 α=0.05 的临界值是 31.41，但单次固定 seed 的卡方本身有 ±√(2·df)≈6 的波动，
# 所以用更保守的 α=0.001 阈值 45.32。

# %%
N_COMPARE = 50_000


def compare_distributions(q: torch.Tensor, p: torch.Tensor, n: int = N_COMPARE, seed: int = 0):
    gen = torch.Generator(device=device).manual_seed(seed)
    direct = torch.multinomial(q, n, replacement=True, generator=gen)
    spec = spec_sample_once(q, p, n, gen)
    top = q.topk(20).indices
    def grouped(x):
        cnt = torch.bincount(x, minlength=V).float()
        return torch.cat([cnt[top], (n - cnt[top].sum())[None]])
    exp = torch.cat([q[top] * n, ((1 - q[top].sum()) * n)[None]])
    return top, grouped(direct), grouped(spec), exp


q_fixed = last_probs(target, prompt, 1.0)[0]
p_uniform = torch.full_like(q_fixed, 1.0 / V)
top20, hist_direct, hist_spec, expected = compare_distributions(q_fixed, p_uniform)
chi2_direct, chi2_spec = chi2_stat(hist_direct, expected), chi2_stat(hist_spec, expected)
crit20 = 45.32   # df=20，α=0.001 的保守临界值（常用 0.05 是 31.41，单次固定 seed 容易偶发超调）
assert max(chi2_direct, chi2_spec) < crit20, "输出分布被算法改变了！"
max_gap = float((hist_spec[:20] / N_COMPARE - q_fixed[top20]).abs().max())
assert max_gap < 4e-3, f"投机采样与 target 分布的频率差过大：{max_gap:.4f}"
print(f"卡方：直接采样 {chi2_direct:.1f}，投机采样 {chi2_spec:.1f}（阈值 {crit20}）；最大频率差 {max_gap:.4f}")

# %%
fig, ax = plt.subplots()
x = np.arange(12)
ax.bar(x - 0.27, q_fixed[top20[:12]].cpu(), 0.27, color="#7C3AED", label="target probability q")
ax.bar(x, (hist_direct[:12] / N_COMPARE).cpu(), 0.27, color="#2563EB", label="direct sampling")
ax.bar(x + 0.27, (hist_spec[:12] / N_COMPARE).cpu(), 0.27, color="#10B981", label="speculative sampling")
ax.set_xticks(x)
ax.set_xticklabels([f"{i.item()}" for i in top20[:12]], fontsize=8)
bar_max = max(q_fixed.max().item(), (hist_spec[:12] / N_COMPARE).max().item())
ax.set_ylim(0, bar_max * 1.45)
ax.set_xlabel("token id (top-12 by target probability)")
ax.set_ylabel("probability / frequency")
ax.set_title(f"Same distribution before/after: chi2 {chi2_spec:.1f} < critical {crit20}")
ax.legend(fontsize=9)
save(fig, "ch08_dist_check")

# %% [markdown]
# **读图**：三组柱子在 1/√n ≈ 0.7% 的采样误差内重合。**接受率不到 1（温度 1.0 下只有 ~0.87），
# 但输出分布完全没变**——被拒绝的那部分质量由 `(q−p)₊` 精确补回。这就是投机解码"免费加速"的前提：
# 它不是近似采样，而是**精确采样的一种更省串行步数的实现**。

# %% [markdown]
# ## 08.5 对照工业实现（vLLM）
#
# 本课程锁定 **vLLM 0.28.x**（v1 engine 默认）。本章 ~60 行手写循环，vLLM 把它拆成
# **proposer（猜）+ scheduler（排）+ rejection sampler（验）** 三层：
#
# | 我们的实现 | vLLM 中的对应物 | 真实文件（0.28.x） |
# | --- | --- | --- |
# | `model_drafter` | draft model / EAGLE / MTP / n-gram / suffix 等多种 proposer | `vllm/v1/spec_decode/`（`eagle.py`、`ngram_proposer.py`、`medusa.py`、`mlp_speculator.py`、`suffix_decoding.py`） |
# | `spec_decode` 的接受/拒绝 | `RejectionSampler`：批量化的 `min(1,q/p)` + 残差重采样 | `vllm/v1/sample/rejection_sampler.py` |
# | `(q−p)₊` 残差重采样 | 上述文件里的 modified rejection sampling kernel | 同上 |
# | KV cache 回滚 | block manager 释放/复用未接受的块 | `vllm/v1/core/block_pool.py`、`vllm/v1/core/kv_cache_manager.py` |
# | 每轮 γ 个 draft | `num_speculative_tokens` | `vllm/config/speculative.py`（`SpeculativeConfig`） |
# | 调度与验证位置 | scheduler 把 draft token 作为"待验证"计算 | `vllm/v1/core/sched/scheduler.py`、`vllm/v1/worker/gpu_model_runner.py` |
#
# 对应的 CLI（0.28.x）：
#
# ```bash
# # 独立 draft 模型（小模型猜、大模型验）
# vllm serve Qwen/Qwen3-8B \
#   --speculative-config '{"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 5}'
#
# # EAGLE / EAGLE-3 草稿头（c 极小，工业上最常用）
# vllm serve Qwen/Qwen3-8B \
#   --speculative-config '{"model": "Qwen/Qwen3-8B-EAGLE3", "method": "eagle3", "num_speculative_tokens": 3}'
#
# # MTP（DeepSeek 系模型自带的多 token 预测头）/ n-gram（零成本，适合重复性强的输出）
# vllm serve deepseek-ai/DeepSeek-V3 \
#   --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
# vllm serve Qwen/Qwen3-8B --speculative-config '{"method": "ngram", "num_speculative_tokens": 4}'
# ```

# %% [markdown]
# 关键区别：
#
# - **draft 成本 c**：我们手写的 draft 与 target 同尺寸（c≈1），所以只能验证"α=1 时无收益"这个负结果；
#   工业 draft 是一个 0.6B 小模型（c≈0.05–0.1）或 EAGLE 的 1 层 draft 头（c≈0.02–0.05），
#   同样的 α 能从 1x 变成 2–3x。**这正是本章热力图右上角的意义。**
# - **kernel**：我们的接受/拒绝是 Python 逐 token 循环 + `.item()` 同步；vLLM 把整个 batch 的
#   接受/拒绝、残差采样、贪心校验写成 Triton/CUDA kernel，一次处理所有请求，几乎没有 Python 开销。
# - **调度**：我们一次只服务 1 条请求；vLLM 要把 draft token 计入 `num_computed_tokens`、
#   给未接受的块"退款"，还要在 batch 里混入不同 γ 的请求（spec token 数可以每请求不同）。
# - **贪心与采样**：temperature=0 时 vLLM 走专门的 greedy 校验（argmax 一致就接受），
#   不采样、不修正；本章的采样版是更一般的情况。
# - **怎么盯指标**：vLLM 暴露 draft/accepted token 计数（Prometheus 里 `vllm:spec_decode_*` 系列，
#   0.28.x 具体名字见 `vllm/v1/metrics/`），用 `accepted / draft` 得到实际 α——
#   **α 掉下来就该降 `num_speculative_tokens` 或关掉投机**，这就是本章加速比模型的生产用法。
#
# 一句话总结：**投机解码不改模型、不改精度，只改"谁来写草稿、以什么成本写"。** 它把 decode 从
# "每 token 一次大前向"变成"每轮一次大前向 + 一堆便宜的猜测"，收益上限 = γ+1，实际收益由 α 与 c 决定。

# %% [markdown]
# ## 08.6 练习与小结
#
# > **一句话记住**：投机解码是拿「便宜的猜测」换「昂贵的串行步数」——
# > α 是赔率、c 是成本；α 不够高时，猜得越多亏得越快。
#
# **练习**（参考答案见 `solutions/08_exercises.py`）
#
# 1. ⭐ 把 `gamma` 从 1 扫到 8、温度从 1.0 扫到 0.1，记录 `实测加速比`。解释为什么高 α 时 γ 越大越好，
#    而低 α 时 γ 增大反而没有收益（提示：看 `(1−α^{γ+1})/(1−α)` 的饱和速度与 `1+γc` 的增长）。
# 2. ⭐ 把 `uniform_drafter` 换成"复制最后一个 token"的 draft（成本同样约 0），在温度 0.1 下重跑。
#    α 会怎么变？为什么在随机权重 TinyGPT 上它会异常高（提示：看 logits 的自点积结构）？
# 3. ⭐⭐ 实现 **贪心版**（temperature=0）投机解码：接受条件变成 `argmax q == argmax p == draft token`，
#    拒绝后直接把 target 的 argmax 作为输出。验证输出序列与 `generate_cached(temperature=0)` 完全一致。
# 4. ⭐⭐ 实现 **树状 draft**（EAGLE 风格）：draft 一次生成一棵深度 2–3 的树，验证时 target 一次前向
#    评估所有分支，接受最长合法前缀。给出每轮平均接受 token 数的对比数字。
# 5. ⭐⭐⭐ 设计一次**真实模型实验**（可用 `vllm` + Qwen3-0.6B draft + 8B target）：给出测量方案、
#    需要记录的指标（α、c、TPOT、接受长度分布）、以及"什么条件下应该关闭投机"的判定规则。评分要点：
#    基线是否公平（同 batch、同温度）、α 是否按任务分组、是否解释了 c 的来源、是否给出 SLO 影响。
#
# %% [markdown]
# **自检 3 问**（不看笔记能答出来才算过）：
#
# 1. 拒绝后为什么必须从 `(q−p)₊` 重采？直接从 `q` 重采会偏向谁，为什么？
# 2. 同权重 draft 的 α=1，为什么加速比不是 γ+1 而是 1×？要让 α=1 兑现成 γ+1，还缺什么条件？
# 3. α=0.5、c=0.1 时，把 γ 从 2 加到 8 是更快还是更慢？用 08.2 的公式说出判断依据。
#
# **工作场景题**（不是练习题，是你在公司会遇到的那种问题）：
#
# 你的代码补全服务上了 EAGLE 投机，SLO 是 P99 < 300ms。某天 P99 变差，`accepted / draft` 指标
# 掉了一半。写出排查顺序：先看 α 的事后分布（按任务类型 / 温度分组），再看 c 是否变了，
# 最后检查 `num_speculative_tokens` 是否还停在原来的值。给出「先降 γ 还是先换 draft」的判据，
# 并说明为什么不能直接关掉投机了事。
#
# **你现在应该能回答：**
#
# - `min(1, q/p)` + `(q−p)₊` 为什么能保证输出分布不变？不做修正会怎样？
# - α、γ、c 如何共同决定加速比；为什么同尺寸 draft 永远不赚钱。
# - 实测 TinyGPT 的 α 随温度怎么变，为什么低 α 时投机反而比 baseline 慢。
# - vLLM 的 `--speculative-config`、EAGLE/MTP/n-gram 对应本章哪个部件，源码在 `vllm/v1/spec_decode/`
#   与 `vllm/v1/sample/rejection_sampler.py`。
#
# **小结与下一章**
#
# 投机解码把「每 token 一次 target 前向」拆成「draft 猜 + target 一次验证」，收益由 α 与 c 决定，
# 正确性由残差修正保证。07 章的量化省带宽、08 章的投机省串行步数，两者都可以叠加在同一个服务上。
# 09 章开始读 vLLM 源码，看这些机制在工业实现里如何被拆成 scheduler / worker / sampler 三层。
#
# **延伸阅读**
#
# - Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*（2022）
# - Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*（2023）
# - EAGLE / EAGLE-2 / EAGLE-3、DeepSeek-V3 的 MTP 头
# - vLLM 文档：Speculative Decoding；源码 `vllm/v1/spec_decode/` 与 `vllm/v1/sample/rejection_sampler.py`
