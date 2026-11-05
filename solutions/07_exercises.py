"""第 07 章练习参考答案（对应 notebooks/07_quantization.py 的 5 道练习）。

直接用 ``.venv/bin/python solutions/07_exercises.py`` 运行，全部是 CPU 上的小实验，几秒内跑完。
正文里的函数与章节里一致（``absmax_scale`` / ``quantize`` / ``dequantize`` / ``quant_error``），
为了自包含这里重新实现一遍。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch.manual_seed(0)


def absmax_scale(x, bits, granularity="tensor", group_size=None):
    qmax = 2 ** (bits - 1) - 1
    if granularity == "tensor":
        amax = x.abs().amax()
    elif granularity == "channel":
        amax = x.abs().amax(dim=1, keepdim=True)
    else:
        out_f, in_f = x.shape
        amax = x.reshape(out_f, in_f // group_size, group_size).abs().amax(-1, keepdim=True)
        amax = amax.expand(out_f, in_f // group_size, group_size).reshape(out_f, in_f)
    return (amax / qmax).clamp_min(1e-8), qmax


def quantize(x, bits, granularity="tensor", group_size=None):
    scale, qmax = absmax_scale(x, bits, granularity, group_size)
    return torch.clamp(torch.round(x / scale), -qmax, qmax).to(torch.int8), scale


def dequantize(q, scale):
    return q.to(torch.float32) * scale


def quant_error(x, x_hat):
    err = (x - x_hat).float()
    return {
        "MSE": err.pow(2).mean().item(),
        "最大绝对误差": err.abs().max().item(),
        "相对误差": (err.norm() / x.norm()).item(),
    }


def quant_bytes(shape, bits, granularity="tensor", group_size=None):
    n = shape[0] * shape[1]
    if granularity == "tensor":
        n_scales = 1
    elif granularity == "channel":
        n_scales = shape[0]
    else:
        n_scales = shape[0] * (shape[1] // group_size)
    return n * bits / 8 + n_scales * 2


w = torch.randn(768, 256) * 0.03
w[:, 5] *= 25.0
w[3, ::41] *= 18.0


# --------------------------------------------------------------------------- #
# 练习 1（⭐）：bits 8/4/2 只用 per-tensor，误差怎么变？
# --------------------------------------------------------------------------- #
def exercise_1() -> None:
    print("=" * 70)
    print("练习 1：per-tensor 下位宽对相对误差的影响")
    rows = []
    for bits in (8, 4, 2):
        x_hat = dequantize(*quantize(w, bits, "tensor"))
        rows.append({"bits": bits, "qmax": 2 ** (bits - 1) - 1, **quant_error(w, x_hat)})
    print(pd.DataFrame(rows).round(5).to_string(index=False))
    print(
        "答案：相对误差大约按 2 倍位宽比例恶化。原因：s = absmax/qmax，"
        "qmax 从 127 → 7 → 1，步长分别放大 18x、127x；量化噪声 ~ s/√12，"
        "而 b=2 时 qmax=1，权重几乎只剩符号，误差接近 100%。"
    )


# --------------------------------------------------------------------------- #
# 练习 2（⭐）：group_size 16/32/64/128/256/512 的误差-字节权衡
# --------------------------------------------------------------------------- #
def exercise_2() -> None:
    print("=" * 70)
    print("练习 2：int4 group_size 扫描（误差 vs 每行字节）")
    rows = []
    for gs in (16, 32, 64, 128, 256):
        x_hat = dequantize(*quantize(w, 4, "group", gs))
        rows.append(
            {
                "group_size": gs,
                "相对误差": round(quant_error(w, x_hat)["相对误差"], 4),
                "每行字节": round(quant_bytes(w.shape, 4, "group", gs) / w.shape[0], 1),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "答案：误差随 group_size 减小单调下降，从 128→16 误差降了 2.6 倍，"
        "但每行字节从 132 涨到 160（scale 开销 ~3% → ~25%）。128 附近是"
        "'误差-字节'曲线的拐点；再细要重新设计 kernel（Marlin 支持 32/64/128 的 group）。"
    )


# --------------------------------------------------------------------------- #
# 练习 3（⭐⭐）：非对称量化（带 zero-point）
# --------------------------------------------------------------------------- #
def quantize_asymmetric(x, bits, granularity="tensor", group_size=128):
    """q = clamp(round(x/s) + z, 0, 2^b-1)，z = round(-min/s)。返回 (q, s, z)。"""
    qmax = 2**bits - 1
    if granularity == "tensor":
        lo, hi = x.min(), x.max()
    elif granularity == "channel":
        lo, hi = x.min(dim=1, keepdim=True).values, x.max(dim=1, keepdim=True).values
    else:
        out_f, in_f = x.shape
        lo = x.reshape(out_f, in_f // group_size, group_size).min(-1, keepdim=True).values
        hi = x.reshape(out_f, in_f // group_size, group_size).max(-1, keepdim=True).values
        lo = lo.expand(out_f, -1, group_size).reshape(out_f, in_f)
        hi = hi.expand(out_f, -1, group_size).reshape(out_f, in_f)
    scale = ((hi - lo) / qmax).clamp_min(1e-8)
    zero = torch.round(-lo / scale).clamp(0, qmax)
    q = torch.clamp(torch.round(x / scale) + zero, 0, qmax)
    return q, scale, zero


def dequantize_asymmetric(q, scale, zero):
    return (q.to(torch.float32) - zero) * scale


def exercise_3() -> None:
    print("=" * 70)
    print("练习 3：对称 vs 非对称量化（ReLU 后的偏斜分布）")
    w_relu = torch.relu(torch.randn(512, 256))  # 一半是 0，右偏
    rows = []
    for name, x_hat in [
        ("对称 int8 per-tensor", dequantize(*quantize(w_relu, 8, "tensor"))),
        ("非对称 int8 per-tensor", dequantize_asymmetric(*quantize_asymmetric(w_relu, 8, "tensor"))),
        ("非对称 int8 group-128", dequantize_asymmetric(*quantize_asymmetric(w_relu, 8, "group", 128))),
    ]:
        rows.append({"方案": name, **{k: round(v, 6) for k, v in quant_error(w_relu, x_hat).items()}})
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "答案：对 ReLU 这类全非负、长尾的分布，非对称量化把 [0, max] 映射到 256 个格点，"
        "而对称量化把 [-max, max] 映射到 256 个格点——有效分辨率差一倍，误差也大约差一倍。"
        "但非对称要额外存 zero-point，且 kernel 多一次加法；在 int4 + 分组场景下"
        "工业实现仍偏向对称（AWQ/GPTQ 的 group 量化），用分组而不是零点来解决偏斜。"
    )


# --------------------------------------------------------------------------- #
# 练习 4（⭐⭐）：简版 GPTQ —— 逐列量化 + 误差补偿
# --------------------------------------------------------------------------- #
def gptq_quantize(w: torch.Tensor, hessian: torch.Tensor, bits: int = 4, damp: float = 0.01) -> torch.Tensor:
    """简版 GPTQ：逐列量化，并用 H⁻¹ 把每列的量化误差补偿到剩余列。

    w: (out, in)，hessian: (in, in) 近似 H = 2·XᵀX。返回反量化后的权重。
    """
    qmax = 2 ** (bits - 1) - 1
    w_work = w.clone().float()
    h = hessian.clone().float()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    idx = torch.arange(h.shape[0])
    h[idx, idx] += damp * h[idx, idx].mean()          # damp 防止奇异
    hinv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    q_hat = torch.zeros_like(w_work)
    for j in range(w_work.shape[1]):
        col = w_work[:, j]
        scale = (col.abs().amax() / qmax).clamp_min(1e-8)
        q = torch.clamp(torch.round(col / scale), -qmax, qmax)
        q_hat[:, j] = q * scale
        err = (col - q * scale) / hinv[j, j].clamp_min(1e-8)   # 本列误差按 H⁻¹ 传播
        w_work[:, j:] -= err[:, None] * hinv[j, j:][None, :]
    return q_hat


def exercise_4() -> None:
    print("=" * 70)
    print("练习 4：naive round vs GPTQ 式误差补偿（看输出误差，而不是权重误差）")
    torch.manual_seed(0)
    x = torch.randn(512, 256)
    w_lin = torch.randn(512, 256) * 0.03
    hessian = 2 * (x.T @ x) / x.shape[0]
    scale = (w_lin.abs().amax(dim=1, keepdim=True) / 7).clamp_min(1e-8)
    naive = torch.clamp(torch.round(w_lin / scale), -7, 7) * scale
    gptq = gptq_quantize(w_lin, hessian, 4)

    def output_error(w_hat: torch.Tensor) -> float:
        return ((w_lin - w_hat) @ x.T).norm().item() / (w_lin @ x.T).norm().item()

    rows = [
        {"方案": "naive round (per-channel int4)",
         "输出相对误差": round(output_error(naive), 5),
         "权重相对误差": round(((w_lin - naive).norm() / w_lin.norm()).item(), 5)},
        {"方案": "GPTQ 式误差补偿 (int4)",
         "输出相对误差": round(output_error(gptq), 5),
         "权重相对误差": round(((w_lin - gptq).norm() / w_lin.norm()).item(), 5)},
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "答案：GPTQ 优化的是输出误差 ‖(W−Ŵ)X‖，不是权重 MSE。所以常见现象是"
        "『权重误差反而变大、输出误差变小』——上面两行正是这个反直觉结果。"
        "完整 GPTQ 还会配合 group-128、按 block 更新 H⁻¹，并在真实校准集上跑，"
        "LLaMA 类模型 int4 的困惑度退化因此能被压到很小。"
    )


# --------------------------------------------------------------------------- #
# 练习 5（⭐⭐⭐）：AWQ 式激活感知缩放（设计题，评分要点）
# --------------------------------------------------------------------------- #
def exercise_5() -> None:
    print("=" * 70)
    print("练习 5（开放题）参考设计：AWQ 式激活感知缩放")
    print(
        """
    目标：寻找 per-input-channel 缩放 s_j，把权重侧离群值搬走。

    做法（等价变换，不改变数学结果）：
        y = W x = (W · diag(s)^{-1}) · (diag(s) x)
    对 W' = W·diag(s)^{-1} 做分组 int4，x' = diag(s)x 在 fp16 里算。

    s_j 的搜索：AWQ 论文用激活幅值 A_j = mean(|x_j|)（在少量校准数据上统计），
    令 s_j = A_j^α，α ∈ [0,1] 网格搜索，目标是最小化输出误差 ‖W x − Ŵ' x'‖。
    α=0 退化为不变换，α=1 完全按激活缩放。

    评分要点：
    1. 是否真的降低了 per-tensor / per-group int4 的权重相对误差（给出数字）；
    2. 是否报告了输出误差/困惑度，而不只是权重 MSE；
    3. 是否说明 kernel 代价：缩放可以融合进前一层的输出（残差前乘），几乎零额外开销；
    4. 是否讨论了"显著通道保留 fp16"这一扩展（AWQ 的 salient channel）；
    5. 是否在长尾/离群值构造的合成权重上做消融，而不是只报一个总数。
    """
    )


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
