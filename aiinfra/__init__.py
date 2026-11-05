"""AI Infra 从零构建 · 课程共享库。

Notebook 只依赖这里提供的四类工具：
- ``aiinfra.tinygpt``：随机权重的微型 GPT（02–08 章的实验对象）
- ``aiinfra.bench``：计时、环境快照、显存/带宽估算
- ``aiinfra.plotting``：统一图表风格与保存
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["tinygpt", "bench", "plotting"]
