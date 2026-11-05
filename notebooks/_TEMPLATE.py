# %% [markdown]
# # 第 NN 章 · 章节标题
#
# > 一句话：本章解决什么问题，为什么值得学。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 60 分钟 |
# | 运行环境 | 🏠 Mac（CPU/MPS）· 无需 GPU |
# | 前置章节 | NN-1 |
# | 本章接口 | 复用 `aiinfra.tinygpt` 的 `TinyGPT` / `KVCache` |
#
# **学完你能回答：**
# 1. 问题一
# 2. 问题二
# 3. 问题三
#
# <!-- 作者注：写完删除本行以上所有占位符；标题编号必须与文件名一致 -->

# %% [markdown]
# ## NN.1 问题与直觉
#
# <!-- 作者注：先给结论，再给推导。必须引用一张 assets/figures/chNN_xxx.svg -->
#
# 先看一张图：
#
# ![](../assets/figures/chNN_topic.svg)
#
# 用 2–4 段话解释图里的三个关键点，并给出一个**反例**：不做这件事会怎样。

# %% [markdown]
# ## NN.2 从零实现
#
# <!-- 作者注：核心实现 ≤ 40 行/单元，逐块解释；每个关键不变量用 assert 写出来 -->

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path.cwd().parent))

from aiinfra.bench import bench, pick_device, sync
from aiinfra.plotting import save, setup_style

setup_style()
device = pick_device()
torch.manual_seed(0)
np.random.seed(0)

print("device =", device)

# %% [markdown]
# ### 数据结构与接口
#
# <!-- 作者注：先定义接口（dataclass / 类签名），再填实现 -->
# %%
# TODO(作者): 定义本章核心类/函数


# %% [markdown]
# ### 最小可运行实现
# %%
# TODO(作者): 实现 + assert


# %% [markdown]
# ## NN.3 实测
#
# <!-- 作者注：固定 seed，warmup=3，reps≥5；先出表格，再出图；解释趋势 -->

# %%
rows = []
# TODO(作者): 跑实验，收集数据
df = pd.DataFrame(rows)
df

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
# TODO(作者): 画图
save(fig, "chNN_xxx")

# %% [markdown]
# 读图：**现象** → **原因** → **对 vLLM 参数的启示**。
#
# <!-- 作者注：必须给出理论值与实测值的对比，并解释差距 -->

# %% [markdown]
# ## NN.4 对照工业实现（vLLM）
#
# <!-- 作者注：给出 vLLM 0.28.x 的真实路径/类名/参数；说明我们 100 行 vs 工业实现差在哪 -->
#
# | 我们的实现 | vLLM 中的对应物 | 文件 |
# | --- | --- | --- |
# | `XXX` | `YYY` | `vllm/.../zzz.py` |

# %% [markdown]
# ## NN.5 练习与小结
#
# **练习**
#
# 1. ⭐ 改一个参数，解释为什么。
# 2. ⭐⭐ 改一段实现，跑通并给出对比数字。
# 3. ⭐⭐⭐ 开放设计题（给出评分要点）。
#
# **你现在应该能回答：**
#
# - ...
#
# **小结与下一章**
#
# 本章留下了什么问题？下一章如何解决？
#
# **延伸阅读**
#
# - vLLM 源码路径 / 论文 / 官方文档
