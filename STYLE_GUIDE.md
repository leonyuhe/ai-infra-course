# 写作与协作规范（STYLE GUIDE）

本文件是多人协作的**契约**。所有章节必须遵守，否则会在集成时被打回。

---

## 1. 目录与命名

- 章节源文件：`notebooks/NN_snake_case_title.py`（percent 格式），构建产物同名 `.ipynb`。
  - 例：`notebooks/03_paged_attention.py` → `notebooks/03_paged_attention.ipynb`
- 概念图：`assets/figures/chNN_topic.svg`（全小写 + 下划线），例 `ch03_block_table.svg`
- 运行生成的图：用 `aiinfra.plotting.save(fig, "ch03_fragmentation")` → 存到 `outputs/ch03_fragmentation.png`
- 练习答案：`solutions/NN_exercises.py` 或 `solutions/NN_exercises.ipynb`
- 云上资产：`cloud/scripts/NN_xxx.sh`、`cloud/docker/`、`cloud/k8s/`

**禁止**修改不属于自己写作用域的文件（见第 9 节）。共享库 `aiinfra/` 只由 Lead 修改；需要新工具先在群里提。

---

## 2. Notebook 结构（严格模板）

从 `notebooks/_TEMPLATE.py` 复制。顺序与标题必须一致：

```
# %% [markdown]
# # 第 NN 章 · 标题
#
# > 一句话说明本章解决什么问题。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 45–60 分钟 |
# | 运行环境 | 🏠 Mac (CPU/MPS) · 无需 GPU |
# | 前置章节 | 02 |
# | 本章代码 | `aiinfra/tinygpt.py` 的 `KVCache` 接口 |
#
# **学完你能回答：**
# 1. ...
# 2. ...

# %% [markdown]
# ## NN.1 问题与直觉
# （图 + 2–4 段文字，先给结论再给推导）
# ![](../assets/figures/chNN_topic.svg)

# %% [markdown]
# ## NN.2 从零实现
# %% 
# 核心实现（带 assert）
# %% [markdown]
# ### 逐段解释
# %%
# 小实验 1

# %% [markdown]
# ## NN.3 实测
# %% 
# 表格 + 图

# %% [markdown]
# ## NN.4 对照工业实现（vLLM）
# （真实路径 + 类名 + 参数；说明差距）

# %% [markdown]
# ## NN.5 练习与小结
# **练习**（⭐ 改参数 / ⭐⭐ 改实现 / ⭐⭐⭐ 开放设计）
# 1. ⭐ ...
# **你现在应该能回答：**
# - ...
# **延伸阅读**
# - ...
```

规则：

- **每个代码 cell ≤ 40 行**，超过就拆；拆完在 markdown 里解释上一段。
- 每个代码 cell 必须能单独运行（只依赖前面 cell + `aiinfra`），**不要**依赖「先运行最后一段」。
- 不写 `plt.show()` 之外的输出魔法；表格优先用 `pandas.DataFrame` 直接显示。
- 章节必须**自包含**：不 import 其他章节的文件。共享代码进 `aiinfra/`，否则在各章内联重写（教学上允许重复，读者不用跳章）。

---

## 3. 代码规范

- Python 3.11+，`from __future__ import annotations` 可选；类型标注用于公开接口。
- 依赖只允许：`torch`、`numpy`、`pandas`、`matplotlib`、`psutil`、标准库。**禁止** transformers/vllm 作为本地章的运行时依赖。
- 随机性：每个 notebook 顶部 `torch.manual_seed(0); np.random.seed(0)`；实验前 `torch.use_deterministic_algorithms` 不强求。
- 设备选择统一：

```python
from aiinfra.bench import pick_device, sync
device = pick_device()          # mps > cuda > cpu
```

- 计时统一用 `aiinfra.bench.bench(fn, warmup=3, reps=5)`，禁止 `time.time()` 裸测。
- 显存/内存统计统一用 `aiinfra.bench.mem_report()`。
- 断言即测试：关键不变量用 `assert` 写出来（形状、块数、命中率、等价性）。
- 所有实验规模必须**本地 < 3 分钟**：TinyGPT 约 2.8M 参数；序列默认 ≤ 512，缓存类章节（KV cache / 分页 / 前缀缓存）可放宽到 4096–8192，但单章总时长仍须 < 3 分钟。
- **小模型测量地板**：MPS 上每次 forward 有约 1–1.5ms 的固定开销（kernel 启动 + 命令缓冲）。凡是收益来自「减少计算量」的实验（前缀缓存、分页 gather），必须同时给出理论值、实测值，并注明固定开销占比；必要时补一组 CPU 对照（CPU 无异步命令缓冲，更能暴露计算收益）。

---

## 4. 图表规范

### 4.1 概念图（手写 SVG）

- 画布：`viewBox="0 0 960 520"`，`width="960"`，白底。
- 配色（只允许这 6 色 + 灰阶）：

| 语义 | 色值 | 用途 |
| --- | --- | --- |
| 主色 | `#2563EB` | 核心模块、数据流主线 |
| 辅色 | `#7C3AED` | 次要素、缓存 |
| 强调 | `#F59E0B` | 关键差异、警告点 |
| 成功 | `#10B981` | 命中/复用/优化收益 |
| 危险 | `#EF4444` | 浪费/碎片/失败 |
| 文字 | `#111827` | 正文 |
| 浅底 | `#F3F4F6` | 容器背景 |

- 字体：`font-family="Inter, 'PingFang SC', 'Noto Sans SC', sans-serif"`；标题 22px，正文 15px，注释 13px。
- 圆角矩形 `rx="8"`，描边 2px；箭头用 `<marker>` 定义一次复用；线宽 2px。
- **图中文字一律用英文**（标题、方框、箭头标注、图例）。原因：课程渲染环境（cairosvg / 无中文字体的容器 / 导出 PDF）会把中文渲染成方格子。notebook 正文仍用中文，只有图内文字必须英文。
- 英文写法优先用术语原名：`KV cache`、`decode step`、`block table`、`prefill`，不要用拼音或过长句子。
- 必须包含 `<title>` 与 `<desc>`（无障碍 + 可搜索）。
- 手写 SVG 容易错，写完用浏览器/`qlmanage -t` 截图自检，或至少确认 XML 合法（`xmllint --noout`）。

### 4.2 实测图（matplotlib）

- 统一入口：

```python
from aiinfra.plotting import setup_style, save
setup_style()
fig, ax = plt.subplots(figsize=(7.2, 4.2))
...
save(fig, "chNN_xxx")   # -> outputs/chNN_xxx.png，自动 300dpi
```

- `setup_style()` 会设置字体、网格、配色、尺寸；不要自己 `rcParams`。
- 每张图必须：坐标轴有单位（`tokens/s`、`ms`）、图例、标题写「现象」而不是「图 1」。
- 图内所有文字（title / xlabel / ylabel / legend / annotate）一律英文；DataFrame 表格与 notebook 正文可用中文。
- 数据点 ≥ 3 个；若曲线有噪声，画 5 次重复的中位数与 min–max 阴影。
- **中文列名 workaround**：`check_notebooks.py` 会检查含 `label=` / `set_title` 等绘图语句的行是否出现中文。若 DataFrame 列名是中文，请先取到英文局部变量再画（如 `tps = df["吞吐"]`），不要把中文列名直接写进绘图调用。

---

## 5. 数字与结论的写法

- 每个实测结论必须带**条件**：「在 M3 Ultra / 96GB / torch 2.x / batch=8 下，TTFT 中位数从 120ms 降到 45ms」。
- 禁止只写「显著提升」。必须给绝对值和相对值。
- 理论值与实测值并排给出，并解释差距来源（kernel 启动、内存分配、Python 开销…）。
- 不确定的地方写「待验证」，不要编造数字。

---

## 6. 练习与答案

- 每题标注难度和预计用时。
- ⭐ 题：改一个参数，回答「为什么」。
- ⭐⭐ 题：改一段实现，要求跑通并给出对比数字。
- ⭐⭐⭐ 题：开放设计，给出评分要点（3–5 条）。
- 参考答案放 `solutions/`，正文只给提示，不贴答案。

---

## 7. 版本锁定

- `requirements.txt` 锁定：`torch`、`matplotlib`、`numpy`、`pandas`、`nbformat`、`nbclient`。
- vLLM 相关章节（09–14）标注适配版本（写进章节头表格，例：`vLLM 0.28.x`），源码路径按该版本给出。
- Mac 上真服务用 `vllm-metal`（标注版本），云上用官方 `vllm` 镜像。

---

## 8. 构建与验证（每章提交前必须跑）

```bash
cd ai-infra-course
# 1) 生成 ipynb
.venv/bin/python tools/build_notebooks.py            # 全部
.venv/bin/python tools/build_notebooks.py --only 03  # 只构建第 3 章

# 2) 无头执行（会真正跑代码，失败即报错）
.venv/bin/python tools/execute_notebooks.py --only 03 --timeout 300

# 3) 校验
.venv/bin/python tools/check_notebooks.py --only 03
```

通过标准：执行无异常、耗时 < 3 分钟、`outputs/` 里有图、无 `TODO` 字样。

---

## 9. 协作与写作用域

| 角色 | 写作用域 |
| --- | --- |
| Lead | `aiinfra/`、`tools/`、`COURSE_DESIGN.md`、`STYLE_GUIDE.md`、`README.md`、`notebooks/00*`、`notebooks/01*`、`notebooks/_TEMPLATE.py`、最终集成 |
| cache chapters | `notebooks/02*`、`03`、`04`、对应 `assets/figures/ch02-04_*.svg`、`solutions/02-04*` |
| sched chapters | `notebooks/05*`、`06`、对应图与答案 |
| vllm chapters | `notebooks/09*`、`10`、`11`、`cloud/`、对应图与答案 |
| optimization | `notebooks/07*`、`08`、对应图与答案 |
| training/cloud | `notebooks/12*`、`13`、`14`、`cloud/`（与 vllm 组按文件前缀分工） |

规则：

1. 只改自己作用域内的文件；共享文件（`aiinfra/`、`tools/`）只读。
2. 新增共享工具需求 → 消息 Lead，由 Lead 加，避免并行冲突。
3. 提交前跑第 8 节三条命令，把「执行耗时 / 关键数字 / 生成的图」写进完成报告。
4. 图表编号全局唯一，避免覆盖别人的 `outputs/` 文件。

---

## 10. 语气与语言

- 正文中文，术语首次出现给英文（例：「分页注意力（PagedAttention）」）。
- 第二人称、短句、先结论后论证；禁止「众所周知」「显然」。
- 每章开头一句话回答「为什么这章值得学」，结尾一句话回答「下一章要解决它留下的什么问题」。
- 代码注释中文，变量名英文，参数名与 vLLM 保持一致（`max_num_seqs`、`block_size`…）。
