# AI Infra 从零构建 · 课程设计

> 参考 *Build a Large Language Model (From Scratch)*（Sebastian Raschka）的教学法：
> **先建立直觉 → 从零手写最小实现 → 亲手测量 → 对照工业实现 → 练习迁移**。
> 区别在于：那本书「从零造模型」，这套课「从零造**推理与训练基础设施**」，vLLM 是最后一站的工业参照物，而不是第一站的 API。

---

## 1. 课程定位

| 项目 | 内容 |
| --- | --- |
| 面向 | 已有 Python 基础、了解 Transformer 大致结构、需要真正吃透 LLM 服务化的工程师 / 平台 / 算法同学 |
| 目标 | 学完能独立回答：显存怎么算、吞吐为什么上不去、该调哪个参数、vLLM 源码该看哪里、上云怎么选卡算钱 |
| 不假设 | 不要求会写 CUDA、不要求有 GPU、不要求读过 vLLM 源码 |
| 硬件 | 一台 Apple Silicon Mac（M3 Ultra / 96GB 已验证）可完成 12/15 章；3 章标注「云上执行」 |
| 形态 | 每章一个独立 Jupyter Notebook（`.ipynb` 由 `.py` 源文件生成），配概念 SVG 图 + 实测图表 + 练习 |

### 为什么不是「vLLM 使用教程」

vLLM 的命令行 30 分钟就能讲完，但学员记不住。真正让人「吃透」的是**可观测的因果关系**：

> 把 `max-num-batched-tokens` 从 2048 调到 8192 → 看 TTFT 和 TPOT 怎么变 → 用 KV cache 公式解释为什么 → 再去源码里找到那个决定这一行为的 scheduler 分支。

每章都必须产出**数字**：延迟、吞吐、显存、误差、加速比。没有数字的章节不算完成。

---

## 2. 教学法：Raschka 风格的五段式

每章固定五段，Notebook 内用小标题分隔：

1. **问题与直觉**（图 + 一句话）
   - 一张概念 SVG：数据流 / 内存布局 / 时间轴，先让读者"看见"。
   - 一个反例：如果不这么做会发生什么（例如不用 KV cache 会重复计算 attention）。
2. **从零实现**（50–150 行核心代码）
   - 只用 PyTorch 基础算子，不引入框架魔法；变量命名与 vLLM 源码尽量对齐。
   - 每一小块都有「为什么这么写」的注释与断言（`assert` 即测试）。
3. **实测**（表格 + 图）
   - 固定 seed、warmup、重复 5 次取中位数；记录硬件信息。
   - 必须画出至少一张随参数变化的曲线。
4. **对照工业实现**（vLLM 源码 / 参数 / 论文）
   - 给出真实文件路径 + 类名 + 关键参数，说明"我们 100 行 vs 它 10000 行"的差距在哪。
5. **练习与小结**
   - 3–5 题，标 ⭐（改参数）/ ⭐⭐（改实现）/ ⭐⭐⭐（开放设计）。
   - 参考答案放在 `solutions/`，正文不剧透。

---

## 3. 课程地图

图例：🏠 本地可跑（Mac） · ☁️ 需要 NVIDIA GPU（代码与文档先写好，上云执行） · 🧪 以模拟/推算为主

### Part 0 · 度量（没有度量就没有 infra）

| # | 章节 | 核心产出 | 图 | 环境 |
| --- | --- | --- | --- | --- |
| 00 | 环境、硬件与度量方法 | 统一的 benchmark 工具：warmup、分位数、同步、环境快照 | 测量误差来源图 | 🏠 |
| 01 | 一次前向：从 matmul 到 token 采样 | 手写 tiny GPT 前向 + 采样；roofline 与算术强度 | 计算图 / roofline | 🏠 |

### Part 1 · 推理内核：从零到 PagedAttention

| # | 章节 | 核心产出 | 图 | 环境 |
| --- | --- | --- | --- | --- |
| 02 | KV Cache | 手写 KV cache；显存公式 vs 实测；prefill/decode 两阶段 | 显存布局、时间轴 | 🏠 |
| 03 | PagedAttention | 手写 block manager + block table + 分页 attention；碎片实验 | 连续分配 vs 分页 | 🏠 |
| 04 | 前缀缓存 | 手写 hash 前缀缓存；TTFT 命中/未命中对比 | 前缀树与 block 复用 | 🏠 |
| 05 | 调度器 | 手写静态批处理 vs 连续批处理调度器；吞吐/延迟曲线 | 时间轴甘特图 | 🏠 |
| 06 | Chunked Prefill 与抢占 | 手写 chunked prefill、抢占重算；长 prompt 尾延迟 | 调度时间轴 | 🏠 |

### Part 2 · 推理优化

| # | 章节 | 核心产出 | 图 | 环境 |
| --- | --- | --- | --- | --- |
| 07 | 量化 | 手写 int8/int4 对称量化、per-channel/group；误差与带宽收益 | 量化误差分布、roofline 移动 | 🏠 |
| 08 | 投机解码 | 手写 draft-verify 循环；接受率 → 加速比模型与实测 | 验证过程图 | 🏠 |

### Part 3 · vLLM：工业级实现

| # | 章节 | 核心产出 | 图 | 环境 |
| --- | --- | --- | --- | --- |
| 09 | vLLM 架构与源码地图 | 从 `LLMEngine` 到 `Scheduler`/`BlockManager`/`Worker` 的调用链；和我们手写实现的逐项对照 | 分层架构图、请求生命周期 | 🏠（读码） |
| 10 | 用 vLLM 提供服务 | 起服务、OpenAI API、LoRA、结构化输出、工具调用；接口契约与压测基线 | 部署拓扑图 | 🏠 vllm-metal 或 ☁️ |
| 11 | 压测与调优 | `vllm bench serve`、Prometheus 指标、参数扫描、瓶颈判定决策树 | 调优决策树、扫描热力图 | 🏠/☁️ |

### Part 4 · 上云与训练侧（只写代码，后续上云执行）

| # | 章节 | 核心产出 | 图 | 环境 |
| --- | --- | --- | --- | --- |
| 12 | 多卡并行 | TP/PP/DP 的分片与通信；本地用线程模拟 all-reduce；云上 `--tensor-parallel-size` 实测脚本 | 切分示意图、通信时间轴 | ☁️ |
| 13 | 训练侧 infra | DDP/ZeRO 显存账本模拟 + LoRA 微调闭环（训练→合并→vLLM 部署） | 显存账本、闭环图 | ☁️ |
| 14 | Capstone：容量规划 | 给定业务 SLO，选模型/精度/卡型/实例数，出成本表与压测方案 | 决策流程图 | 🧪 |

**总计**：15 章 ≈ 6 天集训（每天 2–3 章）或 12 个半天；其中 3 章云上内容可压缩为 1 天。

---

## 4. 章节之间的依赖

```
00 ── 01 ── 02 ── 03 ── 04
              │
              └── 05 ── 06
                        │
        07 ── 08 ───────┤
                        └── 09 ── 10 ── 11 ── 12 ── 13 ── 14
```

- 01 提供的 `TinyGPT` 是 02–08 的共用实验对象（随机权重，只关心形状/带宽，不关心生成质量）。
- 02 的 `KVCache` 接口是 03/04 的输入契约；05 的 `Request`/`Scheduler` 接口是 06 的输入契约。
- 09 起不再造轮子，改为「读工业实现 + 用工业实现做实验」。

**接口契约（必须严格一致，否则后章无法复用）**：

```python
# ch02 定义，ch03/ch04/ch06 复用
class KVCache:
    def append(self, layer: int, k: Tensor, v: Tensor) -> None: ...
    def get(self, layer: int, upto: int | None = None) -> tuple[Tensor, Tensor]: ...

# ch05 定义，ch06 复用
@dataclass
class Request:
    prompt_len: int
    max_new_tokens: int
    arrival: float
    ...
```

---

## 5. 交付物清单

```
ai-infra-course/
├── README.md                 # 快速开始、学习路径、硬件要求
├── COURSE_DESIGN.md          # 本文件
├── STYLE_GUIDE.md            # 写作/代码/图表/命名规范（多人协作的契约）
├── requirements.txt / pyproject.toml
├── aiinfra/                  # 共享 Python 包（Notebook import）
│   ├── tinygpt.py            # 随机权重微型 GPT（MHA + KV cache 接口）
│   ├── bench.py              # 计时、分位数、环境快照、roofline 工具
│   └── plotting.py           # 统一图表风格
├── notebooks/                # 源文件 .py（percent 格式）→ 构建 .ipynb
│   ├── 00_setup_and_measurement.py/.ipynb
│   ├── 01_forward_and_sampling.py/.ipynb
│   └── ... 14_capstone.py/.ipynb
├── assets/figures/           # 手绘 SVG 概念图 (chXX_*.svg)
├── outputs/                  # Notebook 运行生成的 png/csv（gitignore）
├── solutions/                # 练习参考答案（Notebook 或 .py）
├── cloud/                    # 云上执行资产
│   ├── README.md             # AutoDL / RunPod / Lambda 操作手册
│   ├── docker/               # Dockerfile + compose
│   ├── scripts/              # 起服务、压测、多卡实验脚本
│   └── k8s/                  # 生产部署示例
└── tools/
    ├── build_notebooks.py    # .py → .ipynb
    └── execute_notebooks.py  # 无头执行 + 超时校验
```

---

## 6. 云资源策略（用户要求：只写内容与代码）

- **本地章（00–11，除 10/11 的模型服务部分）**：全部可在 Mac 上跑，用随机权重 TinyGPT + 可选 vllm-metal 小模型。
- **云上章（12–13）**：
  - Notebook 内所有 CUDA/多卡代码放在 `if RUN_ON_CLOUD:` 分支，本地执行时走**模拟路径**（线程模拟 all-reduce、用 psutil 模拟显存账本），保证 notebook 仍能一键跑通。
  - `cloud/` 下提供可直接复制到 AutoDL/RunPod 的脚本、镜像、启动命令、成本估算表。
  - 每个云实验都有「**预期现象**」清单（例如 TP=2 时通信占比 15%±5%），学员上云后自行核对。
- **成本护栏**：脚本默认 `--hours 2` 自动关机提示；提供「关机/退租」checklist；给出 4090/A100/H100 的小时价与整课预算估算。

---

## 6.5 统一的章节写作模式

02–14 章统一采用 `CHAPTER_PATTERN.md` 的结构：
**NN.0 目标与路线 → NN.1 问题与直觉 → NN.2 高阶设计（组件/数据流/接口/验收）→ NN.3 分 N 步实现（每步 目标→做法→代码→验收）→ NN.4 实验 A/B/C → NN.5 vLLM 对照 → NN.6 练习**。
00/01 章保留各自的教学结构，但都补了 `NN.0 本章目标与路线`。

---

## 6.6 叙述设计：让技术内容「读得下去」（v0.2）

结构解决「信息全不全」，叙述解决「读不读得下去」。v0.2 引入 10 个可检查的叙述动作，
完整定义与 before/after 示例见 [`POLISH_GUIDE.md`](POLISH_GUIDE.md)，这里只列设计意图：

| # | 动作 | 设计意图 |
| --- | --- | --- |
| 1 | 开场场景钩子（60–140 字） | 先给一个读者真遇到过的现象，再点破机制；从「目录」变成「悬念」 |
| 2 | 先猜后验 | 每个关键实验前先让读者下注，猜错的那次记得最牢；必须解释错误选项为什么像对的 |
| 3 | 类比 + 数字落点 | 类比只负责建立直觉，必须立刻换算成真实字节数/倍数，否则就是鸡汤 |
| 4 | ⚠️ 坑 callout | 把「差一个数量级」的错误单独拎出来，写明错多少倍、错成什么现象 |
| 5 | 时间轴 / 小剧场 | 有状态机（调度、分页、投机解码）必须让事件顺序可见 |
| 6 | 反直觉结论 | 「感觉该是 X，其实是 Y，因为 Z」——最容易被记住的结构 |
| 7 | 标题去公文腔 | 保留 NN.x 编号与关键词，但修饰语带悬念 |
| 8 | 问题接力 | 每节结尾抛出下一节要解决的问题，形成阅读推力 |
| 9 | 收尾三件套 | **一句话记住 / 自检 3 问 / 工作场景题**，把「读完」变成「能答」 |
| 10 | 数字人感化 | 「144 GiB」→「两张 80GB 卡的整卡显存，只为缓存 32 个会话」 |

**硬约束**（比十个动作更重要）：叙述可以重写，**实测数字、断言、结论、代码语义、五段式
小节名与 NN.x 编号一律不动**。改完必须 `build` + `check` + `execute` 全绿。

**验收口径**：每章中文正文增幅 +25%~+50%（超出即视为注水）；至少 2 处先猜后验、2 处类比
落点、1 处 ⚠️ 坑、1 组收尾三件套。

## 7. 验收标准（Definition of Done）

每章必须同时满足：

1. `python tools/build_notebooks.py` 能生成 `.ipynb`，`nbformat.validate` 通过。
2. 在 Mac（CPU 或 MPS）上无头执行 **< 3 分钟**（云章走模拟路径同样 < 3 分钟）。
3. ≥ 2 张图：1 张概念 SVG（`assets/figures/`）+ 1 张实测图表（Notebook 内生成）。
4. ≥ 1 张数字表格（延迟/吞吐/显存/误差），且注明硬件与重复次数。
5. 练习 ≥ 3 题，`solutions/` 有对应答案或解题提示。
6. 「对照工业实现」小节给出 vLLM 真实文件路径/类名/参数名，并标注版本。
7. 无 `TODO`、无未定义变量、无网络依赖（模型下载除外且可选）。

全课程验收：`python tools/execute_notebooks.py --all` 全绿，README 学习路径可走通。

---

## 8. 6 天集训排期（建议）

| Day | 上午 | 下午 | 产出 |
| --- | --- | --- | --- |
| 1 | 00 度量 + 01 前向 | 02 KV Cache | 能自己算显存、跑出 prefill/decode 曲线 |
| 2 | 03 PagedAttention | 04 前缀缓存 | 能解释 block table、复现碎片实验 |
| 3 | 05 调度器 | 06 Chunked Prefill | 能画出吞吐-延迟权衡曲线 |
| 4 | 07 量化 | 08 投机解码 | 能量化误差、算加速比 |
| 5 | 09 架构与源码 | 10 vLLM 服务 + 11 压测 | 能定位一个真实性能问题 |
| 6 | 12 多卡（云） | 13 训练 infra + 14 Capstone | 能出一份容量规划与成本表 |

---

## 9. 风险与对策

| 风险 | 对策 |
| --- | --- |
| 学员机器性能差异大 | 随机权重 TinyGPT 只有 ~2M 参数，CPU 也能秒级跑；所有曲线标注「绝对值随机器变化，趋势才重要」 |
| vLLM 版本迭代快 | 全课程锁定 vLLM 版本（写在 `requirements.txt`），源码路径按该版本校验 |
| Mac 无 CUDA | 云章代码 + 模拟路径双轨；vllm-metal 作为可选的「真服务」体验 |
| Notebook 变成「能跑但看不懂」 | 每段代码前必须有一句「为什么」，每章末尾有「你现在应该能回答」清单 |
