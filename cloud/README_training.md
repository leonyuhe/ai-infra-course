# cloud/ · 训练侧与多卡实验手册（第 12–13 章 + 第 14 章校准）

> 这是第 12、13、14 章的「上云执行包」：开机器 → 跑多卡 / 训练实验 → 把实测数字填回容量规划 → **关机**。
> **平台操作、镜像、起服务、压测、关机语义统一看 [`cloud/README.md`](README.md)**，本文件不重复，
> 只补充训练侧独有的事：选几张卡、跑哪个脚本、看什么指标、花多少钱。
>
> 本地已完成的模拟路径：`notebooks/12_multi_gpu.py`、`13_training_infra.py`、`14_capstone.py`
> 全部可在 Mac 上无头执行（`tools/execute_notebooks.py --only 12 13 14`），上云是为了**用真实数字校准结论**，
> 不是补课。价格为 2026 年示例价，以平台页面为准。

---

## 0. 一页决策：这三章哪些必须上云

| 章节 | 云上实验 | 最低配置 | 预计耗时 | 预算（A100 80G @ $1.9/h） | 本地替代 |
| --- | --- | --- | --- | --- | --- |
| 12 多卡 | TP=1/2/4 的吞吐、TTFT、TPOT 对比 | 单机 4×A100（TP=4 需要 4 卡） | 40–60 min | $10–16 | ✅ 线程模拟 + 带宽模型（结论已给） |
| 13 训练 | LoRA 微调 → 合并 → vLLM 双路线部署 | 1×A100 80G | 30–50 min | $2–4 | ✅ TinyGPT 训练 + merge all-close（闭环已跑通） |
| 14 Capstone | 用真实压测校准容量表（复用 12 章结果） | 1×A100 80G（与 13 共享） | 30–60 min | $2–4 | 🧪 纯推算（三方案表已给） |
| **合计** | | | **≤ 3 h** | **$14–24** | |

> 只想花最少的钱：**跳过 12 章的多卡实验**（模拟结论够用），用一个 A100 跑完 13 + 14，
> 总预算约 $4–6。**12 章只在你真要给团队选卡/选拓扑时才值得上云实测。**

---

## 1. 前置：镜像与额外依赖

平台/镜像/SSH 的完整流程见 [`cloud/README.md`](README.md) 的 §2（AutoDL）、§3（RunPod）、§4（Lambda）。
这里只列训练侧差异：

```bash
# 1) 基础镜像：与 cloud/docker/Dockerfile 一致
#    vllm/vllm-openai:v0.28.0（自带 vLLM / transformers / torch CUDA）
# 2) 第 13 章 LoRA 额外需要（vLLM 镜像通常不带 peft/datasets）：
python -m pip install -U "peft>=0.13" "datasets>=2.21" "accelerate>=0.34"
# 3) 第 12 章只需要 vllm 自带的 bench 客户端，无需额外安装
# 4) 校验
python -c "import vllm, torch, peft; print(vllm.__version__, torch.__version__, peft.__version__)"
nvidia-smi
```

模型下载慢时，在实例内设置镜像（`HF_ENDPOINT=https://hf-mirror.com`），或提前把
`~/.cache/huggingface` 挂到持久盘（RunPod Network Volume / AutoDL 数据盘），避免每次重下 16 GB。

---

## 2. 上云步骤（训练侧精简版）

```bash
# ① 开机器（平台细节看 cloud/README.md；第 12 章选「单机 4 卡」，13/14 选「单卡 A100 80G」）
# ② 拿代码与依赖
git clone <your-repo> ai-infra-course && cd ai-infra-course
python -m pip install -e .        # 本地包 aiinfra，用于第 14 章重算；训练脚本不依赖它
# ③ 第 12 章：TP 扫描（内部会起 vllm serve 再压测，结束自动 kill）
MODEL=Qwen/Qwen3-8B TP_LIST="1 2 4" HOURS=2 bash cloud/scripts/12_tp_bench.sh
# ④ 第 13 章：LoRA 训练 → merge → 双路线部署（需要 JSONL 数据）
DATA_FILE=$HOME/data/kb_qa_train.jsonl MAX_STEPS=200 HOURS=1 \
  bash cloud/scripts/13_lora_finetune.sh
# ⑤ 第 14 章：把 ③④ 的实测值填回 notebook 14.2 的 plan_capacity（见 §5）
# ⑥ 拉回结果、关机（见 §6 与本文件末尾 checklist）
```

---

## 3. 第 12 章：TP 扫描看什么

脚本：[`scripts/12_tp_bench.sh`](scripts/12_tp_bench.sh)。它会为每个 TP 值启动一次
`vllm serve --tensor-parallel-size N`，再用 `vllm bench serve` 打同一份 2K/256 随机负载，
最后输出 markdown 对比表（`output_throughput`、TTFT、TPOT 的 mean/p99、完成请求数）。

**核对清单（与 notebook 12.3 的模拟结论对照）：**

| 指标 | 预期现象（单机 NVLink，A100/H100） | 不符合时先查 |
| --- | --- | --- |
| output tok/s 扩展效率 | TP=2 → 1.7–1.9×；TP=4 → 3.0–3.6×（相对 TP=1） | batch 太小 / 跨 NUMA / MIG |
| mean TTFT | TP 越高越低（prefill 被切开），TP=4 约降 30–50% | `--max-num-batched-tokens` 是否过小 |
| mean TPOT | 改善但边际递减；TP=4 通常好于 TP=2 | KV cache 是否吃满、是否触发抢占 |
| p99 TPOT | 明显高于 mean 说明有排队/抢占 | 降 `--max-num-seqs` 或加实例 |

> 结论落点：**TP=8 在 NVLink 上扩展效率约 0.89；换到以太网会掉到约 0.50**（notebook 12.3 模型）。
> 云上实测如果 TP=4 效率 < 0.8，不要急着加卡——先看是不是 batch 太小、或卡不在同一 NVLink 域。

---

## 4. 第 13 章：LoRA 闭环看什么

脚本：[`scripts/13_lora_finetune.sh`](scripts/13_lora_finetune.sh)。四步：训练 → 验证冻结 →
合并 → vLLM 双路线（合并权重 / `--enable-lora` 热插拔）。

**核对清单（与 notebook 13.3 的本地闭环对照）：**

| 检查点 | 预期 | 不符合时先查 |
| --- | --- | --- |
| trainable params | < 1%（7B + r=16 约 0.2–0.5%） | `target_modules` 是否写错（q/k/v/o） |
| 训练 loss | 前 50 步明显下降，之后趋缓 | lr 太大（>1e-3）会震荡；数据格式/标签 mask |
| merge 前后 greedy 输出 | **逐 token 完全一致** | 是否对 shared/tied 权重做了 LoRA（如 lm_head=tok_emb） |
| 合并模型 vLLM 启动 | 正常加载、无 adapter 开销 | 合并后 dtype 是否 bf16/fp16；config 是否带 `peft` 残留 |
| adapter 热插拔 | `model=base` 与 `model=kb-lora` 都返回 200 | `--max-lora-rank` 是否 ≥ r；adapter 路径格式 |

> 成本对照：7B LoRA（r=16，0.5% 参数）训练 200 步 × batch 2 × 2K 序列，1×A100 约 20–40 分钟，
> 按 $1.9/h 计 **$0.6–1.3**；加上模型下载与两次 vLLM 启动，整章 1 小时内可控在 **$2–4**。

---

## 5. 第 14 章：把实测数字填回容量表

打开 `notebooks/14_capstone.py`，修改 `plan_capacity()` 的输入（或用脚本结果反推），只填**实测值**：

| 符号 | 含义 | 从哪来 | 本课程默认假设 |
| --- | --- | --- | --- |
| `hbm_bw_gbps` | 实测可达带宽 | 第 01 章的 `measure_bandwidth()` 或厂商规格 × 0.6 | A100 ≈ 2000 GB/s |
| `tflops_fp16` | 实测有效算力 | 大 matmul 实测（第 01 章） | A100 ≈ 150 TFLOP/s |
| `单实例 QPS` | 实际压测吞吐 ÷ 输出长度 | `12_tp_bench.sh` / `bench_sweep.sh` 的 `output_throughput` | 由公式推 |
| `TTFT/TPOT` | p95 而非 mean | `vllm bench serve` 的 p99/mean 字段 | 由公式推 |
| `实例数` | `ceil(QPS ÷ 单实例 QPS ÷ 0.8)` | 用实测值重算 | 见 notebook |

**复算示例**（把 12 章 TP=1 的实测 `output_throughput ≈ 4000 tok/s` 填回方案 A）：

```python
# notebook 14.2 的 plan_capacity 里，把 qps_per_instance 换成实测：
measured_qps = 4000 / 300            # 输出 300 token/请求 → 13.3 req/s
instances = ceil(100 / (measured_qps * 0.8))   # → 10 台，而不是估算的 8 台
```

实测通常比公式**更慢**（排队、prefill 抢占、CUDA Graph 覆盖不全），所以第一版容量表
建议在估算基础上 **×1.2–1.5 的安全系数**，然后用第二次压测收敛。

---

## 6. 成本表（本课程三章，示例价）

> 平台单价与整门课预算见 [`cloud/README.md`](README.md) §6；下表只算 12–14 章的训练侧开销。

| 实验 | 配置 | 时长 | 单价 | 小计 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 12 多卡 TP 扫描 | 4×A100 80G | 1.0 h | $1.9/卡/h | **$7.6** | TP=1 时另外 3 卡空闲，可让 TP=1 在 1 卡实例上跑省 $5.7 |
| 12 多卡（省钱版） | 1×A100 80G，只跑 TP=1 | 0.5 h | $1.9/h | **$0.95** | 只能得到 baseline，无法算扩展效率 |
| 13 LoRA 闭环 | 1×A100 80G | 1.0 h | $1.9/h | **$1.9** | 含下载模型、训练 200 步、两次 serve |
| 14 压测校准 | 1×A100 80G | 1.0 h | $1.9/h | **$1.9** | 可与 13 共用实例 |
| 存储/网络（估） | HF 缓存 50 GB + 流量 | — | — | **$0.5–1** | 数据盘/Volume 按平台计 |
| **合计（推荐路径）** | 4×A100 跑 12 + 1×A100 跑 13/14 | ≤ 3 h | | **$12–15** | 比整门课预算（README §6）略高，因为多卡只跑一次 |

**省钱三条**：
1. 12 章先跑 TP=1/2，足够判断「NVLink 域内是否线性」；TP=4 只在要给团队出报告时跑。
2. 13/14 共用同一个实例，模型缓存复用，省一次 16 GB 下载。
3. 训练步数用 `MAX_STEPS` 控制（默认 200 步 ≈ 30 分钟），别一上来跑 3 epoch。

---

## 7. 关机 checklist（训练侧补充版）

> 完整版（含三个平台的 Stop/Terminate 语义）见 [`cloud/README.md`](README.md) §5。训练侧多三件事：

- [ ] 1. **停训练进程**：`pkill -f "13_lora_finetune"`，确认 `nvidia-smi` 没有 python 占卡；
- [ ] 2. **存产物**：`adapter/`、`merged/`、`server*.log`、`tp_bench_results_*/` 全部 `scp` 回本地
      （adapter 只有几十 MB，别丢；merged 可重新 merge，不必传）；
- [ ] 3. **存数据**：训练 JSONL 若含业务数据，确认本地有副本并从实例删除（合规）；
- [ ] 4. **平台关机**：AutoDL 关机/释放；RunPod Stop/Terminate；Lambda Terminate；
- [ ] 5. **核对账单**：Billing 页面「运行实例数 = 0」，数据盘/Volume 是否还需保留；
- [ ] 6. **回填成本**：把本次「时长 × 单价」写进第 14 章成本表，作为下一版的校准数据。

---

## 8. 训练侧故障排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| LoRA 训练 OOM | `per_device_batch × MAX_LEN` 太大 / 无 gradient checkpointing | 降 batch 到 1，开 `gradient_checkpointing=True`，或把 `MAX_LEN` 降到 1024 |
| `trainable params` 占比很高 | `target_modules` 匹配到了太多层（或 `modules_to_save`） | 只保留 q/k/v/o；打印 `model.named_modules()` 确认 |
| merge 前后输出不一致 | 对 tied 权重（lm_head = tok_emb）做了 LoRA | 训练前解绑 lm_head，或不要对 lm_head 做 LoRA |
| vLLM 加载 adapter 报 `max_lora_rank` | `--max-lora-rank` < r | 设为 ≥ r（默认 16，r=16 时显式给 32 更稳） |
| TP 扫描时 server 起不来 | `--tensor-parallel-size` 不能整除 head 数 | 换 TP 或换模型（12.4 有约束说明） |
| NCCL 卡住/超时 | 多卡不在同一 NVLink 域，或容器缺 IPC | `--ipc=host`；确认 `nvidia-smi topo -m` 的 NVLink 拓扑 |
| 压测结果忽高忽低 | 同机双跑 / 压测客户端抢 CPU | 一个实例只跑一个 server；压测用 `--num-prompts 200` 起步 |

---

## 9. 安全与合规

- **不要**把 HF token、平台 API Key、业务数据写进脚本或 notebook；用环境变量 `HF_TOKEN` 注入；
- 第 13 章的训练数据来自企业知识库，**属于敏感数据**：adapter 可能记忆原文，训练前脱敏，训练后按合规要求删除实例内副本；
- 只把 8000/8001/8002 端口暴露给压测机，不要直接开公网；对外服务加 `--api-key`（见 `cloud/README.md` §8）；
- 实验结果里的内网地址、实例 ID、账单截图不要提交到公开仓库。

---

## 10. 文件清单（训练侧）

```
cloud/
├── README.md                     # 平台/镜像/起服务/压测/关机（vllm 组维护，本文件大量引用）
├── README_training.md            # 本文件：12–14 章训练侧执行手册
├── scripts/
│   ├── 12_tp_bench.sh            # TP=1/2/4 扫描 + 对比表模板（第 12 章）
│   └── 13_lora_finetune.sh       # LoRA 训练 → merge → vLLM 双路线（第 13 章）
└── fixtures/ docker/             # 见 cloud/README.md §7
```

> 注：`cloud/README.md` §7 的目录清单里把第 12 章脚本预留为 `tp_experiment.sh`；
> 本课程实际交付为 `12_tp_bench.sh`（按章节号命名、与 13 章对齐），两者指同一件事。
