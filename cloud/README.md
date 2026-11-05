# cloud/ · 云上执行手册（AutoDL / RunPod / Lambda）

> 这份手册是第 10–13 章的「上云执行包」：开机器 → 拉镜像 → 起服务 → 压测 → **关机**。
> 第 12–14 章训练侧的增补内容（多卡 TP 实测、LoRA 训练、容量校准）见 [`README_training.md`](README_training.md)，两份文档共用本文件的平台流程与关机 checklist。
> 本地 Mac 能跑的部分不要上云；只有多卡（第 12 章）和真实吞吐（第 11 章）值得花钱。
>
> ⚠️ 价格为 2026 年示例价，区域/时段/库存差异很大，**下单前以平台页面为准**。所有命令里的版本号 `v0.28.0` 与课程锁定版本一致；镜像不存在时用 `latest` 并把实际版本回填。

---

## 0. 30 秒流程

```
开机器 → 确认 GPU/驱动 → 起服务（compose / docker run）→ 等 /health 200
      → 压测（scripts/bench_sweep.sh）→ 拉回 results/ → 关机（见 §5 checklist）
```

---

## 1. 三个平台怎么选

| 平台 | 计费粒度 | 典型卡 | 关机语义 | 适合 | 坑 |
| --- | --- | --- | --- | --- | --- |
| **AutoDL** | 按小时（有按量/包周） | 4090 24G / A100 80G / H100 80G | 「关机」保留数据盘（少量存储费）；「释放」删实例和数据盘 | 国内网络、便宜、学生多 | 数据盘另计费；镜像/环境可能不自带 vLLM，优先用「社区镜像」或自建 |
| **RunPod** | 按秒 | 4090 / A100 80G / H100 80G | **Stop 仍收存储费**；Terminate 才彻底释放 | 秒级计费、镜像生态好 | 网络盘（Network Volume）单独计费；Stop 后 GPU 不保留但 volume 会 |
| **Lambda** | 按分钟 | A100 40/80G、H100 80G | Terminate 实例即停止计费；持久存储单独计费 | 官方 vLLM 镜像、稳定 | 4090 库存少；开机后要自己装 CUDA 驱动校验 |

选型建议（本课程）：
- 只跑 10/11 章单卡实验：**4090 24G（AutoDL）** 最便宜，8B 用 fp16 或 int8；
- 要跑 8B 高并发/27B：**A100 80G**；要测 TP=4 或 27B fp16：**2×/4× A100 80G**；
- H100 只在第 12 章对比通信/带宽时有意义，单价高，别用来只跑单卡。

---

## 2. AutoDL：完整流程

### 2.1 开机器

1. 控制台 → 容器实例 → 租用 GPU：选 `RTX 4090 / A100-SXM / H100-SXM`，镜像选 **`vllm` 社区镜像**（搜 `vllm`）或 `PyTorch 2.x + CUDA 12.x` 基础镜像；
2. 数据盘：≥ 50GB（HF 缓存一个 8B 模型约 16GB）；
3. 创建后记下：SSH 登录指令、实例 ID、每小时价格。

### 2.2 进机器、确认驱动

```bash
nvidia-smi                     # 必须能看到卡和 CUDA Version
python -c "import vllm; print(vllm.__version__)"   # 社区镜像自带；基础镜像需自己装
df -h /root/autodl-tmp         # 确认数据盘挂载点
```

### 2.3 起服务（二选一）

```bash
# 方式 A：本仓库脚本（推荐，参数集中）
git clone <your-repo> ai-infra-course && cd ai-infra-course
MODE=docker MODEL=Qwen/Qwen3-8B bash cloud/scripts/serve_qwen3.sh

# 方式 B：直接 docker
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.28.0 \
  --model Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.90 --max-model-len 8192
```

首次启动要下载模型（8B 约 16GB，视网络 5–20 分钟）。看到 `Application startup complete` 后：

```bash
curl -s localhost:8000/health          # 期望 200
curl -s localhost:8000/v1/models | head
```

### 2.4 压测（另开一个终端 / tmux）

```bash
cd ai-infra-course
BASE_URL=http://127.0.0.1:8000 MODEL=Qwen/Qwen3-8B NUM_PROMPTS=200 \
  bash cloud/scripts/bench_sweep.sh
# 结果在 cloud/results/*.json
```

> AutoDL 的容器实例里不要用 `localhost` 之外的地址暴露 8000；需要自测就让脚本在实例内跑。

### 2.5 关机（**必做**）

AutoDL 控制台 → 容器实例 → **关机**（保留数据盘）或 **释放**（连数据盘一起删）。
如果是「释放」，先确认 `cloud/results/` 和 HF 模型缓存都已 `scp` 回本地，否则数据没了。

---

## 3. RunPod：完整流程

### 3.1 开机器

1. Pods → Deploy：GPU 选 `A100 80GB / H100 80GB / RTX 4090`；
2. Template 选 **`RunPod PyTorch 2.x`** 或直接填 Docker image `vllm/vllm-openai:v0.28.0`（容器自带 vLLM 与启动入口）；
3. Container disk ≥ 30GB，Volume ≥ 60GB（挂到 `/workspace` 或 `/root/.cache/huggingface`），暴露 HTTP 端口 8000；
4. 记下：Pod ID、`$0.xx/hr`、Volume 的计费方式。

### 3.2 起服务

```bash
# 如果模板已经用 vllm 镜像，直接改启动参数即可；否则：
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -v /workspace/hf:/root/.cache/huggingface \
  vllm/vllm-openai:v0.28.0 \
  --model Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 --max-model-len 8192
```

RunPod 的 HTTP 代理地址形如 `https://<pod-id>-8000.proxy.runpod.net`，可以用它做远程压测：

```bash
BASE_URL=https://<pod-id>-8000.proxy.runpod.net bash cloud/scripts/bench_sweep.sh
```

### 3.3 压测

同 §2.4。远程压测时 `BASE_URL` 换成代理地址，注意网络 RTT 会污染 TTFT，报告里要标注「压测机在公网」。

### 3.4 关机（**必做**）

- **Stop**：GPU 释放，但 Volume 仍按 GB/月计费；
- **Terminate**：实例彻底删除（不可恢复），Volume 若要保留则单独保留；
- 确认 Billing 页面当小时费用归零。

---

## 4. Lambda：完整流程

### 4.1 开机器

1. Lambda Cloud → Launch instance：`A100 80GB` / `H100 80GB`，镜像选 `vLLM` 官方镜像；
2. 挂载持久文件系统（`/lambda/nfs/...`）用于 HF 缓存；
3. 记下实例 IP、`$0.xx/hr`。

### 4.2 起服务与压测

```bash
ssh ubuntu@<ip>
nvidia-smi
python -c "import vllm; print(vllm.__version__)"
# 镜像已含 vLLM CLI；或用本仓库脚本
bash cloud/scripts/serve_qwen3.sh
# 另开终端
bash cloud/scripts/bench_sweep.sh
```

### 4.3 关机（**必做**）

Lambda 控制台 → Instances → **Terminate**（不是关机，Lambda 没有「关机保留」概念，实例在就计费）。
先 `scp` 拉回 `cloud/results/`，再 Terminate。

---

## 5. 关机 checklist（每次都照着做）

> 这一节存在的唯一目的：让你不要在半夜为一个闲置的 A100 付钱。

- [ ] 1. **停服务**：`docker compose down -v`（或 `docker rm -f <container>`），确认 `nvidia-smi` 里没有进程；
- [ ] 2. **拉回数据**：`scp -r <host>:~/ai-infra-course/cloud/results ./cloud/results-cloud/`，确认文件数对得上；
- [ ] 3. **平台关机**：AutoDL 关机/释放；RunPod **Terminate**（Stop 仍收 volume 费）；Lambda **Terminate**；
- [ ] 4. **清附属资源**：Network Volume / 数据盘 / 快照 / 公网 IP，不用就删；
- [ ] 5. **核对账单**：平台的 Billing / 费用中心，确认「当前运行实例数 = 0」；
- [ ] 6. **记录本次花费**：写进实验报告（时长 × 单价），为第 14 章容量规划攒数据。

**三个最常见的「以为关了其实没关」**：
1. RunPod 只点了 Stop —— 实例不跑 GPU 了，但 Volume 继续按 GB/月收费；
2. AutoDL 只关机没释放 —— 数据盘继续按天收费（金额小，但会累积）；
3. 本地终端 `Ctrl+C` 停了 `docker run`，但 `-d` 起的容器还在跑（`--rm` 才会自动删）。

---

## 6. 成本表（示例价，务必以平台页面为准）

单位：USD 或 CNY，按「单卡每小时」计。

| 卡型 | AutoDL（¥/h） | RunPod（$/h） | Lambda（$/h） | 本课程单章 2h 参考 |
| --- | --- | --- | --- | --- |
| RTX 4090 24G | ~1.9 | ~0.5 | 不供 | ¥3.8 / $1.0 |
| A100 80G | ~7.5 | ~1.99 | ~1.29 | ¥15 / $2.6–4.0 |
| H100 80G | ~14 | ~2.99 | ~2.49 | ¥28 / $5.0–6.0 |

> 换算按 1 USD ≈ 7.2 CNY 粗算；价格会随供需波动，表中只用于「量级判断」。

### 整门课的云上预算（按 3 个云实验 × 2 小时估算）

| 方案 | 配置 | 估算 | 说明 |
| --- | --- | --- | --- |
| 最省 | AutoDL 4090 × 2h × 3 次 | ≈ ¥12 | 只跑单卡 8B（第 10/11 章），int8/fp16 |
| 标准 | RunPod A100 80G × 2h × 3 次 | ≈ $12（≈¥86） | 第 10/11 章 + 27B 显存实验 |
| 多卡 | 4×A100 80G × 2h × 2 次（第 12 章） | ≈ $32–48 | TP=1/2/4 对比，按实际卡数计费 |
| 不建议 | 单卡任务用 H100 | 比 A100 贵 50–100% | 单卡带宽收益有限，性价比低 |

**预算护栏**：本目录所有脚本都不会自动关机（它们没有平台 API 权限）——请在日历/手机上设一个「2 小时后」的提醒，或在平台开启「余额不足/自动关机」策略。

---

## 7. 目录与文件

```
cloud/
├── README.md                  # 本文件：平台流程 / 起服务 / 压测 / 成本 / 关机 checklist
├── README_training.md         # 训练侧增补（第 12–14 章）：多卡 TP、LoRA 训练、容量校准
├── docker/
│   ├── Dockerfile             # 基于 vllm/vllm-openai:v0.28.0 的最小镜像
│   └── docker-compose.yml     # 单卡起服务 + HF 缓存挂载 + healthcheck
├── scripts/
│   ├── serve_qwen3.sh         # 统一起服务入口（local / docker）
│   ├── bench_sweep.sh         # 并发扫描 1/4/8/16/32 → vllm bench serve
│   ├── 12_tp_bench.sh         # 第 12 章正式脚本：TP=1/2/4 实测 + 成本提示（训练侧维护）
│   ├── 13_lora_finetune.sh    # 第 13 章正式脚本：LoRA 微调 → 合并 → vLLM 双路线部署（训练侧维护）
│   └── tp_experiment.sh       # 本手册附带的 TP 扫描最小版（与 12_tp_bench.sh 同思路，做对照用）
├── fixtures/                  # 10/11 章的回放数据（示例录制，非本机实测）
│   ├── chat_completions.json
│   ├── embeddings.json
│   └── benchmark_results.json
└── results/                   # 压测输出（运行时生成，gitignore）
```

> **两个 TP 脚本的关系**：`12_tp_bench.sh` 是第 12 章的正式实验入口（多卡重启、成本提示、结果命名规范都在里面）；
> `tp_experiment.sh` 是本手册早期提供的最小版（只做「起服务 → 压测 → 清理」），保留它是为了让学员对照
> 「一个最小可控脚本」和「一个正式实验脚本」的差别。做第 12 章实验请优先用 `12_tp_bench.sh`。

## 8. 安全与合规（上公网前必读）

- **不要**把 8000 端口直接暴露到公网；必须暴露时加 `--api-key`，并只允许压测机 IP；
- 用 `--served-model-name` 隐藏模型路径；不要开 `--trust-remote-code` 跑来路不明的权重；
- 压测流量会跑满 GPU 并产生真实账单，脚本默认 `NUM_PROMPTS=200`，先小规模验证再放大；
- 实验结果（`cloud/results/`）不要提交到公开仓库，里面可能带内网地址与 API Key。

## 9. 常见故障

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `docker: could not select device driver` | 没装 nvidia-container-toolkit | 云上官方镜像一般自带；自建机器按 NVIDIA 文档装 |
| 启动卡在 `Loading safetensors` 很久 | 首次下模型 / 网络慢 | 提前把 HF 缓存挂到持久盘，或换国内镜像源 |
| `CUDA out of memory` | `gpu_memory_utilization` 太高或 `max_model_len` 太大 | 降 util 到 0.85、减 `--max-model-len` |
| `/health` 一直 503 | 模型还在加载 / 显存不足 | 看容器日志；8B fp16 至少 24G 卡 |
| `vllm bench serve` 报 404 | `--base-url` 少了 `/v1` 或端口不对 | `curl $BASE_URL/v1/models` 先验证 |
| 压测结果波动巨大 | 压测机在公网 / 别的进程抢 GPU | 压测机与服务器同区域；压测前后 `nvidia-smi` 确认无其他进程 |
