#!/usr/bin/env bash
# 第 12 章 · 云上 TP 实测：用 vLLM 的 benchmark 客户端量 TP=1/2/4 的扩展效率
#
# 用途：在 1×8 卡 NVIDIA 机器（A100/H100/L40S）上，固定模型与负载，只改
#       --tensor-parallel-size，对比 output throughput / TTFT / TPOT。
#
# 用法：
#   bash cloud/scripts/12_tp_bench.sh                 # 默认 Qwen3-8B，TP=1 2 4
#   MODEL=meta-llama/Llama-3.1-8B-Instruct TP_LIST="1 2 4 8" bash cloud/scripts/12_tp_bench.sh
#   HOURS=2 HOURLY_USD=1.9 bash cloud/scripts/12_tp_bench.sh   # 成本提示
#
# 成本护栏：脚本结束会打印预计费用与关机 checklist；不会替你关机。
# 前置：镜像内已装 vllm==0.28.x（见 cloud/README.md），模型已下载或可联网拉取。

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
TP_LIST="${TP_LIST:-1 2 4}"
PORT="${PORT:-8000}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
INPUT_LEN="${INPUT_LEN:-2048}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
HOURS="${HOURS:-2}"
HOURLY_USD="${HOURLY_USD:-1.9}"
RESULT_DIR="${RESULT_DIR:-$HOME/tp_bench_results_$(date +%m%d_%H%M)}"
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\n❌ %s\n' "$*" >&2; exit 1; }

log "检查环境"
command -v nvidia-smi >/dev/null || die "没有 nvidia-smi：这台机器不是 NVIDIA GPU 环境"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
python3 -c "import vllm; print('vllm', vllm.__version__)" || die "vllm 未安装，见 cloud/README.md"
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
log "GPU 数 = $GPU_COUNT；TP 列表 = $TP_LIST；模型 = $MODEL"
echo "$HOURS 小时预计费用：\$$(python3 -c "print(round($HOURS*$HOURLY_USD,2))")（$HOURLY_USD/h，仅实例费）"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "停止 vLLM server (pid=$SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health/" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

for TP in $TP_LIST; do
  if (( TP > GPU_COUNT )); then
    log "跳过 TP=$TP（GPU 只有 $GPU_COUNT 张）"; continue
  fi
  log "========== TP=$TP =========="
  cleanup
  # 单机 TP：不要加 --enable-prefix-caching，保持与 baseline 一致
  vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" >>"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_for_server || die "TP=$TP server 180s 内没起来，看 $SERVER_LOG"
  log "server ready，开始压测"
  vllm bench serve \
    --backend openai-chat \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --save-result --result-dir "$RESULT_DIR" \
    --result-filename "tp${TP}.json"
  cleanup
  SERVER_PID=""
  sleep 10  # 等显存回收，避免下一个 TP 的 profile_run 失败
done

log "汇总对比表"
python3 - "$RESULT_DIR" <<'PY'
import glob, json, os, sys

rows = []
for path in sorted(glob.glob(os.path.join(sys.argv[1], "tp*.json"))):
    tp = int(os.path.basename(path)[2:-5])
    d = json.load(open(path))
    rows.append((tp, d.get("output_throughput"), d.get("mean_ttft_ms"), d.get("p99_ttft_ms"),
                 d.get("mean_tpot_ms"), d.get("p99_tpot_ms"), d.get("completed")))
base = rows[0][1] if rows and rows[0][1] else 1.0
print("| TP | output tok/s | 扩展效率 | mean TTFT(ms) | p99 TTFT(ms) | mean TPOT(ms) | p99 TPOT(ms) | 完成请求 |")
print("| --- | --- | --- | --- | --- | --- | --- | --- |")
for tp, thr, ttft, p99t, tpot, p99p, done in rows:
    eff = (thr or 0) / base / tp
    print(f"| {tp} | {thr:.1f} | {eff:.2f} | {ttft:.1f} | {p99t:.1f} | {tpot:.2f} | {p99p:.2f} | {done} |")
PY

log "预期现象（对照 12.3 的模拟结论）"
cat <<'TXT'
- TP=2/4 的 output tok/s 通常接近线性，扩展效率 0.85–0.95（NVLink 域内）。
- TP 越高，mean TTFT 通常越低（prefill 被切开），但 TPOT 的改善会变缓。
- 如果 TP=4 的扩展效率 < 0.8，先查：是否跨 NUMA、是否在 MIG 上、batch 是否太小。
- 如果 server 启动报 head 数不能被 TP 整除：换 TP 或换模型（见 12.4 的约束）。
TXT

log "结果目录：$RESULT_DIR"
log "成本提示：本次实验预算 ≈ \$$(python3 -c "print(round($HOURS*$HOURLY_USD,2))")（$HOURS h × \$$HOURLY_USD/h）"
cat <<'TXT'

================= 关机 checklist =================
[ ] 1. vLLM server 已停止（本脚本 trap 会 kill；再 ps -ef | grep vllm 确认）
[ ] 2. nvidia-smi 显存回到基线（无残留进程）
[ ] 3. 结果已下载到本地：scp -r <实例>:$HOME/tp_bench_results_* ./
[ ] 4. AutoDL：控制台「关机」（不是「释放」，释放会丢数据盘）
[ ] 5. RunPod：Stop Pod（不是 Terminate，Terminate 会删卷）
[ ] 6. Lambda：Terminate instance 后确认账单页实例数归零
[ ] 7. 记一笔实际费用与耗时（下节课的成本表用它校准）
==================================================
TXT
