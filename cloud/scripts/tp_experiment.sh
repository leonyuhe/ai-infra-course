#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# tp_experiment.sh — TP 扫描（第 12 章预留）：--tensor-parallel-size 1/2/4
#
# TP 是启动参数，所以每个 TP 都要「重启服务 → 压测 → 收结果 → 关服务」。
# 本脚本只做单机多卡；跨机 PP/DP 不在范围内。
#
# 用法：
#   nvidia-smi -L                          # 先确认卡数
#   bash cloud/scripts/tp_experiment.sh    # TP=1,2,4 逐个跑
#   TPS="1 2" CONCURRENCY="1 8 32" bash cloud/scripts/tp_experiment.sh
#
# 环境变量：
#   MODEL       模型（默认 Qwen/Qwen3-8B）
#   TPS         TP 列表（默认 "1 2 4"）
#   CONCURRENCY 每个 TP 下的并发扫描（默认 "1 8 32"）
#   NUM_PROMPTS 每轮请求数（默认 200）
#   IMAGE       镜像（默认 vllm/vllm-openai:v0.28.0）
#   PORT        端口（默认 8000）
#   HF_CACHE    宿主机 HF 缓存（默认 $HOME/.cache/huggingface）
#   RESULT_DIR  结果目录（默认 <repo>/cloud/results/tp）
#   DRY_RUN     1 = 只打印命令
#
# 预期现象（上云后自行核对，第 12 章会给理论模型）：
#   TP=1 → TP=2：吞吐提升明显（每卡权重减半、KV 减半）
#   TP=2 → TP=4：提升变小，通信占比上升（小 batch 时甚至变慢）
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
TPS="${TPS:-1 2 4}"
CONCURRENCY="${CONCURRENCY:-1 8 32}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.28.0}"
PORT="${PORT:-8000}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULT_DIR="${RESULT_DIR:-${REPO_DIR}/results/tp}"
DRY_RUN="${DRY_RUN:-0}"
CONTAINER="vllm-tp-sweep"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"

log() { printf '[tp_experiment] %s\n' "$*"; }

run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] %s\n' "$*"
  else
    "$@"
  fi
}

run_bench() {
  # $1 = tp, $2 = 并发数, $3 = 输出文件名
  local tp="$1" c="$2" out="$3"
  local args=(bench serve
    --backend openai --base-url "http://127.0.0.1:${PORT}" --model "${MODEL}"
    --dataset-name random --random-input-len 512 --random-output-len 128
    --num-prompts "${NUM_PROMPTS}" --max-concurrency "${c}"
    --metric-percentiles 50,90,99 --save-result --result-filename "${out}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] vllm %s\n' "${args[*]}"
    return 0
  fi
  if command -v vllm >/dev/null 2>&1; then
    vllm "${args[@]}" --result-dir "${RESULT_DIR}"
  else
    log "宿主机没有 vllm CLI，改用容器跑 bench"
    docker run --rm --network host \
      -v "${RESULT_DIR}:/results" \
      --entrypoint vllm "${IMAGE}" "${args[@]}" --result-dir /results
  fi
}

command -v nvidia-smi >/dev/null 2>&1 || { log "没有 nvidia-smi：本脚本只能在云上 GPU 机器跑"; exit 1; }
command -v docker >/dev/null 2>&1 || { log "找不到 docker"; exit 1; }
N_GPU="$(nvidia-smi -L | wc -l | tr -d ' ')"
log "检测到 ${N_GPU} 张 GPU；计划 TPS=${TPS}"
mkdir -p "${HF_CACHE}" "${RESULT_DIR}"

cleanup() {
  log "清理容器 ${CONTAINER}"
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

wait_health() {
  local waited=0
  while (( waited < HEALTH_TIMEOUT )); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      log "服务就绪（等待 ${waited}s）"
      return 0
    fi
    sleep 10
    (( waited += 10 ))
  done
  log "等待 ${HEALTH_TIMEOUT}s 仍未就绪，放弃；看 docker logs ${CONTAINER}"
  docker logs --tail 80 "${CONTAINER}" || true
  return 1
}

for tp in ${TPS}; do
  if (( tp > N_GPU )); then
    log "跳过 TP=${tp}（只有 ${N_GPU} 张卡）"
    continue
  fi
  log "=== TP=${tp}：启动服务 ==="
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  run docker run -d --name "${CONTAINER}" --gpus all --ipc=host \
    -p "${PORT}:8000" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    "${IMAGE}" \
    --model "${MODEL}" --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size "${tp}" \
    --gpu-memory-utilization 0.90 --max-num-seqs 256

  if [[ "${DRY_RUN}" != "1" ]]; then
    wait_health || exit 1
  fi

  for c in ${CONCURRENCY}; do
    STAMP="$(date +%Y%m%d_%H%M%S)"
    log "TP=${tp} 并发=${c}"
    run_bench "${tp}" "${c}" "tp${tp}_c${c}_${STAMP}.json"
    sleep 5
  done
  cleanup
done

log "完成：结果在 ${RESULT_DIR}"
log "关机 checklist：docker rm -f ${CONTAINER} → 平台控制台关机/释放 → 核对账单"
