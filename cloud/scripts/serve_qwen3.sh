#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# serve_qwen3.sh — 起一个 Qwen3 服务（Mac 本机 / 云上 Docker 二选一）
#
# 用法：
#   bash cloud/scripts/serve_qwen3.sh                     # 本机（vllm-metal）默认 Qwen/Qwen3-8B
#   MODEL=Qwen/Qwen3-0.6B PORT=8000 bash cloud/scripts/serve_qwen3.sh
#   MODE=docker TP=4 MAX_LEN=8192 bash cloud/scripts/serve_qwen3.sh
#
# 环境变量：
#   MODE          local | docker           （默认 local）
#   MODEL         模型 ID                  （默认 Qwen/Qwen3-8B）
#   PORT          宿主机端口              （默认 8000）
#   TP            tensor-parallel-size     （默认 1，仅 docker 模式）
#   MAX_LEN       --max-model-len          （默认 8192）
#   GPU_UTIL      --gpu-memory-utilization （默认 0.90）
#   MAX_SEQS      --max-num-seqs           （默认 256）
#   MAX_BATCHED   --max-num-batched-tokens （默认 8192）
#   IMAGE         docker 镜像              （默认 vllm/vllm-openai:v0.28.0）
#   HF_CACHE      宿主机 HF 缓存目录       （默认 $HOME/.cache/huggingface）
#   DRY_RUN       1 = 只打印命令不执行
#
# 退出后记得按 cloud/README.md §5 的 checklist 关机。
# ---------------------------------------------------------------------------
set -euo pipefail

MODE="${MODE:-local}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_SEQS="${MAX_SEQS:-256}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.28.0}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
DRY_RUN="${DRY_RUN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

log() { printf '[serve_qwen3] %s\n' "$*"; }
run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] %s\n' "$*"
  else
    "$@"
  fi
}

case "${MODE}" in
  local)
    command -v vllm >/dev/null 2>&1 || {
      log "找不到 vllm CLI。Mac 上先安装："
      log "  brew tap vllm-project/vllm-metal && brew install vllm-metal"
      log "（命令以 vllm-metal tap 的 README 为准，版本待核对）"
      exit 1
    }
    log "本机模式：${MODEL}  port=${PORT}  max-model-len=${MAX_LEN}  gpu-util=${GPU_UTIL}"
    log "起好后另开终端验证：curl -s http://127.0.0.1:${PORT}/health"
    run vllm serve "${MODEL}" \
      --host 0.0.0.0 --port "${PORT}" \
      --max-model-len "${MAX_LEN}" \
      --gpu-memory-utilization "${GPU_UTIL}"
    ;;
  docker)
    command -v docker >/dev/null 2>&1 || { log "找不到 docker，请先安装 Docker"; exit 1; }
    mkdir -p "${HF_CACHE}" "${REPO_DIR}/results"
    log "Docker 模式：${MODEL}  port=${PORT}  tp=${TP}  image=${IMAGE}"
    log "起好后另开终端验证：curl -s http://127.0.0.1:${PORT}/v1/models"
    log "关机：docker compose down -v 或 docker rm -f vllm-qwen3"
    run docker run --rm --gpus all --ipc=host \
      --name vllm-qwen3 -p "${PORT}:8000" \
      -v "${HF_CACHE}:/root/.cache/huggingface" \
      -v "${REPO_DIR}/results:/workspace/results" \
      -e HF_HOME=/root/.cache/huggingface \
      "${IMAGE}" \
      --model "${MODEL}" --host 0.0.0.0 --port 8000 \
      --tensor-parallel-size "${TP}" \
      --max-model-len "${MAX_LEN}" \
      --gpu-memory-utilization "${GPU_UTIL}" \
      --max-num-seqs "${MAX_SEQS}" \
      --max-num-batched-tokens "${MAX_BATCHED}" \
      --enable-prefix-caching --enable-chunked-prefill
    ;;
  *)
    log "MODE 只支持 local | docker（收到：${MODE}）"
    exit 2
    ;;
esac
