#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# bench_sweep.sh — 并发扫描：对已启动的 vLLM 服务依次压 1/4/8/16/32 并发
#
# 用法：
#   bash cloud/scripts/bench_sweep.sh
#   BASE_URL=http://127.0.0.1:8000 MODEL=Qwen/Qwen3-8B NUM_PROMPTS=200 \
#     bash cloud/scripts/bench_sweep.sh
#   CONCURRENCY="1 4 8 16 32 64" bash cloud/scripts/bench_sweep.sh
#
# 环境变量：
#   BASE_URL       服务地址（默认 http://127.0.0.1:8000）
#   MODEL          模型 ID（默认 Qwen/Qwen3-8B）
#   NUM_PROMPTS    每轮请求数（默认 200）
#   INPUT_LEN      随机输入长度（默认 512）
#   OUTPUT_LEN     随机输出长度（默认 128）
#   CONCURRENCY    并发列表（默认 "1 4 8 16 32"）
#   RESULT_DIR     结果目录（默认 <repo>/cloud/results）
#   BENCH_IMAGE   宿主机没装 vllm 时，用这个镜像跑 bench（默认同服务镜像）
#   DRY_RUN        1 = 只打印命令
#
# 输出：每个并发一个 JSON（vllm bench serve --save-result），可回填到
#      cloud/fixtures/benchmark_results.json 的同构结构，供第 11 章回放。
# ---------------------------------------------------------------------------
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
CONCURRENCY="${CONCURRENCY:-1 4 8 16 32}"
BENCH_IMAGE="${BENCH_IMAGE:-vllm/vllm-openai:v0.28.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${RESULT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)/results}"
DRY_RUN="${DRY_RUN:-0}"

log() { printf '[bench_sweep] %s\n' "$*"; }

run_bench() {
  # $1 = 并发数, $2 = 输出文件名
  local c="$1" out="$2"
  local args=(bench serve
    --backend openai --base-url "${BASE_URL}" --model "${MODEL}"
    --dataset-name random --random-input-len "${INPUT_LEN}" --random-output-len "${OUTPUT_LEN}"
    --num-prompts "${NUM_PROMPTS}" --max-concurrency "${c}"
    --metric-percentiles 50,90,99 --save-result --result-filename "${out}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] vllm %s\n' "${args[*]}"
    return 0
  fi
  if command -v vllm >/dev/null 2>&1; then
    vllm "${args[@]}" --result-dir "${RESULT_DIR}"
  else
    log "宿主机没有 vllm CLI，改用容器跑 bench（结果挂载到 ${RESULT_DIR}）"
    docker run --rm --network host \
      -v "${RESULT_DIR}:/results" \
      --entrypoint vllm "${BENCH_IMAGE}" "${args[@]}" --result-dir /results
  fi
}

if [[ "${DRY_RUN}" != "1" ]]; then
  curl -fsS "${BASE_URL}/health" >/dev/null || {
    log "服务未就绪：${BASE_URL}/health 不是 200；先起服务（serve_qwen3.sh）"
    exit 1
  }
fi
mkdir -p "${RESULT_DIR}"

log "目标=${BASE_URL} 模型=${MODEL} prompts=${NUM_PROMPTS} in/out=${INPUT_LEN}/${OUTPUT_LEN}"
log "并发序列：${CONCURRENCY}"

for c in ${CONCURRENCY}; do
  STAMP="$(date +%Y%m%d_%H%M%S)"
  OUT="bench_c${c}_${STAMP}.json"
  log "→ 并发 ${c} → ${RESULT_DIR}/${OUT}"
  run_bench "${c}" "${OUT}"
  # 让队列排空、KV 回收，避免上一轮的长尾污染下一轮
  sleep 5
done

log "完成。结果目录：${RESULT_DIR}"
log "下一步：把 JSON 里的 TTFT/TPOT/吞吐/KV 字段汇总成 fixtures，重跑第 11 章 notebook"
log "别忘了关机：docker compose -f cloud/docker/docker-compose.yml down -v"
