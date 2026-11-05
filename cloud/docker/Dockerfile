# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# 基于 vLLM 官方镜像的最小封装层。
#
# 构建（仅在上云机器上需要；本地无需构建）：
#   docker build -f cloud/docker/Dockerfile -t aiinfra/vllm:0.28.0 cloud/docker
# 运行：
#   bash cloud/scripts/serve_qwen3.sh        # MODE=docker
#
# 注意：tag 必须与课程锁定版本一致（vLLM 0.28.x）。若该 tag 拉不到，
# 用 vllm/vllm-openai:latest 并把 docker run 日志里的实际版本回填到实验报告。
# ---------------------------------------------------------------------------
FROM vllm/vllm-openai:v0.28.0

ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    VLLM_LOGGING_LEVEL=INFO \
    PYTHONUNBUFFERED=1

# 实测与压测产物都落在这里，运行时用 -v 挂到宿主机
RUN mkdir -p /workspace/results

WORKDIR /workspace

EXPOSE 8000

# 官方镜像自带 curl；首次启动要加载模型，start-period 给足 10 分钟
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# 覆盖 ENTRYPOINT 成 vllm serve，CMD 给默认模型与端口；
# docker run 时追加的参数会接在 CMD 后面。
ENTRYPOINT ["vllm", "serve"]
CMD ["Qwen/Qwen3-8B", "--host", "0.0.0.0", "--port", "8000"]
