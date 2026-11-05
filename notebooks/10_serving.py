# %% [markdown]
# # 第 10 章 · 用 vLLM 提供服务
#
# > 09 章看的是内部结构；这一章把服务真的跑起来：Mac 上用 `vllm-metal`，云上用官方镜像，然后用 `httpx` 把 OpenAI 兼容 API 逐个打一遍。
#
# | 项目 | 内容 |
# | --- | --- |
# | 预计用时 | 60–75 分钟 |
# | 运行环境 | 🏠 Mac（有 `vllm-metal` 时在线）· 无服务时自动回放 `cloud/fixtures/` |
# | 前置章节 | 09 |
# | 锁定版本 | vLLM 0.28.x（Mac: vllm-metal；云: vllm/vllm-openai 镜像） |
# | 本章产物 | 部署拓扑、启动命令、OpenAI 接口契约表、10 条 smoke test、显存/成本估算 |
#
# **学完你能回答：**
# 1. 在 Mac 和云上分别怎么起一个 OpenAI 兼容服务？关键参数有哪些？
# 2. `/v1/chat/completions`、`/v1/completions`、`/v1/embeddings` 的请求和响应长什么样？
# 3. LoRA 多适配器、结构化输出、工具调用分别要开哪些参数？
# 4. Qwen3-8B / 27B 在 24/48/80GB 卡上、不同精度下能开多少并发？每百万 token 大概多少钱？

# %% [markdown]
# ## 10.1 问题与直觉
#
# 把模型跑起来不难，难的是**让所有客户端不用改代码就能用**。业务方只认 OpenAI 的 JSON 格式：`messages` 进、`choices[].message` 出、流式走 SSE、错误码是 400/422/429。vLLM 的 API Server 负责把这套「世界语」翻译成引擎内部的 `EngineCoreRequest`。
#
# ![](../assets/figures/ch10_topology.svg)
#
# 图里三个要点：
#
# 1. **两种部署形态共用同一套 API**：Mac 上是单进程 Metal worker，云上是「API+EngineCore 容器 + 多个 GPU worker」，客户端代码一行不用改；
# 2. **能力是叠加的**：LoRA 适配器、embeddings 实例、Prometheus 指标都是并列的附加件，不是重写服务；
# 3. **关机是流程的一部分**：图底部的 checklist 不是玩笑，按小时计费的卡忘关一晚，预算就没了。
#
# **反例**：如果不用兼容层，每个业务方都要自己实现 tokenize、流式拼接、重试、token 计数；换模型时全部客户端重写。这就是「协议」的价值：把变化关在服务端。

# %% [markdown]
# ## 10.2 从零实现：启动命令 + 一个会降级的客户端
#
# 先写启动方式。下面三种任选其一，**不要同时跑**（8000 端口会冲突）：
#
# | 场景 | 命令 | 说明 |
# | --- | --- | --- |
# | Mac 单机体验 | `brew tap vllm-project/vllm-metal` → `brew install vllm-metal` → `vllm serve ...` | 需要 Apple Silicon，Metal 后端；命令以 tap 的 README 为准（版本待核对） |
# | 云上 Docker | `cd cloud && docker compose up -d` | 镜像 `vllm/vllm-openai:v0.28.0`，挂载 HF cache |
# | 云上裸 docker run | 见下面输出 | 临时调试用，验证完就删 |

# %%
MODEL = "Qwen/Qwen3-8B"
SERVE = {
    "Mac（vllm-metal，单机体验）": [
        "brew tap vllm-project/vllm-metal",
        "brew install vllm-metal",
        f"vllm serve {MODEL} --port 8000 --max-model-len 4096",
    ],
    "云上（官方镜像 + compose）": [
        "cd cloud",
        "docker compose up -d",
        "curl -s http://127.0.0.1:8000/health",
    ],
    "云上（裸 docker run，临时调试）": [
        "docker run --rm --gpus all -p 8000:8000 \\",
        "  -v ~/.cache/huggingface:/root/.cache/huggingface \\",
        f"  vllm/vllm-openai:v0.28.0 --model {MODEL}",
    ],
}
for mode, cmds in SERVE.items():
    print(f"[{mode}]")
    for c in cmds:
        print("  " + c)

# %% [markdown]
# 生产上不会只给一个 `--model`。下面是一条「能力全开」的命令，每个参数都在本章后面有对应小节：
#
# - `--enable-lora`：开多适配器；`--max-loras` / `--max-lora-rank` 是容量上限；
# - `--enable-auto-tool-choice --tool-call-parser hermes`：让模型按工具 schema 返回 `tool_calls`；
# - `--structured-outputs-config.backend xgrammar`：约束解码；请求侧用 `response_format` 触发；
# - `--gpu-memory-utilization` / `--max-num-seqs` / `--max-num-batched-tokens`：第 11 章重点调的三个旋钮。

# %%
PROD_FLAGS = f"""vllm serve {MODEL} \\
  --host 0.0.0.0 --port 8000 \\
  --max-model-len 8192 \\
  --gpu-memory-utilization 0.90 \\
  --max-num-seqs 256 --max-num-batched-tokens 8192 \\
  --enable-prefix-caching --enable-chunked-prefill \\
  --enable-lora --max-loras 4 --max-lora-rank 64 \\
  --enable-auto-tool-choice --tool-call-parser hermes \\
  --structured-outputs-config.backend xgrammar \\
  --api-key YOUR_KEY"""
print(PROD_FLAGS)
assert "--enable-lora" in PROD_FLAGS and "--enable-auto-tool-choice" in PROD_FLAGS

# %% [markdown]
# 客户端只用 `httpx`，不依赖 `openai` 包。核心设计是**自动降级**：启动时探测 `/health`，通了就走真实 HTTP；不通就读 `cloud/fixtures/` 里录好的响应，并在表里标注「回放」。这样同一份 notebook 在没装 vLLM 的 Mac 上也能跑通，上云后无需改代码。

# %%
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import httpx
except ImportError:          # 没装 httpx 也能回放；装了才可能走在线
    httpx = None

from aiinfra.bench import human_bytes, kv_cache_bytes
from aiinfra.plotting import save, setup_style

setup_style()
ROOT = Path.cwd().parent
FIXTURES = ROOT / "cloud" / "fixtures"
print("fixtures:", sorted(p.name for p in FIXTURES.glob("*.json")))
assert (FIXTURES / "chat_completions.json").exists()

# %%
def _post_stream(base_url: str, path: str, payload: dict) -> dict:
    """流式 POST：记录首 token 时间与 token 数（一个 SSE data 块 ≈ 一个 token）。"""
    t0 = time.perf_counter()
    with httpx.Client(timeout=60) as client:
        with client.stream("POST", base_url + path, json=payload) as r:
            if r.status_code != 200:
                return {"status_code": r.status_code, "ttft_ms": None,
                        "body": r.read().decode(), "n_tokens": 0, "replay": False}
            ttft, n_tokens = None, 0
            for line in r.iter_lines():
                if not line.startswith("data: ") or "[DONE]" in line:
                    continue
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                json.loads(line[6:])
                n_tokens += 1
    return {"status_code": 200, "ttft_ms": ttft, "body": None,
            "n_tokens": n_tokens, "replay": False}

# %%
class VLLMClient:
    """OpenAI 兼容客户端：有服务走真实 HTTP，没有服务回放 cloud/fixtures。"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", fixtures: Path = FIXTURES):
        self.base_url = base_url.rstrip("/")
        self.chat_fx = json.loads((fixtures / "chat_completions.json").read_text(encoding="utf-8"))
        self.embed_fx = json.loads((fixtures / "embeddings.json").read_text(encoding="utf-8"))
        self.online = False
        if httpx is not None:
            try:
                self.online = httpx.get(self.base_url + "/health", timeout=0.4).status_code == 200
            except Exception:
                self.online = False

    def mode(self) -> str:
        return "在线" if self.online else "回放"

    def chat(self, payload: dict, idx: int) -> dict:
        if not self.online:
            rec = self.chat_fx["records"][idx]
            usage = rec["response"].get("usage", {})
            return {"status_code": rec["status_code"], "ttft_ms": rec["ttft_ms"],
                    "body": rec["response"], "n_tokens": usage.get("completion_tokens", 0),
                    "replay": True}
        return _post_stream(self.base_url, "/v1/chat/completions", payload | {"stream": True})

    def embed(self, record: dict) -> dict:
        if not self.online:
            return {"status_code": record["status_code"], "ttft_ms": record["latency_ms"],
                    "body": record["response"], "n_tokens": 0, "replay": True}
        r = httpx.post(self.base_url + "/v1/embeddings",
                       json={"model": record["response"]["model"], "input": record["input"]},
                       timeout=30)
        return {"status_code": r.status_code, "ttft_ms": None, "body": r.json(),
                "n_tokens": 0, "replay": False}

# %%
client = VLLMClient()
if not client.online:
    print("WARNING 当前为回放模式：未检测到 http://127.0.0.1:8000/health")
    print("   起服务（Mac）: brew tap vllm-project/vllm-metal && brew install vllm-metal")
    print("                  vllm serve Qwen/Qwen3-8B --port 8000")
    print("   起服务（云） : cd cloud && docker compose up -d")
    print("   起好后重跑本 cell，会自动切换为在线模式（客户端代码不用改）。")
print("客户端模式 =", client.mode())
print("fixture 录制说明：", client.chat_fx["provenance"]["note"][:48], "...")

# %% [markdown]
# ## 10.3 实测：10 条 smoke test
#
# 覆盖 8 种正常 chat 场景 + 1 条按预期失败的请求 + 1 条 embeddings。回放模式下「首 token/首包时间」和「输出 tokens」直接来自 fixture 的录制字段；在线模式下由客户端实测（流式首块 = TTFT）。

# %%
rows = []
cases = [(rec["case"], "chat", i) for i, rec in enumerate(client.chat_fx["records"])]
cases.append((client.embed_fx["records"][0]["case"], "embed", 0))
for n, (case, kind, idx) in enumerate(cases, 1):
    if kind == "chat":
        rec = client.chat_fx["records"][idx]
        out = client.chat(rec["request"], idx)
    else:
        rec = client.embed_fx["records"][idx]
        out = client.embed(rec)
    status = out["status_code"]
    note = "OK" if status == 200 else str(out["body"])[:64]
    rows.append({"#": n, "场景": case, "状态码": status,
                 "首 token/首包(ms)": None if out["ttft_ms"] is None else round(out["ttft_ms"], 1),
                 "输出 tokens": out["n_tokens"], "模式": client.mode(), "备注": note})
smoke_df = pd.DataFrame(rows)
smoke_df

# %%
assert len(smoke_df) == 10, "smoke test 应有 10 条请求"
assert (smoke_df["状态码"] == 200).sum() == 9, "应有 9 条成功"
assert (smoke_df["状态码"] == 400).sum() == 1, "应有 1 条按预期返回 400"
assert smoke_df["首 token/首包(ms)"].notna().sum() == 9, "成功请求都应有首包时间"
ok = smoke_df[smoke_df["状态码"] == 200]
print(f"成功 {len(ok)}/10；首包中位数 {ok['首 token/首包(ms)'].median():.1f} ms；"
      f"输出 token 合计 {int(ok['输出 tokens'].sum())}")

# %% [markdown]
# 三条「高级能力」在响应里的形状，回放与在线完全一致，下面用 fixture 里的真实响应直接验证：

# %%
json_rec = next(r for r in client.chat_fx["records"] if r["case"] == "json-schema")
parsed = json.loads(json_rec["response"]["choices"][0]["message"]["content"])
assert set(parsed) == {"model", "weights_gb", "kv_budget_gb"}
print("structured output（response_format=json_schema）解析成功：", parsed)

tool_rec = next(r for r in client.chat_fx["records"] if r["case"] == "tool-call")
call = tool_rec["response"]["choices"][0]["message"]["tool_calls"][0]
assert call["function"]["name"] == "get_weather"
print("tool calling：", call["function"]["name"], call["function"]["arguments"],
      "| finish_reason =", tool_rec["response"]["choices"][0]["finish_reason"])

# %%
emb = client.embed(client.embed_fx["records"][0])
vec = np.asarray(emb["body"]["data"][0]["embedding"], dtype=float)
assert vec.shape[0] == client.embed_fx["embedding_dim"]
pd.DataFrame([{"输入": client.embed_fx["records"][0]["input"], "维度": vec.shape[0],
               "前 4 维": np.round(vec[:4], 3).tolist(), "首包(ms)": emb["ttft_ms"],
               "usage": emb["body"]["usage"]}])

# %% [markdown]
# ### 显存估算：权重 × 精度 × `gpu_memory_utilization`
#
# 公式只有三行，但每个参数都能解释一个线上事故：
#
# ```
# 权重显存 = 参数量 × 每参数字节（fp16=2, int8=1, int4=0.5）
# KV 显存 = 2 × layers × kv_heads × head_dim × 2 字节 × token 数
# 可用 KV = 卡容量 × gpu_memory_utilization − 权重 − 框架/激活预留(约 2GB)
# ```
#
# **关键陷阱**：量化只压缩权重，**不压缩 KV cache**（KV 默认仍是 fp16/bf16）。所以 int4 的 8B 在 24GB 卡上能跑，KV 空间主要来自「省下来的权重」。

# %%
MODELS = {
    "Qwen3-8B": {"params": 8.2e9, "n_layer": 36, "n_kv_head": 8, "head_dim": 128},
    "Qwen3-27B": {"params": 27.0e9, "n_layer": 64, "n_kv_head": 8, "head_dim": 128},
}
WEIGHT_BYTES = {"fp16": 2, "int8": 1, "int4": 0.5}
UTIL, OVERHEAD_GB = 0.90, 2.0

rows = []
for name, m in MODELS.items():
    kv_per_tok = kv_cache_bytes(m["n_layer"], m["n_kv_head"], m["head_dim"], 1, 1, 2)
    for dt, wb in WEIGHT_BYTES.items():
        weights = m["params"] * wb / 1e9
        kv_80 = 80 * UTIL - weights - OVERHEAD_GB
        rows.append({"模型": name, "权重精度": dt, "权重(GB)": round(weights, 1),
                     "KV/token(KB)": round(kv_per_tok / 1024, 1),
                     "24GB 能放下": "是" if 24 * UTIL - weights - OVERHEAD_GB > 1 else "否",
                     "80GB KV 预算(GB)": round(kv_80, 1),
                     "80GB 可容纳 token": int(max(kv_80, 0) * 1e9 // kv_per_tok),
                     "80GB@2048ctx 并发": int(max(kv_80, 0) * 1e9 // kv_per_tok // 2048)})
df_mem = pd.DataFrame(rows)
df_mem

# %%
rows = []
for name, m in MODELS.items():
    kv_per_tok = kv_cache_bytes(m["n_layer"], m["n_kv_head"], m["head_dim"], 1, 1, 2)
    weights = m["params"] * 2 / 1e9
    for util in [0.80, 0.85, 0.90, 0.95]:
        kv_gb = 80 * util - weights - OVERHEAD_GB
        rows.append({"模型": name, "gpu_memory_utilization": util,
                     "权重(GB)": round(weights, 1), "KV 预算(GB)": round(kv_gb, 1),
                     "可容纳 token": int(max(kv_gb, 0) * 1e9 // kv_per_tok),
                     "2048 ctx 并发": int(max(kv_gb, 0) * 1e9 // kv_per_tok // 2048)})
df_util = pd.DataFrame(rows)
df_util

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
labels = (df_mem["模型"] + " " + df_mem["权重精度"]).tolist()
w_gb = df_mem["权重(GB)"].to_numpy()
kv_gb = df_mem["80GB KV 预算(GB)"].to_numpy()
ax.bar(labels, w_gb, color="#2563EB", label="weights")
ax.bar(labels, kv_gb, bottom=w_gb, color="#7C3AED", label="KV budget")
ax.bar(labels, [OVERHEAD_GB] * len(df_mem), bottom=w_gb + kv_gb,
       color="#9CA3AF", label="framework / activation")
ax.axhline(80, color="#EF4444", ls="--", lw=1.5, label="80GB card capacity")
ax.set_ylabel("GPU memory (GB)")
ax.set_title("80GB / util=0.90: weight precision decides how much KV fits")
ax.tick_params(axis="x", rotation=12, labelsize=9)
ax.legend(ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
save(fig, "ch10_memory_budget")

# %%
fig, ax = plt.subplots()
utils = np.linspace(0.75, 0.95, 9)
for name, m in MODELS.items():
    kv_per_tok = kv_cache_bytes(m["n_layer"], m["n_kv_head"], m["head_dim"], 1, 1, 2)
    weights = m["params"] * 2 / 1e9
    tokens = np.maximum(80 * utils - weights - OVERHEAD_GB, 0) * 1e9 / kv_per_tok
    ax.plot(utils, tokens / 1000, "o-", label=name)
ax.set_xlabel("gpu_memory_utilization")
ax.set_ylabel("KV token capacity (thousands)")
ax.set_title("Higher gpu_memory_utilization: more KV capacity, higher OOM risk")
ax.legend()
save(fig, "ch10_kv_vs_util")

# %% [markdown]
# **读表**：
#
# - 8B 在 fp16 下：24GB 卡只剩约 3.2GB 给 KV（约 2.2 万 token、10 个 2048 上下文并发）；80GB 卡能到 36 万 token、177 个并发。这就是「同样模型换卡能多开 17 倍并发」的来源。
# - 27B 在 fp16 下 54GB 权重，24GB/48GB 直接放不下；int4 后 13.5GB，才能在 24GB 卡上跑，但 KV 只剩约 6GB（11 个并发）。
# - `gpu_memory_utilization` 从 0.80 调到 0.95，8B 的 KV 容量涨约 25%；再往上就会被 CUDA graph、激活和碎片吃掉——**0.90 是常用折中，0.95 以上只在确认无峰值时用**。
#
# ### 成本估算：每百万输出 token 多少钱
#
# 用 11 章 fixture 的回放吞吐做量级估算（8B 在 4090/A100/H100 上分别为 420/812/1350 tok/s；27B 按参数量线性外推）。**价格是示例价，下单前以平台页面为准。**

# %%
CARD_PRICE = {"4090 24GB": 0.69, "A100 80GB": 1.99, "H100 80GB": 2.99}   # USD/小时（示例）
REF_8B = {"4090 24GB": 420, "A100 80GB": 812, "H100 80GB": 1350}        # tok/s（回放量级）
rows = []
for card, price in CARD_PRICE.items():
    for model, toks in [("Qwen3-8B", REF_8B[card]), ("Qwen3-27B", REF_8B[card] * 8.2 / 27)]:
        rows.append({"卡型": card, "模型": model, "小时价(USD)": price,
                     "参考吞吐(tok/s)": round(toks),
                     "每百万输出 token(USD)": round(price / (toks * 3600) * 1e6, 2)})
df_cost = pd.DataFrame(rows)
df_cost

# %% [markdown]
# 结论：**每 token 成本 ≈ 小时价 ÷ (吞吐 × 3600)**。4090 单卡单价低、但吞吐也低，算下来不一定比 A100 贵；真正决定成本的是「你的并发能不能把卡喂饱」——这正是第 11 章压测要回答的问题。

# %% [markdown]
# ## 10.4 对照工业实现（vLLM 0.28.x）
#
# ### 接口契约表
#
# | Endpoint | 关键请求字段 | 关键响应字段 | vLLM 实现（0.28.x） |
# | --- | --- | --- | --- |
# | `POST /v1/chat/completions` | `model` `messages` `max_tokens` `temperature` `stream` `tools` `tool_choice` `response_format` `stop` | `choices[].message`、`finish_reason`、`usage` | `vllm/entrypoints/openai/api_server.py` + `serving_chat.py` |
# | `POST /v1/completions` | `model` `prompt` `max_tokens` `n` `stream` | `choices[].text`、`usage` | 同目录 `serving_completion.py` |
# | `POST /v1/embeddings` | `model` `input` `encoding_format` | `data[].embedding`、`usage` | `serving_embedding.py`；需 `--task embed`（`--runner pooling`，版本待核对） |
# | `GET /v1/models` | — | 已加载模型与 LoRA 适配器列表 | `serving_models.py`（待核对） |
# | `GET /health` | — | 200 = 就绪 | 用于客户端探测/负载均衡 |
# | `GET /metrics` | — | Prometheus 指标（`vllm:*`） | 第 11 章的瓶颈判定靠它 |

# %% [markdown]
# ### 高级能力的开关与请求形状
#
# | 能力 | 服务端参数 | 请求侧写法 | 响应侧特征 |
# | --- | --- | --- | --- |
# | LoRA 多适配器 | `--enable-lora --max-loras 4 --max-lora-rank 64 --lora-modules name=path` | `model` 填适配器名 | 与基座同结构，`model` 字段回显适配器名 |
# | 动态加载 LoRA | 无需重启 | `POST /v1/load_lora_adapter`（vLLM 扩展） | 返回成功；`/v1/models` 可查 |
# | 结构化输出 | `--structured-outputs-config.backend xgrammar`（旧版 `--guided-decoding-backend`，待核对） | `response_format={"type":"json_schema","json_schema":{...}}` | `message.content` 是合法 JSON |
# | 引导解码（legacy） | 同上 | 请求里带 `guided_json` / `guided_regex` 等扩展字段 | 同上 |
# | 工具调用 | `--enable-auto-tool-choice --tool-call-parser hermes` | `tools=[...]`、`tool_choice="auto"` | `choices[].message.tool_calls[]`，`finish_reason="tool_calls"` |
#
# 注意：`--tool-call-parser` 必须与模型训练时的工具格式匹配（Qwen 常用 `hermes`，Llama 常用 `llama3_json`）。选错不会报错，只会退化成普通文本。

# %% [markdown]
# ### 差距在哪：兼容层不是「加个路由」
#
# 一个能上生产的 OpenAI 兼容层还要处理：流式背压与客户端断连、`usage` 的精确统计（含缓存命中 token）、错误码映射（400/422/429/503）、多模态输入、tokenizer 差异、请求取消（`abort`）、API Key 鉴权与限流。这些在 vLLM 里散落在 `vllm/entrypoints/openai/` 与 `vllm/v1/engine/` 中，也是「自己写个 FastAPI 包一层」和工业实现的主要差距。

# %% [markdown]
# ## 10.5 练习与小结
#
# **练习**（⭐ 改参数 / ⭐⭐ 改实现 / ⭐⭐⭐ 开放设计）
#
# 1. ⭐ 把 `--max-model-len` 从 8192 改成 4096，重跑 `df_mem`/`df_util` 的 KV 计算。并发数为什么不是翻倍？提示：`--max-model-len` 影响的是单请求上限，不是块总数。
# 2. ⭐ 用 `--gpu-memory-utilization 0.95` 起服务，再压 32 并发；观察是否 OOM 或吞吐下降。解释 0.95 为什么比 0.90 危险。
# 3. ⭐⭐ 给 `VLLMClient` 加 `completion()`（`/v1/completions`）与 `list_models()`，用 fixture 回放验证；再在真服务上各打一条。
# 4. ⭐⭐ 写一个 `load_lora_adapter(client, name, path)`：启动时 `--enable-lora`，运行中动态加载，然后对比「基座 / 适配器」对同一 prompt 的输出与 TTFT。
# 5. ⭐⭐⭐ 设计一份「同一服务同时提供 chat + embeddings + LoRA」的部署方案：几个进程、几个端口、GPU 怎么分、K8s 怎么探活、如何限流。（评分要点：资源隔离、端口与探活、失败降级、成本、监控 5 条）
#
# **你现在应该能回答：**
#
# - Mac 与云上的起服务命令差在哪？为什么客户端代码可以不变？
# - 显存账本的三个公式是什么？为什么 int4 不省 KV cache？
# - 什么情况下该用 `response_format`，什么情况下该用 `tool_calls`？两者能一起用吗？
#
# **小结与下一章**
#
# 到这里服务已经能接真实流量了。但「能跑」不等于「跑得好」：并发加到哪里开始亏、TTFT 和 TPOT 谁先崩、瓶颈在算力还是 KV——下一章用压测和决策树回答。

# %% [markdown]
# **延伸阅读**
#
# - vLLM OpenAI 兼容服务器文档（`docs/serving/openai_compatible_server.md`，版本随 0.28.x）；
# - `vllm/entrypoints/openai/serving_chat.py`：`response_format` 与 tool parser 的分支；
# - `cloud/README.md`：AutoDL / RunPod / Lambda 的开机、起服务、压测、关机全流程；
# - 第 11 章：压测、指标与调参决策树。
