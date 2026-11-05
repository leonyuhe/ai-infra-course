#!/usr/bin/env bash
# 第 13 章 · LoRA 微调 → 合并 → vLLM 部署 的云上闭环（示例脚本，需在 NVIDIA 实例上执行）
#
# 流程：
#   0. 环境检查（GPU / vllm / peft / 成本护栏）
#   1. LoRA 微调：基座冻结，只训练 A/B（PEFT）
#   2. 验证「只更新 LoRA 参数」：基座权重哈希不变 + 合并前后贪心输出一致
#   3. 合并 adapter → 导出完整模型（路线 A：零推理开销）
#   4. vLLM 起服务并 smoke test 合并模型
#   5. 起「基座 + adapter 热插拔」服务（路线 B：多租户共享基座）
#   6. 成本结算 + 关机 checklist
#
# 用法：
#   bash cloud/scripts/13_lora_finetune.sh                 # 全流程，默认 200 步、2h 预算
#   SKIP_TRAIN=1 bash cloud/scripts/13_lora_finetune.sh    # 复用已训好的 adapter
#   HOURS=1 HOURLY_USD=1.9 bash cloud/scripts/13_lora_finetune.sh
#
# 前置（镜像见 cloud/README.md）：
#   python -m pip install -U peft datasets accelerate   # vllm 镜像通常已带 transformers
#  训练数据：JSONL，每行 {"prompt": "...", "response": "..."}；这是 RAG 问答的简化格式。

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DATA_FILE="${DATA_FILE:-$HOME/data/kb_qa_train.jsonl}"
WORK_DIR="${WORK_DIR:-$HOME/lora_run_$(date +%m%d_%H%M)}"
ADAPTER_DIR="$WORK_DIR/adapter"
MERGED_DIR="$WORK_DIR/merged"
MAX_STEPS="${MAX_STEPS:-200}"
MAX_LEN="${MAX_LEN:-2048}"
LORA_R="${LORA_R:-16}"
HOURS="${HOURS:-2}"
HOURLY_USD="${HOURLY_USD:-1.9}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
PORT_MERGED="${PORT_MERGED:-8001}"
PORT_ADAPTER="${PORT_ADAPTER:-8002}"

mkdir -p "$WORK_DIR"
PY="${PYTHON_BIN:-python3}"
log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\n❌ %s\n' "$*" >&2; exit 1; }

log "0) 环境检查"
command -v nvidia-smi >/dev/null || die "没有 nvidia-smi：请在 NVIDIA 实例上执行"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
"$PY" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || die "vllm/torch 不可用，先看 cloud/README.md 的镜像说明"
"$PY" -c "import peft, datasets, transformers; print('peft', peft.__version__, '| transformers', transformers.__version__)" \
  || die "缺 peft/datasets：pip install -U peft datasets accelerate"
[[ -f "$DATA_FILE" ]] || die "训练数据不存在：$DATA_FILE（格式见文件头注释）"
log "数据 = $DATA_FILE；输出 = $WORK_DIR；预计费用 ≈ \$$("$PY" -c "print(round($HOURS*$HOURLY_USD,2))")"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local port="$1"
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$port/health/" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

if [[ "$SKIP_TRAIN" != "1" ]]; then
  log "1) LoRA 微调（基座冻结，仅训练 A/B，最多 $MAX_STEPS 步）"
  "$PY" - <<'PY'
import json, os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

model_id = os.environ["MODEL"]
data_file = os.environ["DATA_FILE"]
out_dir = os.environ["ADAPTER_DIR"]
max_steps = int(os.environ["MAX_STEPS"])
max_len = int(os.environ["MAX_LEN"])
lora_r = int(os.environ["LORA_R"])

tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
model.config.use_cache = False

cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, bias="none",
                 task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
model = get_peft_model(model, cfg)
model.print_trainable_parameters()  # 期望：可训练参数 < 1%

raw = load_dataset("json", data_files=data_file, split="train")


def encode(ex):
    prompt = tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}],
                                     tokenize=False, add_generation_prompt=True)
    text = prompt + ex["response"] + tok.eos_token
    ids = tok(text, truncation=True, max_length=max_len)["input_ids"]
    p_ids = tok(prompt, truncation=True, max_length=max_len)["input_ids"]
    return {"input_ids": ids, "labels": [-100] * len(p_ids) + ids[len(p_ids):]}


ds = raw.map(encode, remove_columns=raw.column_names)


class Collator:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, feats):
        pad = self.tok.pad_token_id
        n = max(len(f["input_ids"]) for f in feats)
        input_ids = [f["input_ids"] + [pad] * (n - len(f["input_ids"])) for f in feats]
        labels = [f["labels"] + [-100] * (n - len(f["labels"])) for f in feats]
        attn = [[1] * len(f["input_ids"]) + [0] * (n - len(f["input_ids"])) for f in feats]
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}


args = TrainingArguments(
    output_dir=out_dir, max_steps=max_steps,
    per_device_train_batch_size=2, gradient_accumulation_steps=8,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
    logging_steps=10, save_strategy="no", bf16=True,
    report_to=[], remove_unused_columns=False,
)
Trainer(model=model, args=args, train_dataset=ds, data_collator=Collator(tok)).train()
model.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
print("adapter 已保存 ->", out_dir)
PY
else
  log "1) SKIP_TRAIN=1，复用 $ADAPTER_DIR"
fi

log "2) 验证：基座冻结 + 合并前后 greedy 输出一致"
"$PY" - <<'PY'
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id, adapter_dir = os.environ["MODEL"], os.environ["ADAPTER_DIR"]
tok = AutoTokenizer.from_pretrained(adapter_dir)
base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, adapter_dir)
model.eval()
prompt = tok.apply_chat_template([{"role": "user", "content": "用一句话解释 KV cache。"}],
                                 tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    before = model.generate(**inputs, max_new_tokens=24, do_sample=False)
with torch.no_grad():
    merged = model.merge_and_unload()
    merged.eval()
    after = merged.generate(**inputs, max_new_tokens=24, do_sample=False)
equal = torch.equal(before, after)
print("合并前后 greedy token 序列一致:", equal)
print("before:", tok.decode(before[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[:120])
print("after :", tok.decode(after[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[:120])
assert equal, "merge 前后输出不一致：检查是否对共享权重（如 tied lm_head）做了 LoRA"
merged.save_pretrained(os.environ["MERGED_DIR"])
tok.save_pretrained(os.environ["MERGED_DIR"])
print("合并模型已保存 ->", os.environ["MERGED_DIR"])
PY

log "3) 起 vLLM 服务（合并模型，路线 A）"
vllm serve "$MERGED_DIR" --served-model-name merged --host 127.0.0.1 --port "$PORT_MERGED" \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.85 >>"$WORK_DIR/serve_merged.log" 2>&1 &
SERVER_PID=$!
wait_for_server "$PORT_MERGED" || die "合并模型服务没起来，看 $WORK_DIR/serve_merged.log"
curl -s "http://127.0.0.1:$PORT_MERGED/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"merged","messages":[{"role":"user","content":"你好，介绍一下你自己"}],"max_tokens":64}' \
  | "$PY" -c "import json,sys; d=json.load(sys.stdin); print('smoke answer:', d['choices'][0]['message']['content'][:100])"
cleanup

log "4) 路线 B：基座 + adapter 热插拔（多租户共享一份基座）"
vllm serve "$MODEL" --served-model-name base --host 127.0.0.1 --port "$PORT_ADAPTER" \
  --enable-lora --lora-modules kb-lora="$ADAPTER_DIR" --max-lora-rank "$LORA_R" \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.85 >>"$WORK_DIR/serve_lora.log" 2>&1 &
SERVER_PID=$!
wait_for_server "$PORT_ADAPTER" || die "LoRA 服务没起来，看 $WORK_DIR/serve_lora.log"
for M in base kb-lora; do
  printf '  model=%s -> ' "$M"
  curl -s "http://127.0.0.1:$PORT_ADAPTER/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"max_tokens\":32}" \
    | "$PY" -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:60])"
done
cleanup

log "5) 预期现象（对照 13.3 的本地闭环）"
cat <<'TXT'
- 训练日志：trainable params 应小于总参数的 1%；loss 在前 50 步明显下降。
- 合并验证：greedy token 序列必须完全一致；不一致通常是 LoRA 打到了 tied 权重上。
- 合并后模型：vLLM 启动时间略长（要加载完整权重），但推理零 adapter 开销。
- 热插拔模型：同一份基座 + 多个 adapter，切换 model 名即可；adapter 越多，Punica 批处理越省显存。
- 如果 merge 后质量下降：检查 merge 用的是 bf16 还是 fp32（merge_and_unload 建议 fp32 基座）。
TXT

log "6) 结果目录：$WORK_DIR"
log "本次预计费用 ≈ \$$("$PY" -c "print(round($HOURS*$HOURLY_USD,2))")（$HOURS h × \$$HOURLY_USD/h）"
cat <<'TXT'

================= 关机 checklist =================
[ ] 1. vLLM 进程已停止：ps -ef | grep vllm，再 nvidia-smi 确认显存归零
[ ] 2. adapter / merged 模型 / 训练日志已下载：
       scp -r <实例>:$HOME/lora_run_* ./
[ ] 3. 若用云盘：确认 adapter 已同步（adapter 只有几十 MB，值得存）
[ ] 4. AutoDL：控制台「关机」；RunPod：Stop Pod；Lambda：Terminate；Vast：Destroy
[ ] 5. 核对账单页实例数为 0，数据盘按需保留
[ ] 6. 记下实际训练步数/耗时/费用，回填到第 14 章的成本表
==================================================
TXT
