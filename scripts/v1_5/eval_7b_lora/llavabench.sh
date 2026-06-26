#!/bin/bash
# LLaVA-Bench-in-the-Wild evaluation (GPT-assisted, needs OPENAI_API_KEY)

# --- 数据完整性检查 ---
LB_DIR="./playground/data/eval/llava-bench-in-the-wild"
if [ ! -f "$LB_DIR/questions.jsonl" ]; then
    echo "[ERROR] LLaVA-Bench annotations missing. Run: bash scripts/v1_5/download_eval_data.sh"
    exit 1
fi
if [ ! -d "$LB_DIR/images" ] || [ "$(ls -A $LB_DIR/images 2>/dev/null | wc -l)" -lt 10 ]; then
    echo "[ERROR] LLaVA-Bench images missing. Run: bash scripts/v1_5/download_eval_data.sh"
    exit 1
fi


MODEL_PATH="./checkpoints/llava-v1.5-7b-lora-8-20260624-merged"
# (merged model, no base needed)
CKPT="llava-v1.5-7b-lora"

python -m llava.eval.model_vqa \
    --model-path "$MODEL_PATH" \
      \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

OPENAI_API_KEY="${OPENAI_API_KEY:-}" python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/$CKPT.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/$CKPT.jsonl

python llava/eval/summarize_gpt_review.py \
    -f playground/data/eval/llava-bench-in-the-wild/reviews/$CKPT.jsonl
