#!/bin/bash
# MMBench evaluation for LLaVA-v1.5-7B-LoRA (single-GPU, needs online submission)

# --- 数据完整性检查 ---
SPLIT="mmbench_dev_20230712"
if [ ! -f "./playground/data/eval/mmbench/$SPLIT.tsv" ]; then
    echo "[ERROR] mmbench_dev_20230712.tsv missing. Run: bash scripts/v1_5/download_eval_data.sh"
    exit 1
fi

MODEL_PATH="./checkpoints/llava-v1.5-7b-lora"
MODEL_BASE="lmsys/vicuna-7b-v1.5"
CKPT="llava-v1.5-7b-lora"
SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path "$MODEL_PATH" \
    --model-base "$MODEL_BASE" \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT

echo "Upload to: https://opencompass.org.cn/leaderboard-multimodal"
