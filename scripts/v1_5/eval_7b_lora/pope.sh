#!/bin/bash
# POPE evaluation for LLaVA-v1.5-7B-LoRA (single-GPU, local evaluation)

# --- 数据完整性检查 ---
if [ ! -f "./playground/data/eval/pope/llava_pope_test.jsonl" ]; then
    echo "[ERROR] pope annotations missing. Run: bash scripts/v1_5/download_eval_data.sh"
    exit 1
fi
if [ ! -d "./playground/data/eval/pope/val2014" ] || [ "$(ls -A ./playground/data/eval/pope/val2014 2>/dev/null | wc -l)" -lt 100 ]; then
    echo "[ERROR] pope/val2014 images missing. Run: bash scripts/v1_5/download_eval_data.sh"
    exit 1
fi
MODEL_PATH="./checkpoints/llava-v1.5-7b-lora-8-20260624-merged"
# (merged model, no base needed)
CKPT="llava-v1.5-7b-lora"

python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
      \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$CKPT.jsonl
