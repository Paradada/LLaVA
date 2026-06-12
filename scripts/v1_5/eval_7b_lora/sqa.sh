#!/bin/bash
# ScienceQA evaluation for LLaVA-v1.5-7B-LoRA (single-GPU, local evaluation)

# --- 数据完整性检查 ---
if [ ! -d "./playground/data/eval/scienceqa/images/test" ] || [ "$(ls -A ./playground/data/eval/scienceqa/images/test 2>/dev/null | wc -l)" -lt 100 ]; then
    echo "[ERROR] ScienceQA images missing. See download instructions in scripts/v1_5/download_eval_data.sh"
    exit 1
fi

MODEL_PATH="./checkpoints/llava-v1.5-7b-lora"
MODEL_BASE="lmsys/vicuna-7b-v1.5"
CKPT="llava-v1.5-7b-lora"

python -m llava.eval.model_vqa_science \
    --model-path "$MODEL_PATH" \
    --model-base "$MODEL_BASE" \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT}_result.json
