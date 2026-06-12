#!/bin/bash
# MM-Vet evaluation for LLaVA-v1.5-7B-LoRA (single-GPU, needs external jupyter notebook for scoring)

# --- 数据完整性检查 ---
if [ ! -d "./playground/data/eval/mm-vet/images" ] || [ "$(ls -A ./playground/data/eval/mm-vet/images 2>/dev/null | wc -l)" -lt 10 ]; then
    echo "[ERROR] MM-Vet images missing. Run: bash scripts/v1_5/download_eval_data.sh"
    exit 1
fi

MODEL_PATH="./checkpoints/llava-v1.5-7b-lora"
MODEL_BASE="lmsys/vicuna-7b-v1.5"
CKPT="llava-v1.5-7b-lora"

python -m llava.eval.model_vqa \
    --model-path "$MODEL_PATH" \
    --model-base "$MODEL_BASE" \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$CKPT.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$CKPT.json

echo "Result file: ./playground/data/eval/mm-vet/results/$CKPT.json"
echo "Use MM-Vet official jupyter notebook to score this file."
