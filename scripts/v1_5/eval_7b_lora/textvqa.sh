#!/bin/bash
# TextVQA evaluation for LLaVA-v1.5-7B-LoRA (single-GPU, local evaluation)

# --- 数据完整性检查 ---
if [ ! -f "./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl" ]; then
    echo "[ERROR] textvqa annotations missing. Run: bash scripts/v1_5/download_eval_data.sh"
    exit 1
fi
if [ ! -f "./playground/data/eval/textvqa/TextVQA_0.5.1_val.json" ]; then
    echo "[ERROR] TextVQA_0.5.1_val.json missing. Run: bash scripts/v1_5/download_eval_data.sh"
    exit 1
fi
if [ ! -d "./playground/data/eval/textvqa/train_images" ] || [ "$(ls -A ./playground/data/eval/textvqa/train_images 2>/dev/null | wc -l)" -lt 100 ]; then
    echo "[ERROR] textvqa/train_images missing. Run: bash scripts/v1_5/download_eval_data.sh"
    exit 1
fi

MODEL_PATH="./checkpoints/llava-v1.5-7b-lora-official-merged"
# (merged model, no base needed)
CKPT="llava-v1.5-7b-lora"

python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
      \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl
