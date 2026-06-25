#!/bin/bash
# MME evaluation for LLaVA-v1.5-7B-LoRA (single-GPU, local evaluation)

# --- 数据完整性检查 ---
if [ ! -f "./playground/data/eval/MME/llava_mme.jsonl" ]; then
    echo "[ERROR] MME annotations missing. Run: bash scripts/v1_5/download_eval_data.sh"
    exit 1
fi
if [ ! -d "./playground/data/eval/MME/MME_Benchmark_release_version" ] || [ "$(ls -A ./playground/data/eval/MME/MME_Benchmark_release_version 2>/dev/null | wc -l)" -lt 10 ]; then
    echo "[ERROR] MME images missing. See download instructions in scripts/v1_5/download_eval_data.sh"
    exit 1
fi

MODEL_PATH="./checkpoints/llava-v1.5-7b-lora-8-20260614-merged"
# (merged model, no base needed)
CKPT="llava-v1.5-7b-lora"

python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
      \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py --results_dir answers/$CKPT
