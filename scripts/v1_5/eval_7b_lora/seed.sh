#!/bin/bash
# SEED-Bench evaluation for LLaVA-v1.5-7B-LoRA (multi-GPU, local evaluation)

# --- 数据完整性检查 ---
SEED_IMG="./playground/data/eval/seed_bench/SEED-Bench-image"
if [ ! -d "$SEED_IMG" ] || [ "$(ls -A $SEED_IMG 2>/dev/null | wc -l)" -lt 100 ]; then
    echo "[ERROR] SEED-Bench images missing. See download instructions in scripts/v1_5/download_eval_data.sh"
    exit 1
fi

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b-lora"

MODEL_PATH="./checkpoints/llava-v1.5-7b-lora-8-20260624-merged"
# (merged model, no base needed)

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path "$MODEL_PATH" \
          \
        --question-file ./playground/data/eval/seed_bench/llava-seed-bench.jsonl \
        --image-folder ./playground/data/eval/seed_bench \
        --answers-file ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/seed_bench/answers/$CKPT/merge.jsonl
> "$output_file"
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file ./playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file ./playground/data/eval/seed_bench/answers_upload/$CKPT.jsonl

echo "Result file: ./playground/data/eval/seed_bench/answers_upload/$CKPT.jsonl"
