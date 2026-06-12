#!/bin/bash
# GQA evaluation for LLaVA-v1.5-7B-LoRA (multi-GPU, local evaluation)

# --- 数据完整性检查 ---
GQA_IMG="./playground/data/eval/gqa/data/images"
if [ ! -d "$GQA_IMG" ] || [ "$(ls -A $GQA_IMG 2>/dev/null | wc -l)" -lt 100 ]; then
    echo "[ERROR] GQA images missing. Run: bash scripts/v1_5/download_eval_data.sh"
    exit 1
fi

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b-lora"
MODEL_PATH="./checkpoints/llava-v1.5-7b-lora"
MODEL_BASE="lmsys/vicuna-7b-v1.5"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path "$MODEL_PATH" \
        --model-base "$MODEL_BASE" \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl
> "$output_file"
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
