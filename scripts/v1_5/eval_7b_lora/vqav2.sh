#!/bin/bash
# VQAv2 evaluation for LLaVA-v1.5-7B-LoRA (multi-GPU)

# --- 数据完整性检查 ---
VQAv2_IMG="./playground/data/eval/vqav2/test2015"
if [ ! -d "$VQAv2_IMG" ] || [ "$(ls -A $VQAv2_IMG 2>/dev/null | wc -l)" -lt 100 ]; then
    echo "[ERROR] VQAv2 test2015 images missing. Run: bash scripts/v1_5/download_eval_data.sh"
    exit 1
fi

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b-lora"
MODEL_PATH="./checkpoints/llava-v1.5-7b-merged"
# (merged model, no base needed)
SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path "$MODEL_PATH" \
          \
        --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl
> "$output_file"
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT
echo "Upload file: ./playground/data/eval/vqav2/answers_upload/$SPLIT/${CKPT}_vqav2_mscoco_2015_test-dev2015.json"
echo "Submit to: https://eval.ai/web/challenges/challenge-page/830/my-submission"
