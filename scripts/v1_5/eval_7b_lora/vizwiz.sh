#!/bin/bash
# VisWiz evaluation for LLaVA-v1.5-7B-LoRA (single-GPU, needs online submission)

# --- 数据完整性检查 ---
if [ ! -d "./playground/data/eval/vizwiz/test" ] || [ "$(ls -A ./playground/data/eval/vizwiz/test 2>/dev/null | wc -l)" -lt 100 ]; then
    echo "[ERROR] VisWiz test images missing. Run: bash scripts/v1_5/download_eval_data.sh"
    exit 1
fi

MODEL_PATH="./checkpoints/llava-v1.5-7b-lora-8-20260614-merged"
# (merged model, no base needed)
CKPT="llava-v1.5-7b-lora"

python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
      \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/$CKPT.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/$CKPT.json

echo "Upload file: ./playground/data/eval/vizwiz/answers_upload/$CKPT.json"
echo "Submit to: https://eval.ai/web/challenges/challenge-page/2185/my-submission"
