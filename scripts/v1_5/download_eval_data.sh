#!/bin/bash
# ============================================================================
# LLaVA-v1.5 评估数据统一下载脚本
# 自动检测已存在的数据，只下载缺失部分
# 所有数据下载到 ./playground/data/eval/
# ============================================================================

set -e

EVAL_DIR="./playground/data/eval"
mkdir -p "$EVAL_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
check() { echo -e "  ${GREEN}[✓]${NC} $1"; }

echo ""
echo "============================================"
echo "  LLaVA-v1.5 Evaluation Data Downloader"
echo "============================================"
echo ""

# ======================================================================
# 1. VQAv2 — test2015 images (~6.2GB)
# ======================================================================
info "1/11 VQAv2 — test2015 images"
VQAv2_DIR="$EVAL_DIR/vqav2"
mkdir -p "$VQAv2_DIR/test2015"

if [ "$(ls -A $VQAv2_DIR/test2015 2>/dev/null | wc -l)" -gt 100 ]; then
    check "vqav2/test2015 already exists"
else
    warn "Downloading COCO test2015 (~6.2GB)..."
    wget -c http://images.cocodataset.org/zips/test2015.zip -O /tmp/test2015.zip
    unzip -qo /tmp/test2015.zip -d "$VQAv2_DIR/"
    rm /tmp/test2015.zip
    check "vqav2/test2015 done"
fi

# ======================================================================
# 2. GQA — images + eval scripts (~20GB)
# ======================================================================
info "2/11 GQA — images"
GQA_DIR="$EVAL_DIR/gqa/data"
mkdir -p "$GQA_DIR"

if [ "$(ls -A $GQA_DIR/images 2>/dev/null | wc -l)" -gt 100 ]; then
    check "gqa/data/images already exists"
else
    warn "Downloading GQA images (~20GB, large)..."
    wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -O /tmp/gqa_images.zip
    unzip -qo /tmp/gqa_images.zip -d "$GQA_DIR/"
    rm /tmp/gqa_images.zip
    check "gqa/data/images done"
fi

if [ -f "$GQA_DIR/eval/eval.py" ]; then
    check "gqa evaluation scripts exist"
else
    warn "GQA evaluation scripts need manual download:"
    echo "  https://cs.stanford.edu/people/dorarad/gqa/evaluate.html"
    echo "  Extract eval/ to: $GQA_DIR/"
    echo "  Note: may need to patch eval.py: https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187"
fi

# ======================================================================
# 3. VisWiz — annotations + test images (~12GB)
# ======================================================================
info "3/11 VisWiz"
VIZWIZ_DIR="$EVAL_DIR/vizwiz"
mkdir -p "$VIZWIZ_DIR"

if [ -f "$VIZWIZ_DIR/llava_test.jsonl" ]; then
    check "vizwiz annotation exists (from eval.zip)"
else
    warn "llava_test.jsonl not found, download eval.zip first"
fi

if [ "$(ls -A $VIZWIZ_DIR/test 2>/dev/null | wc -l)" -gt 100 ]; then
    check "vizwiz/test already exists"
else
    warn "Downloading VisWiz test images (~12GB)..."
    wget -c https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip -O /tmp/vizwiz_test.zip
    unzip -qo /tmp/vizwiz_test.zip -d "$VIZWIZ_DIR/"
    rm /tmp/vizwiz_test.zip
    check "vizwiz/test done"
fi

# ======================================================================
# 4. ScienceQA — images + json
# ======================================================================
info "4/11 ScienceQA"
SQA_DIR="$EVAL_DIR/scienceqa"
mkdir -p "$SQA_DIR"

if [ -f "$SQA_DIR/llava_test_CQM-A.json" ]; then
    check "scienceqa annotation exists (from eval.zip)"
fi

if [ "$(ls -A $SQA_DIR/images/test 2>/dev/null | wc -l)" -gt 100 ]; then
    check "scienceqa images already exist"
else
    warn "ScienceQA images need manual download from official repo:"
    echo "  https://github.com/lupantech/ScienceQA/tree/main/data/scienceqa"
    echo "  Download: images/, pid_splits.json, problems.json"
    echo "  Place in: $SQA_DIR/"
fi

# ======================================================================
# 5. TextVQA — val json + images
# ======================================================================
info "5/11 TextVQA"
TEXT_VQA_DIR="$EVAL_DIR/textvqa"
mkdir -p "$TEXT_VQA_DIR"

if [ -f "$TEXT_VQA_DIR/llava_textvqa_val_v051_ocr.jsonl" ]; then
    check "textvqa annotation exists (from eval.zip)"
fi

if [ ! -f "$TEXT_VQA_DIR/TextVQA_0.5.1_val.json" ]; then
    warn "Downloading TextVQA val json..."
    wget -c https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json -O "$TEXT_VQA_DIR/TextVQA_0.5.1_val.json"
    check "TextVQA json done"
fi

if [ "$(ls -A $TEXT_VQA_DIR/train_images 2>/dev/null | wc -l)" -gt 100 ]; then
    check "textvqa/train_images already exists"
else
    warn "Downloading TextVQA train_val_images (~4GB)..."
    wget -c https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip -O /tmp/textvqa_images.zip
    unzip -qo /tmp/textvqa_images.zip -d "$TEXT_VQA_DIR/"
    rm /tmp/textvqa_images.zip
    check "textvqa/train_images done"
fi

# ======================================================================
# 6. POPE — coco annotations + val2014 images (~6.2GB)
# ======================================================================
info "6/11 POPE"
POPE_DIR="$EVAL_DIR/pope"
mkdir -p "$POPE_DIR"

if [ -f "$POPE_DIR/llava_pope_test.jsonl" ]; then
    check "pope annotation exists (from eval.zip)"
fi

if [ "$(ls -A $POPE_DIR/coco 2>/dev/null | wc -l)" -gt 1 ]; then
    check "pope/coco already exists"
else
    warn "POPE coco annotations need manual download:"
    echo "  https://github.com/AoiDragon/POPE/tree/main/output/coco"
    echo "  Download all 3 files to: $POPE_DIR/coco/"
fi

if [ "$(ls -A $POPE_DIR/val2014 2>/dev/null | wc -l)" -gt 100 ]; then
    check "pope/val2014 already exists"
else
    warn "Downloading COCO val2014 (~6.2GB)..."
    wget -c http://images.cocodataset.org/zips/val2014.zip -O /tmp/val2014.zip
    unzip -qo /tmp/val2014.zip -d "$POPE_DIR/"
    rm /tmp/val2014.zip
    check "pope/val2014 done"
fi

# ======================================================================
# 7. MME — images
# ======================================================================
info "7/11 MME"
MME_DIR="$EVAL_DIR/MME"
mkdir -p "$MME_DIR/MME_Benchmark_release_version"

if [ -f "$MME_DIR/llava_mme.jsonl" ]; then
    check "mme annotation exists (from eval.zip)"
fi

if [ "$(ls -A $MME_DIR/MME_Benchmark_release_version 2>/dev/null | wc -l)" -gt 10 ]; then
    check "MME images already exist"
else
    warn "MME images need manual download:"
    echo "  https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation"
    echo "  Place images in: $MME_DIR/MME_Benchmark_release_version/"
fi

# ======================================================================
# 8. MMBench — dev tsv files
# ======================================================================
info "8/11 MMBench"
MMB_DIR="$EVAL_DIR/mmbench"
mkdir -p "$MMB_DIR"

if [ -f "$MMB_DIR/mmbench_dev_20230712.tsv" ]; then
    check "mmbench_dev_20230712.tsv already exists"
else
    warn "Downloading mmbench_dev_20230712.tsv..."
    wget -c https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv -O "$MMB_DIR/mmbench_dev_20230712.tsv"
    check "mmbench_dev_20230712.tsv done"
fi

if [ -f "$MMB_DIR/mmbench_dev_cn_20231003.tsv" ]; then
    check "mmbench_dev_cn_20231003.tsv already exists"
else
    warn "Downloading mmbench_dev_cn_20231003.tsv..."
    wget -c https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv -O "$MMB_DIR/mmbench_dev_cn_20231003.tsv"
    check "mmbench_dev_cn_20231003.tsv done"
fi

# ======================================================================
# 9. SEED-Bench — images + videos
# ======================================================================
info "9/11 SEED-Bench"
SEED_DIR="$EVAL_DIR/seed_bench"
mkdir -p "$SEED_DIR"

if [ -f "$SEED_DIR/llava-seed-bench.jsonl" ]; then
    check "seed-bench annotation exists (from eval.zip)"
fi

if [ "$(ls -A $SEED_DIR/SEED-Bench-image 2>/dev/null | wc -l)" -gt 100 ]; then
    check "SEED-Bench images already exist"
else
    warn "SEED-Bench images need manual download:"
    echo "  https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md"
    echo "  Place images in: $SEED_DIR/SEED-Bench-image/"
    echo "  Place video frames in: $SEED_DIR/SEED-Bench-video-image/"
fi

# ======================================================================
# 10. LLaVA-Bench-in-the-Wild — images
# ======================================================================
info "10/11 LLaVA-Bench-in-the-Wild"
LLaVA_BENCH_DIR="$EVAL_DIR/llava-bench-in-the-wild"
mkdir -p "$LLaVA_BENCH_DIR"

if [ -f "$LLaVA_BENCH_DIR/questions.jsonl" ]; then
    check "llava-bench annotations exist (from eval.zip)"
fi

if [ "$(ls -A $LLaVA_BENCH_DIR/images 2>/dev/null | wc -l)" -gt 10 ]; then
    check "llava-bench images already exist"
else
    warn "LLaVA-Bench images need manual download from HuggingFace:"
    echo "  pip install huggingface_hub"
    echo "  huggingface-cli download liuhaotian/llava-bench-in-the-wild --local-dir $LLaVA_BENCH_DIR --repo-type dataset"
    echo "  (If HF is blocked, use: export HF_ENDPOINT=https://hf-mirror.com)"
fi

# ======================================================================
# 11. MM-Vet — mm-vet.zip
# ======================================================================
info "11/11 MM-Vet"
MMVET_DIR="$EVAL_DIR/mm-vet"
mkdir -p "$MMVET_DIR"

if [ -f "$MMVET_DIR/llava-mm-vet.jsonl" ]; then
    check "mm-vet annotation exists (from eval.zip)"
fi

if [ "$(ls -A $MMVET_DIR/images 2>/dev/null | wc -l)" -gt 10 ]; then
    check "mm-vet/images already exist"
else
    warn "Downloading mm-vet.zip (~2GB)..."
    wget -c https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip -O /tmp/mm-vet.zip
    unzip -qo /tmp/mm-vet.zip -d "$MMVET_DIR/"
    rm /tmp/mm-vet.zip
    # mm-vet.zip extracts to ./mm-vet/images/
    if [ -d "$MMVET_DIR/mm-vet/images" ]; then
        mv "$MMVET_DIR/mm-vet/images" "$MMVET_DIR/images" 2>/dev/null || true
    fi
    check "mm-vet done"
fi

echo ""
echo "============================================"
echo "  Download Summary"
echo "============================================"
echo ""

# 统计各目录状态
summary() {
    local dir="$1" label="$2" expect="$3"
    if [ -d "$dir" ]; then
        local count=$(ls -A "$dir" 2>/dev/null | wc -l)
        if [ "$count" -ge "$expect" ]; then
            echo -e "  ${GREEN}[✓]${NC} $label ($count items)"
        else
            echo -e "  ${YELLOW}[!]${NC} $label — $count items, need more (expect ≥$expect)"
        fi
    else
        echo -e "  ${RED}[✗]${NC} $label — directory missing"
    fi
}

summary "$VQAv2_DIR/test2015"            "VQAv2 test2015"        100
summary "$GQA_DIR/images"                "GQA images"            100
summary "$VIZWIZ_DIR/test"               "VisWiz test images"    100
summary "$SQA_DIR/images/test"           "ScienceQA images"      100
summary "$TEXT_VQA_DIR/train_images"     "TextVQA images"        100
summary "$POPE_DIR/val2014"             "POPE val2014"         100
summary "$MME_DIR/MME_Benchmark_release_version" "MME images"   10
summary "$SEED_DIR/SEED-Bench-image"     "SEED-Bench images"     100
summary "$LLaVA_BENCH_DIR/images"        "LLaVA-Bench images"     10
summary "$MMVET_DIR/images"              "MM-Vet images"          10

echo ""
echo "Items marked [!] or [✗] still need attention."
echo "Re-run this script after manual downloads to verify."
