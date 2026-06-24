#!/bin/bash

# ============================================================================
# LLaVA-v1.5-7B LoRA DeepSpeed 测试脚本 — 2x RTX 4090 24GB
# 目的：在 2 卡上稳定跑通几百 step，验证数据/模型/训练链路完整，不 OOM 不 crash
# ============================================================================
#
# === Global Batch Size ===
# global_batch_size = per_device_train_batch_size × gradient_accumulation_steps × num_gpus
#                   = 4 × 4 × 2
#                   = 32
# （推荐范围 32~64；如需增大，优先上调 per_device_train_batch_size 到 6 或 8）
#
# === 显存优化思路 ===
# 1. Zero-2 + optimizer offload + param offload（见 zero2_offload.json）
#    → 优化器状态 + 模型参数卸载到 CPU，单卡显存节省约 40~50%
# 2. per_device_train_batch_size=4
#    → 单卡 micro batch 从原始 16 降到 4，大幅降低激活显存
# 3. gradient_accumulation_steps=4
#    → 通过梯度累积补偿小 batch，保持等效 global batch size=32
# 4. LoRA r=128, alpha=256
#    → 只训练 ~1% 参数（~100M），全参数微调 7B 需 >40GB 显存
# 5. bf16 + gradient_checkpointing + tf32
#    → 混合精度 + 激活重计算，以计算换显存
# 6. dataloader_num_workers=2
#    → 降低 CPU 内存占用，避免数据加载成为瓶颈
#
# === NCCL 通信稳定性 ===
# - 禁用 InfiniBand（消费级显卡无 IB 硬件）
# - 启用 P2P 直连（PCIe 通道，2 卡场景延迟更低）
# - DEBUG=WARN 便于排查通信异常
# ============================================================================

# --- NCCL 环境变量 ---
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN
# 如果有多网卡，取消下面注释并指定通信网卡：
# export NCCL_SOCKET_IFNAME=eth0

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-lora-2x4090-test \
    --num_train_epochs 1 \
    --max_steps 500 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none
