# LLaVA-v1.5-7B LoRA 指令微调 (Stage 2) — 完整操作指南

> **适用硬件**: 8x RTX 4090 48GB
> **训练方式**: LoRA 低秩适配 + DeepSpeed ZeRO-2
> **预计耗时**: ~8-10 小时
> **可训参数**: ~1% (LoRA 适配器 + mm_projector)

---

## 一、为什么选 LoRA

在 8x RTX 4090 (无 NVLink) 上尝试全量微调时遇到的已知问题：

| 问题 | 原因 |
|------|------|
| NCCL reduce_scatter 报错 | 8 卡 PCIe P2P 拓扑下 ZeRO-3 通信不稳定 |
| loss 持续为 0.0 | Gradient checkpointing + DeepSpeed ZeRO-2 下参数被脱钩 |
| 关掉 checkpointing 后 OOM | 全量参数 + 优化器状态超过 48GB |

**LoRA 方案解决这些问题的原理**：
- 冻结 LLM 主体，只训练低秩适配器（LoRA weights + mm_projector）
- 可训参数极小 → 显存占用大幅降低 → `gradient_checkpointing` 可以安全开启
- 梯度通信量极小 → ZeRO-2 下无 NCCL 压力

---

## 二、使用脚本

| 文件 | 用途 |
|------|------|
| `scripts/v1_5/download_projector_7b.sh` | 下载 Stage 1 预训练的 mm_projector |
| `scripts/v1_5/finetune_7b_lora.sh` | LoRA 指令微调训练脚本 |

---

## 三、服务器端操作

### 第一步：同步代码

```bash
cd ~/data/LLaVA
git pull origin main
```

### 第二步：下载预训练 Projector

```bash
bash scripts/v1_5/download_projector_7b.sh
```

### 第三步：启动训练

```bash
cd ~/data/LLaVA
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
bash scripts/v1_5/finetune_7b_lora.sh
```

---

## 四、LoRA 训练参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--lora_enable` | True | 启用 LoRA |
| `--lora_r` | 128 | LoRA 秩 |
| `--lora_alpha` | 256 | LoRA 缩放因子 |
| `--mm_projector_lr` | 2e-5 | mm_projector 学习率（低于 LoRA） |
| `--learning_rate` | 2e-4 | LoRA 适配器学习率 |
| `--deepspeed` | zero2.json | ZeRO-2（梯度+优化器分片） |
| `--per_device_train_batch_size` | 16 | 每卡 batch size |
| `--gradient_accumulation_steps` | 1 | 无梯度累积 |
| `--gradient_checkpointing` | True | 节省显存 |
| `--bf16` | True | BFloat16 混合精度 |

**全局 Batch Size** = 16 × 1 × 8 = **128**（与官方一致）

---

## 五、显存预估（每卡）

| 组件 | 占用 |
|------|------|
| 基座模型 (Vicuna-7B, bf16, 冻结) | ~14 GB |
| LoRA 适配器 (r=128) | ~0.2 GB |
| CLIP ViT-L/14 (冻结) | ~1.7 GB |
| 优化器状态 (LoRA + projector) | ~2 GB |
| 激活值 (batch=16, flash_attn_2) | ~8-12 GB |
| **合计** | **~26-30 GB** |

48GB 卡绰绰有余。

---

## 六、训练完成后

### 产物

```
./checkpoints/llava-v1.5-7b-lora/
├── adapter_config.json     # LoRA 配置
├── adapter_model.bin       # LoRA 权重
├── non_lora_trainables.bin # mm_projector 等非 LoRA 权重
└── ...
```

### 合并 LoRA 权重（可选，用于推理）

```bash
python scripts/merge_lora_weights.py \
    --model-path ./checkpoints/llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --save-model-path ./checkpoints/llava-v1.5-7b-merged
```

### 不合并直接推理

```python
from llava.model.builder import load_pretrained_model

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="./checkpoints/llava-v1.5-7b-lora",
    model_base="lmsys/vicuna-7b-v1.5",   # 基座 LLM
    model_name="llava-v1.5-7b-lora",
)
```

---

## 七、常见问题

### 如果 OOM

```bash
# 将 finetune_7b_lora.sh 中改为：
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 2 \
# 全局 BS = 8 × 2 × 8 = 128
```

### 如果 NCCL 报错

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO   # 如需诊断
```

### 如果要跑 13B LoRA

将脚本中 `vicuna-7b-v1.5` 换为 `vicuna-13b-v1.5`，projector 对应换成 13B 版本即可。

---

## 八、LoRA vs 全量微调性能

根据官方报告，LLaVA-1.5 LoRA 与全量微调性能几乎持平：

| 方案 | VQAv2 | GQA | MME | MM-Bench |
|------|-------|-----|-----|----------|
| 7B 全量 | 78.5 | 62.0 | 1510.7 | 64.3 |
| 7B LoRA | 79.1 | 63.0 | 1476.9 | 66.1 |
| 13B 全量 | 80.0 | 63.3 | 1531.3 | 67.7 |
| 13B LoRA | 80.0 | 63.3 | 1541.7 | 68.5 |
