# LLaVA-v1.5-7B 指令微调 (Stage 2) — 完整操作指南

> **适用硬件**: 8x RTX 4090 48GB  
> **训练方式**: 全量微调 (full fine-tuning) + DeepSpeed ZeRO-3  
> **预计耗时**: ~12-15 小时

---

## 一、架构回顾

LLaVA 训练分为两个阶段：

```
Stage 1 (Pretrain / Feature Alignment)      Stage 2 (Visual Instruction Tuning)
──────────────────────────────────────      ──────────────────────────────────────
冻结 LLM + 冻结 Vision Encoder               全参数微调 LLM + mm_projector
仅训练 mm_projector (MLP connector)          Vision Encoder 仍然冻结
数据: 558K 图文对 (BLIP captions)           数据: 665K 多模态指令数据
DeepSpeed ZeRO-2                             DeepSpeed ZeRO-3
→ 产出 mm_projector.bin                     → 产出最终对话模型
```

**本次只跑 Stage 2**，Stage 1 的产物 (mm_projector.bin) 直接从 HuggingFace 下载官方预训练版本。

Stage 2 的 Loss 只计算在 Assistant 回复部分（human 指令部分被 mask 为 IGNORE_INDEX），这是标准的 instruction tuning 做法。

---

## 二、文件说明

| 文件 | 用途 |
|------|------|
| `scripts/v1_5/download_projector_7b.sh` | 下载 Stage 1 预训练的 mm_projector |
| `scripts/v1_5/finetune_7b.sh` | Stage 2 指令微调训练脚本 |
| `scripts/zero3.json` | DeepSpeed ZeRO-3 配置文件 |

---

## 三、服务器端操作步骤

### 第一步：同步代码

```bash
# 在服务器上
cd ~/data/LLaVA
git pull origin main
```

### 第二步：下载预训练 Projector

Stage 2 依赖 Stage 1 训练的 mm_projector。运行以下脚本自动下载：

```bash
cd ~/data/LLaVA
bash scripts/v1_5/download_projector_7b.sh
```

执行成功后，会生成文件：
```
./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin
```

### 第三步：验证数据目录结构

确保数据目录与脚本中的路径对齐：

```bash
cd ~/data/LLaVA

echo "=== 顶层数据目录 ==="
ls playground/data/

echo ""
echo "=== COCO train2017 ==="
ls playground/data/coco/train2017/ | head -3

echo ""
echo "=== GQA images ==="
ls playground/data/gqa/images/ | head -3

echo ""
echo "=== OCR-VQA images ==="
ls playground/data/ocr_vqa/images/ | head -3

echo ""
echo "=== TextVQA train_images ==="
ls playground/data/textvqa/train_images/ | head -3

echo ""
echo "=== VisualGenome VG_100K ==="
ls playground/data/vg/VG_100K/ | head -3

echo ""
echo "=== VisualGenome VG_100K_2 ==="
ls playground/data/vg/VG_100K_2/ | head -3

echo ""
echo "=== 标注文件 ==="
ls -lh playground/data/llava_v1_5_mix665k.json
```

期望的目录结构：

```
playground/data/
├── coco/
│   └── train2017/
├── gqa/
│   └── images/
├── ocr_vqa/
│   └── images/
├── textvqa/
│   └── train_images/
├── vg/
│   ├── VG_100K/
│   └── VG_100K_2/
└── llava_v1_5_mix665k.json
```

### 第四步：确认环境依赖

```bash
conda activate llava   # 激活虚拟环境

# 检查关键包
python -c "
import torch; print(f'torch: {torch.__version__}')
import transformers; print(f'transformers: {transformers.__version__}')
import deepspeed; print(f'deepspeed: {deepspeed.__version__}')
import flash_attn; print(f'flash_attn: {flash_attn.__version__}')
"

# 期望输出:
# torch: 2.1.2
# transformers: 4.37.2
# deepspeed: 0.12.6
# flash_attn: 2.x.x
```

### 第五步：启动训练

```bash
cd ~/data/LLaVA
bash scripts/v1_5/finetune_7b.sh
```

---

## 四、训练参数详解

| 参数 | 值 | 说明 |
|------|----|------|
| `--model_name_or_path` | `lmsys/vicuna-7b-v1.5` | 7B Chat LLM（自动从 HuggingFace 下载） |
| `--version` | `v1` | Vicuna v1 对话模板 |
| `--data_path` | `./playground/data/llava_v1_5_mix665k.json` | 665K 混合指令数据标注 |
| `--image_folder` | `./playground/data` | 图片根目录（与 JSON 中相对路径拼接） |
| `--vision_tower` | `openai/clip-vit-large-patch14-336` | CLIP ViT-L/14 336px |
| `--pretrain_mm_mlp_adapter` | `./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin` | Stage 1 产出 |
| `--mm_projector_type` | `mlp2x_gelu` | 两层 MLP 视觉-语言连接器 |
| `--mm_vision_select_layer` | `-2` | 取 CLIP 倒数第二层特征 |
| `--image_aspect_ratio` | `pad` | 非方形图片用 padding 而非 crop |
| `--group_by_modality_length` | `True` | 训练采样时按模态分组，加速 ~25% |
| `--bf16` | `True` | BFloat16 混合精度 |
| `--learning_rate` | `2e-5` | 学习率（官方推荐值） |
| `--per_device_train_batch_size` | `16` | 每 GPU batch size |
| `--gradient_accumulation_steps` | `1` | 梯度累积步数 |
| `--num_train_epochs` | `1` | 训练 1 个 epoch |
| `--model_max_length` | `2048` | 序列最大长度 |
| `--gradient_checkpointing` | `True` | 梯度检查点，节省显存 |
| `--deepspeed` | `./scripts/zero3.json` | ZeRO Stage 3（参数/梯度/优化器状态分片） |
| `--report_to` | `wandb` | 日志上报（可选，如不用 wandb 可去掉） |

**全局 Batch Size** = 16 × 1 × 8 = **128**（与官方一致）

---

## 五、DeepSpeed ZeRO-3 配置

文件：`scripts/zero3.json`

```json
{
    "bf16": { "enabled": "auto" },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

ZeRO-3 将模型参数、梯度和优化器状态分片到 8 张 GPU 上，每张卡的实际显存占用：

| 组件 | 每卡占用 (估算) |
|------|----------------|
| 模型参数 (bf16) | ~1.75 GB |
| 梯度 (bf16) | ~1.75 GB |
| 优化器状态 (Adam fp32×2) | ~3.5 GB |
| 激活值 (batch=16, seq=2048) | ~8-12 GB |
| **合计** | **~15-19 GB** |

48GB 显存绰绰有余。

---

## 六、训练过程监控

### WandB (如果配置了)

训练脚本中包含了 `--report_to wandb`，自动上报 loss、学习率等指标。

### 命令行日志

DeepSpeed 会输出每个 step 的 loss：

```
{'loss': 1.2345, 'learning_rate': 1.8e-5, 'epoch': 0.05}
```

### GPU 利用率

```bash
# 另开一个终端窗口
watch -n 1 nvidia-smi
# 正常情况下 8 张卡 GPU-Util 接近 100%
```

---

## 七、异常处理

### 如果 OOM (显存溢出)

将 `scripts/v1_5/finetune_7b.sh` 中的两个参数改为：

```bash
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 2 \
```

全局 batch size 保持 = 8 × 2 × 8 = 128，不影响训练效果。

### 如果某个 GPU 挂了

DeepSpeed + NCCL 会自动检测并报错。排查方向：
1. 检查 PCIe 拓扑和 NVLink 连接
2. 降低 `per_device_train_batch_size`
3. 使用 `zero3_offload.json` 替代 `zero3.json`（将部分参数卸载到 CPU 内存，速度变慢但更稳定）

### 断点续训

训练脚本中已包含断点续训逻辑（`train.py` 第 966-968 行）：

```python
if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
```

中断后直接重新运行 `bash scripts/v1_5/finetune_7b.sh` 即可自动恢复。

### 手动指定 GPU

```bash
# 只使用前 4 张卡
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/finetune_7b.sh
```

此时需要调整梯度累积以保持全局 batch size：
```bash
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 2 \
# 全局 batch = 16 × 2 × 4 = 128
```

---

## 八、训练完成后

### 产物

```
./checkpoints/llava-v1.5-7b/
├── config.json                # 模型配置
├── model-0000x-of-0000x.safetensors  # 模型权重（分片存储）
├── trainer_state.json         # 训练器状态（可恢复）
└── training_args.bin          # 训练参数记录
```

### 快速推理测试

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "./checkpoints/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": "请描述这张图片里的内容",
    "conv_mode": None,
    "image_file": "test.jpg",
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)
```

---

## 九、参考资料

- [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)
- [LLaVA-v1.5 Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)
- [LLaVA Data Documentation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)
- [LLaVA-v1.5 Technical Report](https://arxiv.org/abs/2310.03744)
