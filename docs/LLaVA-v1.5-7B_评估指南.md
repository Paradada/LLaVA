# LLaVA-v1.5-7B LoRA 评估指南

> **模型**: LLaVA-v1.5-7B LoRA（自训练）
> **模型路径**: `./checkpoints/llava-v1.5-7b-lora`
> **基座 LLM**: `lmsys/vicuna-7b-v1.5`

---

## 一、评估概览

LLaVA-1.5 在 **12 个 benchmark** 上评估，分为三类：

### 本地直接出分（推荐先跑）

| Benchmark | 评估方式 | GPU | 指标 |
|-----------|---------|-----|------|
| POPE | 本地脚本 | 单卡 | Accuracy / F1 |
| TextVQA | 本地脚本 | 单卡 | Accuracy |
| MME | 本地脚本 | 单卡 | Perception / Cognition |
| ScienceQA | 本地脚本 | 单卡 | Accuracy |
| GQA | 本地脚本 | 多卡 | Accuracy |
| SEED-Bench | 本地脚本 | 多卡 | Accuracy |

### 需要在线提交

| Benchmark | 评估方式 | GPU | 提交地址 |
|-----------|---------|-----|---------|
| VQAv2 | 在线服务器 | 多卡 | eval.ai |
| VisWiz | 在线服务器 | 单卡 | eval.ai |
| MMBench | 在线服务器 | 单卡 | opencompass |
| MMBench-CN | 在线服务器 | 单卡 | opencompass |

### 需要 GPT-4 / 外部工具

| Benchmark | 评估方式 | 依赖 |
|-----------|---------|------|
| LLaVA-Bench-Wild | GPT-4 打分 | OPENAI_API_KEY |
| MM-Vet | Jupyter 打分 | 官方 Notebook |

---

## 二、前置准备（两步完成）

### 2.1 第一步：下载 eval.zip 并解压

```bash
cd ~/data/LLaVA

# 下载 eval.zip（约 1.5GB，含所有自定义标注、脚本、目录结构）
# Google Drive: https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view

# 解压到 ./playground/data/eval/
```

eval.zip 提供的基础目录结构：

```
playground/data/eval/
├── vqav2/          ├── gqa/            ├── vizwiz/
├── scienceqa/      ├── textvqa/        ├── pope/
├── MME/            ├── mmbench/        ├── seed_bench/
├── llava-bench-in-the-wild/            └── mm-vet/
```

### 2.2 第二步：统一下载所有评估数据

```bash
cd ~/data/LLaVA
bash scripts/v1_5/download_eval_data.sh
```

这个脚本会：

- **自动检测**已有数据，跳过已存在的，只下载缺失部分
- 自动下载的内容：COCO test2015、GQA 图片、TextVQA 图片/标注、VisWiz 图片、POPE val2014、MMBench tsv、MM-Vet 图片
- 需要手动下载的会**明确提示**下载地址：ScienceQA 图片、MME 图片、SEED-Bench 图片、POPE coco 标注、LLaVA-Bench 图片、GQA 评估脚本
- 跑完后打印**每个 dataset 的状态表**（✓ 就绪 / ! 缺失）

### 2.3 确认模型和环境

```bash
# 确认 checkpoint
ls ./checkpoints/llava-v1.5-7b-lora/
# 期望: adapter_config.json  adapter_model.bin  non_lora_trainables.bin ...

# 如需 HF 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 三、各 Benchmark 运行命令

> 每个评估脚本都自带数据完整性检查，缺数据会直接报错提示。

### 3.1 POPE ⭐ 本地出分（推荐第一个跑）

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval_7b_lora/pope.sh
```

### 3.2 TextVQA ⭐ 本地出分

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval_7b_lora/textvqa.sh
```

### 3.3 MME ⭐ 本地出分

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval_7b_lora/mme.sh
```

### 3.4 ScienceQA ⭐ 本地出分

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval_7b_lora/sqa.sh
```

### 3.5 GQA ⭐ 本地出分（多卡）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval_7b_lora/gqa.sh
```

### 3.6 SEED-Bench ⭐ 本地出分（多卡）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval_7b_lora/seed.sh
```

### 3.7 VQAv2（多卡 → 在线提交）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval_7b_lora/vqav2.sh
# 提交到: https://eval.ai/web/challenges/challenge-page/830/my-submission
```

### 3.8 VisWiz（单卡 → 在线提交）

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval_7b_lora/vizwiz.sh
# 提交到: https://eval.ai/web/challenges/challenge-page/2185/my-submission
```

### 3.9 MMBench / MMBench-CN（单卡 → 在线提交）

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval_7b_lora/mmbench.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval_7b_lora/mmbench_cn.sh
# 提交到: https://opencompass.org.cn/leaderboard-multimodal
```

### 3.10 LLaVA-Bench-in-the-Wild（GPT-4 打分）

```bash
OPENAI_API_KEY="sk-****" bash scripts/v1_5/eval_7b_lora/llavabench.sh
```

### 3.11 MM-Vet（Jupyter 打分）

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval_7b_lora/mmvet.sh
# 然后用 MM-Vet 官方 jupyter notebook 打开打分:
# ./playground/data/eval/mm-vet/results/llava-v1.5-7b-lora.json
```

---

## 四、推荐的跑分顺序

```
1. POPE      (单卡, ~5min,  本地出分)  ─┐
2. TextVQA   (单卡, ~10min, 本地出分)   ├─ 快速验证（~1h）
3. MME       (单卡, ~15min, 本地出分)   │
4. ScienceQA (单卡, ~20min, 本地出分)  ─┘
5. GQA       (多卡, ~30min, 本地出分)  ─┐
6. SEED-Bench(多卡, ~30min, 本地出分)   │
7. VQAv2     (多卡, ~20min, 在线提交)   ├─ 完整评估（~3h）
8. VisWiz    (单卡, ~15min, 在线提交)   │
9. MMBench   (单卡, ~10min, 在线提交)   │
10. MM-Vet   (单卡, ~20min, Jupyter)   ─┘
11. LLaVA-Bench (单卡, ~20min, GPT-4)  ── 需要 API key
```

---

## 五、脚本文件清单

```
scripts/v1_5/
├── download_eval_data.sh          ← 统一下载所有评估数据
└── eval_7b_lora/                  ← 12 个评估脚本（均含数据检查）
    ├── pope.sh         # 单卡, 本地出分
    ├── textvqa.sh      # 单卡, 本地出分
    ├── mme.sh          # 单卡, 本地出分
    ├── sqa.sh          # 单卡, 本地出分
    ├── gqa.sh          # 多卡, 本地出分
    ├── seed.sh         # 多卡, 本地出分
    ├── vqav2.sh        # 多卡, 在线提交
    ├── vizwiz.sh       # 单卡, 在线提交
    ├── mmbench.sh      # 单卡, 在线提交
    ├── mmbench_cn.sh   # 单卡, 在线提交
    ├── llavabench.sh   # 单卡, GPT-4
    └── mmvet.sh        # 单卡, Jupyter 打分
```

---

## 六、官方参考分数（LLaVA-v1.5-7B-LoRA）

| Benchmark | 官方分数 | 指标说明 |
|-----------|---------|---------|
| VQAv2 | 79.1 | test-dev Accuracy |
| GQA | 63.0 | test-dev Balanced Accuracy |
| VisWiz | 47.8 | test-dev Accuracy |
| ScienceQA | 68.4 | Image Accuracy |
| TextVQA | 58.2 | Val Accuracy |
| POPE | 86.4 | F1 Score |
| MME | 1476.9 | Perception + Cognition |
| MMBench | 66.1 | dev Accuracy |
| MMBench-CN | 58.9 | dev Accuracy |
| SEED-Bench | 60.1 | Image Accuracy |
| LLaVA-Bench-Wild | 67.9 | GPT-4 Relative Score |
| MM-Vet | 30.2 | GPT-4 Score |

---

## 七、常见问题

### 评估时 OOM

评估比训练轻很多（batch_size=1, 无梯度），单卡 48GB 不会 OOM。如果出现，`CUDA_VISIBLE_DEVICES=0` 只用一张卡。

### 数据缺失

每个脚本自带检查。缺数据时会打印 `[ERROR] ... missing. Run: bash scripts/v1_5/download_eval_data.sh`。

### LoRA 未合并如何评估

所有脚本已包含 `--model-base lmsys/vicuna-7b-v1.5`，无需合并即可评估。

### HuggingFace 连不上

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### GQA eval.py 报错

官方 GQA eval.py 有缺失资源问题，需按[此 gist](https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187)修改。
