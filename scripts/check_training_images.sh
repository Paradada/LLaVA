#!/bin/bash
# 检查 llava_v1_5_mix665k.json 中所有引用的图片是否存在
# 用法: bash scripts/check_training_images.sh

cd "$(dirname "$0")/.."   # 切到项目根目录

ANNOTATION="./playground/data/llava_v1_5_mix665k.json"
IMAGE_BASE="./playground/data"

if [ ! -f "$ANNOTATION" ]; then
    echo "[ERROR] Annotation not found: $ANNOTATION"
    exit 1
fi

echo "============================================"
echo "  Scanning training images..."
echo "  Annotation: $ANNOTATION"
echo "============================================"

# 用 Python 做精确检查
python3 -c "
import json, os
from collections import defaultdict

ann = '$ANNOTATION'
base = '$IMAGE_BASE'

with open(ann, 'r') as f:
    data = json.load(f)

missing = []
by_dataset = defaultdict(lambda: {'total': 0, 'missing': []})

for item in data:
    if 'image' not in item:
        continue
    img = item['image']
    dataset = img.split('/')[0]
    full_path = os.path.join(base, img)
    by_dataset[dataset]['total'] += 1
    if not os.path.exists(full_path):
        missing.append(img)
        by_dataset[dataset]['missing'].append(img)

# 输出报告
print()
total_items = sum(v['total'] for v in by_dataset.values())
total_missing = len(missing)
print(f'Total images referenced: {total_items}')
print(f'Total images MISSING:   {total_missing}')
print()

for ds in sorted(by_dataset.keys()):
    v = by_dataset[ds]
    m = len(v['missing'])
    status = 'OK' if m == 0 else f'MISSING {m}/{v[\"total\"]}'
    print(f'  {ds:15s}  {v[\"total\"]:>6d} images  [{status}]')

if total_missing == 0:
    print()
    print('All images present. Training data is complete.')
elif total_missing <= 20:
    print()
    print('--- Missing files ---')
    for f in missing:
        print(f'  {f}')
else:
    print()
    print(f'Too many missing ({total_missing}), showing first 20:')
    for f in missing[:20]:
        print(f'  {f}')
"

echo ""
echo "Done. If images are missing, re-download the corresponding dataset."
