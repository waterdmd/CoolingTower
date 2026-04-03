# 🏗️ YOLO Training Guide — Cooling Tower Detection

> **Part of the Cooling Tower Detection Pipeline**  
> Repository: [github.com/waterdmd/CoolingTower](https://github.com/waterdmd/CoolingTower)

This guide covers how to fine-tune a YOLO model to detect cooling towers from aerial imagery. Once trained, the resulting `best.pt` weights are used in **Step 5** of the main pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dataset Structure](#dataset-structure)
4. [Label Format](#label-format)
5. [Step 1 — Inspect an Existing Checkpoint](#step-1--inspect-an-existing-checkpoint)
6. [Step 2 — Create the Data Config YAML](#step-2--create-the-data-config-yaml)
7. [Step 3 — Load the Base YOLO Model](#step-3--load-the-base-yolo-model)
8. [Step 4 — Train the Model](#step-4--train-the-model)
9. [Step 5 — Validate and Export](#step-5--validate-and-export)
10. [Key Variables Reference](#key-variables-reference)
11. [Output Files](#output-files)
12. [Tips & Troubleshooting](#tips--troubleshooting)

---

## Overview

The training pipeline fine-tunes a **YOLOv11n** (nano) base model on a custom labeled dataset of aerial cooling tower images. Training runs for 200 epochs at 1024×1024 image resolution. The best weights (`best.pt`) are saved automatically and used in the detection pipeline.

**Pipeline summary:**

```
Labeled Images (images/ + labels/)
         ↓
   data.yaml config
         ↓
  Load base YOLO model
         ↓
    yolo.train()   ← 200 epochs, imgsz=1024
         ↓
  runs/new_ct_train/weights/best.pt
         ↓
  Use in detection pipeline (Step 5)
```

---

## Prerequisites

- Python 3.9+
- CUDA GPU (recommended: A100 or V100 on HPC)
- Packages:

```bash
pip install torch torchvision
pip install ultralytics
pip install pyyaml
```

- A labeled dataset of aerial images (see [Dataset Structure](#dataset-structure) below)
- A base YOLO checkpoint to fine-tune from (e.g., `yolo11n.pt`)

---

## Dataset Structure

Your `data/` folder must follow this exact layout:

```
data/
├── data.yaml                  ← config file (auto-generated in Step 2)
├── train/
│   ├── images/
│   │   ├── tile_21_XXXXX_YYYYY.png
│   │   └── ...
│   └── labels/
│       ├── tile_21_XXXXX_YYYYY.txt
│       └── ...
└── valid/
    ├── images/
    │   ├── tile_21_AAAAA_BBBBB.png
    │   └── ...
    └── labels/
        ├── tile_21_AAAAA_BBBBB.txt
        └── ...
```

> **Rule:** Every image file in `images/` must have a corresponding `.txt` label file with the same name in `labels/`. Images with no cooling towers still need an empty `.txt` file.

---

## Label Format

Each `.txt` label file uses **YOLO format** — one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

- All values are **normalized** to `[0, 1]` relative to image width and height
- `class_id` is always `0` (there is only one class: `ct` = cooling tower)
- Coordinates are the **center** of the bounding box, not the top-left corner

**Example label file** for an image with two cooling towers:

```
0 0.512 0.374 0.083 0.091
0 0.731 0.228 0.076 0.085
```

**Conversion from pixel coordinates:**

```python
x_center = (x1 + x2) / 2 / image_width
y_center = (y1 + y2) / 2 / image_height
width    = (x2 - x1) / image_width
height   = (y2 - y1) / image_height
```

---

## Step 1 — Inspect an Existing Checkpoint

If you have an existing `best.pt` and want to check what settings it was trained with before retraining:

```python
import torch

pt_path = "/path/to/best.pt"   # ← change this

ckpt = torch.load(pt_path, map_location='cpu')
args = ckpt.get('train_args', {})

print("Architecture Base:", args.get('model', 'Unknown'))
print("Epochs:           ", args.get('epochs'))
print("Batch Size:       ", args.get('batch'))
print("Optimizer:        ", args.get('optimizer'))
print("Initial LR (lr0): ", args.get('lr0'))
```

Also verify your GPU is available:

```python
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available: ", torch.cuda.is_available())
```

---

## Step 2 — Create the Data Config YAML

This script writes the `data/data.yaml` file that tells YOLO where your images are and what classes exist.

**Variables to change:**

| Variable | Example Value | Description |
|---|---|---|
| `train` | `/your/path/data/train/images` | Absolute path to training images folder |
| `val` | `/your/path/data/valid/images` | Absolute path to validation images folder |
| `names` | `['ct']` | List of class names — keep as-is for cooling towers |
| `nc` | `1` | Number of classes — keep as `1` |

```python
import yaml

config = {
    'names': ['ct'],          # class names — do not change
    'nc': 1,                  # number of classes — do not change
    'train': '/your/path/data/train/images',   # ← UPDATE THIS
    'val':   '/your/path/data/valid/images'    # ← UPDATE THIS
}

with open('data/data.yaml', 'w') as f:
    yaml.dump(config, f)

print("data.yaml written successfully.")
```

> **Note:** Use absolute paths. Relative paths can break when the working directory changes.

---

## Step 3 — Load the Base YOLO Model

This step loads the base YOLOv11n weights. A compatibility shim for `C3k2` is registered first to handle custom layer types that may appear in some checkpoints.

**Variables to change:**

| Variable | Example Value | Description |
|---|---|---|
| `model path` | `/your/path/yolo11n.pt` | Path to the base YOLO model file to fine-tune from |

```python
import torch
import torch.nn as nn
from ultralytics.nn.modules.block import C3
from ultralytics import YOLO
import builtins

# Register C3k2 compatibility shim
class C3k2(C3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

builtins.C3k2 = C3k2

# Load the base model
model = YOLO('/your/path/yolo11n.pt')   # ← UPDATE THIS
print("Model loaded successfully.")
```

> **Note:** `yolo11n.pt` is the nano variant (fastest, least VRAM). You can swap to `yolo11s.pt` (small) or `yolo11m.pt` (medium) for higher accuracy at the cost of training time.

---

## Step 4 — Train the Model

Runs fine-tuning on your labeled dataset. This is the main training call.

**Variables to change:**

| Variable | Example Value | Description |
|---|---|---|
| `data` | `'/your/path/data/data.yaml'` | Path to the YAML created in Step 2 |
| `epochs` | `200` | Number of training epochs — 200 is a good starting point |
| `name` | `'new_ct_train'` | Name of the training run — output saved under `runs/<name>/` |
| `imgsz` | `1024` | Training image size in pixels — keep at 1024 to match tile resolution |

```python
from ultralytics import YOLO

yolo = YOLO(model)   # model loaded in Step 3

yolo.train(
    data   = '/your/path/data/data.yaml',   # ← UPDATE THIS
    epochs = 200,
    name   = 'new_ct_train',                # ← rename per experiment
    imgsz  = 1024
)
```

Training progress is printed to console and logged automatically. On a single A100, 200 epochs typically takes **2–4 hours** depending on dataset size.

---

## Step 5 — Validate and Export

After training, validate on the held-out validation set and print metrics.

```python
valid_results = yolo.val()
print(valid_results)
```

**Key metrics to check:**

| Metric | What It Means | Target |
|---|---|---|
| `mAP50` | Mean Average Precision at IoU=0.50 | > 0.85 is good |
| `mAP50-95` | mAP averaged over IoU thresholds 0.5–0.95 | > 0.60 is good |
| `Precision` | Of all predicted boxes, fraction that are correct | Higher is better |
| `Recall` | Of all actual cooling towers, fraction detected | Higher is better |

**Best weights are saved automatically at:**

```
runs/new_ct_train/weights/best.pt
```

Copy this file to your pipeline's model path (used in the main pipeline Step 5).

---

## Key Variables Reference

Quick summary of every variable you need to update for a new training run:

| Script | Variable | What to Set |
|---|---|---|
| Step 1 | `pt_path` | Path to an existing checkpoint to inspect (optional) |
| Step 2 | `train` in config | Absolute path to `data/train/images/` |
| Step 2 | `val` in config | Absolute path to `data/valid/images/` |
| Step 3 | `model path` | Path to base YOLO weights (`yolo11n.pt`) |
| Step 4 | `data` | Path to `data/data.yaml` |
| Step 4 | `epochs` | Number of training epochs (200 default) |
| Step 4 | `name` | Run name — change per experiment to avoid overwriting |
| Step 4 | `imgsz` | Image size (1024 for this pipeline) |

---

## Output Files

After training completes, the `runs/new_ct_train/` folder contains:

```
runs/new_ct_train/
├── weights/
│   ├── best.pt       ← USE THIS in the detection pipeline
│   └── last.pt       ← weights from the final epoch
├── results.csv       ← per-epoch loss and metric curves
├── confusion_matrix.png
├── PR_curve.png
├── F1_curve.png
└── val_batch*.jpg    ← sample validation predictions
```

---

## Tips & Troubleshooting

**Training is very slow**
- Make sure CUDA is available: `torch.cuda.is_available()` should return `True`
- Lower `imgsz` to 768 to reduce VRAM usage (slight accuracy tradeoff)
- Use a smaller base model (`yolo11n` is already nano/fastest)

**CUDA out of memory**
- Reduce `imgsz` or add `batch=8` (or lower) to the `train()` call
- Close other GPU processes before launching training

**`C3k2` ImportError or unknown layer**
- Make sure the `builtins.C3k2 = C3k2` shim in Step 3 runs **before** loading any model
- This only applies when loading checkpoints trained with newer Ultralytics versions

**`data.yaml` not found**
- Always use absolute paths in the YAML
- Confirm the file was written: `cat data/data.yaml`

**Low mAP after 200 epochs**
- Check that labels are correct (visualize a few with `yolo.val(plots=True)`)
- Make sure train/val images don't overlap
- Try training for more epochs or with a larger base model (`yolo11s.pt`)

**Overwriting a previous run**
- Change `name='new_ct_train'` to a unique name each time (e.g., `name='ct_train_v2'`)
- Or add `exist_ok=True` if you want to resume/overwrite intentionally

---

*Water DMD Lab — Arizona State University*  
*Repository: [github.com/waterdmd/CoolingTower](https://github.com/waterdmd/CoolingTower)*
