# Usage Guide

Complete guide for using the Cooling Tower Detection pipeline.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Detection Pipeline](#detection-pipeline)
3. [Human Review (HITL)](#human-review-hitl)
4. [Segmentation Pipeline](#segmentation-pipeline)
5. [Python API](#python-api)
6. [Advanced Usage](#advanced-usage)
7. [Examples](#examples)

---

## Basic Usage

### Complete Workflow

```bash
# 1. Run detection
python scripts/run_detection.py \
    --input_dir data/sample_images \
    --output_dir output/detections \
    --save_viz

# 2. Review detections (Jupyter notebook)
jupyter notebook notebooks/02_hitl_review.ipynb

# 3. Run segmentation on reviewed images
python scripts/run_segmentation.py \
    --detections_file output/detections/detections.pkl \
    --reviewed_file output/reviewed_images.pkl \
    --output_dir output/masks
```

---

## Detection Pipeline

### Command Line

```bash
python scripts/run_detection.py \
    --input_dir PATH_TO_IMAGES \
    --output_dir OUTPUT_PATH \
    --model_path models/best.pt \
    --conf_threshold 0.4 \
    --img_size 768 \
    --augment \
    --save_viz \
    --viz_dir output/visualizations
```

**Parameters:**
- `--input_dir`: Directory containing input images (required)
- `--output_dir`: Where to save detection results (default: output/detections)
- `--model_path`: Path to YOLO weights (default: models/best.pt)
- `--conf_threshold`: Confidence threshold 0-1 (default: 0.4)
- `--img_size`: Input image size (default: 768)
- `--augment`: Enable test-time augmentation
- `--save_viz`: Save visualization images with boxes
- `--viz_dir`: Directory for visualizations (default: output/visualizations)

### Jupyter Notebook

```python
from src.detection import CoolingTowerDetector

# Initialize
detector = CoolingTowerDetector(
    model_path="models/best.pt",
    conf_threshold=0.4
)

# Run detection
results = detector.process_directory(
    input_dir="data/sample_images",
    output_file="output/detections.pkl",
    save_visualizations=True
)

# Results is a list of (image_path, boxes, confidences)
for img_path, boxes, confs in results:
    print(f"{img_path}: {len(boxes)} detections")
```

---

## Human Review (HITL)

### Interactive Notebook

1. Open `notebooks/02_hitl_review.ipynb`
2. Run all cells up to the review interface
3. Use buttons to accept/reject detections:
   - **✓ Accept**: Good detection, will be used for segmentation
   - **✗ Reject**: False positive, will be excluded
   - **→ Skip**: Unsure, move to next

4. Results are saved to `output/reviewed_images.pkl`

### Custom Review Script

```python
import pickle
from pathlib import Path

# Load detections
with open("output/detections/detections.pkl", "rb") as f:
    detections = pickle.load(f)

# Custom filtering logic
approved = []
for img_path, boxes, confs in detections:
    # Your criteria
    if len(boxes) > 0 and all(c > 0.5 for c in confs):
        approved.append(Path(img_path).name)

# Save approved list
with open("output/reviewed_images.pkl", "wb") as f:
    pickle.dump(approved, f)
```

---

## Segmentation Pipeline

### Command Line

```bash
python scripts/run_segmentation.py \
    --detections_file output/detections/detections.pkl \
    --output_dir output/masks \
    --checkpoint models/sam2/checkpoints/sam2.1_hiera_base_plus.pt \
    --config_dir models/sam2/configs/sam2.1 \
    --reviewed_file output/reviewed_images.pkl
```

**Parameters:**
- `--detections_file`: Path to detections pickle (required)
- `--output_dir`: Where to save masks (default: output/masks)
- `--checkpoint`: SAM2 checkpoint path
- `--config_dir`: SAM2 config directory
- `--reviewed_file`: Optional filter for reviewed images
- `--mask_size`: Output mask size (default: 768)

### Jupyter Notebook

```python
from src.segmentation import SAM2Segmenter
import pickle

# Load detections
with open("output/detections/detections.pkl", "rb") as f:
    detections = pickle.load(f)

# Initialize segmenter
segmenter = SAM2Segmenter(
    checkpoint_path="models/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
    config_dir="models/sam2/configs/sam2.1"
)

# Generate masks
masks = segmenter.process_detections(
    detections=detections,
    output_dir="output/masks"
)
```

---

## Python API

### Detection

```python
from src.detection import CoolingTowerDetector

# Initialize detector
detector = CoolingTowerDetector(
    model_path="models/best.pt",
    conf_threshold=0.4,
    img_size=768,
    augment=True,
    device="cuda"  # or "cpu"
)

# Detect in single image
image_rgb, boxes, confs = detector.detect_image("image.png")

# Process directory
results = detector.process_directory(
    input_dir="images/",
    output_file="detections.pkl",
    max_workers=4
)

# Load saved detections
detections = CoolingTowerDetector.load_detections("detections.pkl")
```

### Segmentation

```python
from src.segmentation import SAM2Segmenter
import numpy as np

# Initialize
segmenter = SAM2Segmenter(
    checkpoint_path="models/sam2/checkpoint.pt",
    config_dir="models/sam2/configs/sam2.1",
    config_name="sam2.1_hiera_b+.yaml"
)

# Segment single image
boxes = np.array([[x1, y1, x2, y2], ...])
mask = segmenter.segment_image(
    image_path="image.png",
    boxes=boxes
)

# Process multiple detections
masks = segmenter.process_detections(
    detections=detection_results,
    output_dir="masks/"
)
```

### Utilities

```python
from src.utils import (
    visualize_detections,
    visualize_mask,
    draw_boxes_on_image,
    compute_iou,
    calculate_metrics
)

# Visualize detections
visualize_detections(
    image_path="image.png",
    boxes=boxes,
    confs=confs,
    save_path="visualization.png"
)

# Visualize mask
visualize_mask(
    image_path="image.png",
    mask=mask,
    save_path="mask_overlay.png",
    alpha=0.5
)

# Calculate metrics
metrics = calculate_metrics(
    pred_masks=[pred_mask1, pred_mask2],
    gt_masks=[gt_mask1, gt_mask2]
)
print(f"Mean IoU: {metrics['mean_iou']:.3f}")
```

---

## Advanced Usage

### Custom Configuration

Create a custom config file:

```yaml
# custom_config.yaml
detection:
  model_path: "path/to/custom_model.pt"
  confidence_threshold: 0.5
  image_size: 1024

segmentation:
  sam2_checkpoint: "path/to/checkpoint.pt"
  box_padding_ratio: 0.3
  fixed_padding: 100

processing:
  max_workers: 8
  device: "cuda"
```

Use it:

```bash
python scripts/run_detection.py --config custom_config.yaml
```

### Batch Processing Large Datasets

```python
from pathlib import Path
from src.detection import CoolingTowerDetector

detector = CoolingTowerDetector(model_path="models/best.pt")

# Process in batches
image_dirs = [
    "dataset/batch1",
    "dataset/batch2",
    "dataset/batch3"
]

for batch_dir in image_dirs:
    print(f"Processing {batch_dir}...")
    results = detector.process_directory(
        input_dir=batch_dir,
        output_file=f"output/{Path(batch_dir).name}_detections.pkl"
    )
```

### Custom Box Expansion

```python
from src.segmentation import SAM2Segmenter

# Custom expansion parameters
mask = segmenter.segment_image(
    image_path="image.png",
    boxes=boxes,
    padding_ratio=0.5,  # 50% expansion
    fixed_padding=100   # Plus 100 pixels
)
```

### Export Results

```python
import json
import pickle
from pathlib import Path

# Load results
with open("output/detections.pkl", "rb") as f:
    detections = pickle.load(f)

# Export to JSON
results = []
for img_path, boxes, confs in detections:
    results.append({
        "image": str(Path(img_path).name),
        "detections": [
            {
                "box": [int(x) for x in box],
                "confidence": float(conf)
            }
            for box, conf in zip(boxes, confs)
        ]
    })

with open("output/detections.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## Examples

### Example 1: Simple Detection

```bash
# Detect cooling towers in a folder
python scripts/run_detection.py \
    --input_dir my_images/ \
    --output_dir results/
```

### Example 2: High-Precision Detection

```bash
# Use higher threshold and augmentation
python scripts/run_detection.py \
    --input_dir my_images/ \
    --output_dir results/ \
    --conf_threshold 0.6 \
    --augment \
    --save_viz
```

### Example 3: Full Pipeline with Review

```bash
# Step 1: Detect
python scripts/run_detection.py \
    --input_dir data/aerial_images/ \
    --output_dir output/detections/ \
    --save_viz --viz_dir output/review_images/

# Step 2: Review (open notebook)
jupyter notebook notebooks/02_hitl_review.ipynb

# Step 3: Segment approved only
python scripts/run_segmentation.py \
    --detections_file output/detections/detections.pkl \
    --reviewed_file output/reviewed_images.pkl \
    --output_dir output/final_masks/
```

### Example 4: Python Script Integration

```python
#!/usr/bin/env python3
"""Custom detection and segmentation pipeline."""

from src.detection import CoolingTowerDetector
from src.segmentation import SAM2Segmenter
from src.utils import create_summary_report

# Configuration
INPUT_DIR = "data/my_project"
OUTPUT_DIR = "output/my_project"

# Detection
detector = CoolingTowerDetector(model_path="models/best.pt")
detections = detector.process_directory(
    input_dir=INPUT_DIR,
    output_file=f"{OUTPUT_DIR}/detections.pkl"
)

# Create report
create_summary_report(detections, f"{OUTPUT_DIR}/report.txt")

# Segmentation
segmenter = SAM2Segmenter(
    checkpoint_path="models/sam2/checkpoint.pt",
    config_dir="models/sam2/configs/sam2.1"
)
masks = segmenter.process_detections(
    detections=detections,
    output_dir=f"{OUTPUT_DIR}/masks"
)

print(f"✓ Processed {len(masks)} images")
print(f"✓ Results saved to {OUTPUT_DIR}")
```

---

## Tips and Best Practices

1. **Start with visualizations**: Always use `--save_viz` first to check detection quality
2. **Tune confidence threshold**: Start with 0.4, increase if too many false positives
3. **Use HITL for important projects**: Manual review improves results significantly
4. **Monitor GPU memory**: Reduce batch size if running out of memory
5. **Save intermediate results**: Don't skip the pickle files, they're useful for debugging

## Troubleshooting

See [SETUP.md](SETUP.md#troubleshooting) for common issues and solutions.
