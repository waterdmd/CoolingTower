# Cooling Tower Detection with YOLO and SAM2

A comprehensive pipeline for detecting and segmenting cooling towers in aerial/satellite imagery using YOLO for object detection and SAM2 (Segment Anything Model 2) for precise segmentation.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Overview

This project provides an end-to-end solution for:
- **Detection**: Identifying cooling towers in large aerial images using YOLO
- **Segmentation**: Precise pixel-level segmentation using fine-tuned SAM2
- **Human-in-the-Loop**: Interactive review interface for quality assurance
- **Batch Processing**: Efficient multi-threaded processing of large datasets

## ✨ Features

- 🚀 Multi-threaded batch processing for high-speed inference
- 🎯 YOLO-based detection with configurable confidence thresholds
- 🖼️ SAM2 integration for high-quality segmentation masks
- 👁️ Interactive Jupyter-based review interface (HITL)
- 📊 Automatic mask generation and export
- 🔧 Configurable pipeline via YAML/command-line arguments

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for SAM2)
- 16GB+ RAM
- 10GB+ free disk space

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/cooling-tower-detection.git
cd cooling-tower-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download model weights

Download the required model files and place them in the appropriate directories:

- **YOLO weights**: Place `best.pt` in `models/yolo/`
- **SAM2 checkpoint**: Download from [SAM2 repository](https://github.com/facebookresearch/segment-anything-2) and place in `models/sam2/checkpoints/`

```bash
# Create model directories
mkdir -p models/yolo models/sam2/checkpoints

# Download SAM2 checkpoint (example)
wget -P models/sam2/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/sam2.1_hiera_base_plus.pt
```

## 📁 Project Structure

```
cooling-tower-detection/
├── config/
│   └── config.yaml              # Main configuration file
├── data/
│   ├── sample_images/           # Sample input images
│   └── README.md                # Data documentation
├── models/
│   ├── yolo/
│   │   └── best.pt              # YOLO model weights
│   └── sam2/
│       ├── checkpoints/         # SAM2 model checkpoints
│       └── configs/             # SAM2 configuration files
├── src/
│   ├── __init__.py
│   ├── detection.py             # YOLO detection pipeline
│   ├── segmentation.py          # SAM2 segmentation
│   ├── utils.py                 # Utility functions
│   └── visualization.py         # Visualization tools
├── notebooks/
│   ├── 01_detection_demo.ipynb
│   ├── 02_hitl_review.ipynb     # Human-in-the-loop interface
│   └── 03_analysis.ipynb
├── scripts/
│   ├── run_detection.py         # Main detection script
│   ├── run_segmentation.py      # Main segmentation script
│   └── process_pipeline.py      # Full pipeline
├── outputs/                     # Generated outputs (gitignored)
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### 1. Run detection on sample data

```bash
python scripts/run_detection.py \
    --input_dir data/sample_images \
    --output_dir outputs/detections \
    --conf_threshold 0.4
```

### 2. Run segmentation on detected objects

```bash
python scripts/run_segmentation.py \
    --input_dir data/sample_images \
    --detection_file outputs/detections/detections.pkl \
    --output_dir outputs/masks
```

### 3. Run full pipeline

```bash
python scripts/process_pipeline.py \
    --config config/config.yaml \
    --input_dir data/sample_images \
    --output_dir outputs
```

## 📊 Usage Examples

### Detection Only

```python
from src.detection import CoolingTowerDetector

detector = CoolingTowerDetector(
    model_path='models/yolo/best.pt',
    conf_threshold=0.4
)

results = detector.detect_batch(
    image_dir='data/sample_images',
    num_workers=4
)
```

### Detection + Segmentation

```python
from src.detection import CoolingTowerDetector
from src.segmentation import SAM2Segmentor

# Detect
detector = CoolingTowerDetector('models/yolo/best.pt')
detections = detector.detect_batch('data/images')

# Segment
segmentor = SAM2Segmentor(
    checkpoint='models/sam2/checkpoints/sam2.1_hiera_base_plus.pt',
    config='models/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml'
)
masks = segmentor.segment_from_detections(detections)
```

### Human-in-the-Loop Review

Use the interactive Jupyter notebook:

```bash
jupyter notebook notebooks/02_hitl_review.ipynb
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize the pipeline:

```yaml
detection:
  model_path: models/yolo/best.pt
  conf_threshold: 0.4
  img_size: 768
  augment: true

segmentation:
  sam2_checkpoint: models/sam2/checkpoints/sam2.1_hiera_base_plus.pt
  sam2_config: models/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml
  box_expansion: 0.25
  fixed_padding: 50

processing:
  num_workers: 4
  batch_size: 8
  output_mask_size: 768
```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2)
