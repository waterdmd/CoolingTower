# Setup Instructions

Detailed guide for setting up the Cooling Tower Detection pipeline.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **CUDA**: 11.7 or higher (for GPU support)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for models and data

### Check Your System

```bash
# Check Python version
python --version  # Should be 3.8+

# Check CUDA availability (if using GPU)
nvidia-smi

# Check available disk space
df -h
```

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/cooling-tower-detection.git
cd cooling-tower-detection
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n cooling-tower python=3.8
conda activate cooling-tower
```

### 3. Install Dependencies

```bash
# Install PyTorch (choose based on your CUDA version)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision

# Install other requirements
pip install -r requirements.txt
```

### 4. Install SAM2

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 5. Set Up SAM2 Configuration

```bash
python scripts/setup_sam2.py
```

This script will:
- Create necessary directory structure
- Copy SAM2 configuration files
- Optionally download model checkpoints

### 6. Download Model Weights

#### YOLO Model
Place your trained YOLO weights (`best.pt`) in the `models/` directory.

If you don't have weights yet, you'll need to train a model or use pre-trained weights.

#### SAM2 Checkpoints

Download from [SAM2 releases](https://github.com/facebookresearch/segment-anything-2/releases):

```bash
# Example: Download base+ model
cd models/sam2/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

### 7. Configure Paths

Edit `config/config.yaml` to match your setup:

```yaml
detection:
  model_path: "models/best.pt"  # Your YOLO weights path

segmentation:
  sam2_checkpoint: "models/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
  sam2_config_dir: "models/sam2/configs/sam2.1"

paths:
  input_dir: "data/sample_images"
  output_dir: "output"
```

### 8. Prepare Sample Data

Place test images in `data/sample_images/`:

```bash
# Example structure
data/sample_images/
├── image1.png
├── image2.png
└── image3.png
```

### 9. Verify Installation

```bash
# Test YOLO detection
python -c "from ultralytics import YOLO; print('YOLO OK')"

# Test SAM2
python -c "from sam2.build_sam import build_sam2; print('SAM2 OK')"

# Test imports
python -c "from src import CoolingTowerDetector, SAM2Segmenter; print('All OK')"
```

## Quick Test

Run a quick test to ensure everything works:

```bash
# 1. Run detection
python scripts/run_detection.py \
    --input_dir data/sample_images \
    --output_dir output/test

# 2. Check results
ls output/test/
```

## Troubleshooting

### CUDA/GPU Issues

If GPU is not detected:

```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### SAM2 Installation Issues

If SAM2 installation fails:

```bash
# Try installing dependencies separately
pip install hydra-core
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Import Errors

If you get import errors:

```bash
# Ensure you're in the project root and the package is installed
cd /path/to/cooling-tower-detection
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Memory Issues

If you run out of memory:

1. Reduce batch size in config
2. Process fewer images at once
3. Use CPU instead of GPU
4. Close other applications

### Configuration Issues

If paths are not working:

```bash
# Use absolute paths in config.yaml
model_path: "/full/path/to/models/best.pt"

# Or ensure you run commands from project root
cd /path/to/cooling-tower-detection
python scripts/run_detection.py ...
```

## Next Steps

1. **Test with Sample Data**: Run detection on provided samples
2. **Review Results**: Use the HITL notebook to review detections
3. **Run Segmentation**: Generate masks for approved detections
4. **Train Custom Model**: If needed, train YOLO on your own data

## Additional Resources

- [YOLO Documentation](https://docs.ultralytics.com/)
- [SAM2 Repository](https://github.com/facebookresearch/segment-anything-2)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

## Getting Help

- Check [Issues](https://github.com/yourusername/cooling-tower-detection/issues)
- Read [CONTRIBUTING.md](CONTRIBUTING.md)
- Open a new issue with details
