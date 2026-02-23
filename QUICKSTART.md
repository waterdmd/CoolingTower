# Quick Start Guide

Get up and running with Cooling Tower Detection in 10 minutes!

## Prerequisites Check

```bash
# Verify Python version (3.8+)
python --version

# Check if you have a CUDA-capable GPU (optional but recommended)
nvidia-smi
```

## Installation (5 minutes)

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/cooling-tower-detection.git
cd cooling-tower-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install SAM2
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 2. Download Models

```bash
# Set up SAM2 (automated)
python scripts/setup_sam2.py

# Download SAM2 checkpoint
cd models/sam2/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
cd ../../..

# Add your YOLO weights
# Place your best.pt file in models/ directory
```

### 3. Add Sample Data

```bash
# Create sample images directory if not exists
mkdir -p data/sample_images

# Add your test images to data/sample_images/
# (PNG or JPG format, aerial imagery)
```

## Quick Test (2 minutes)

### Run Detection

```bash
python scripts/run_detection.py \
    --input_dir data/sample_images \
    --output_dir output/quick_test \
    --save_viz
```

**Expected Output:**
```
✓ YOLO model loaded from models/best.pt
Processing 10 images...
100%|████████████████████| 10/10 [00:15<00:00,  1.50s/it]

✓ Detection complete!
  Total images processed: 10
  Images with detections: 7
  Total cooling towers detected: 23
✓ Results saved to output/quick_test/detections.pkl
```

### View Results

```bash
# Check detection results
ls output/quick_test/

# View visualizations
open output/quick_test/visualizations/  # macOS
xdg-open output/quick_test/visualizations/  # Linux
```

## Complete Workflow (3 minutes)

### 1. Detection

```bash
python scripts/run_detection.py \
    --input_dir data/sample_images \
    --output_dir output/project1 \
    --save_viz
```

### 2. Review (Optional but Recommended)

```bash
# Open review notebook
jupyter notebook notebooks/02_hitl_review.ipynb

# Use the interactive interface to accept/reject detections
# Results saved to output/reviewed_images.pkl
```

### 3. Segmentation

```bash
python scripts/run_segmentation.py \
    --detections_file output/project1/detections.pkl \
    --output_dir output/project1/masks \
    --reviewed_file output/reviewed_images.pkl  # Optional: only if you did review
```

**Expected Output:**
```
Loading SAM2 from models/sam2/checkpoint.pt...
✓ SAM2 model loaded successfully

Processing 7 images...
100%|████████████████████| 7/7 [00:45<00:00,  6.43s/it]

✓ Saved 7 masks to output/project1/masks
```

### 4. Check Results

```bash
# View generated masks
ls output/project1/masks/

# Masks are saved as PNG files (white=cooling tower, black=background)
```

## Common Issues

### Issue: "Model not found"
**Solution:**
```bash
# Verify model exists
ls models/best.pt
ls models/sam2/checkpoints/*.pt

# If missing, download or add your models
```

### Issue: "CUDA out of memory"
**Solution:**
```bash
# Use CPU instead
python scripts/run_detection.py --input_dir data/sample_images --output_dir output

# Or reduce batch size in config.yaml
```

### Issue: "SAM2 import error"
**Solution:**
```bash
# Reinstall SAM2
pip uninstall sam2
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Run setup again
python scripts/setup_sam2.py
```

## Next Steps

### Explore the Notebooks

```bash
jupyter notebook
```

Navigate to `notebooks/` and open:
1. `01_detection_pipeline.ipynb` - Interactive detection
2. `02_hitl_review.ipynb` - Human review interface
3. `03_segmentation_pipeline.ipynb` - Segmentation workflow

### Customize Configuration

Edit `config/config.yaml`:

```yaml
detection:
  confidence_threshold: 0.5  # Adjust threshold
  image_size: 1024           # Larger images

processing:
  max_workers: 8             # More parallel workers
```

### Process Your Own Data

```bash
# Place your images in a directory
# Run full pipeline
python scripts/run_detection.py --input_dir /path/to/your/images --output_dir output/my_project
```

## Resources

- **Full Documentation**: See [README.md](README.md)
- **Detailed Setup**: See [SETUP.md](SETUP.md)
- **Usage Examples**: See [USAGE.md](USAGE.md)
- **API Reference**: Check docstrings in `src/` modules

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/cooling-tower-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/cooling-tower-detection/discussions)
- **Documentation**: Check all `.md` files in repository

## What's Next?

1. ✅ Successfully run detection
2. ✅ Review and approve results
3. ✅ Generate segmentation masks
4. 📊 Analyze your results
5. 🚀 Scale to larger datasets

**Happy detecting! 🎯**

---

*Completed the quick start? Star ⭐ the repo if this was helpful!*
