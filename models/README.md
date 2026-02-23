# Models Directory

This directory contains model weights and configurations for the cooling tower detection pipeline.

## Directory Structure

```
models/
├── yolo/
│   ├── best.pt                 # YOLO detection model weights
│   └── README.md              # YOLO model details
└── sam2/
    ├── checkpoints/
    │   └── sam2.1_hiera_base_plus.pt  # SAM2 segmentation weights
    └── configs/
        └── sam2.1/            # SAM2 configuration files
            ├── sam2.1_hiera_b+.yaml
            ├── sam2.1_hiera_l.yaml
            └── sam2.1_hiera_s.yaml
```

## Required Models

### 1. YOLO Detection Model

**File**: `yolo/best.pt`

This is a custom-trained YOLOv8/v11 model for cooling tower detection.

**How to obtain**:
- Use your own trained model
- Contact the repository maintainers for a pre-trained version
- Train your own following YOLO training documentation

**Model details**:
- Architecture: YOLOv8/v11
- Input size: 768×768
- Classes: 1 (cooling_tower)
- Training dataset: Custom cooling tower imagery

### 2. SAM2 Segmentation Model

**File**: `sam2/checkpoints/sam2.1_hiera_base_plus.pt`

This is the SAM2 (Segment Anything 2) model from Meta AI.

**How to download**:

```bash
# Create directory
mkdir -p models/sam2/checkpoints

# Download base+ model (recommended)
wget -P models/sam2/checkpoints/ \
  https://dl.fbaipublicfiles.com/segment_anything_2/sam2.1_hiera_base_plus.pt
```

**Alternative models**:

- **Small** (faster, less accurate):
  ```bash
  wget -P models/sam2/checkpoints/ \
    https://dl.fbaipublicfiles.com/segment_anything_2/sam2.1_hiera_small.pt
  ```

- **Large** (slower, more accurate):
  ```bash
  wget -P models/sam2/checkpoints/ \
    https://dl.fbaipublicfiles.com/segment_anything_2/sam2.1_hiera_large.pt
  ```

### 3. SAM2 Configuration Files

**Directory**: `sam2/configs/sam2.1/`

These configuration files define the SAM2 model architecture.

**How to obtain**:

```bash
# Clone SAM2 repository (if not already done)
git clone https://github.com/facebookresearch/segment-anything-2.git

# Copy configs
cp -r segment-anything-2/sam2/configs models/sam2/
```

## Model Sizes

| Model | Size | VRAM Required | Speed | Accuracy |
|-------|------|---------------|-------|----------|
| YOLO best.pt | ~50MB | 2GB | Fast | High |
| SAM2 Small | ~150MB | 4GB | Fast | Good |
| SAM2 Base+ | ~300MB | 8GB | Medium | High |
| SAM2 Large | ~600MB | 12GB | Slow | Highest |

## Using Custom Models

### Using a Different YOLO Model

1. Place your model in `models/yolo/`
2. Update `config/config.yaml`:
   ```yaml
   paths:
     yolo_model: "models/yolo/your_model.pt"
   ```

### Using a Different SAM2 Model

1. Download the model to `models/sam2/checkpoints/`
2. Update `config/config.yaml`:
   ```yaml
   paths:
     sam2_checkpoint: "models/sam2/checkpoints/your_model.pt"
     sam2_config_name: "appropriate_config.yaml"
   ```

## Fine-tuning

### Fine-tuning YOLO

See [Ultralytics YOLO documentation](https://docs.ultralytics.com/modes/train/) for training custom YOLO models.

### Fine-tuning SAM2

See [SAM2 fine-tuning guide](https://github.com/facebookresearch/segment-anything-2) for adapting SAM2 to your domain.

## Model Licensing

- **YOLO**: Check the license of your specific YOLO model
- **SAM2**: Licensed under Apache 2.0 by Meta AI

Please ensure you comply with all model licenses when using this pipeline.

## Troubleshooting

### Model Not Found

If you get "Model not found" errors:

1. Check file paths in `config/config.yaml`
2. Ensure models are in correct directories
3. Verify file names match exactly (case-sensitive)

### Out of Memory

If you run out of GPU memory:

1. Use a smaller SAM2 model (small instead of base+)
2. Reduce `num_workers` in config
3. Enable `clear_cuda_cache: true` in config
4. Process fewer images at once

### Slow Inference

To speed up inference:

1. Use GPU (CUDA) instead of CPU
2. Reduce image resolution
3. Use smaller models
4. Increase `num_workers` for detection

## Need Help?

- Check [INSTALL.md](../INSTALL.md) for setup instructions
- Open an issue if models don't download correctly
- See [README.md](../README.md) for usage examples
