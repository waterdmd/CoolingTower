# 🛰️ Cooling Tower Detection Pipeline Guide

> **YOLO + SAM2 Geospatial Detection on Aerial Imagery**  
> Repository: [github.com/waterdmd/CoolingTower](https://github.com/waterdmd/CoolingTower)  
> Water DMD Lab — Arizona State University

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1 — Download Aerial Imagery Tiles](#step-1--download-aerial-imagery-tiles)
4. [Step 2 — Stitch Tiles into 3×3 Grid PNGs](#step-2--stitch-tiles-into-3x3-grid-pngs)
5. [Step 3 — Convert Stitched PNG to GeoTIFF](#step-3--convert-stitched-png-to-geotiff)
6. [Step 4 — Load YOLO and SAM2 Models](#step-4--load-yolo-and-sam2-models)
7. [Step 5 — Run YOLO on Grid Images](#step-5--run-yolo-on-grid-images)
8. [Step 6 — Human-in-the-Loop Review (HITL)](#step-6--human-in-the-loop-review-hitl)
9. [Step 7 — Run SAM2 Segmentation on Accepted Grids](#step-7--run-sam2-segmentation-on-accepted-grids)
10. [Output Summary](#output-summary)
11. [Quick Checklist for a New City](#quick-checklist-for-a-new-city)

---

## Overview

This document describes the entire process of detecting cooling towers using a pipeline that:
- Downloads aerial imagery tiles from a GIS server
- Stitches them together into a large mosaic
- Identifies cooling towers using a YOLO object detection model
- Produces georeferenced mask outputs using a SAM2 segmentation algorithm

No deep ML experience is required — follow the steps in order, update the highlighted variables for your target city, and execute the scripts.

### Pipeline at a Glance

| Step | Stage | What Happens |
|---|---|---|
| 1 | Download Tiles | Fetch aerial imagery tiles from a GIS tile server using lat/lon bounding box |
| 2 | Stitch to PNG | Assemble individual 256×256 tiles into 3×3 grid PNGs (768×768 px each) |
| 3 | Convert to GeoTIFF | Georeference the stitched mosaic so it carries real-world coordinates |
| 4 | Load Models | Load the YOLO detector and SAM2 segmentation model into GPU memory |
| 5 | Run YOLO on Grids | Detect cooling tower bounding boxes across all grid images; save results |
| 6 | Human Review (HITL) | Manually accept or reject YOLO detections using an interactive widget |
| 7 | Run SAM on Masks | Generate pixel-level segmentation masks for every accepted detection |

---

## Prerequisites

### Environment

- Python 3.9+ with pip
- CUDA-capable GPU (recommended: A100 / V100 on an HPC cluster)
- Access to a Jupyter / JupyterLab environment or terminal
- Internet access to the target GIS tile server

### Required Python Packages

```bash
pip install requests tqdm pillow numpy rasterio mercantile
pip install torch torchvision ultralytics opencv-python
pip install ipywidgets matplotlib
# SAM2 — install from source (see GitHub repo README for instructions)
```

### Model Checkpoints

Place the following files in your working directories before running:

- `best.pt` — your trained YOLO model (see [YOLO_Training_Guide.md](YOLO_Training_Guide.md))
- `sam2.1_hiera_base_plus.pt` — download from [Meta's SAM2 release](https://github.com/facebookresearch/segment-anything-2)
- `sam2.1_hiera_b+.yaml` — SAM2 config, located in `configs/sam2.1/`

> **📝 Note:** You can also fine-tune SAM2 on your own imagery; the fine-tuned checkpoint (`checkpoint.pt`) is used in Step 7.

---

## Step 1 — Download Aerial Imagery Tiles

This script downloads individual map tiles from a GIS tile server (Web Mercator / WMTS) for a given bounding box and zoom level. Tiles are saved as JPEG files named `tile_{ZOOM}_{x}_{y}.jpeg`.

### Key Variables to Change

| Variable | Example Value | Description |
|---|---|---|
| `OUTPUT_DIR` | `/your/path/tiles/` | Folder where tile JPEGs will be saved |
| `MIN_LAT` / `MAX_LAT` | `33.413`, `33.431` | Latitude bounds of the target city/area |
| `MIN_LON` / `MAX_LON` | `-111.937`, `-111.916` | Longitude bounds of the target city/area |
| `ZOOM` | `21` | Tile zoom level — 21 gives very high resolution (~6 cm/px) |
| `TILE_URL_TEMPLATE` | *(GIS server URL)* | Swap in the correct ArcGIS / WMS tile endpoint for your region |
| `THREADS` | `30` | Number of parallel download threads — lower if server rate-limits you |

### How to Find Your Bounding Box

1. Go to [bboxfinder.com](http://bboxfinder.com) or use Google Maps
2. Draw a rectangle around your city / area of interest
3. Copy the lat/lon min/max values into `MIN_LAT`, `MAX_LAT`, `MIN_LON`, `MAX_LON`

### How to Find the Tile URL

The URL template follows ArcGIS REST API conventions. Replace the base path for a different county or state GIS portal. Example for Maricopa County:

```
https://gis.maricopa.gov/imagery/rest/services/
  Aerial2022Sep2022OctOrtho/MapServer/tile/{z}/{y}/{x}
```

Search your county/city GIS portal for "Imagery MapServer" or "Aerial Ortho" REST endpoints.

### Running the Script

```bash
python download_tiles.py
# Output: tile_21_XXXXX_YYYYY.jpeg files in OUTPUT_DIR
# Failed tiles are logged to error_log_*.txt
```

> **📝 Note:** If many tiles fail, reduce `THREADS` or add a longer `RETRY_DELAY`. The script automatically skips tiles that already exist, so you can safely re-run it.

---

## Step 2 — Stitch Tiles into 3×3 Grid PNGs

Groups downloaded tiles into non-overlapping 3×3 blocks and stitches each block into a single 768×768 PNG. Also produces one large composite PNG of the entire area and a JSON mapping of which tiles went into which grid.

### Key Variables to Change

| Variable | Example Value | Description |
|---|---|---|
| `TILE_DIR` | `/your/path/tiles/` | Same folder used in Step 1 (source tiles) |
| `OUTPUT_GRID_DIR` | `/your/path/grids/` | Where individual 768×768 grid PNGs are saved |
| `FINAL_PNG` | `area_stitched.png` | Filename for the full-area composite PNG |
| `GRID_MAP_JSON` | `/your/path/grid_tile_map.json` | Path for the tile→grid mapping file (needed in Step 3) |
| `ZOOM` | `21` | Must match the zoom used in Step 1 |

### Running the Script

```bash
python stitch_tiles.py
# Output:
#   grid_XXXXX_YYYYY.png             — individual 768px grid images
#   area_stitched_all_3x3_grids.png  — full composite
#   grid_tile_map.json               — mapping used by Step 3
```

> **📝 Note:** Tiles at the edges that don't form complete 3×3 groups are skipped. The JSON mapping is critical — don't delete it.

---

## Step 3 — Convert Stitched PNG to GeoTIFF

Reads the composite PNG and the grid tile map to compute real-world geographic coordinates (Web Mercator, EPSG:3857). Writes a GeoTIFF that can be opened in QGIS, ArcGIS, or any GIS tool.

### Key Variables to Change

| Variable | Example Value | Description |
|---|---|---|
| `INPUT_PNG` | `/your/path/area_stitched.png` | The composite PNG from Step 2 |
| `GRID_MAP_JSON` | `/your/path/grid_tile_map.json` | The JSON mapping from Step 2 |
| `OUTPUT_TIF` | `area_georef.tif` | Output filename for the GeoTIFF |
| `ZOOM` | `21` | Must match Steps 1 and 2 |

### Running the Script

```bash
python png_to_geotiff.py
# Output: area_georef.tif  (RGB GeoTIFF, EPSG:3857)
# Open in QGIS or ArcGIS to verify alignment
```

> **📝 Note:** The script uses `mercantile` to convert tile indices to Web Mercator bounding boxes. Coordinate accuracy depends on the zoom level and tile server's alignment.

---

## Step 4 — Load YOLO and SAM2 Models

This step loads both models into GPU memory before running inference. Run this in a Jupyter notebook or at the top of your inference script so the models are ready for Steps 5 and 7.

### Key Variables to Change

| Variable | Example Value | Description |
|---|---|---|
| `checkpoint_path` | `/path/to/sam2.1_hiera_base_plus.pt` | Path to SAM2 model weights |
| `config_dir` | `/path/to/configs/sam2.1/` | Directory containing SAM2 YAML configs |
| `config_name` | `sam2.1_hiera_b+.yaml` | SAM2 config filename (must be in `config_dir`) |
| YOLO path | `/path/to/best.pt` | Your YOLO model weights (set in Step 5 script) |

### Expected Output

```
📂 Loading Config from: /path/to/configs/sam2.1/
📂 Loading Weights from: /path/to/sam2.1_hiera_base_plus.pt
🏗️  Building model architecture...
⚖️  Loading state dict...
   (Unwrapping 'model' key from checkpoint...)
🚀 SUCCESS: SAM2 Model Loaded & Ready!
```

### Troubleshooting Model Loading

| Error | Fix |
|---|---|
| Config dir missing | Copy `configs/` from `sam2_hack_install/sam2/configs/` into `sam2_final/configs/` |
| Checkpoint not found | Verify `checkpoint_path` points to the `.pt` file exactly |
| CUDA out of memory | Lower batch size, or free other GPU processes before loading |
| `ImportError: sam2` | Add the `sam2_hack_install` path to `sys.path` (shown in script header) |

---

## Step 5 — Run YOLO on Grid Images

Runs the YOLO model on every 768×768 grid PNG. Detections (bounding boxes + confidence scores) are saved to a `.pkl` file for downstream use. Images with no detections are skipped automatically.

### Key Variables to Change

| Variable | Example Value | Description |
|---|---|---|
| `image_directory` | `/your/path/grids/` | Folder of 768px grid PNGs from Step 2 |
| YOLO model path | `/your/path/best.pt` | Your trained YOLO weights |
| `conf` threshold | `0.4` | Minimum confidence to keep a detection (0–1) |
| `imgsz` | `768` | Inference image size — keep at 768 to match tile size |
| `max_workers` | `40` | Number of parallel threads — adjust to your GPU count |
| `.pkl` output path | `/your/path/all_box_coords.pkl` | Where detection results are saved |

### Running the Script

```bash
python run_yolo.py
# Output: all_box_coords_and_images_latest.pkl
# Contains: list of (image_path, boxes_tensor, conf_tensor)
# Prints total detected cooling towers and elapsed time
```

> **📝 Note:** The script uses `ThreadPoolExecutor`, so a fresh YOLO model is loaded in each thread. This avoids CUDA context conflicts but uses more VRAM. Lower `max_workers` if you run out of memory.

---

## Step 6 — Human-in-the-Loop Review (HITL)

Before running the expensive SAM segmentation, this step lets you manually accept or reject each YOLO detection using an interactive Jupyter widget. This prevents false positives from propagating into the final masks.

### Part A — Generate Review Images

Draws bounding boxes (with confidence scores) on each detected grid and saves them to a review folder for visual inspection.

| Variable | Example Value | Description |
|---|---|---|
| `all_data` pkl | `/your/path/all_box_coords.pkl` | YOLO results from Step 5 |
| `REVIEW_DIR` | `/your/path/hitl_review_grids/` | Where annotated review images are saved |
| `CONF_THRESHOLD` | `0.4` | Only draw boxes at or above this confidence |

```bash
python save_review_images.py
# Produces annotated PNG files in REVIEW_DIR
```

### Part B — Interactive Accept / Reject Widget

Run this cell in a Jupyter notebook. Each grid image is shown with its detected boxes. Click the buttons to sort images into accepted or rejected folders.

| Variable | Example Value | Description |
|---|---|---|
| `REVIEW_DIR` | `/your/path/hitl_review_grids/` | Folder from Part A |
| `SAVE_DIR_YES` | `/your/path/accepted_grids/` | Accepted images copied here |
| `SAVE_DIR_NO` | `/your/path/rejected_grids/` | Rejected images copied here |

**Widget Controls:**

| Button | Action |
|---|---|
| ✔ Accept | Copy to `SAVE_DIR_YES`; add to `selected_images` list |
| ✘ Reject | Copy to `SAVE_DIR_NO`; skip for SAM processing |
| ➡ Skip | Move to next image without copying |

> **📝 Note:** After reviewing, save the accepted list — this file is required in Step 7:
> ```python
> import pickle
> pickle.dump(selected_images, open('new_tempe_selected_images.pkl', 'wb'))
> ```

---

## Step 7 — Run SAM2 Segmentation on Accepted Grids

For each YOLO-detected and human-accepted grid, this script crops the region around each bounding box, runs SAM2 to produce a precise pixel mask, and saves a binary mask PNG alongside each grid image. Grids with no detections receive a black (all-zero) mask.

### Key Variables to Change

| Variable | Example Value | Description |
|---|---|---|
| `GRID_DIR` | `/your/path/grids/` | Same grid folder from Step 2 |
| `MASK_DIR` | `/your/path/masks/` | Output folder for binary mask PNGs |
| `CHECKPOINT` | `/path/to/sam2_finetune/checkpoint.pt` | Fine-tuned SAM2 weights (or base if no fine-tuning) |
| `selected_images` pkl | `/your/path/new_tempe_selected_images.pkl` | Accepted filenames from Step 6 |
| `all_data` pkl | `/your/path/all_box_coords.pkl` | YOLO results from Step 5 |
| `MASK_SIZE` | `768` | Output mask size in pixels — must match grid size |

### expand_box Parameters

Each YOLO bounding box is expanded before being sent to SAM2 to give the model more context:

| Parameter | Default | Description |
|---|---|---|
| `padding` | `0.25` | Fractional expansion (25% of box width/height on each side) |
| `fixed_padding` | `50` | Additional fixed pixel padding on each side |

### Running the Script

```bash
python run_sam_masks.py
# For each accepted grid:
#   → Crop + expand each YOLO box region
#   → Run SAM2 predict() on the region
#   → Union all masks into one 768×768 binary PNG
# Non-detected grids → black mask saved (0-filled)
# Output: MASK_DIR/*.png  (white = cooling tower, black = background)
```

> **📝 Note:** SAM2 returns `multimask_output=True` (3 masks per box). All masks are OR-combined into the final binary mask. `torch.cuda.empty_cache()` is called before each prediction to prevent memory buildup.

---

## Output Summary

| File / Folder | Created In | Contents |
|---|---|---|
| `tile_21_X_Y.jpeg` | Step 1 | Raw aerial image tiles from GIS server |
| `grid_X_Y.png` | Step 2 | 768×768 stitched grid images (3×3 tiles) |
| `area_stitched.png` | Step 2 | Full composite aerial image of the area |
| `grid_tile_map.json` | Step 2 | Mapping of grid keys to source tile filenames |
| `area_georef.tif` | Step 3 | Georeferenced GeoTIFF (EPSG:3857, RGB) |
| `all_box_coords.pkl` | Step 5 | YOLO bounding boxes and confidence scores |
| `hitl_review_grids/` | Step 6A | Annotated grid images for human review |
| `accepted_grids/` | Step 6B | Human-accepted detection images |
| `selected_images.pkl` | Step 6B | List of accepted grid filenames for SAM |
| `masks/*.png` | Step 7 | Binary segmentation masks (white = CT, black = BG) |

---

## Quick Checklist for a New City

When running on a new city, update only these variables — everything else stays the same:

- [ ] **Step 1** — `OUTPUT_DIR`: set a new folder name for this city's tiles
- [ ] **Step 1** — `MIN_LAT`, `MAX_LAT`, `MIN_LON`, `MAX_LON`: paste the bounding box for the new city
- [ ] **Step 1** — `TILE_URL_TEMPLATE`: update if the GIS server is different for the new county/state
- [ ] **Step 2** — `TILE_DIR`, `OUTPUT_GRID_DIR`, `FINAL_PNG`, `GRID_MAP_JSON`: use the new city name prefix
- [ ] **Step 3** — `INPUT_PNG`, `GRID_MAP_JSON`, `OUTPUT_TIF`: point to the new city's Step 2 outputs
- [ ] **Step 5** — `image_directory`, `.pkl` output path: use new city folder and filename
- [ ] **Step 6** — all path variables: use new city folder names
- [ ] **Step 7** — `GRID_DIR`, `MASK_DIR`, `.pkl` paths: use new city names consistently

Model paths and parameters stay the same unless you have new model weights.

---

*Water DMD Lab — Arizona State University*  
*Repository: [github.com/waterdmd/CoolingTower](https://github.com/waterdmd/CoolingTower)*
