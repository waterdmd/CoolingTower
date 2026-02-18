# Automated Cooling Tower Detection & Segmentation Pipeline (Maricopa County)

This repository implements an end-to-end deep learning workflow for identifying and segmenting cooling towers in high-resolution aerial imagery.

The pipeline integrates:

- YOLOv11 for object detection  
- SAM2 (Segment Anything Model 2) for pixel-level segmentation  
- Human-in-the-Loop (HITL) verification for quality control  

---

## 📋 Table of Contents

- [Data Acquisition](#data-acquisition)
- [Preprocessing & Stitching](#preprocessing--stitching)
- [Georeferencing](#georeferencing)
- [Model Setup (SAM2)](#model-setup-sam2)
- [Object Detection (YOLO)](#object-detection-yolo)
- [Review Preparation](#review-preparation)
- [Human-in-the-Loop Verification](#human-in-the-loop-verification)
- [Semantic Segmentation (SAM2)](#semantic-segmentation-sam2)

---

## Data Acquisition

### Goal
Efficiently download high-resolution aerial imagery tiles from the Maricopa County GIS server.

### Source
Aerial2022Sep2022OctOrtho MapServer

- Resolution: Zoom Level 21 (~7 cm/pixel)
- Region of Interest: Configurable Latitude/Longitude bounding boxes (e.g., Tempe, Glendale)

### Implementation Details

- Converts Latitude/Longitude coordinates to Web Mercator Tile IDs
- Uses ThreadPoolExecutor (30 threads) for rapid parallel downloading
- Includes retry logic (3 retries per tile)
- Logs failed downloads for troubleshooting

---

## Preprocessing & Stitching

### Goal
Create context-aware image grids to improve detection accuracy.

### Logic

- Stitches individual 256x256 tiles into larger 3x3 grids
- Output resolution: 768x768 pixels

Cooling towers are large structures. Single tiles often crop them. Stitching ensures the model sees the entire structure and its surroundings.

### Mapping

Generates:

tempe_grid_tile_map_latest.json

This file links each stitched grid back to its original raw tile filenames for geospatial tracking.

### Output

- Individual stitched PNG grids
- Optional full-area mosaic for visualization

---

## Georeferencing

### Goal
Convert PNG grids into GeoTIFFs for GIS integration.

### Libraries Used

- rasterio
- mercantile
- PIL

### Process

1. Read the JSON grid mapping
2. Identify tile X/Y coordinates
3. Compute Web Mercator bounds (EPSG:3857)
4. Construct an Affine Transform
5. Write spatial metadata into the image header

### Outcome

- Geo-referenced .tif files
- Compatible with ArcGIS Pro and QGIS

---

## Model Setup (SAM2)

### Goal
Initialize Segment Anything Model 2 in a restricted cluster environment.

### Custom Loading

- Implements custom loading due to restricted pip paths
- Manually invokes Hydra configuration

Loads:

- sam2.1_hiera_b+.yaml
- sam2.1_hiera_base_plus.pt

### Checkpoint Handling

Includes logic to extract weights from the "model" key to prevent state dictionary mismatch errors.

---

## Object Detection (YOLO)

### Goal
Rapidly identify potential cooling towers.

### Model

YOLOv11 (custom trained)

### Inference

- Runs on 768x768 stitched grids
- Uses ThreadPoolExecutor or YOLO stream mode
- Confidence threshold: 0.3 – 0.5

### Output

- Bounding boxes (xyxy format)
- Confidence scores
- Serialized .pkl file for downstream stages

---

## Review Preparation

### Goal
Visualize detections for manual verification.

### Visualization

- Loads images and YOLO bounding box data
- Draws bounding boxes using matplotlib
- Labels boxes with confidence scores

### Filtering

Only saves images where detections meet the confidence threshold to a review directory.

---

## Human-in-the-Loop Verification

### Goal
Clean the dataset by manually verifying YOLO detections before segmentation.

### Interface

Interactive GUI built with ipywidgets running in Jupyter Notebook.

### Workflow

For each image:

- ✔ Accept
- ✘ Reject
- ➡ Skip

Images are sorted into:

- new_accepted_grids/
- new_rejected_grids/

### Benefit

Reduces false positives such as:

- HVAC units
- Circular roof vents
- Rooftop equipment

---

## Semantic Segmentation (SAM2)

### Goal
Generate precise pixel-level masks for verified cooling towers.

### Input

Only human-verified accepted grids.

### Prompt Engineering

- Uses YOLO bounding boxes as box prompts
- Applies configurable padding to fully enclose the structure

### Zero-Shot Inference

- Runs SAM2 without fine-tuning
- Generates high-fidelity binary masks (0 or 255)

### Output

Binary mask images where white pixels represent the exact cooling tower footprint, excluding shadows and rooftop clutter.

---

## Pipeline Summary

Tile Download  
→ Tile Stitching (3x3 Grids)  
→ YOLO Detection  
→ Human Verification (HITL)  
→ SAM2 Segmentation  
→ GeoTIFF Export  

---

## Key Features

- High-resolution aerial imagery (~7 cm/pixel)
- Multi-threaded scalable downloading and inference
- Human-verified detections
- Zero-shot segmentation
- GIS-compatible outputs
- Designed for restricted cluster environments
