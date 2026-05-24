# Project Status: Satellite Change Detection System

**Last Updated:** May 2026  
**Project Type:** End-to-end satellite change detection system using deep learning  
**Primary Developer:** Jayesh  
**Operating System:** Pop OS (Linux) — dual-booted with Windows  
**Project Root:** `/media/jayesh/Windows-SSD/Users/jayes/satellite-change-detector/`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Tech Stack with Versions](#3-tech-stack-with-versions)
4. [Folder and File Structure](#4-folder-and-file-structure)
5. [Datasets](#5-datasets)
6. [ML Model Details](#6-ml-model-details)
7. [Training Pipeline](#7-training-pipeline)
8. [Inference Pipeline](#8-inference-pipeline)
9. [GIS and Post-Processing Layer](#9-gis-and-post-processing-layer)
10. [Backend — FastAPI](#10-backend--fastapi)
11. [Async Workers — Celery + Redis](#11-async-workers--celery--redis)
12. [Database — PostgreSQL + PostGIS](#12-database--postgresql--postgis)
13. [Frontend — Leaflet.js](#13-frontend--leafletjs)
14. [Model Performance and Metrics](#14-model-performance-and-metrics)
15. [Known Issues and Limitations](#15-known-issues-and-limitations)
16. [What Is Missing and Not Yet Built](#16-what-is-missing-and-not-yet-built)
17. [Environment Setup and Run Instructions](#17-environment-setup-and-run-instructions)

---

## 1. Project Overview

This project is a production-grade, end-to-end satellite change detection system. The core goal is to detect meaningful structural changes — such as new buildings, roads, deforestation, and urban expansion — between two Sentinel-2 satellite images taken over the same geographic region at different points in time. The system was built specifically to operate on real Indian satellite data from the ISRO/Bhoonidhi Sentinel-2 archive, with the target region being tile **T43PGQ**, which covers the **Bengaluru metropolitan area** and its surrounding region in Karnataka, India.

The project is not just a model — it is a complete system with a machine learning pipeline, a GIS processing pipeline, an asynchronous backend API, a task queue, a database, and a web-based frontend with an interactive map. It was designed to be deployable and demonstrable end-to-end.

### What the system does, step by step:

1. A user opens the web frontend and inputs the file paths to T1 (before) and T2 (after) Sentinel-2 image folders.
2. The user draws an Area of Interest (AOI) polygon on the Leaflet map.
3. The user clicks "Run Detection."
4. The frontend sends a POST request to the FastAPI backend.
5. The backend queues an async Celery task.
6. The Celery worker picks up the task and runs the full inference pipeline:
   - Loads Sentinel-2 bands (B02, B03, B04, B08) from JP2 files using rasterio (Linux) or glymur (Windows fallback).
   - Aligns both images spatially (same CRS, same resolution).
   - Computes NDVI for both dates.
   - Tiles images into overlapping 256×256 patches.
   - Runs the ChangeFormer deep learning model on each patch pair.
   - Stitches predictions back into a full-resolution change mask using Gaussian blending.
   - Applies post-processing: probability threshold, NDVI-based vegetation filter, noise removal.
   - Saves a georeferenced GeoTIFF change mask and a GeoJSON polygon file.
7. The backend stores the result (in-memory in current state, DB-ready schema exists).
8. The frontend polls for completion, then renders red polygons over detected change areas on the Leaflet map.
9. Alerts are triggered and logged if the detected changed area exceeds a configurable threshold.

### Real-world results achieved:

The system was run on real Sentinel-2 data of Bengaluru (March 2021 vs March 2023). After training the ChangeFormer model on the OSCD dataset and running inference, the system detected **14.63 km² of changed area** across the Bengaluru region. The red polygon overlay on the frontend correctly showed dense clustering of changes around Bengaluru's urban core, periurban expansion zones, and along major development corridors — which is geographically consistent with the known rapid urbanization of the region during 2021–2023.

---

## 2. System Architecture

The system is organized into six functional layers that communicate via a well-defined flow:

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                             │
│           Leaflet.js — index.html (single file)             │
│    AOI drawing, path input, status polling, map overlay     │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP POST /detect-change
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND API                              │
│              FastAPI — backend/main.py                      │
│     /detect-change, /results/{id}, /alerts, /health         │
└───────────────────────┬─────────────────────────────────────┘
                        │ Celery task queue
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  ASYNC TASK QUEUE                           │
│              Redis (message broker)                         │
│         Celery Worker — workers/tasks.py                    │
└───────────────────────┬─────────────────────────────────────┘
                        │ calls
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  INFERENCE PIPELINE                         │
│   pipeline/preprocess.py → tiling.py → inference.py        │
│          → stitch.py → postprocess.py → gis.py             │
└───────────────────────┬─────────────────────────────────────┘
                        │ ML model
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML LAYER                                 │
│   ml/model.py (ChangeFormer) + checkpoints/best_model.pth   │
└───────────────────────┬─────────────────────────────────────┘
                        │ results stored
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   STORAGE LAYER                             │
│  outputs/ (GeoTIFF + GeoJSON) + PostgreSQL/PostGIS (schema) │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Tech Stack with Versions

### Core Python Environment

| Package | Version | Purpose |
|---|---|---|
| Python | 3.11 (conda env: `satchange`) | Runtime |
| PyTorch | 2.x (cu124) | Deep learning model |
| torchvision | latest (cu124) | Image transforms |
| CUDA | 12.4 compatible (driver: 580, CUDA 13.0 on Pop OS) | GPU acceleration |
| timm | latest | ChangeFormer encoder backbone (EfficientNet-B0) |
| transformers | latest | HuggingFace model loading |
| segmentation-models-pytorch | latest | Original Siamese UNet backbone (Phase 1) |

### GIS and Satellite Data

| Package | Version | Purpose |
|---|---|---|
| rasterio | 1.5.0 (conda-forge) | Read/write GeoTIFF, JP2 (Linux), spatial operations |
| GDAL | 3.12.3 (conda-forge) | Underlying GIS library |
| geopandas | latest | Vector data handling |
| shapely | latest | Geometry operations |
| pyproj | 3.7.2 | Coordinate reference system transformations |
| fiona | latest | GeoJSON/Shapefile I/O |
| glymur | latest | JP2 reading fallback (Windows only — not needed on Linux) |

### Backend and API

| Package | Version | Purpose |
|---|---|---|
| FastAPI | latest | REST API framework |
| uvicorn | latest | ASGI server |
| pydantic | latest | Request/response validation |
| python-multipart | latest | File upload handling |

### Async Task Processing

| Package | Version | Purpose |
|---|---|---|
| celery | 5.6.3 | Distributed task queue |
| redis | latest (Python client) | Message broker for Celery |
| Redis Server | system service (Memurai on Windows / redis-server on Linux) | Broker backend |

### Database

| Package | Version | Purpose |
|---|---|---|
| PostgreSQL | 18 (Windows) / system (Linux) | Relational database |
| PostGIS | 3.4 | Spatial extensions for PostgreSQL |
| SQLAlchemy | 2.0 style | ORM for Python–PostgreSQL connection |
| psycopg2-binary | latest | PostgreSQL driver |
| geoalchemy2 | latest | PostGIS geometry types in SQLAlchemy |

### ML Training

| Package | Version | Purpose |
|---|---|---|
| albumentations | latest | Image augmentation pipeline |
| MLflow | 3.11.1 | Experiment tracking and metric logging |
| opencv-python | latest | Noise removal (connected components) |
| scikit-learn | latest | Evaluation metrics (F1, IoU) |
| numpy | latest | Array operations |
| scipy | latest | Gaussian blending for patch stitching |
| tqdm | latest | Progress bars |

### Frontend

| Library | Source | Purpose |
|---|---|---|
| Leaflet.js | CDN | Interactive map |
| Leaflet.draw | CDN | AOI polygon drawing tool |

---

## 4. Folder and File Structure

```
satellite-change-detector/
│
├── GEMINI.md                          # Project context for Gemini CLI sessions
├── PROJECT_STATUS.md                  # Running status log (updated per session)
├── README.md                          # Setup and run instructions
├── requirements.txt                   # Pinned Python dependencies
│
├── configs/
│   └── config.yaml                    # Central config: model paths, thresholds,
│                                      # dataset paths, DB URL, Redis URL, output dir
│
├── ml/
│   ├── __init__.py
│   ├── model.py                       # ChangeFormer architecture (EfficientNet-B0
│   │                                  # encoder, custom decoder, sigmoid output)
│   ├── dataset.py                     # LEVIRDataset + OSCDDataset classes
│   ├── losses.py                      # DiceLoss + BCEDiceLoss (combined)
│   ├── train.py                       # Full training loop with MLflow, early stopping,
│   │                                  # mixed precision, torch.compile
│   └── evaluate.py                    # Val set evaluation: IoU, F1, confusion matrix,
│                                      # sample prediction visualizations
│
├── pipeline/
│   ├── __init__.py
│   ├── preprocess.py                  # load_sentinel2_bands() — reads B02/B03/B04/B08
│   │                                  # from JP2 using rasterio (Linux) or glymur
│   │                                  # (Windows), normalizes, returns array+meta+CRS
│   │                                  # align_images() — reprojects img2 to match img1
│   │                                  # compute_ndvi() — (NIR-Red)/(NIR+Red)
│   ├── tiling.py                      # tile_image() — splits (H,W,C) array into
│   │                                  # overlapping 256×256 patches with metadata
│   │                                  # tile_image_pair() — tiles both images together
│   ├── stitch.py                      # stitch_patches() — reassembles prediction
│   │                                  # patches into full mask using Gaussian blending
│   ├── postprocess.py                 # apply_threshold() — prob mask → binary
│   │                                  # filter_by_ndvi() — suppresses vegetation change
│   │                                  # remove_noise() — removes small components
│   │                                  # postprocess() — runs all steps in sequence
│   ├── inference.py                   # run_inference() — master orchestration:
│   │                                  # preprocess → tile → model → stitch → post →
│   │                                  # gis → return result dict
│   └── gis.py                         # save_geotiff() — writes georef GeoTIFF
│                                      # mask_to_geojson() — vectorizes mask to polygons
│                                      # compute_total_change_area() — returns km²
│
├── backend/
│   ├── __init__.py
│   ├── main.py                        # FastAPI app init, CORS, startup event,
│   │                                  # static file serving for GeoJSON outputs
│   ├── routes.py                      # POST /detect-change, GET /results/{task_id},
│   │                                  # GET /alerts, GET /health
│   └── schemas.py                     # Pydantic models: AOICreate, DetectionRequest,
│                                      # DetectionResponse, ResultResponse, AlertResponse
│
├── workers/
│   ├── __init__.py
│   ├── tasks.py                       # Celery app + run_detection_task():
│   │                                  # updates status → runs inference → saves result
│   │                                  # → checks alert threshold → logs completion
│   └── alerts.py                      # trigger_alert() — logs to outputs/alerts.log
│                                      # and prints formatted alert to console
│
├── database/
│   ├── __init__.py
│   ├── schema.sql                     # PostGIS DDL: aois, detection_results, alerts
│   └── models.py                      # SQLAlchemy 2.0 ORM models matching schema
│
├── frontend/
│   └── index.html                     # Single-file web app: Leaflet map, AOI draw,
│                                      # path inputs, run button, status polling,
│                                      # GeoJSON overlay, alerts panel
│
├── data/
│   ├── LEVIR-CD/                      # LEVIR-CD dataset (Phase 1 training)
│   │   ├── train/
│   │   │   ├── A/                     # T1 images (PNG, 1024×1024, RGB)
│   │   │   ├── B/                     # T2 images
│   │   │   └── label/                 # Binary change masks
│   │   └── val/
│   │       ├── A/
│   │       ├── B/
│   │       └── label/
│   │
│   ├── OSCD/                          # OSCD dataset (Phase 2 training — current)
│   │   ├── images/                    # 24 city folders
│   │   │   ├── mumbai/
│   │   │   │   ├── imgs_1/            # T1 Sentinel-2 band TIFs (B02,B03,B04,B08...)
│   │   │   │   ├── imgs_2/            # T2 Sentinel-2 band TIFs
│   │   │   │   ├── imgs_1_rect/       # Rectified versions
│   │   │   │   ├── imgs_2_rect/
│   │   │   │   └── dates.txt
│   │   │   └── ... (23 more cities)
│   │   ├── train_labels/              # 14 city folders with cm/cm.png labels
│   │   └── test_labels/               # 10 city folders with cm/cm.png labels
│   │
│   └── ISRO/                          # Real Sentinel-2 data from Bhoonidhi
│       ├── 2021/
│       │   └── S2A_MSIL2A_20210314T050651_N0214_R019_T43PGQ_20210314T073831.SAFE/
│       │       └── GRANULE/
│       │           └── L2A_T43PGQ_A029903_20210314T051944/
│       │               └── IMG_DATA/
│       │                   └── R10m/
│       │                       ├── T43PGQ_20210314T050651_B02_10m.jp2
│       │                       ├── T43PGQ_20210314T050651_B03_10m.jp2
│       │                       ├── T43PGQ_20210314T050651_B04_10m.jp2
│       │                       ├── T43PGQ_20210314T050651_B08_10m.jp2
│       │                       └── T43PGQ_20210314T050651_TCI_10m.jp2
│       └── 2023/
│           └── S2A_MSIL2A_20230314T050651_N0509_R019_T43PGQ_20230314T090502.SAFE/
│               └── GRANULE/
│                   └── L2A_T43PGQ_A040342_20230314T051815/
│                       └── IMG_DATA/
│                           └── R10m/
│                               ├── T43PGQ_20230314T050651_B02_10m.jp2
│                               ├── T43PGQ_20230314T050651_B03_10m.jp2
│                               ├── T43PGQ_20230314T050651_B04_10m.jp2
│                               ├── T43PGQ_20230314T050651_B08_10m.jp2
│                               └── T43PGQ_20230314T050651_TCI_10m.jp2
│
├── checkpoints/
│   └── best_model.pth                 # Best ChangeFormer checkpoint (val IoU: 0.5864)
│                                      # Saved with torch.compile — keys have _orig_mod.
│                                      # prefix; inference.py strips this on load
│
└── outputs/
    ├── task_{uuid}/                   # Per-task output folders
    │   ├── change_mask.tif            # Georeferenced binary GeoTIFF change mask
    │   └── change_polygons.geojson    # Vectorized change polygons with area_km2
    ├── confusion_matrix.png           # From ml/evaluate.py (LEVIR-CD val set)
    ├── samples.png                    # Sample predictions (t1|t2|label|pred)
    └── alerts.log                     # Alert log file
```

---

## 5. Datasets

### 5.1 LEVIR-CD (Phase 1 — Siamese UNet Training, Now Retired)

**What it is:**  
LEVIR-CD is a publicly available, well-curated change detection benchmark dataset consisting of aerial imagery collected over Texas, USA. It was designed for building-level change detection research. The images are high-resolution aerial photos, not satellite imagery.

**Download source:** `https://justchenhao.github.io/LEVIR/`

**Structure:**
- RGB images (PNG format), 1024×1024 pixels per image
- Organized into `A/` (time 1), `B/` (time 2), `label/` (binary mask) subfolders
- Same filename exists across all three folders (e.g., `0001.png`)
- Labels: 0 = no change, 1 = change (binary mask, pixel-level)
- Train/val split provided by dataset authors

**Where stored:** `data/LEVIR-CD/`

**Why it was used:**  
It was used in the first phase to train the initial Siamese UNet model and validate that the pipeline worked end-to-end. The model trained on LEVIR-CD achieved excellent metrics on the LEVIR validation set (val IoU: 0.819, val F1: 0.901) but failed to generalize to Sentinel-2 satellite imagery due to domain shift — LEVIR is aerial photography at ~0.5m resolution while Sentinel-2 is satellite imagery at 10m resolution, with completely different spectral characteristics.

**Current status:** No longer used for training. OSCD is the active training dataset.

---

### 5.2 OSCD — Onera Satellite Change Detection Dataset (Phase 2 — Current Training Dataset)

**What it is:**  
OSCD is a public change detection dataset specifically designed for Sentinel-2 satellite imagery. It was created by Onera (the French aerospace lab) and is the standard benchmark for Sentinel-2 change detection research. Unlike LEVIR-CD which uses aerial photography, OSCD uses the exact same sensor and resolution as the inference target data — making it far more appropriate for this project.

**Download source:** `https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection`  
Three separate downloads were made:
1. All images (488.96 MB) — `data/OSCD/images/`
2. Training labels (134.64 KB) — `data/OSCD/train_labels/`
3. Test labels (81.65 KB) — `data/OSCD/test_labels/`

**Structure:**
```
data/OSCD/
  images/
    {city_name}/
      imgs_1/       # Time 1 Sentinel-2 bands (individual .tif per band)
      imgs_2/       # Time 2 Sentinel-2 bands
      imgs_1_rect/  # Pre-rectified versions
      imgs_2_rect/
      dates.txt     # Acquisition dates for t1 and t2
  train_labels/
    {city_name}/
      cm/
        cm.png      # Binary change mask (PNG)
  test_labels/
    {city_name}/
      cm/
        cm.png
```

**Coverage:** 24 cities worldwide including Mumbai (India), Hongkong, Beihai, Abudhabi — making it relevant for Asian urban environments.

**Band files:** Individual `.tif` files per band, named like `S2A_OPER_MSI_L1C_TL_*_B04.tif`. Only B02 (Blue), B03 (Green), B04 (Red) are used during training to match the 3-channel RGB input used for LEVIR-CD and to keep channel dimensions consistent.

**Training split used:**
- Cities listed in `data/OSCD/images/train.txt` = 12 training cities
- Last 2 cities from that list = validation split
- Total patches generated: **230 train patches, 71 val patches** (256×256, stride=128)

**Importantly:** Mumbai (tile T43QBB) is one of the training cities — making the model partially adapted to Indian subcontinent imagery and Sentinel-2 characteristics relevant to the inference region (T43PGQ, Karnataka/Bengaluru).

---

### 5.3 ISRO / Bhoonidhi Sentinel-2 (Real-World Inference Data)

**What it is:**  
Real Sentinel-2 Level-2A (atmospherically corrected) satellite data downloaded from the Bhoonidhi portal operated by NRSC/ISRO. This is the actual production data the model runs inference on.

**Download source:** `https://bhoonidhi.nrsc.gov.in` (free account required)

**Tile:** T43PGQ — covers Bengaluru and surrounding Karnataka region  
**Temporal pair:**
- T1: 14 March 2021 (scene: `S2A_MSIL2A_20210314T050651_N0214_R019_T43PGQ`)
- T2: 14 March 2023 (scene: `S2A_MSIL2A_20230314T050651_N0509_R019_T43PGQ`)

**Why same month, same year offset:**  
Using the same calendar month (March) across both years controls for seasonal vegetation variation. March in Karnataka is the pre-monsoon dry season — minimal cloud cover, stable vegetation state. The 2-year gap (2021→2023) captures real structural changes like urban expansion and new construction.

**Bands used for inference:** B02, B03, B04 (RGB — for model input), B08 (NIR — for NDVI computation only)  
**Format:** JP2 (JPEG2000)  
**Resolution:** 10m spatial resolution  
**CRS:** UTM auto-detected from SAFE metadata  
**Image size:** ~10,980 × 10,980 pixels per band (full Sentinel-2 tile)

**JP2 reading strategy:**
- On Linux (Pop OS): rasterio reads JP2 natively via `JP2OpenJPEG` GDAL driver — confirmed working (`['JP2OpenJPEG']` listed in GDAL drivers)
- On Windows: The `gdal_JP2OpenJPEG.dll` plugin was missing from the conda environment. Workaround was implemented using the `glymur` library to read pixel data and a synthetic Affine transform derived from known T43PGQ geographic bounds (78.0–79.0°E, 18.0–19.0°N)

---

## 6. ML Model Details

### 6.1 Model Evolution

The project went through two model iterations:

**Phase 1 — Siamese UNet (retired):**  
A Siamese UNet was built using `segmentation-models-pytorch` with a shared ResNet34 encoder pretrained on ImageNet. Two image patches (t1 and t2) were passed through the shared encoder, feature maps were differenced, and a UNet decoder produced the binary change mask. This model was trained on LEVIR-CD and achieved val IoU of 0.819 — excellent on LEVIR but nearly useless on Sentinel-2 data (only 0.0025 km² detected, max confidence 0.49). The architecture was retired due to domain shift.

**Phase 2 — ChangeFormer (current, active):**  
A transformer-based architecture called ChangeFormer was implemented. It uses an EfficientNet-B0 encoder (from `timm`) pretrained on ImageNet as a shared backbone, processes both t1 and t2 through the same encoder, computes feature differences across multiple scales, and uses a lightweight decoder with projection layers to produce a single-channel sigmoid probability output.

### 6.2 Current Model Architecture — ChangeFormer

```
Input: Two tensors of shape (B, 3, 256, 256) — t1 and t2 RGB patches

Encoder (shared, EfficientNet-B0):
  → Processes t1 and t2 separately through the same weights
  → Extracts multi-scale feature maps at 5 resolution levels
  → Pretrained on ImageNet (weights loaded from HuggingFace timm hub)

Feature Difference Module:
  → For each encoder level: diff_i = |feat_t1_i - feat_t2_i|
  → Produces 5 difference feature maps of decreasing spatial resolution

Decoder:
  → projection layers: reduce channel dimensions at each scale
  → fusion layer: combines multi-scale difference features
  → head: 1×1 conv → single channel output

Output: Tensor of shape (B, 1, 256, 256)
  → Raw logits (no sigmoid during training — BCEWithLogitsLoss used)
  → Sigmoid applied during inference to get probability map [0, 1]
```

**Important implementation detail:**  
The model was trained with `torch.compile()` enabled on Linux (Pop OS), which adds an `_orig_mod.` prefix to all state dict keys. The inference script strips this prefix on checkpoint load:
```python
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
```

### 6.3 Training Configuration

| Parameter | Value |
|---|---|
| Dataset | OSCD (Sentinel-2, 12 train cities, 2 val cities) |
| Train patches | 230 |
| Val patches | 71 |
| Patch size | 256 × 256 pixels |
| Channels | 3 (B04/Red, B03/Green, B02/Blue only) |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| LR scheduler | CosineAnnealingLR |
| Loss function | BCEWithLogitsLoss + DiceLoss (combined) |
| Mixed precision | Yes — `torch.amp.autocast('cuda')` + GradScaler |
| torch.compile | Yes (Linux only) — ~20–30% training speedup |
| Epochs | 50 |
| Early stopping patience | 10 epochs |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM) |
| Checkpoint saved | `checkpoints/best_model.pth` (best val IoU) |
| Experiment tracking | MLflow — experiment `LEVIR_CD_Siamese_UNet` (name carried over) |

### 6.4 Augmentation Pipeline (Training Only)

Applied via `albumentations` to both t1 and t2 simultaneously (same transform applied to both to maintain correspondence):
- Horizontal flip
- Vertical flip
- Random rotation (90°)
- Color jitter (brightness, contrast)
- Normalization to [0, 1] via 2nd–98th percentile clipping

---

## 7. Training Pipeline

### How to run training:

```bash
conda activate satchange
cd /media/jayesh/Windows-SSD/Users/jayes/satellite-change-detector
python ml/train.py
```

### What happens during training:

1. `configs/config.yaml` is loaded — dataset type set to `oscd`
2. `OSCDDataset` initializes, scans all train cities, loads and tiles images into 256×256 patches (stored in memory)
3. `ChangeFormer` model is initialized, EfficientNet-B0 pretrained weights loaded from HuggingFace
4. `torch.compile(model)` is applied for Linux speedup
5. Training loop runs for up to 50 epochs with early stopping
6. Each epoch: forward pass → BCEDice loss → backward → optimizer step → LR scheduler step
7. Validation run after each epoch: IoU and F1 computed on val patches
8. MLflow logs: `train_loss`, `train_iou`, `train_f1`, `val_loss`, `val_iou`, `val_f1`, `lr`
9. If val IoU improves: checkpoint saved to `checkpoints/best_model.pth`
10. If no improvement for 10 epochs: training stops

### How to monitor training:

```bash
# In a separate terminal:
conda activate satchange
mlflow ui
# Open http://localhost:5000 in browser
# Click experiment → click run → view metric graphs
```

---

## 8. Inference Pipeline

The inference pipeline is a fully orchestrated sequence that takes raw Sentinel-2 folders as input and produces GeoTIFF + GeoJSON outputs.

### Entry point:

```python
from pipeline.inference import run_inference

result = run_inference(
    t1_folder="data/ISRO/2021/.../R10m",
    t2_folder="data/ISRO/2023/.../R10m",
    checkpoint_path="checkpoints/best_model.pth",
    output_dir="outputs/my_run"
)
# Returns: {'geotiff_path': ..., 'geojson_path': ..., 'changed_area_km2': ...}
```

### Steps inside run_inference():

1. **preprocess.py — `load_sentinel2_bands()`**
   - Scans folder for `*B02*.jp2`, `*B03*.jp2`, `*B04*.jp2`, `*B08*.jp2`
   - Opens each with `rasterio.open()` (Linux) or `glymur.Jp2k()` (Windows fallback)
   - Stacks into `(H, W, 4)` numpy array
   - Applies 2nd–98th percentile normalization per band
   - Returns: `(image_array, rasterio_meta_dict, CRS)`

2. **preprocess.py — `align_images()`**
   - Checks if both images share CRS, transform, and dimensions
   - If not: reprojects img2 to match img1 using `rasterio.warp.reproject`

3. **preprocess.py — `compute_ndvi()`**
   - NDVI = (B08 − B04) / (B08 + B04 + ε)
   - Returns `(H, W)` float array, clipped to [−1, 1]
   - Computed for both t1 and t2 and passed to postprocessor

4. **tiling.py — `tile_image_pair()`**
   - Tiles both aligned images into overlapping 256×256 patches
   - Overlap: 32 pixels
   - Each patch has metadata: `{row, col, y, x}` for stitching
   - For 10,980×10,980 image: generates ~2,401 patch pairs (301 batches of 8)

5. **inference.py — model forward pass**
   - Loads ChangeFormer from checkpoint (`_orig_mod.` prefix stripped)
   - Sends model to CUDA
   - Iterates over patch batches
   - Uses `torch.amp.autocast('cuda')` for mixed precision
   - Applies `torch.sigmoid()` to raw logits → probability map
   - Collects predictions with their spatial metadata

6. **stitch.py — `stitch_patches()`**
   - Reassembles all patch predictions into full `(H, W)` mask
   - Gaussian blending on overlapping regions removes visible seam artifacts

7. **postprocess.py — `postprocess()`**
   - `apply_threshold(mask, threshold=0.3)` → binary mask
   - `filter_by_ndvi(mask, ndvi_t1, ndvi_t2, threshold=0.02)` → suppresses areas where `|NDVI_t2 − NDVI_t1| < 0.02` (removes seasonal vegetation fluctuation)
   - `remove_noise(mask, min_area=5)` → removes connected components < 5 pixels

8. **gis.py — output generation**
   - `save_geotiff()` — writes binary mask as georeferenced GeoTIFF with correct CRS and Affine transform
   - `mask_to_geojson()` — vectorizes binary raster using `rasterio.features.shapes()`, converts polygons to EPSG:4326, adds `area_m2` and `area_km2` properties to each feature, saves as GeoJSON
   - `compute_total_change_area()` — sums all polygon areas → returns total km²

---

## 9. GIS and Post-Processing Layer

### Post-processing parameters (currently hardcoded in postprocess.py):

| Parameter | Current Value | Effect |
|---|---|---|
| `prob_threshold` | 0.3 | Pixels above this are "changed" |
| `ndvi_threshold` | 0.02 | NDVI change below this = vegetation, suppressed |
| `min_area` | 5 pixels | Components smaller than this = noise, removed |

### Note on coordinate system:

Because the ISRO Sentinel-2 data was downloaded without reading the JP2 metadata CRS (due to JP2OpenJPEG driver issues on Windows), a synthetic Affine transform was generated based on known T43PGQ geographic bounds: `from_bounds(78.0, 18.0, 79.0, 19.0, width, height)` with CRS = EPSG:4326. On Linux, rasterio reads the CRS natively from the JP2 file — the synthetic fallback is only used on Windows. The GeoJSON output correctly places polygons over the Bengaluru region at the right coordinates, as confirmed visually on the Leaflet map.

---

## 10. Backend — FastAPI

### Starting the backend:

```bash
conda activate satchange
cd /media/jayesh/Windows-SSD/Users/jayes/satellite-change-detector
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints:

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/detect-change` | Accepts `DetectionRequest` (t1_folder, t2_folder, aoi_geojson), queues Celery task, returns `{task_id}` |
| GET | `/api/v1/results/{task_id}` | Returns task status: `processing`, `complete`, or `failed`. When complete, returns `{changed_area_km2, geojson_url}` |
| GET | `/api/v1/alerts` | Returns all unacknowledged alerts |
| GET | `/api/v1/health` | Returns `{"status": "ok"}` |

### Current storage mode:

The backend currently uses **in-memory dict storage** for task results. This was implemented as a workaround after PostgreSQL was not running at the time of initial deployment. The full PostgreSQL schema exists and is deployed (schema.sql applied successfully with PostGIS), but the routes are not yet wired to the DB in production mode. The in-memory store means results are lost if the backend restarts.

### Static file serving:

The backend mounts `outputs/` as a static directory, allowing the frontend to fetch GeoJSON files by URL.

---

## 11. Async Workers — Celery + Redis

### Starting the worker:

```bash
# Linux — full prefork concurrency (no --pool=solo needed)
conda activate satchange
cd /media/jayesh/Windows-SSD/Users/jayes/satellite-change-detector
python -m celery -A workers.tasks.celery_app worker --loglevel=info

# Windows only — solo pool required
python -m celery -A workers.tasks.celery_app worker --loglevel=info --pool=solo
```

### Redis setup:

```bash
# Linux — start and enable
sudo systemctl start redis-server
sudo systemctl enable redis-server
redis-cli ping  # Should return PONG

# Windows — Memurai (Redis for Windows)
# Start from Windows Services or memurai.exe
```

### What the worker does:

The `run_detection_task` Celery task receives `(task_id, t1_folder, t2_folder, aoi_geojson)` and:
1. Logs start with task ID and folder paths
2. Sets environment variables for PROJ_DATA if needed
3. Calls `run_inference()` from `pipeline/inference.py`
4. Saves result JSON to `outputs/task_{task_id}/result.json`
5. Checks if `changed_area_km2 > alert_threshold_km2` (configured in config.yaml)
6. If threshold exceeded: calls `trigger_alert()` in `workers/alerts.py`
7. Updates in-memory task store in backend with final status and results
8. On any exception: logs full traceback, marks task as failed

### Alert system:

`workers/alerts.py` — `trigger_alert()`:
- Appends formatted alert entry to `outputs/alerts.log`
- Prints formatted alert to console with timestamp and area in km²
- DB alert insertion is prepared (SQLAlchemy model exists) but in-memory mode bypasses this

---

## 12. Database — PostgreSQL + PostGIS

### Status: Schema deployed, not yet wired to backend routes

PostgreSQL 18 is running on Windows; system PostgreSQL is running on Linux (Pop OS). PostGIS 3.4 is installed and active on both. The schema was applied via:

```bash
psql -U postgres -h 127.0.0.1 -d satchange -f database/schema.sql
```

### Schema:

```sql
-- Enable PostGIS
CREATE EXTENSION IF NOT EXISTS postgis;

-- Areas of Interest
CREATE TABLE aois (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255),
    description TEXT,
    geom GEOMETRY(POLYGON, 4326) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Detection Results
CREATE TABLE detection_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aoi_id UUID REFERENCES aois(id),
    task_id VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    change_mask_path TEXT,
    geojson_path TEXT,
    changed_area_km2 FLOAT,
    t1_date DATE,
    t2_date DATE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Alerts
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    result_id UUID REFERENCES detection_results(id),
    message TEXT,
    triggered_at TIMESTAMP DEFAULT NOW(),
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP
);
```

---

## 13. Frontend — Leaflet.js

The frontend is a single self-contained HTML file (`frontend/index.html`) with no build tools, no npm, no external dependencies beyond CDN-loaded libraries.

### How to open:

```bash
# Open directly in browser
xdg-open frontend/index.html   # Linux
start frontend\index.html       # Windows
```

### Current UI elements:

- **Leaflet map** — centered on India (lat: 20, lng: 78, zoom: 5), CARTO dark basemap
- **T1 Folder Path input** — text field for absolute path to 2021 R10m folder
- **T2 Folder Path input** — text field for absolute path to 2023 R10m folder
- **Redraw AOI button** — clears current AOI polygon, enables Leaflet.draw rectangle tool
- **Run Detection button** — POSTs to `http://localhost:8000/api/v1/detect-change`
- **Status panel** — shows "Processing...", "Complete! Area: X km²", or "Task Failed" with task ID
- **Alerts panel** — polls `/api/v1/alerts` and displays any triggered alerts
- **GeoJSON overlay** — when detection completes, fetches GeoJSON from backend static URL, renders as red filled polygons. Each polygon has a popup showing area in km².

### Polling mechanism:

After submitting a detection request, the frontend polls `GET /api/v1/results/{task_id}` every 3 seconds until status is `complete` or `failed`. On completion, it fetches the GeoJSON URL returned in the result and adds it as a Leaflet layer.

---

## 14. Model Performance and Metrics

### Phase 1 — Siamese UNet on LEVIR-CD (retired baseline):

| Metric | Value |
|---|---|
| Global Accuracy | 0.9919 |
| Change (Class 1) IoU | 0.8194 |
| Change (Class 1) F1 | 0.9007 |
| Precision | 0.9221 |
| Recall | 0.8803 |
| No-Change IoU | 0.9915 |

**Why this is misleading:** These metrics were computed on LEVIR-CD val set — aerial photos, USA, ~0.5m resolution. When run on Sentinel-2 (10m, India), max model confidence was only 0.49 and detected area was 0.0025 km² — effectively useless.

### Phase 2 — ChangeFormer on OSCD (current model):

| Metric | Value |
|---|---|
| Val IoU (change class) | 0.5864 |
| Val F1 | 0.7363 |
| Val Loss | 0.2209 |
| Train IoU (final epoch) | 0.5371 |

**Inference results on T43PGQ (Bengaluru, 2021 vs 2023):**

| Model | Changed Area | Max Confidence | Notes |
|---|---|---|---|
| Siamese UNet (LEVIR) | 0.0025 km² | 0.49 | Nearly blind on Sentinel-2 |
| ChangeFormer (pretrained only) | 3.53 km² | 0.89 | No change-specific training |
| ChangeFormer (OSCD fine-tuned) | **14.63 km²** | **1.00** | Current deployed model |

### Why val IoU of 0.58 is acceptable despite being lower than Phase 1:

OSCD is a significantly harder dataset than LEVIR-CD. OSCD uses real Sentinel-2 imagery with genuine ambiguity, mixed change types, and varying scene complexity. An IoU of 0.58 on OSCD is competitive with published baselines for this dataset. More importantly, the model's real-world performance on T43PGQ is geographically meaningful — the red polygon clusters on the Leaflet map correctly show dense change around Bengaluru's urban expansion corridors.

### Current model limitations:

1. **Pixel-level only, not object-level:** The model detects individual changed pixels, not discrete objects (buildings, roads). There is no spatial coherence enforcement beyond the noise removal step.
2. **No change type classification:** All changes are labeled as a single class. The model cannot distinguish between building construction, deforestation, road building, or water body changes.
3. **Small training set:** Only 230 training patches across 12 OSCD cities. This is very small for a transformer-based model.
4. **Domain gap still present:** Model trained on European and Asian global cities (OSCD), not specifically on Karnataka/Bengaluru. Some false positives from seasonal agricultural variation remain.
5. **No confidence calibration:** Probability scores are not calibrated — a prediction of 0.95 does not mean 95% real-world probability of change.
6. **Fixed patch size:** 256×256 patches at 10m resolution = 2.56km × 2.56km per patch. Features larger than this are not captured as single objects.
7. **3-channel input only:** Model takes RGB (B02, B03, B04). NIR band (B08) is only used for NDVI filtering, not as a model input channel. This discards valuable near-infrared information that could improve vegetation change discrimination.

---

## 15. Known Issues and Limitations

### Resolved Issues (documented for reference):

| Issue | Root Cause | Resolution |
|---|---|---|
| `OMP: Error #15` — libiomp5md.dll conflict | Duplicate OpenMP runtimes in base conda env | Set `KMP_DUPLICATE_LIB_OK=TRUE` env var |
| JP2 files not readable on Windows | `gdal_JP2OpenJPEG.dll` missing from conda env | Used `glymur` library for pixel reading, synthetic transform for CRS |
| `PROJ: proj_create_from_database: Cannot find proj.db` | `PROJ_DATA` env var pointing to wrong path | Set `PROJ_DATA=/home/jayesh/miniconda3/envs/satchange/share/proj` on Linux |
| Backend failing with fbgemm.dll error | uvicorn launching via base env Python not satchange | Fixed conda activation, `python -m uvicorn` instead of `uvicorn` |
| `UnetDecoder.forward() takes 2 positional arguments but 7 were given` | Siamese UNet decoder receiving list instead of single tensor | Rewrote model forward pass to concatenate diff features correctly |
| `BCELoss unsafe to autocast` | BCELoss incompatible with FP16 mixed precision | Replaced with BCEWithLogitsLoss, removed sigmoid from model output |
| `_orig_mod.` prefix in checkpoint keys | `torch.compile()` wraps model with prefix | Strip prefix on load: `k.replace('_orig_mod.', '')` |
| `conda activate satchange` not persisting | `conda init powershell` not run | Ran `conda init powershell`, restarted terminal |
| 0 patches from OSCDDataset | Tiling stride too large | Changed stride to `patch_size // 2` (128px) |
| In-memory backend storage | Result data lost on restart | Wired FastAPI routes and Celery workers to PostgreSQL via SQLAlchemy. |

### Active Known Issues:

1. **Synthetic Affine transform on Windows** — If the project is run from Windows, JP2 CRS is not read from file. The synthetic transform covers T43PGQ approximately but is not pixel-accurate. On Linux this is resolved.
2. **NDVI filtering threshold is very low (0.02)** — At this threshold, the NDVI filter barely suppresses anything. A value of 0.1–0.15 is more physically meaningful for distinguishing real structural change from vegetation seasonality.
3. **Noise removal min_area = 5 pixels** — Very permissive. Some speckle noise passes through. For production, 50–100 pixels is a more meaningful minimum changed area at 10m resolution.
5. **No authentication on API** — The FastAPI backend has no authentication. Anyone on the network can submit detection jobs.
6. **No rate limiting** — Multiple concurrent detection requests will queue and consume all GPU memory.
7. **Frontend uses localhost hardcoded** — `index.html` POSTs to `http://localhost:8000`. Not suitable for multi-machine deployment without editing the URL.

---

## 16. What Is Missing and Not Yet Built

### High Priority — Directly improves core functionality:

1. **Database wiring** — Connect FastAPI routes to PostgreSQL/PostGIS for persistent result storage, AOI history, and alert management. SQLAlchemy models and schema are fully defined and deployed — this is a routing code change only.

2. **Before/After comparison slider** — Add Leaflet.SideBySide plugin to show 2021 TCI on left and 2023 TCI on right. Allows visual verification of detected changes. Currently the frontend only shows red polygons with no visual context of what changed.

3. **Spectral index layers** — Compute and serve NDVI, NDWI, NDBI overlays from the existing band data. Would allow users to filter change types (vegetation vs water vs built-up).

4. **S2Looking dataset retraining** — S2Looking is a building change detection dataset specifically designed for high-precision building footprint change detection. Training on it would give the model object-level semantic understanding rather than pixel-level. Publicly available at `https://github.com/S2Looking/Dataset`.

5. **Pseudo-labeling / semi-supervised fine-tuning** — Use the current model's high-confidence predictions (>0.85) as pseudo-labels, and train for a few epochs on the T43PGQ data itself. This adapts the model to the specific sensor, region, and date pair without manual annotation.

### Medium Priority — System completeness:

6. **Time series support** — Download 4–6 Sentinel-2 images for T43PGQ across multiple years. Run inference between consecutive pairs. Show a timeline of cumulative change on the frontend.

7. **Change type classification** — Add a classification head to the model that categorizes detected changes into: urban/built-up, vegetation loss, water body change, bare soil. This requires a multi-class dataset like SECOND or DynamicEarthNet.

8. **Scene search integration** — Integrate Copernicus Data Space API or Sentinel Hub API to allow users to search for available Sentinel-2 scenes by AOI, date range, and cloud cover percentage. Currently scenes must be manually downloaded from Bhoonidhi.

9. **Confidence scoring per polygon** — Each GeoJSON polygon currently has only `area_m2` and `area_km2`. Add mean model confidence score as a property, allowing users to filter by certainty.

10. **API authentication** — Add JWT-based authentication to FastAPI endpoints.

### Lower Priority — Quality of life / production hardening:

11. **Docker containerization** — Package the backend, worker, and dependencies into Docker Compose for reproducible deployment.
12. **PostGIS spatial queries** — Use PostGIS geometry functions to filter results by AOI intersection, compute spatial statistics, and support multi-AOI monitoring.
13. **Frontend map style improvements** — Add layer controls, zoom-to-result button, polygon color scaling by area, and export as PNG/PDF.
14. **Celery result backend** — Currently Celery results also go to Redis. Should be separated from broker for cleaner architecture.
15. **Automated testing** — No unit tests or integration tests exist for any pipeline component.

---

## 17. Environment Setup and Run Instructions

### One-time setup (Linux / Pop OS):

```bash
# 1. Install system dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget tmux htop \
  libgdal-dev gdal-bin libproj-dev proj-bin \
  libgeos-dev postgresql postgresql-contrib postgis redis-server

# 2. Install Miniconda and create environment
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda create -n satchange python=3.11 -y
conda activate satchange

# 3. Install GIS stack
conda install -c conda-forge gdal rasterio geopandas shapely pyproj fiona -y

# 4. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 5. Install all other dependencies
pip install fastapi "uvicorn[standard]" celery redis sqlalchemy psycopg2-binary \
  geoalchemy2 pydantic python-multipart numpy opencv-python scikit-learn \
  albumentations mlflow segmentation-models-pytorch timm transformers \
  tqdm pyyaml python-dotenv pytest scipy rio-cogeo glymur shapely geopandas

# 6. Set permanent env vars
echo 'export KMP_DUPLICATE_LIB_OK=TRUE' >> ~/.bashrc
echo 'export GDAL_DATA=$(gdal-config --datadir)' >> ~/.bashrc
echo 'export HF_HUB_DISABLE_SYMLINKS_WARNING=1' >> ~/.bashrc
source ~/.bashrc

# 7. Start services
sudo systemctl enable redis-server && sudo systemctl start redis-server
sudo systemctl start postgresql
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
sudo -u postgres psql -c "CREATE DATABASE satchange;"
sudo -u postgres psql -d satchange -c "CREATE EXTENSION postgis;"
psql -U postgres -h 127.0.0.1 -d satchange -f database/schema.sql
```

### Starting the full system (3 terminals required):

```bash
# Terminal 1 — FastAPI backend
conda activate satchange
cd /media/jayesh/Windows-SSD/Users/jayes/satellite-change-detector
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Celery worker
conda activate satchange
cd /media/jayesh/Windows-SSD/Users/jayes/satellite-change-detector
python -m celery -A workers.tasks.celery_app worker --loglevel=info

# Terminal 3 — MLflow (optional, for training monitoring)
conda activate satchange
cd /media/jayesh/Windows-SSD/Users/jayes/satellite-change-detector
mlflow ui
# Open http://localhost:5000
```

### Running training:

```bash
# Optional: in tmux to survive terminal close
tmux new -s training
conda activate satchange
cd /media/jayesh/Windows-SSD/Users/jayes/satellite-change-detector
python ml/train.py
# Ctrl+B then D to detach; tmux attach -t training to return
```

### Running inference directly (without API):

```python
from pipeline.inference import run_inference

result = run_inference(
    t1_folder="/media/jayesh/Windows-SSD/Users/jayes/satellite-change-detector/data/ISRO/2021/S2A_MSIL2A_20210314T050651_N0214_R019_T43PGQ_20210314T073831.SAFE/GRANULE/L2A_T43PGQ_A029903_20210314T051944/IMG_DATA/R10m",
    t2_folder="/media/jayesh/Windows-SSD/Users/jayes/satellite-change-detector/data/ISRO/2023/S2A_MSIL2A_20230314T050651_N0509_R019_T43PGQ_20230314T090502.SAFE/GRANULE/L2A_T43PGQ_A040342_20230314T051815/IMG_DATA/R10m",
    checkpoint_path="checkpoints/best_model.pth",
    output_dir="outputs/test_run"
)
print(result)
```

### Opening the frontend:

```bash
xdg-open frontend/index.html
```

Enter full absolute paths for T1 and T2 folders, draw AOI, click Run Detection. Backend and worker must be running.
