# Project Roadmap: Satellite Change Detection System
# Full Development Plan — All Phases (Version 2.0)

**Document Version:** 2.0 (Updated — includes AOI fix, email alerts, remaining original phases)
**Project:** Satellite Change Detection System
**Current State:** ChangeFormer trained on OSCD, basic Leaflet frontend, FastAPI backend, Celery workers, 14.63 km² detected over Bengaluru (T43PGQ, 2021 vs 2023). Two critical bugs: AOI clipping not working, email alerts not implemented.
**Target State:** LandViewer-grade satellite platform with AOI-clipped results, email alerts, comparison slider, spectral indices, time series, scene search, and building-level change detection.

---

## Table of Contents

1. [Datasets Roadmap](#1-datasets-roadmap)
2. [IMMEDIATE Fix A — AOI Clipping Bug](#2-immediate-fix-a--aoi-clipping-bug)
3. [IMMEDIATE Fix B — Email Alert System](#3-immediate-fix-b--email-alert-system)
4. [Phase 0 — Pseudo-Labeling Fine-Tune on T43PGQ](#4-phase-0--pseudo-labeling-fine-tune-on-t43pgq)
5. [Phase 0B — Final Production Inference](#5-phase-0b--final-production-inference)
6. [Phase 0C — Database Wiring and Linux JP2 Cleanup](#6-phase-0c--database-wiring-and-linux-jp2-cleanup)
7. [Phase 1 — Frontend Upgrade](#7-phase-1--frontend-upgrade-comparison-slider--layer-panel)
8. [Phase 2 — Spectral Indices](#8-phase-2--spectral-indices-ndvi-ndwi-ndbi)
9. [Phase 3 — Time Series Analysis](#9-phase-3--time-series-analysis)
10. [Phase 4 — Scene Search Integration](#10-phase-4--scene-search-integration)
11. [Phase 5 — Better Model with S2Looking](#11-phase-5--better-model-with-s2looking)
12. [CLI Tools — When to Use Which](#12-cli-tools--when-to-use-which)
13. [Complete Future Folder Structure](#13-complete-future-folder-structure)
14. [Execution Order Summary](#14-execution-order-summary)

---

## 1. Datasets Roadmap

### 1.1 OSCD — Onera Satellite Change Detection Dataset (Current — Active Baseline)

**What it is:**
OSCD is the standard academic benchmark for Sentinel-2 satellite change detection. Created by the French aerospace agency Onera, it consists of 24 city pairs worldwide, each with two Sentinel-2 multispectral images at different dates and a binary pixel-level change mask (0 = no change, 1 = change) at 10m resolution.

**Why we continue using it:**
It is the only publicly available labeled dataset built specifically on Sentinel-2 imagery matching our inference sensor, resolution, and band configuration. The Mumbai city pair (tile T43QBB) is especially relevant as Indian subcontinent Sentinel-2 data.

**Current storage:** `data/OSCD/` — already downloaded, no action needed.
**Phase:** Active baseline throughout all training phases.

---

### 1.2 S2Looking — Building Change Detection Dataset (Phase 5)

**What it is:**
S2Looking is a large-scale building change detection dataset on Sentinel-2 imagery with 5,000 image pairs at 1024×1024 pixels, specifically labeling building appearance and disappearance across multiple cities including Asian urban areas.

**Why we use it:**
Enables training a model that detects discrete building footprints rather than individual pixels, directly addressing the current model's core limitation of pixel-level-only detection with no semantic understanding.

**Download:** `https://github.com/S2Looking/Dataset`
**Storage:** `data/S2Looking/train/`, `data/S2Looking/val/`, `data/S2Looking/test/`
**Phase:** Phase 5.

---

### 1.3 MUMUCD — Multi-Modal Urban Change Detection (Future Research Only)

**What it is:**
Combines optical Sentinel-2 with SAR Sentinel-1 data, enabling cloud-penetrating change detection. Useful for monsoon season when optical imagery is cloud-contaminated over Indian subcontinent.

**Download:** `https://github.com/multimodal-cd/MUMUCD`
**Phase:** Post Phase 5, future research — not in current roadmap.

---

## 2. IMMEDIATE Fix A — AOI Clipping Bug

### Problem Statement

**Current broken behavior:** When the user draws an AOI polygon on the Leaflet map and clicks Run Detection, the system completely ignores the AOI. It runs inference on the entire T43PGQ tile (10,980 × 10,980 pixels, approximately 110km × 110km) and returns ALL detected changes across the whole image. The 14.63 km² result and hundreds of red dots cover all of greater Bengaluru — not the specific area the user drew.

**Root cause:** The AOI GeoJSON polygon drawn on the frontend is sent to the backend in the `DetectionRequest` body, passes through to the Celery task parameters, but when `run_inference()` is called in `pipeline/inference.py`, the `aoi_geojson` parameter is never used. The model tiles and processes all 2,401 patches regardless of where the AOI is drawn.

**Why this matters critically:** The entire purpose of drawing an AOI is to get targeted results. A user selecting a 5km² area around Whitefield should see only changes within that boundary. Without AOI clipping, the system cannot be used for any targeted monitoring use case.

### Fix Timeline: 1 day

### Tech Stack for This Fix

| Technology | Purpose |
|---|---|
| shapely | AOI polygon geometry — intersection, containment |
| rasterio.mask | Clip raster array to polygon boundary |
| geopandas | CRS reprojection of AOI from EPSG:4326 to image UTM CRS |
| pyproj | Coordinate transformation |

### How Correct AOI Clipping Must Work

```
User draws AOI polygon on map → GeoJSON string sent in POST body
              ↓
Backend receives AOI → passes to Celery task
              ↓
Worker receives AOI → passes to run_inference()
              ↓
preprocess: reproject AOI from EPSG:4326 → image CRS (UTM)
              ↓
rasterio.mask.mask() clips both T1 and T2 to AOI bounding box
              ↓
Only the clipped region is tiled → far fewer patches
              ↓
Model runs inference only on AOI patches
              ↓
Predictions stitched back → apply exact AOI polygon mask
  (not just bounding box — pixels outside polygon set to 0)
              ↓
GIS: vectorize only polygons inside AOI boundary
              ↓
Frontend: red dots appear ONLY inside drawn polygon
```

### Step-by-Step Implementation

#### Step 1 — Confirm the bug

```bash
grep -n "aoi" pipeline/inference.py
grep -n "aoi" workers/tasks.py
# If aoi_geojson is received but never passed to any clipping function → bug confirmed
```

#### Step 2 — Gemini CLI Prompt for AOI Fix

```
@pipeline/inference.py @pipeline/preprocess.py @pipeline/gis.py @workers/tasks.py @backend/routes.py @backend/schemas.py

CRITICAL BUG FIX: AOI polygon drawn by user is completely ignored.
System runs on full 110km×110km tile instead of user-selected area.
Fix this end-to-end across all files.

CHANGE 1 — pipeline/preprocess.py, add two new functions:

def clip_image_to_aoi(
    image_array: np.ndarray,
    meta: dict,
    aoi_geojson_str: str
) -> tuple[np.ndarray, dict]:
    """
    Clip a (H, W, C) image array to the bounding box of an AOI polygon.
    
    Steps:
    1. Parse aoi_geojson_str as dict
    2. Extract geometry (type: Polygon or Feature with geometry)
    3. Create shapely polygon from coordinates
    4. Reproject polygon from EPSG:4326 to meta['crs'] using pyproj Transformer
       pyproj.Transformer.from_crs('EPSG:4326', meta['crs'], always_xy=True)
    5. Use rasterio.mask.mask() with [reprojected_polygon], crop=True, all_touched=True
       NOTE: rasterio.mask.mask expects (C, H, W) format — transpose before, transpose back after
    6. Update meta: width, height, transform from the mask output
    7. Return (clipped_array_HWC, updated_meta)
    
    If aoi_geojson_str is None or empty string: return (image_array, meta) unchanged.
    If reprojection fails: log warning and return original (graceful degradation).
    """

def create_aoi_pixel_mask(
    image_shape: tuple,
    meta: dict,
    aoi_geojson_str: str
) -> np.ndarray:
    """
    Burn AOI polygon into a binary numpy array matching image_shape (H, W).
    Returns array with 1 inside AOI polygon, 0 outside.
    Uses rasterio.features.geometry_mask() with invert=True.
    Reprojects AOI to image CRS same as clip_image_to_aoi.
    If aoi_geojson_str is None: return np.ones(image_shape) — all pixels valid.
    """

CHANGE 2 — pipeline/inference.py:
Modify run_inference() signature to accept aoi_geojson: Optional[str] = None

After loading both images with load_sentinel2_bands():
  If aoi_geojson is not None and aoi_geojson.strip() != '':
    img1, meta1 = clip_image_to_aoi(img1, meta1, aoi_geojson)
    img2, meta2 = clip_image_to_aoi(img2, meta2, aoi_geojson)
    Log: f"AOI clipping applied. Clipped image shape: {img1.shape}"
  Else:
    Log: "No AOI provided — processing full tile"

After stitching predictions back (full prediction mask):
  If aoi_geojson is not None:
    aoi_mask = create_aoi_pixel_mask(prediction.shape, meta1, aoi_geojson)
    prediction = prediction * aoi_mask  # Zero out pixels outside exact AOI boundary

Continue with postprocess → gis as normal using the masked prediction.

CHANGE 3 — workers/tasks.py:
Verify aoi_geojson is extracted from task parameters and passed to run_inference().
Add log line: f"AOI GeoJSON received: {'yes' if aoi_geojson else 'no'}"

CHANGE 4 — backend/schemas.py:
Ensure DetectionRequest has:
  aoi_geojson: Optional[str] = None

CHANGE 5 — backend/routes.py:
Ensure aoi_geojson from DetectionRequest is passed to Celery task call.

IMPORTANT NOTES:
- rasterio.mask.mask() expects geometries in SAME CRS as raster
- Shapely polygon must be reprojected before passing to rasterio.mask
- After clip: image is smaller → meta width/height/transform must all update
- The Affine transform of clipped image reflects its new geographic origin
- Test: AOI of 5×5 km → roughly 20×20 image patches (not 2401)
- All functions must have full type hints and Google docstrings
- No placeholders — complete working implementation
```

#### Step 3 — Test the AOI fix

```bash
conda activate satchange
cd /media/jayesh/Windows-SSD/Users/jayes/satellite-change-detector

python -c "
from pipeline.inference import run_inference
import json

# Small 5km x 5km AOI over Electronic City, Bengaluru
aoi = json.dumps({
    'type': 'Feature',
    'geometry': {
        'type': 'Polygon',
        'coordinates': [[
            [77.65, 12.83], [77.72, 12.83],
            [77.72, 12.89], [77.65, 12.89],
            [77.65, 12.83]
        ]]
    }
})

result = run_inference(
    t1_folder='data/ISRO/2021/S2A_MSIL2A_20210314T050651_N0214_R019_T43PGQ_20210314T073831.SAFE/GRANULE/L2A_T43PGQ_A029903_20210314T051944/IMG_DATA/R10m',
    t2_folder='data/ISRO/2023/S2A_MSIL2A_20230314T050651_N0509_R019_T43PGQ_20230314T090502.SAFE/GRANULE/L2A_T43PGQ_A040342_20230314T051815/IMG_DATA/R10m',
    checkpoint_path='checkpoints/best_model.pth',
    output_dir='outputs/aoi_test',
    aoi_geojson=aoi
)
print(result)
# Expected: changed_area_km2 << 14.63 (much smaller, only Electronic City)
# Expected: GeoJSON polygons all within the drawn 5km x 5km box
"
```

#### Step 4 — Verify visually in frontend

1. Start backend + worker
2. Open `frontend/index.html`
3. Draw small AOI polygon over Whitefield only
4. Run Detection
5. Red dots must appear ONLY inside drawn polygon — not across all Bengaluru
6. Changed area must be proportionally smaller than 14.63 km²

---

## 3. IMMEDIATE Fix B — Email Alert System

### Problem Statement

**Current broken behavior:** The alert system logs to `outputs/alerts.log` and prints to Celery console when changed area exceeds the threshold, but no email is sent. The user gets zero real-world notification and must keep the app open or manually check log files.

**What needs to be built:**
1. SMTP-based email sending when change threshold is exceeded
2. User email input in the frontend — user enters their address before running detection
3. Configurable threshold — user sets minimum km² to trigger alert
4. Formatted HTML email with detection summary and result link

### Fix Timeline: 1 day

### Tech Stack for This Fix

| Technology | Purpose |
|---|---|
| smtplib (Python stdlib) | Send emails via SMTP — no external library needed |
| email.mime.multipart | Compose HTML + plain text email |
| Gmail SMTP | Free delivery via Google (500/day limit) |
| FastAPI | Receive email + threshold in DetectionRequest |
| Frontend HTML | Email input field and threshold slider |

### Gmail App Password Setup

1. Go to `https://myaccount.google.com/security`
2. Enable 2-Step Verification if not already active
3. Search for "App passwords" in account settings
4. Create new app password → name it "SatChange"
5. Copy the 16-character password (shown once only)

### Config Setup

Add to `configs/config.yaml`:
```yaml
alerts:
  threshold_km2: 1.0
  email_enabled: true
  smtp_host: "smtp.gmail.com"
  smtp_port: 587
  smtp_user: "your_gmail@gmail.com"
  smtp_password: "your_16char_app_password"
  sender_name: "Satellite Change Detector"
```

Create `.env` file (never commit to git — add to `.gitignore`):
```bash
echo "SMTP_USER=your_gmail@gmail.com" >> .env
echo "SMTP_PASSWORD=your_app_password" >> .env
echo "ALERT_THRESHOLD_KM2=1.0" >> .env
echo ".env" >> .gitignore
```

### Gemini CLI Prompt for Email Alert System

```
@workers/alerts.py @workers/tasks.py @backend/routes.py @backend/schemas.py @frontend/index.html @configs/config.yaml

Implement a complete email alert system. Users enter their email in the frontend.
When detected change area exceeds their threshold, an HTML email is sent to them.

CHANGE 1 — workers/alerts.py (complete rewrite):

import smtplib
import os
import yaml
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from datetime import datetime
from typing import Optional

def load_alert_config() -> dict:
    """
    Load alert config from configs/config.yaml.
    Override smtp_user and smtp_password with env vars SMTP_USER and SMTP_PASSWORD
    if they are set (for security — avoid hardcoding credentials).
    Return config dict with keys: threshold_km2, email_enabled, smtp_host,
    smtp_port, smtp_user, smtp_password, sender_name.
    """

def compose_alert_email_html(
    recipient_email: str,
    task_id: str,
    changed_area_km2: float,
    tile: str,
    t1_date: str,
    t2_date: str,
    threshold_km2: float,
    result_url: str
) -> str:
    """
    Return complete HTML string for the alert email body.
    
    Design requirements:
    - Dark professional theme: background #0d1117, card background #161b22
    - Header: satellite emoji + "Change Detection Alert"
    - Summary card with table:
        Row 1: "Tile" | "T43PGQ (Bengaluru, India)"
        Row 2: "Detection Period" | "{t1_date} → {t2_date}"  
        Row 3: "Changed Area" | "{changed_area_km2:.3f} km²" (in red bold)
        Row 4: "Alert Threshold" | "{threshold_km2} km²"
        Row 5: "Detection Time" | current timestamp
    - Red warning badge: "⚠️ THRESHOLD EXCEEDED"
    - Description: "A significant land change of {km²} km² was detected in 
                    satellite tile {tile}, exceeding your configured alert 
                    threshold of {threshold} km². This may indicate construction,
                    deforestation, or other structural changes."
    - "View Results" button: styled anchor tag linking to result_url
      Background: #e74c3c, text: white, padding: 12px 24px, border-radius: 6px
    - Footer: "This alert was generated by Satellite Change Detection System"
               "To stop receiving alerts, remove your email from the application."
    
    Use only inline CSS (email clients strip style tags).
    """

def send_alert_email(
    recipient_email: str,
    task_id: str,
    changed_area_km2: float,
    tile: str = "T43PGQ",
    t1_date: str = "2021-03-14",
    t2_date: str = "2023-03-14",
    threshold_km2: float = 1.0,
    result_url: str = ""
) -> bool:
    """
    Send HTML alert email via SMTP.
    
    Steps:
    1. Load config via load_alert_config()
    2. If email_enabled is False: log "Email disabled in config" and return False
    3. Compose MIMEMultipart('alternative') message
    4. Set From, To, Subject headers
       Subject: "🛰️ Satellite Change Alert — {changed_area_km2:.2f} km² detected in {tile}"
    5. Attach plain text fallback: "Change detected: {km²} km². View: {url}"
    6. Attach HTML body from compose_alert_email_html()
    7. Connect to smtp_host:smtp_port with SMTP()
    8. Call starttls()
    9. Login with smtp_user and smtp_password
    10. Send message
    11. Quit connection
    12. Return True
    
    Wrap entire send in try/except Exception:
    - If fails: log full error, return False
    - NEVER raise — email failure must never crash the Celery worker
    """

def trigger_alert(
    result_id: str,
    changed_area_km2: float,
    threshold_km2: float,
    recipient_email: Optional[str] = None,
    task_id: str = "",
    tile: str = "T43PGQ",
    t1_date: str = "",
    t2_date: str = ""
) -> bool:
    """
    Main function called by Celery worker after inference.
    
    1. Check: if changed_area_km2 <= threshold_km2, return False (no alert)
    2. If threshold exceeded:
       a. Format alert log entry with timestamp + all details
       b. Append to outputs/alerts.log (create if not exists)
       c. Print formatted alert to console with ⚠️ emoji
       d. If recipient_email is not None:
          result_url = f"http://localhost:8000/api/v1/results/{task_id}"
          email_sent = send_alert_email(recipient_email, task_id, 
                                        changed_area_km2, tile,
                                        t1_date, t2_date, threshold_km2, result_url)
          Log: "Alert email sent to {email}" or "Alert email failed"
       e. Return True
    """

CHANGE 2 — workers/tasks.py:
Modify run_detection_task() function signature to include:
  recipient_email: Optional[str] = None
  alert_threshold_km2: float = 1.0
  t1_date: str = "2021-03-14"
  t2_date: str = "2023-03-14"

After inference completes:
  trigger_alert(
      result_id=task_id,
      changed_area_km2=inference_results['changed_area_km2'],
      threshold_km2=alert_threshold_km2,
      recipient_email=recipient_email,
      task_id=task_id,
      tile="T43PGQ",
      t1_date=t1_date,
      t2_date=t2_date
  )

CHANGE 3 — backend/schemas.py:
Add to DetectionRequest:
  recipient_email: Optional[str] = None
  alert_threshold_km2: Optional[float] = 1.0

CHANGE 4 — backend/routes.py:
In detect_change() endpoint:
  Extract recipient_email and alert_threshold_km2 from request body
  Pass both to the Celery task

CHANGE 5 — frontend/index.html:
Add "Alert Settings" section in sidebar between Run Detection button and Status panel:

  <div class="section-title">📧 Alert Settings</div>
  
  Email input:
  <label>Your Email (for alerts)</label>
  <input type="email" id="alertEmail" placeholder="you@example.com">
  
  Threshold slider:
  <label>Alert if change exceeds <span id="thresholdValue">1.0</span> km²</label>
  <input type="range" id="alertThreshold" min="0.1" max="50" step="0.1" value="1.0">
  (slider updates the span text live as user drags)
  
  Info text:
  <small>You will receive an email when detected change area exceeds your threshold.</small>
  
  When Run Detection button is clicked:
  - Include in POST body: recipient_email = document.getElementById('alertEmail').value || null
  - Include in POST body: alert_threshold_km2 = parseFloat(document.getElementById('alertThreshold').value)
  
  Alerts panel update:
  When GET /api/v1/alerts returns data, show each alert as:
  <div class="alert-card">
    ⚠️ <strong>14.63 km²</strong> detected — threshold 1.0 km² exceeded<br>
    <small>Email sent to: user@example.com | 09 May 2026 15:30</small>
  </div>

Full type hints, Google docstrings everywhere. Email send must never crash worker.
No placeholders.
```

#### Test email sending

```bash
conda activate satchange
cd /media/jayesh/Windows-SSD/Users/jayes/satellite-change-detector

python -c "
from workers.alerts import send_alert_email
success = send_alert_email(
    recipient_email='your_test_email@gmail.com',
    task_id='test-task-123',
    changed_area_km2=14.63,
    tile='T43PGQ',
    t1_date='2021-03-14',
    t2_date='2023-03-14',
    threshold_km2=1.0,
    result_url='http://localhost:8000/api/v1/results/test-task-123'
)
print('Email sent successfully:', success)
"
# Check inbox within 30 seconds
```

---

## 4. Phase 0 — Pseudo-Labeling Fine-Tune on T43PGQ

### Goal

Fine-tune the ChangeFormer model on Bengaluru's own satellite data without any manual labeling, by using the current model's high-confidence predictions as auto-generated training labels. Pixels where the model is very confident (probability > 0.85 = change, < 0.15 = no change) serve as pseudo-labels. Uncertain pixels (0.15–0.85) are masked out and ignored during training.

### Timeline: 1–2 days

### Why This Works

The current OSCD-trained model already captures the most obvious, unambiguous structural changes in Bengaluru. Its high-confidence predictions on large newly built areas and clearly unchanged farmland form a reliable signal. Training on these for 10 epochs with a very low learning rate adapts the model to the specific spectral characteristics of T43PGQ imagery (Indian subcontinent atmosphere, Karnataka land cover types, March dry season signature) without requiring any human annotation.

### Step-by-Step Implementation

#### Step 1 — Generate pseudo-label GeoTIFF

```bash
conda activate satchange
cd /media/jayesh/Windows-SSD/Users/jayes/satellite-change-detector
mkdir -p data/pseudo_labels

python -c "
import numpy as np
import rasterio
import os

with rasterio.open('outputs/oscd_test/change_mask.tif') as src:
    prob_mask = src.read(1).astype(np.float32)
    meta = src.meta.copy()
    print(f'Probability mask shape: {prob_mask.shape}')
    print(f'Min: {prob_mask.min():.3f}, Max: {prob_mask.max():.3f}')

# Create pseudo-labels
pseudo = np.full_like(prob_mask, -1, dtype=np.int8)
pseudo[prob_mask > 0.85] = 1   # Definitely changed
pseudo[prob_mask < 0.15] = 0   # Definitely unchanged
# -1 = uncertain → ignored during training

n_change = (pseudo == 1).sum()
n_nochange = (pseudo == 0).sum()
n_ignore = (pseudo == -1).sum()
total = pseudo.size

print(f'Pseudo-labeled change: {n_change:,} ({n_change/total*100:.1f}%)')
print(f'Pseudo-labeled no-change: {n_nochange:,} ({n_nochange/total*100:.1f}%)')
print(f'Ignored uncertain zone: {n_ignore:,} ({n_ignore/total*100:.1f}%)')

meta.update(dtype='int8', count=1, nodata=-1)
with rasterio.open('data/pseudo_labels/t43pgq_pseudo_label.tif', 'w', **meta) as dst:
    dst.write(pseudo[np.newaxis, :, :])

print('Saved: data/pseudo_labels/t43pgq_pseudo_label.tif')
"
```

#### Gemini CLI Prompt for Phase 0

```
@ml/dataset.py @ml/losses.py @ml/train.py @configs/config.yaml

Add pseudo-label fine-tuning support.

CHANGE 1 — ml/dataset.py, add PseudoLabelDataset class:
- Loads T1 from config pseudo.t1_folder using load_sentinel2_bands()
- Loads T2 from config pseudo.t2_folder using load_sentinel2_bands()
- Loads pseudo-label from data/pseudo_labels/t43pgq_pseudo_label.tif using rasterio
- Tiles T1, T2, and pseudo-label into 256×256 patches (stride=128, overlap=32)
- SKIPS patches where fewer than 10% of pixels have definitive labels (!=−1)
- For remaining patches: keeps the −1 values in the label tensor as-is
  (MaskedBCEDiceLoss will ignore these pixels during training)
- Returns: {
    't1': tensor(3, 256, 256),
    't2': tensor(3, 256, 256),
    'label': tensor(1, 256, 256),  # values: 0, 1, or -1
    'mask': tensor(1, 256, 256)    # 1 where label != -1, 0 where label == -1
  }
- Update get_dataset() to handle dataset_type='pseudo'

CHANGE 2 — ml/losses.py, add MaskedBCEDiceLoss class:
- Accepts (pred, target, mask) tensors
- Computes BCEWithLogitsLoss and DiceLoss ONLY on pixels where mask==1
- Zeros out contribution of masked pixels (mask==0) completely
- Returns combined weighted loss (same weighting as BCEDiceLoss)
- This ensures uncertain pseudo-labeled pixels do not pollute gradients

CHANGE 3 — ml/train.py:
Add fine-tune mode triggered when dataset_type=='pseudo':
- Load pre-trained checkpoint from configs.model.checkpoint_path before training
- Use learning rate = 0.00001 (10x lower than initial LR of 0.0001)
- Use MaskedBCEDiceLoss instead of BCEDiceLoss
- Run max 10 epochs (not 50)
- Save best checkpoint to checkpoints/best_model_finetuned.pth
- Add MLflow tags: {'mode': 'finetune', 'base_dataset': 'oscd', 'finetune_data': 'pseudo_t43pgq'}

CHANGE 4 — configs/config.yaml:
Add pseudo section:
  pseudo:
    t1_folder: "data/ISRO/2021/S2A_MSIL2A_20210314T050651_N0214_R019_T43PGQ_20210314T073831.SAFE/GRANULE/L2A_T43PGQ_A029903_20210314T051944/IMG_DATA/R10m"
    t2_folder: "data/ISRO/2023/S2A_MSIL2A_20230314T050651_N0509_R019_T43PGQ_20230314T090502.SAFE/GRANULE/L2A_T43PGQ_A040342_20230314T051815/IMG_DATA/R10m"
    pseudo_label_path: "data/pseudo_labels/t43pgq_pseudo_label.tif"
    dataset_type: pseudo
    finetune_lr: 0.00001
    finetune_epochs: 10
    checkpoint_output: "checkpoints/best_model_finetuned.pth"

Full type hints and docstrings. No placeholders.
```

#### Run fine-tuning

```bash
tmux new -s finetune
conda activate satchange
cd /media/jayesh/Windows-SSD/Users/jayes/satellite-change-detector
# Set dataset_type: pseudo in config.yaml first
python ml/train.py
# Ctrl+B then D to detach safely
# Monitor at http://localhost:5000 (MLflow)
```

---

## 5. Phase 0B — Final Production Inference

### Goal

Run production-quality inference using the fine-tuned model and compare results against the OSCD-only baseline to quantify the improvement.

### Timeline: 1 hour after Phase 0 completes

```bash
conda activate satchange
cd /media/jayesh/Windows-SSD/Users/jayes/satellite-change-detector

python -c "
from pipeline.inference import run_inference

result = run_inference(
    t1_folder='data/ISRO/2021/S2A_MSIL2A_20210314T050651_N0214_R019_T43PGQ_20210314T073831.SAFE/GRANULE/L2A_T43PGQ_A029903_20210314T051944/IMG_DATA/R10m',
    t2_folder='data/ISRO/2023/S2A_MSIL2A_20230314T050651_N0509_R019_T43PGQ_20230314T090502.SAFE/GRANULE/L2A_T43PGQ_A040342_20230314T051815/IMG_DATA/R10m',
    checkpoint_path='checkpoints/best_model_finetuned.pth',
    output_dir='outputs/final_production'
)
print('Final production result:', result)
"
```

### Expected comparison table

| Model | Changed Area | Max Confidence | Notes |
|---|---|---|---|
| Siamese UNet (LEVIR) | 0.0025 km² | 0.49 | Domain shift — useless on Sentinel-2 |
| ChangeFormer pretrained only | 3.53 km² | 0.89 | No change training |
| ChangeFormer + OSCD | 14.63 km² | 1.00 | Current deployed model |
| ChangeFormer + OSCD + pseudo fine-tune | TBD | TBD | Target: fewer agricultural FP |

---

## 6. Phase 0C — Database Wiring and Linux JP2 Cleanup

### Goal

Two housekeeping tasks: (1) wire FastAPI routes to PostgreSQL for persistent storage, and (2) remove the glymur JP2 workaround since Linux rasterio reads JP2 natively.

### Timeline: 1 day

### 6.1 Database Wiring — Gemini CLI Prompt

```
@GEMINI.md @backend/routes.py @backend/main.py @database/models.py @workers/tasks.py

Wire all backend routes and Celery tasks to PostgreSQL instead of in-memory dict.
DB already running. Schema already deployed. SQLAlchemy models exist.
DB URL: postgresql://postgres:postgres@localhost:5432/satchange

CHANGE 1 — backend/main.py:
Remove in_memory_results = {} global dict.
Add get_db() dependency yielding SQLAlchemy session.
On startup: test DB connection — log warning if fails, still start app.

CHANGE 2 — backend/routes.py:
POST /detect-change:
  Create DetectionResult row with status='pending', task_id=uuid
  Pass result_id (DB UUID) to Celery task
  Return {task_id, result_id}

GET /results/{task_id}:
  Query detection_results WHERE task_id=task_id
  Return status, changed_area_km2, geojson_path from DB

GET /alerts:
  Query alerts WHERE acknowledged=false

Add POST /alerts/{id}/acknowledge to mark alert read

CHANGE 3 — workers/tasks.py:
Accept result_id parameter (DB UUID)
At start: UPDATE detection_results SET status='processing' WHERE id=result_id
After success: UPDATE SET status='complete', geojson_path, changed_area_km2, updated_at
After alert: INSERT INTO alerts (result_id, message, triggered_at)
On failure: UPDATE SET status='failed', updated_at

All DB ops wrapped in try/except — never crash worker on DB failure.
```

### 6.2 Linux JP2 Cleanup — Gemini CLI Prompt

```
@GEMINI.md @pipeline/preprocess.py

We are on Linux (Pop OS) where rasterio reads JP2 natively via JP2OpenJPEG GDAL driver.
Remove ALL glymur imports and usage. Remove the synthetic Affine transform fallback.

Rewrite load_sentinel2_bands() to use rasterio.open() directly for all JP2 files.
Read CRS and transform directly from JP2 metadata (src.crs, src.transform).
Keep all other functions (align_images, compute_ndvi, clip_image_to_aoi) unchanged.

Add platform check at module import:
  Check JP2OpenJPEG is in GDAL driver list
  If not found: raise RuntimeError with install instructions
  Print available JP2 drivers on import

Full type hints and docstrings.
```

---

## 7. Phase 1 — Frontend Upgrade: Comparison Slider + Layer Panel

### Goal

Add a before/after comparison slider showing 2021 TCI vs 2023 TCI, a spectral layer selector panel (Natural Color, NDVI, NDWI, NDBI), and a map type switcher (Satellite, Street, Terrain).

### Timeline: 1–2 days

### Step 1 — Convert TCI to Cloud-Optimized GeoTIFF

```bash
conda activate satchange
mkdir -p outputs/tci outputs/indices

rio cogeo create \
  "data/ISRO/2021/.../R10m/T43PGQ_20210314T050651_TCI_10m.jp2" \
  "outputs/tci/tci_2021.tif"

rio cogeo create \
  "data/ISRO/2023/.../R10m/T43PGQ_20230314T050651_TCI_10m.jp2" \
  "outputs/tci/tci_2023.tif"
```

### Gemini CLI Prompt for Phase 1

```
@frontend/index.html @backend/main.py

Add comparison slider, layer panel, and map type switcher to the frontend.

Add CDN imports:
  Leaflet.SideBySide: https://unpkg.com/leaflet-side-by-side@2.0.0/leaflet-side-by-side.js
  georaster: https://unpkg.com/georaster
  georaster-layer-for-leaflet: https://unpkg.com/georaster-layer-for-leaflet/dist/georaster-layer-for-leaflet.min.js

1. Comparison slider toggle button in sidebar
   ON: load tci_2021.tif and tci_2023.tif as georaster layers
       L.control.sideBySide(left2021, right2023).addTo(map)
   OFF: remove layers and slider

2. Layer panel (top-right, collapsible):
   Buttons: Natural Color | NDVI | NDWI | NDBI | Agriculture | Forestry
   Each loads GeoTIFF from /tci/ or /indices/ backend route
   One layer visible at a time, active button highlighted

3. Map type switcher (top-left):
   Satellite (ESRI World Imagery) | Street (OSM) | Terrain (OpenTopo)

4. backend/main.py:
   app.mount("/tci", StaticFiles(directory="outputs/tci"), name="tci")
   app.mount("/indices", StaticFiles(directory="outputs/indices"), name="indices")
   Create directories on startup if missing

Keep all existing detection + alert functionality intact. Dark theme throughout.
```

---

## 8. Phase 2 — Spectral Indices: NDVI, NDWI, NDBI

### Goal

Compute NDVI, NDWI, NDBI from Sentinel-2 bands and serve as colored GeoTIFF overlays in the Layer panel.

### Timeline: 1 day

### Index Formulas

| Index | Formula | Bands | Colormap |
|---|---|---|---|
| NDVI | (B08 − B04) / (B08 + B04) | NIR, Red (R10m) | RdYlGn |
| NDWI | (B03 − B08) / (B03 + B08) | Green, NIR (R10m) | RdBu |
| NDBI | (B11 − B08) / (B11 + B08) | SWIR (R20m), NIR | RdGy |

### Gemini CLI Prompt for Phase 2

```
@GEMINI.md @pipeline/preprocess.py @configs/config.yaml
Create pipeline/indices.py and scripts/compute_indices.py.

pipeline/indices.py functions:
- load_band(folder_path, band_name) → (array, meta, CRS)
  Searches R10m for B02/B03/B04/B08, R20m for B11/B12
- compute_ndvi(red, nir), compute_ndwi(green, nir), compute_ndbi(swir, nir)
  All return float32 arrays in [-1, 1]
- normalize_index(index, vmin=-1, vmax=1) → uint8 [0,255]
- apply_colormap(uint8_array, colormap_name) → RGBA uint8 (H,W,4)
- save_index_geotiff(array, meta, output_path, colormap) → None
- resample_band_to_10m(band_array, band_meta, target_meta) → array
  Uses rasterio.warp.reproject with Resampling.bilinear

scripts/compute_indices.py:
- Args: --t1-safe-root, --t2-safe-root
- Auto-locate R10m and R20m folders from SAFE root
- Compute NDVI/NDWI/NDBI for both years
- Save to outputs/indices/ with year suffix
- Print progress summary

Run command:
python scripts/compute_indices.py \
  --t1-safe-root "data/ISRO/2021/S2A_MSIL2A_...SAFE" \
  --t2-safe-root "data/ISRO/2023/S2A_MSIL2A_...SAFE"
```

---

## 9. Phase 3 — Time Series Analysis

### Goal

Download Sentinel-2 scenes for 2019, 2020, 2022, 2024 (2021 and 2023 already exist). Run inference between consecutive pairs. Show a bar chart timeline in the frontend.

### Timeline: 2–3 days

### Additional Downloads

Download from Bhoonidhi for years 2019, 2020, 2022, 2024. Same criteria: March, T43PGQ, cloud < 20%, Level 2A.

**Storage:** `data/ISRO/2019/`, `data/ISRO/2020/`, `data/ISRO/2022/`, `data/ISRO/2024/`

### Gemini CLI Prompt for Phase 3

```
@backend/routes.py @backend/main.py @frontend/index.html

Create scripts/run_timeseries_inference.py:
- Scans data/ISRO/ for year folders in sorted order
- Creates consecutive pairs: (2019,2020), (2020,2021), (2021,2022), (2022,2023), (2023,2024)
- Runs run_inference() for each pair
- Saves to outputs/timeseries/{y1}_{y2}/change_mask.tif + change_polygons.geojson + result.json
- Writes outputs/timeseries/summary.json

Add to backend/routes.py:
- GET /api/v1/timeseries → return summary.json
- GET /api/v1/timeseries/{period}/geojson → return GeoJSON for that period
- Static mount: /timeseries → outputs/timeseries/

Add to frontend/index.html:
- Collapsible timeline panel at bottom (200px height)
- Chart.js bar chart: X=period labels, Y=changed_area_km2
- Click bar → load that period's GeoJSON polygons on map
- CDN: https://cdn.jsdelivr.net/npm/chart.js
```

---

## 10. Phase 4 — Scene Search Integration

### Goal

Let users search for available Sentinel-2 scenes via the Copernicus Data Space STAC API by tile, date range, and cloud cover. Users can browse and download scenes directly from the frontend.

### Timeline: 2–3 days

### Setup

1. Register at `https://dataspace.copernicus.eu/` (free)
2. Generate OAuth2 client credentials
3. Add to `configs/config.yaml` under `copernicus:` section

### Gemini CLI Prompt for Phase 4

```
@backend/routes.py @backend/schemas.py @frontend/index.html @configs/config.yaml

Add Copernicus Data Space scene search.

Backend schemas: SceneSearchRequest, SceneResult, SceneSearchResponse, DownloadRequest

Backend routes:
- POST /api/v1/scenes/search:
  Get OAuth2 token from Copernicus
  Query STAC: https://catalogue.dataspace.copernicus.eu/stac/collections/SENTINEL-2/items
  Filter by tile, date range, cloud cover
  Return SceneSearchResponse

- POST /api/v1/scenes/download:
  Celery task to download .SAFE file to data/ISRO/{year}/
  Return task_id

- GET /api/v1/scenes/download/{task_id}: return download progress

Frontend sidebar (Scene Search section):
- Tile ID input (default T43PGQ)
- Date from/to pickers
- Cloud cover slider (0-100%, default 20%)
- Search button → POST /api/v1/scenes/search
- Results: scrollable scene cards with date, cloud %, download button
- Hover card → show bounding box on map
- Download progress bar

Use httpx for async HTTP calls. Full error handling for auth failures.
```

---

## 11. Phase 5 — Better Model with S2Looking

### Goal

Train ChangeFormerV2 (EfficientNet-B2 encoder) on S2Looking for building-level change detection. Add model selector so users choose between pixel-level (OSCD) and building-level (S2Looking) detection modes.

### Timeline: 3–5 days

### Download S2Looking

```bash
pip install gdown
mkdir -p data/S2Looking
# Check https://github.com/S2Looking/Dataset for current Google Drive links
gdown --id <FOLDER_ID> --folder -O data/S2Looking/

# Verify
python -c "
from pathlib import Path
for split in ['train','val','test']:
    n = len(list(Path(f'data/S2Looking/{split}/Image1').glob('*.png')))
    print(f'{split}: {n} pairs')
"
```

### Gemini CLI Prompt for Phase 5

```
@ml/dataset.py @ml/model.py @ml/train.py @pipeline/inference.py @backend/routes.py @frontend/index.html @configs/config.yaml

Add S2Looking dataset and ChangeFormerV2 for building-level detection.

PART 1 — ml/dataset.py:
Add S2LookingDataset:
- Tiles 1024×1024 PNG images to 256×256 patches (stride 128)
- Combines label1 + label2 into binary mask
- Same return format as OSCDDataset
- Update get_dataset() for dataset_type='s2looking'

PART 2 — ml/model.py:
Add ChangeFormerV2 with EfficientNet-B2 encoder
Constructor: encoder_name parameter ('efficientnet_b0'|'efficientnet_b2')

PART 3 — ml/train.py + config:
Support dataset_type='s2looking', encoder='efficientnet_b2'
Save to checkpoints/best_model_s2looking.pth
batch_size: 4 (B2 needs more VRAM)

PART 4 — pipeline/inference.py:
model_type parameter: 'oscd' | 's2looking' | 'finetuned'
Load correct checkpoint and architecture per model_type

PART 5 — backend/routes.py + schemas.py:
Add model_type to DetectionRequest (default 'oscd')
Pass through to Celery task and inference

PART 6 — frontend/index.html:
Add model selector radio buttons in sidebar:
  ○ Pixel-level (OSCD) — faster, all change types
  ● Building-level (S2Looking) — precise building footprints
Pass selected model_type in POST body
```

---

## 12. CLI Tools — When to Use Which

### Gemini CLI (Gemini 2.5 Pro Preview)

**Best for:** Multi-file tasks requiring coordinated changes across many files simultaneously. Its 1M token context lets it read the entire codebase at once.

**Best tasks in this project:**
- AOI clipping fix (preprocess + inference + tasks + routes + frontend simultaneously)
- Email alert system (alerts + tasks + routes + schemas + frontend simultaneously)
- Frontend upgrades (index.html + backend must coordinate)
- Architecture-level integration work
- Tracing bugs across the full call chain

**Session pattern:**
```
# Always start with:
@GEMINI.md @PROJECT_STATUS.md
# Use /plan before complex multi-file changes
# Use /compress when session context is long
# Select "Allow for this session" for Python execution
```

### CodeGPT 5.5 High CLI

**Best for:** Precise single-file implementations requiring algorithmic correctness.

**Best tasks in this project:**
- Spectral index math in `pipeline/indices.py`
- PyTorch model architecture in `ml/model.py`
- Dataset class implementations with exact array shapes
- OAuth2/STAC API client code
- Masked loss functions

### Phase CLI Recommendations

| Task | Primary CLI | Reason |
|---|---|---|
| Fix A: AOI Clipping | **Gemini** | Touches 5 files simultaneously |
| Fix B: Email Alerts | **Gemini** | Touches 5 files simultaneously |
| Phase 0: Pseudo-labeling | **CodeGPT** | Precise masked loss + dataset class |
| Phase 0C: DB Wiring | **Gemini** | Routes + models + tasks + main |
| Phase 0C: JP2 Cleanup | **Gemini** | Single file but needs full context |
| Phase 1: Frontend | **Gemini** | index.html + main.py coordinate |
| Phase 2: Indices | **CodeGPT** | Band math algorithmic precision |
| Phase 3: Time Series | **Gemini** | Backend + frontend + script |
| Phase 4: Scene Search | **Split** | CodeGPT API client, Gemini frontend |
| Phase 5: S2Looking | **CodeGPT** | PyTorch model + dataset |

---

## 13. Complete Future Folder Structure

```
satellite-change-detector/
│
├── GEMINI.md
├── PROJECT_STATUS.md
├── STATUS.md
├── PLAN.md
├── README.md
├── requirements.txt
├── .env                                 # SMTP credentials — NEVER commit
├── .gitignore                           # includes .env
│
├── configs/
│   └── config.yaml                      # + alerts, copernicus, pseudo, models sections
│
├── ml/
│   ├── model.py                         # ChangeFormer + ChangeFormerV2 (EfficientNet-B2)
│   ├── dataset.py                       # LEVIR + OSCD + S2Looking + PseudoLabel
│   ├── losses.py                        # + MaskedBCEDiceLoss
│   ├── train.py                         # + finetune mode, S2Looking support
│   └── evaluate.py
│
├── pipeline/
│   ├── preprocess.py                    # Clean Linux JP2 + clip_image_to_aoi
│   ├── tiling.py
│   ├── stitch.py
│   ├── postprocess.py
│   ├── inference.py                     # + aoi_geojson param, model_type param
│   ├── gis.py
│   └── indices.py                       # NEW — NDVI/NDWI/NDBI computation
│
├── scripts/
│   ├── compute_indices.py               # NEW — compute and save spectral indices
│   ├── run_timeseries_inference.py      # NEW — batch multi-year inference
│   ├── generate_pseudo_labels.py        # NEW — pseudo-label generation script
│   └── convert_tci_to_cog.py           # NEW — TCI JP2 → COG GeoTIFF
│
├── backend/
│   ├── main.py                          # + tci/indices/timeseries static mounts
│   ├── routes.py                        # + scene search, timeseries, DB-wired, alerts ack
│   └── schemas.py                       # + email fields, model_type, SceneResult, TimeSeries
│
├── workers/
│   ├── tasks.py                         # + recipient_email, result_id, DB updates
│   └── alerts.py                        # + full SMTP email sending
│
├── database/
│   ├── schema.sql
│   └── models.py
│
├── frontend/
│   └── index.html                       # + slider, layers, timeline, scene search,
│                                        #   email input, threshold slider, model selector
│
├── data/
│   ├── LEVIR-CD/
│   ├── OSCD/
│   ├── S2Looking/                       # Phase 5
│   │   ├── train/ (Image1/ Image2/ label1/ label2/)
│   │   ├── val/
│   │   └── test/
│   ├── pseudo_labels/                   # Phase 0
│   │   └── t43pgq_pseudo_label.tif
│   └── ISRO/
│       ├── 2019/                        # Phase 3 — new download
│       ├── 2020/                        # Phase 3 — new download
│       ├── 2021/                        # existing
│       ├── 2022/                        # Phase 3 — new download
│       ├── 2023/                        # existing
│       └── 2024/                        # Phase 3 — new download
│
├── checkpoints/
│   ├── best_model.pth                   # ChangeFormer OSCD (current)
│   ├── best_model_finetuned.pth         # ChangeFormer OSCD + pseudo fine-tune
│   └── best_model_s2looking.pth         # ChangeFormerV2 S2Looking (Phase 5)
│
└── outputs/
    ├── task_{uuid}/                     # Per-task detection results
    ├── tci/                             # Phase 1
    │   ├── tci_2021.tif
    │   └── tci_2023.tif
    ├── indices/                         # Phase 2
    │   ├── ndvi_2021.tif   ndvi_2023.tif
    │   ├── ndwi_2021.tif   ndwi_2023.tif
    │   └── ndbi_2021.tif   ndbi_2023.tif
    ├── timeseries/                      # Phase 3
    │   ├── summary.json
    │   ├── 2019_2020/
    │   ├── 2020_2021/
    │   ├── 2021_2022/
    │   ├── 2022_2023/
    │   └── 2023_2024/
    ├── final_production/                # Phase 0B
    ├── aoi_test/                        # Fix A testing
    ├── confusion_matrix.png
    ├── samples.png
    └── alerts.log
```

---

## 14. Execution Order Summary

```
IMMEDIATE (do these first — both are blocking usability):
  Fix A: AOI Clipping         ← detection ignores user-drawn AOI
  Fix B: Email Alerts         ← no notification sent to user

REMAINING ORIGINAL PHASES (do these next):
  Phase 0:  Pseudo-label fine-tune on T43PGQ data
  Phase 0B: Final production inference + model comparison
  Phase 0C: Database wiring to PostgreSQL + Linux JP2 cleanup

NEW LANDVIEWER-INSPIRED PHASES (do these after original phases are complete):
  Phase 1: Frontend comparison slider + layer selector panel
  Phase 2: NDVI / NDWI / NDBI spectral index overlays
  Phase 3: Multi-year time series (2019–2024)
  Phase 4: Copernicus scene search + download
  Phase 5: S2Looking building-level change detection model
```

*End of PLAN.md — Version 2.0*
