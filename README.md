# 🛰️ Satellite Change Detector

> End-to-end deep learning system for detecting urban and structural change in Sentinel-2 satellite imagery — built for real Indian satellite data over Bengaluru (tile T43PGQ), with a full web interface, async pipeline, and time series from 2019 to 2026.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20CUDA-ee4c2c?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## What It Does

Draw an Area of Interest on a map, point it at two Sentinel-2 image folders, and the system will:

1. Clip both images to your AOI
2. Run a ChangeFormer deep learning model on 256×256 patch pairs
3. Post-process with NDVI filtering and noise removal
4. Return georeferenced GeoTIFF + GeoJSON change polygons
5. Render red polygons over changed areas on an interactive Leaflet map
6. Email you if the changed area exceeds your configured threshold

**Real result:** 14.63 km² of change detected over Bengaluru between March 2021 and March 2023 — consistent with known urban expansion corridors in the region.

---

## Features

- **AOI-clipped inference** — draw any polygon, only that area is processed and returned
- **ChangeFormer model** — transformer-based architecture with EfficientNet-B0 encoder, trained on OSCD (Sentinel-2 specific dataset)
- **Timeseries 2019–2026** — 8 years of March imagery for T43PGQ, with year-pair comparison overlays
- **Spectral layers** — TCI, NDVI, NDWI, NDBI served as map tile layers for all years
- **Scene search** — search and download Sentinel-2 scenes directly from Copernicus Data Space
- **Email alerts** — configurable threshold; HTML alert emails sent on detection
- **User auth** — register/login with bcrypt passwords
- **Async pipeline** — FastAPI + Celery + Redis; jobs queue while backend stays responsive
- **PostgreSQL storage** — results, AOIs, and alerts persisted with PostGIS geometry support

---

## Architecture

```
Browser (Leaflet frontend)
        │ POST /api/v1/detect-change
        ▼
FastAPI backend  ──────────────────────────────────────────────────────────┐
        │ Celery task                                                        │
        ▼                                                                    │
Redis broker                                                                 │
        │                                                                    │
        ▼                                                                    │
Celery worker                                                                │
   ├── pipeline/preprocess.py   (load JP2 bands, align, NDVI)               │
   ├── pipeline/tiling.py       (256×256 overlapping patches)                │
   ├── ml/model.py              (ChangeFormer forward pass)                  │
   ├── pipeline/stitch.py       (Gaussian blend reassembly)                  │
   ├── pipeline/postprocess.py  (threshold, NDVI filter, noise removal)      │
   └── pipeline/gis.py          (GeoTIFF + GeoJSON output)                  │
        │                                                                    │
        ▼                                                                    │
PostgreSQL/PostGIS  ◄───────────────────────────────────────────────────────┘
outputs/ (static GeoJSON served by FastAPI)
```

---

## Stack

| Layer | Technology |
|---|---|
| Model | ChangeFormer (EfficientNet-B0 encoder, custom decoder) |
| Training data | OSCD (Sentinel-2, 24 cities), LEVIR-CD (baseline) |
| Inference data | Bhoonidhi/ISRO Sentinel-2 L2A, tile T43PGQ, 2019–2026 |
| Backend | FastAPI + uvicorn |
| Task queue | Celery 5.6 + Redis |
| Database | PostgreSQL 18 + PostGIS 3.4 + SQLAlchemy 2.0 |
| GIS | rasterio, GDAL, geopandas, shapely, pyproj |
| Tile serving | rio-tiler (COG-based XYZ tiles) |
| Scene search | Copernicus Data Space STAC API + OData download |
| Frontend | Leaflet.js + Leaflet.draw (single HTML file, no build step) |
| Training tracking | MLflow |
| GPU | NVIDIA RTX 4050 Laptop (CUDA 12.4) |

---

## Project Structure

```
satellite-change-detector/
├── backend/
│   ├── main.py              # FastAPI app, CORS, startup
│   ├── routes.py            # All API endpoints
│   ├── schemas.py           # Pydantic models
│   ├── copernicus.py        # Copernicus STAC search + download
│   └── tiles.py             # COG tile serving (TCI/NDVI/NDWI/NDBI)
├── pipeline/
│   ├── preprocess.py        # Band loading, alignment, NDVI
│   ├── tiling.py            # Patch generation
│   ├── inference.py         # Master orchestration
│   ├── stitch.py            # Gaussian blend reassembly
│   ├── postprocess.py       # Threshold, filter, denoise
│   ├── gis.py               # GeoTIFF + GeoJSON output
│   └── indices.py           # NDVI/NDWI/NDBI computation
├── ml/
│   ├── model.py             # ChangeFormer architecture
│   ├── dataset.py           # OSCDDataset + LEVIRDataset
│   ├── train.py             # Training loop (MLflow, mixed precision)
│   ├── losses.py            # DiceLoss + BCEDiceLoss
│   └── evaluate.py          # IoU, F1, confusion matrix
├── workers/
│   ├── tasks.py             # Celery tasks (detection + download)
│   └── alerts.py            # Email alert system
├── database/
│   ├── models.py            # SQLAlchemy ORM models
│   └── schema.sql           # PostGIS DDL
├── scripts/
│   ├── compute_indices.py   # Batch spectral index generation
│   └── run_timeseries_inference.py
├── frontend/
│   └── index.html           # Full web app (single file)
├── configs/
│   └── config.yaml.example  # Config template (copy → config.yaml)
└── requirements.txt
```

---

## Model

**ChangeFormer** — a transformer-inspired change detection architecture.

- Shared EfficientNet-B0 encoder processes T1 and T2 separately
- Feature differences computed at 5 resolution scales
- Lightweight decoder fuses multi-scale difference maps
- Sigmoid output: per-pixel change probability in [0, 1]
- Trained with BCEWithLogitsLoss + DiceLoss on OSCD dataset

**Metrics (OSCD val set):**

| Metric | Value |
|---|---|
| Val IoU (change class) | 0.5864 |
| Val F1 | 0.7363 |
| Val Loss | 0.2209 |

**Real-world inference (Bengaluru 2021→2023):**

| Model | Changed Area |
|---|---|
| Siamese UNet (LEVIR pretrain) | 0.0025 km² |
| ChangeFormer (pretrained only) | 3.53 km² |
| ChangeFormer (OSCD fine-tuned) | **14.63 km²** |

---

## Setup

### Prerequisites

- Pop OS / Ubuntu 22+
- Miniconda
- NVIDIA GPU with CUDA 12.x (CPU inference also works, slower)
- PostgreSQL + PostGIS
- Redis

### Install

```bash
# Clone
git clone https://github.com/jayesh037/satellite-change-detector.git
cd satellite-change-detector

# Create conda environment
conda create -n satchange python=3.11 -y
conda activate satchange

# GIS stack (must come first)
conda install -c conda-forge gdal rasterio geopandas shapely pyproj fiona -y

# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# All other dependencies
pip install -r requirements.txt
```

### Configure

```bash
cp configs/config.yaml.example configs/config.yaml
# Edit configs/config.yaml:
#   - smtp_password: your Gmail app password
#   - copernicus client_id / client_secret: from dataspace.copernicus.eu
#   - database URL if different from default

cp .env.example .env   # or create .env manually
# Add:
# SMTP_USER=your@gmail.com
# SMTP_PASSWORD=your_app_password
# COPERNICUS_CLIENT_ID=...
# COPERNICUS_CLIENT_SECRET=...
# COPERNICUS_USERNAME=...
# COPERNICUS_PASSWORD=...
```

### Database

```bash
sudo systemctl start postgresql
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
sudo -u postgres psql -c "CREATE DATABASE satchange;"
sudo -u postgres psql -d satchange -c "CREATE EXTENSION postgis;"
psql -U postgres -h 127.0.0.1 -d satchange -f database/schema.sql
```

### Data

Download Sentinel-2 L2A data for tile T43PGQ from [Bhoonidhi (ISRO)](https://bhoonidhi.nrsc.gov.in) or [Copernicus Data Space](https://dataspace.copernicus.eu). Place under `data/ISRO/{year}/`.

For model training, download [OSCD dataset](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection) and place under `data/OSCD/`.

---

## Running

Three terminals required:

```bash
# Terminal 1 — Backend
conda activate satchange
cd satellite-change-detector
export $(grep -v '^#' .env | xargs)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Celery worker
conda activate satchange
cd satellite-change-detector
export $(grep -v '^#' .env | xargs)
python -m celery -A workers.tasks.celery_app worker --loglevel=info

# Terminal 3 — Frontend
cd satellite-change-detector/frontend
python3 -m http.server 3000
# Open http://localhost:3000
```

### Training

```bash
# Optional: monitor in tmux
tmux new -s train
conda activate satchange
python ml/train.py

# Monitor with MLflow
mlflow ui   # http://localhost:5000
```

### Generate spectral indices (all years)

```bash
for year in 2019 2020 2021 2022 2023 2024 2025 2026; do
  SAFE=$(find data/ISRO/$year -name "*.SAFE" -type d | head -1)
  python scripts/compute_indices.py --t1-safe-root "$SAFE" --t2-safe-root "$SAFE"
done
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/detect-change` | Submit detection job (t1_folder, t2_folder, aoi_geojson) |
| GET | `/api/v1/results/{task_id}` | Poll job status + results |
| GET | `/api/v1/alerts` | Get unacknowledged alerts |
| POST | `/api/v1/alerts/acknowledge-all` | Clear all alerts |
| POST | `/api/v1/auth/register` | Register user |
| POST | `/api/v1/auth/login` | Login |
| POST | `/api/v1/scenes/search` | Search Copernicus scenes |
| POST | `/api/v1/scenes/download` | Queue scene download |
| GET | `/api/v1/scenes/download/{task_id}` | Download progress |
| GET | `/api/v1/tiles/{layer}/{z}/{x}/{y}` | Serve COG tile (tci_2023, ndvi_2021, etc.) |
| GET | `/api/v1/timeseries/{y1}_{y2}/geojson` | Change GeoJSON for year pair |
| GET | `/api/v1/timeseries/summary` | Aggregated timeseries stats |

---

## Datasets Used

| Dataset | Source | Purpose |
|---|---|---|
| LEVIR-CD | [justchenhao.github.io/LEVIR](https://justchenhao.github.io/LEVIR/) | Initial baseline training (aerial) |
| OSCD | [IEEE DataPort](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection) | Main training (Sentinel-2 specific) |
| Bhoonidhi/ISRO | [bhoonidhi.nrsc.gov.in](https://bhoonidhi.nrsc.gov.in) | Real inference data (T43PGQ, 2019–2026) |
| Copernicus Data Space | [dataspace.copernicus.eu](https://dataspace.copernicus.eu) | Scene search + download |

---

## Environment Variables

Create a `.env` file (never commit this):

```env
SMTP_USER=your@gmail.com
SMTP_PASSWORD=your_16char_app_password
COPERNICUS_CLIENT_ID=your_client_id
COPERNICUS_CLIENT_SECRET=your_client_secret
COPERNICUS_USERNAME=your_copernicus_email
COPERNICUS_PASSWORD=your_copernicus_password
```

---

## Known Limitations

- Pixel-level detection only — no object-level building footprints
- No change type classification (construction vs deforestation vs water)
- Small training set (230 patches from OSCD)
- Frontend uses hardcoded `localhost:8000` — not multi-machine ready without editing
- No Docker setup yet

---

## License

MIT
