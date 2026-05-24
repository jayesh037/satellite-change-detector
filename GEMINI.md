\# GEMINI.md — Satellite Change Detection System



\## Project Goal

Detect meaningful changes between satellite images (2021 vs 2023)

and alert users. Real-world deployment on ISRO/Sentinel-2 data.



\## Tech Stack

\- Python 3.11

\- PyTorch 2.x (Siamese UNet model)

\- Rasterio + GDAL (GIS processing)

\- FastAPI (backend)

\- Celery + Redis (async workers)

\- PostgreSQL + PostGIS (database)

\- Leaflet.js (frontend map)

\- MLflow (experiment tracking)

\- segmentation-models-pytorch (pretrained encoders)



\## Datasets

\- Training: LEVIR-CD (RGB, 256x256 patches, binary labels)

\- Inference: ISRO Sentinel-2 (4 bands: B02,B03,B04,B08, JP2 format)



\## Model

\- Siamese UNet with shared ResNet34 encoder (pretrained ImageNet)

\- Loss: BCE + Dice combined

\- Input: two 256x256 patches (t1 and t2)

\- Output: binary change mask

\- Supports 3ch (training) and 4ch (inference) input via config



\## Hardware

\- GPU: RTX 4050 Laptop (6GB VRAM)

\- batch\_size: 8

\- mixed\_precision: true

\- encoder: resnet34



\## Coding Rules

\- Python 3.11, type hints everywhere

\- Google-style docstrings

\- Config via configs/config.yaml only — no hardcoded values

\- No placeholders — complete working code only

\- All files self-contained and importable

