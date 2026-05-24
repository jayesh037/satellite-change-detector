# Satellite Change Detection System

An end-to-end system for detecting meaningful structural changes between multi-temporal satellite images (e.g., 2021 vs 2023) and automatically alerting users based on spatial area thresholds. Designed for ISRO/Sentinel-2 deployment.

## 1. Setup Environment

Create and activate a new Python virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

## 2. Install Dependencies

Install all the pinned requirements:

```bash
pip install -r requirements.txt
```

## 3. Download LEVIR-CD Dataset

Download the [LEVIR-CD Dataset](https://justcheneng.github.io/LEVIR/) and extract it into the `data/` directory so the structure is:
- `data/LEVIR-CD/train/A/`
- `data/LEVIR-CD/train/B/`
- `data/LEVIR-CD/train/label/`
- `data/LEVIR-CD/val/A/`
- `data/LEVIR-CD/val/B/`
- `data/LEVIR-CD/val/label/`

## 4. Train Model

Train the Siamese UNet on the LEVIR-CD dataset. Training progress is logged to MLflow, and the best weights are saved automatically.

```bash
python ml/train.py
```
*(Checkpoint is saved to `checkpoints/best_model.pth`)*

## 5. Evaluate Model

Run inference against the validation split to calculate global IoU, F1 score, and export the confusion matrix alongside sample visual predictions.

```bash
python ml/evaluate.py
```
*(Outputs saved to the `outputs/` directory)*

## 6. Run Inference on Sentinel-2

To test the GIS pipeline directly from Python without the web server, you can import and run the `run_inference` function:

```python
from pipeline.inference import run_inference

results = run_inference(
    t1_folder="data/sentinel2/t1",
    t2_folder="data/sentinel2/t2",
    checkpoint_path="checkpoints/best_model.pth",
    output_dir="outputs/demo"
)
print(results)
```

## 7. Start Redis

Ensure Redis is installed and running. It acts as the message broker for the Celery workers.

```bash
# Linux / macOS
redis-server

# Windows (Using WSL or Memurai)
# wsl redis-server
```

## 8. Start FastAPI Backend

Ensure your PostgreSQL database (with PostGIS enabled) is running and matches the URL configured in `configs/config.yaml`.

Start the API:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## 9. Start Celery Worker

In a separate terminal (with the virtual environment activated), start the background worker to process the heavy ML and GIS inference tasks.

```bash
# Linux / macOS
celery -A workers.tasks.celery_app worker --loglevel=info

# Windows
celery -A workers.tasks.celery_app worker --loglevel=info --pool=solo
```

## 10. Open Frontend

The frontend is a pure HTML/JS file. Simply open it in your browser:

```bash
# Windows
start frontend/index.html

# macOS
open frontend/index.html

# Linux
xdg-open frontend/index.html
```

Use the interface to input your Sentinel-2 folder paths, draw an Area of Interest (AOI), and trigger the detection pipeline.
