# Project Status: Satellite Change Detection System

## 📌 Current State
**Status:** **PHASE 2 READY (DATASET PREPARED & MODEL INTEGRATED)**

*   **Phase 1 Complete:** ChangeFormer model has been successfully integrated into the pipeline, and pretrained weights are loaded.
*   **Phase 2 Ready:** The OSCD dataset has been downloaded and correctly processed.
    *   **Dataset Structure:** `data/OSCD/images/`, `data/OSCD/train_labels/`, `data/OSCD/test_labels/`
    *   **Patches Generated:** 230 train patches (using a stride of 128px) and 71 val patches are now loading successfully.
*   **Current Inference Results:** Preliminary inference yields 3.53 km² of detected change using the pretrained ChangeFormer (with a threshold of 0.7).
*   **Environment Setup:** The following environment variables are required on startup:
    ```powershell
    $env:KMP_DUPLICATE_LIB_OK="TRUE"
    $env:PROJ_DATA="C:\Users\jayes\miniconda3\Library\share\proj"
    ```

---

## 📂 Files Created & Their Purpose

### Configuration & Documentation
*   `configs/config.yaml`: Central configuration hub governing model parameters, training hyperparameters, dataset paths, inference thresholds, and connections.
*   `requirements.txt`: Precisely pinned Python dependencies.
*   `README.md`: Comprehensive step-by-step instructions.
*   `PROJECT_STATUS.md`: This summary document.

### Machine Learning Layer (`ml/`)
*   `ml/model.py`: Implements the model architecture (ChangeFormer integration).
*   `ml/dataset.py`: Implements `OSCDDataset` (and `LEVIRDataset`), handling Sentinel-2 image bands, labeling, tiling, and Albumentations augmentations.
*   `ml/losses.py`: Defines numerically stable loss functions like `BCEDiceLoss`.
*   `ml/train.py`: The complete training loop utilizing mixed precision, optimizer scheduling, early stopping, and MLflow metric tracking.
*   `ml/evaluate.py`: Evaluates the trained model against the validation set.

### GIS & Data Pipeline (`pipeline/`)
*   `pipeline/preprocess.py`: Handles Sentinel-2 ingestion, normalization, spatial reprojection alignment, and NDVI calculation.
*   `pipeline/tiling.py`: Breaks large satellite images into overlapping manageable chunks.
*   `pipeline/stitch.py`: Reassembles predicted patches back into a full-resolution change mask using 2D Gaussian blending.
*   `pipeline/postprocess.py`: Cleans raw probability masks and masks out seasonal vegetation changes using NDVI variance.
*   `pipeline/gis.py`: Vectorizes the cleaned binary raster mask and computes real-world area coverage (`area_km2`).
*   `pipeline/inference.py`: Master orchestration script binding the pipeline into a single `run_inference()` workflow.

### Backend API & Database (`backend/`, `database/`)
*   `database/schema.sql`: Raw SQL establishing PostGIS extension, spatial tables, output logs, and system triggers.
*   `database/models.py`: SQLAlchemy 2.0 declarative ORM models.
*   `backend/schemas.py`: Pydantic models for data validation.
*   `backend/routes.py`: FastAPI endpoints managing database interactions and Celery tasks.
*   `backend/main.py`: Bootstraps the FastAPI application and validates PostgreSQL connectivity.

### Asynchronous Workers (`workers/`)
*   `workers/tasks.py`: Defines the Celery application and background worker functions.
*   `workers/alerts.py`: Evaluates final generated areas against configured thresholds to trigger alerts.

### Frontend Portal (`frontend/`)
*   `frontend/index.html`: A single-file web application utilizing Leaflet.js for AOI definition and visualization.

---

## 🚀 Next Steps
1.  **Train Model:** Run `python ml/train.py` to train the ChangeFormer model on the newly prepared OSCD dataset.
2.  **Evaluate Inference:** After training, run inference again on the T43PGQ Sentinel-2 data to evaluate real-world performance against the baseline.
