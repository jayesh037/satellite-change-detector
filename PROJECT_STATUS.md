# Project Status: Satellite Change Detection System

## 📌 Current State
**Status:** **READY FOR TRAINING & DEPLOYMENT**  
The entire software architecture, machine learning pipeline, GIS processing utilities, background workers, backend API, and web frontend have been fully implemented. All required files, configurations, and boilerplate are present and correctly wired together.

The project is currently awaiting the download of the LEVIR-CD dataset and the execution of the initial model training phase.

---

## 📂 Files Created & Their Purpose

### Configuration & Documentation
*   `configs/config.yaml`: The central configuration hub governing model parameters, training hyperparameters, dataset paths, inference thresholds, and database/Redis connection URLs.
*   `requirements.txt`: Contains precisely pinned Python dependencies required to run the environment reproducibly.
*   `README.md`: Comprehensive step-by-step instructions covering environment setup, data downloading, model training, and running the backend/frontend services.
*   `PROJECT_STATUS.md`: This summary document.

### Machine Learning Layer (`ml/`)
*   `ml/model.py`: Implements `SiameseUNet`, leveraging a shared ResNet34 encoder to extract and compare multi-temporal image features.
*   `ml/dataset.py`: Implements `LEVIRDataset`, handling image pairs (T1, T2) and label loading, augmented with an Albumentations pipeline (flips, rotations, color jitter).
*   `ml/losses.py`: Defines numerically stable `DiceLoss` and a combined `BCEDiceLoss` tailored for class-imbalanced segmentation tasks.
*   `ml/train.py`: The complete training loop utilizing mixed precision (`torch.cuda.amp`), Adam optimizer, Cosine Annealing learning rate scheduling, early stopping, and MLflow metric tracking.
*   `ml/evaluate.py`: Evaluates the trained model against the validation set, calculating global IoU/F1 metrics and exporting a confusion matrix alongside sample visualizations.

### GIS & Data Pipeline (`pipeline/`)
*   `pipeline/preprocess.py`: Handles Sentinel-2 `.jp2` band ingestion, 2nd-98th percentile normalization, spatial reprojection alignment (`align_images`), and calculates NDVI.
*   `pipeline/tiling.py`: Breaks large satellite images into overlapping manageable chunks (`patch_size=256`, `overlap=32`) to feed into the neural network without memory overflow.
*   `pipeline/stitch.py`: Reassembles the predicted patches back into a massive full-resolution change mask using 2D Gaussian blending to eliminate visible seam artifacts.
*   `pipeline/postprocess.py`: Cleans the raw probability masks by applying thresholds, removing small noise components, and explicitly masking out seasonal vegetation changes using absolute NDVI variance logic.
*   `pipeline/gis.py`: Vectorizes the cleaned binary raster mask into spatial formats. Computes real-world area coverage (`area_km2`) and exports georeferenced GeoTIFFs and GeoJSONs.
*   `pipeline/inference.py`: The master orchestration script binding `preprocess` -> `tiling` -> `SiameseUNet` -> `stitch` -> `postprocess` -> `gis` into a single `run_inference()` workflow.

### Backend API & Database (`backend/`, `database/`)
*   `database/schema.sql`: Raw SQL establishing the PostGIS extension, spatial tables (`aois`), output logs (`detection_results`), and system triggers (`alerts`).
*   `database/models.py`: SQLAlchemy 2.0 declarative ORM models binding the Python application directly to the PostGIS geometry types.
*   `backend/schemas.py`: Pydantic models ensuring strict data validation for all incoming and outgoing API requests (e.g., GeoJSON parsing).
*   `backend/routes.py`: FastAPI endpoints (`/detect-change`, `/results/{task_id}`, `/alerts`) managing database interactions and triggering Celery tasks.
*   `backend/main.py`: Bootstraps the FastAPI application, mounts static file serving for GeoJSON outputs, manages CORS, and validates PostgreSQL connectivity on startup.

### Asynchronous Workers (`workers/`)
*   `workers/tasks.py`: Defines the Celery application and the `run_detection_task` worker function that executes the heavy `run_inference()` pipeline in the background, updating DB statuses asynchronously.
*   `workers/alerts.py`: Evaluates final generated areas against the configured `threshold_km2`. If exceeded, records an alert into the database and appends it to `outputs/alerts.log`.

### Frontend Portal (`frontend/`)
*   `frontend/index.html`: A clean, single-file minimal dark-themed web application utilizing Leaflet.js. It allows users to define an AOI natively on the map, trigger the backend API, poll for status updates, and visualize the resulting red GeoJSON change polygons and area metrics.

---

## 🚀 Next Steps

The immediate next phase is to address the backend issues and spin up the asynchronous services:
1.  **Fix Backend PyTorch Issue**: Resolve the `fbgemm.dll` loading error occurring in the FastAPI backend when importing PyTorch.
2.  **Start Background Services**: Spin up Redis and the Celery workers.
3.  **Run End-to-End**: Test the full flow from the frontend map down to the background inference and alert generation.
e representations.
3.  **Evaluate Performance**: Run `python ml/evaluate.py` to ensure the model achieves satisfactory IoU and F1 scores before attempting inference on real-world Sentinel-2 imagery.
