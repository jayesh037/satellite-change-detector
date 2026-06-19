import os
import sys
import yaml
import json
import traceback
import redis
from pathlib import Path
from typing import Optional
from celery import Celery

# Add the project root to the python path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.inference import run_inference
from workers.alerts import trigger_alert
from backend.storage import upload_file, get_signed_url
from backend.copernicus import download_scene
from database.models import SessionLocal, DetectionResult, Alert

# Load configuration for Celery Redis broker
config_path = "configs/config.yaml"
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")  # Use cloud Redis if available

try:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
            redis_url = config.get("celery", {}).get("broker_url", redis_url)
            alert_threshold = config.get("alerts", {}).get("threshold_km2", 1.0)
except Exception as e:
    print(f"Failed to load config for Celery, using defaults. Error: {e}")
    alert_threshold = 1.0

# Initialize Celery App
celery_app = Celery(
    "satellite_workers",
    broker=redis_url,
    backend=redis_url
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


@celery_app.task(bind=True, name="workers.tasks.run_detection_task")
def run_detection_task(
    self, 
    result_id: str, 
    t1_folder: Optional[str] = None, 
    t2_folder: Optional[str] = None, 
    t1_scene_id: Optional[str] = None,
    t2_scene_id: Optional[str] = None,
    aoi_geojson: Optional[str] = None,
    recipient_email: Optional[str] = None,
    alert_threshold_km2: float = 1.0,
    t1_date: Optional[str] = None,
    t2_date: Optional[str] = None
) -> dict:
    """
    Celery task to run the end-to-end change detection pipeline.
    
    Steps:
    1. Update status to 'processing' in Database.
    2. Run pipeline/inference.py run_inference()
    3. Update DB with final results.
    4. Check area threshold -> trigger alert if needed
    5. Update status to 'complete' in Database.
    6. On any error: update status to 'failed' in DB, log error

    Args:
        result_id (str): Primary key of the DetectionResult.
        t1_folder (str): Path to T1 images.
        t2_folder (str): Path to T2 images.
        
    Returns:
        dict: A summary of the task execution.
    """
    task_id = self.request.id
    output_dir = Path("outputs") / f"task_{task_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    db = SessionLocal()
    started_at_dt = None
    
    def update_status(status: str, **kwargs):
        try:
            db_result = db.query(DetectionResult).filter(DetectionResult.id == int(result_id)).first()
            if db_result:
                db_result.status = status
                for k, v in kwargs.items():
                    setattr(db_result, k, v)
                db.commit()
        except Exception as e:
            print(f"Error updating DB status: {e}")
            db.rollback()

    # Record start time and set initial status
    from datetime import datetime as _dt
    started_at_dt = _dt.utcnow()
    update_status("processing", status_message="Initializing job...", started_at=started_at_dt)
    
    try:
        print(f"Starting change detection task {task_id} for Result ID {result_id}")
        
        # Load config if needed
        app_config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                app_config = yaml.safe_load(f) or {}

        # 1b. Download scenes if scene IDs are provided
        if t1_scene_id and t2_scene_id:
            print(f"Downloading T1 scene: {t1_scene_id}")
            update_status("processing", status_message="Downloading satellite bands...")
            t1_year = t1_date[:4] if t1_date else "2021"
            t1_folder = download_scene(t1_scene_id, t1_year, app_config)
            
            print(f"Downloading T2 scene: {t2_scene_id}")
            t2_year = t2_date[:4] if t2_date else "2023"
            t2_folder = download_scene(t2_scene_id, t2_year, app_config)
            
        print(f"T1 Folder: {t1_folder}")
        print(f"T2 Folder: {t2_folder}")
        print(f"AOI GeoJSON received: {'yes' if aoi_geojson else 'no'}")
        
        # 2. Run pipeline/inference.py run_inference()
        # Assume checkpoint is in the default location
        checkpoint_path = "checkpoints/best_model.pth"
        
        update_status("processing", status_message="Preprocessing images...")
        
        update_status("processing", status_message="Running ML inference...")
        inference_results = run_inference(
            t1_folder=t1_folder,
            t2_folder=t2_folder,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            aoi_geojson=aoi_geojson
            # We use default inference parameters, but these could be passed via request if needed
        )
        
        changed_area_km2 = inference_results["changed_area_km2"]
        
        # 3. Upload results to B2 and save to DB
        update_status("processing", status_message="Uploading results...")
        geojson_local = Path(inference_results["geojson_path"])
        geotiff_local = Path(inference_results["geotiff_path"])

        geojson_key = None
        geotiff_key = None

        if geojson_local.exists():
            geojson_key = upload_file(
                str(geojson_local),
                f"results/{task_id}/change_mask.geojson"
            )
            print(f"GeoJSON uploaded to B2: {geojson_key}")

        if geotiff_local.exists():
            geotiff_key = upload_file(
                str(geotiff_local),
                f"results/{task_id}/change_mask.tif"
            )
            print(f"GeoTIFF uploaded to B2: {geotiff_key}")

        # Calculate processing duration in minutes
        from datetime import datetime as _dt2
        processing_minutes = None
        if started_at_dt:
            delta = _dt2.utcnow() - started_at_dt
            processing_minutes = round(delta.total_seconds() / 60, 1)

        update_status("complete",
            status_message=f"Detection complete — {changed_area_km2:.4f} km² changed",
            change_mask_path=geotiff_key or str(geotiff_local),
            geojson_path=geojson_key or str(geojson_local),
            geojson_b2_key=geojson_key,
            geotiff_b2_key=geotiff_key,
            changed_area_km2=changed_area_km2
        )
        
        # 4. Check area threshold -> trigger alert if needed
        if changed_area_km2 >= alert_threshold_km2:
            try:
                message = f"Significant change detected! {changed_area_km2:.4f} km2 changed between {t1_date} and {t2_date}."
                new_alert = Alert(
                    result_id=int(result_id),
                    message=message
                )
                db.add(new_alert)
                db.commit()
                print(f"Alert recorded in database for Result ID {result_id}")
                trigger_alert(
                    result_id=result_id,
                    changed_area_km2=changed_area_km2,
                    threshold_km2=alert_threshold_km2,
                    recipient_email=recipient_email,
                    task_id=task_id,
                    tile="T43PGQ",
                    t1_date=t1_date or "",
                    t2_date=t2_date or "",
                    processing_minutes=processing_minutes
                )
            except Exception as e:
                print(f"Error inserting alert into DB: {e}")
                db.rollback()
        
        print(f"Task {task_id} completed successfully. Changed Area: {changed_area_km2:.4f} km²")
        return {"status": "success", "task_id": task_id, "changed_area_km2": changed_area_km2}
        
    except Exception as e:
        # 6. On any error: update status to "failed", log error
        error_trace = traceback.format_exc()
        print(f"Task {task_id} FAILED with error: {e}")
        print(error_trace)
        
        update_status("failed", status_message=f"Task failed: {str(e)[:200]}")
            
        # Re-raise the exception so Celery marks the task as failed
        raise e
    finally:
        db.close()


@celery_app.task(bind=True)
def download_scene_task(self, download_id, title, year):
    r = redis.Redis.from_url(redis_url)
    key = f"dl_progress:{self.request.id}"
    
    # Store initial state
    r.set(key, json.dumps({"status": "downloading", "progress": 0}))

    def progress_callback(percent):
        r.set(key, json.dumps({"status": "downloading", "progress": percent}))

    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                app_config = yaml.safe_load(f) or {}
        else:
            app_config = {}
            
        safe_path = download_scene(
            download_id=download_id,
            year=year,
            config=app_config,
            progress_callback=progress_callback
        )
        r.set(key, json.dumps({"status": "complete", "progress": 100, "safe_path": str(safe_path)}))
        return {"status": "complete", "safe_path": str(safe_path)}
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Task {self.request.id} FAILED with error: {e}")
        print(error_trace)
        r.set(key, json.dumps({"status": "failed", "progress": 0, "error": str(e)}))
        raise e
