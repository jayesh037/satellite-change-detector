import os
import sys
import yaml
import json
import traceback
from pathlib import Path
from celery import Celery

# Add the project root to the python path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.inference import run_inference
from workers.alerts import trigger_alert


# Load configuration for Celery Redis broker
config_path = "configs/config.yaml"
redis_url = "redis://localhost:6379/0"  # Default fallback

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
def run_detection_task(self, result_id: str, t1_folder: str, t2_folder: str) -> dict:
    """
    Celery task to run the end-to-end change detection pipeline.
    
    Steps:
    1. Update status to 'processing' (save to JSON)
    2. Run pipeline/inference.py run_inference()
    3. Save results to JSON file
    4. Check area threshold -> trigger alert if needed
    5. Update status to 'complete'
    6. On any error: update status to 'failed', log error

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
    result_file = output_dir / "result.json"
    
    def save_state(state_dict):
        with open(result_file, "w") as f:
            json.dump(state_dict, f)
            
    state = {
        "id": result_id,
        "task_id": task_id,
        "status": "processing"
    }
    save_state(state)
    
    try:
        print(f"Starting change detection task {task_id} for Result ID {result_id}")
        print(f"T1 Folder: {t1_folder}")
        print(f"T2 Folder: {t2_folder}")
        
        # 2. Run pipeline/inference.py run_inference()
        # Assume checkpoint is in the default location
        checkpoint_path = "checkpoints/best_model.pth"
        
        inference_results = run_inference(
            t1_folder=t1_folder,
            t2_folder=t2_folder,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            # We use default inference parameters, but these could be passed via request if needed
        )
        
        changed_area_km2 = inference_results["changed_area_km2"]
        
        # 3. Save results to JSON file
        state.update({
            "change_mask_path": str(inference_results["geotiff_path"]),
            "geojson_path": str(inference_results["geojson_path"]),
            "changed_area_km2": changed_area_km2
        })
        
        # 4. Check area threshold -> trigger alert if needed
        # We re-read threshold from config locally in case it changed without restarting worker
        current_threshold = alert_threshold
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                    current_threshold = cfg.get("alerts", {}).get("threshold_km2", alert_threshold)
        except Exception:
            pass
            
        trigger_alert(
            result_id=result_id, 
            changed_area_km2=changed_area_km2, 
            threshold_km2=current_threshold
        )
        
        # 5. Update status to "complete"
        state["status"] = "complete"
        save_state(state)
        
        print(f"Task {task_id} completed successfully. Changed Area: {changed_area_km2:.4f} km²")
        return {"status": "success", "task_id": task_id, "changed_area_km2": changed_area_km2}
        
    except Exception as e:
        # 6. On any error: update status to "failed", log error
        error_trace = traceback.format_exc()
        print(f"Task {task_id} FAILED with error: {e}")
        print(error_trace)
        
        state["status"] = "failed"
        save_state(state)
            
        # Re-raise the exception so Celery marks the task as failed
        raise e