import os
import sys
import json
import uuid
from typing import List
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

# Add the project root to the python path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.schemas import DetectionRequest, DetectionResponse, ResultResponse, AlertResponse
from workers.tasks import run_detection_task

router = APIRouter()

# In-memory storage for tasks
TASKS_DB = {}
ALERTS_DB = []

@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint to verify API availability."""
    return {"status": "ok", "service": "satellite-change-detector"}

@router.post("/detect-change", response_model=DetectionResponse, status_code=status.HTTP_202_ACCEPTED)
def detect_change(request: DetectionRequest):
    """
    Initiates a change detection task.
    Saves the AOI and task info to in-memory dictionary and queues the processing task in Celery.
    """
    result_id = str(uuid.uuid4())
    
    # Queue the Celery task
    task = run_detection_task.delay(
        result_id=result_id,
        t1_folder=request.t1_folder,
        t2_folder=request.t2_folder
    )
    
    TASKS_DB[task.id] = {
        "id": result_id,
        "task_id": task.id,
        "status": "pending",
        "t1_date": request.t1_date,
        "t2_date": request.t2_date,
        "aoi_name": request.aoi.name
    }

    return DetectionResponse(
        task_id=task.id,
        message="Change detection task queued successfully."
    )

@router.get("/results/{task_id}", response_model=ResultResponse)
def get_result(task_id: str):
    """
    Retrieves the status and output metadata of a change detection task using its task_id.
    """
    result = TASKS_DB.get(task_id)
    
    if not result:
        # Worker might have saved to output JSON before this API restarted (in-memory lost)
        # So check JSON
        result_file = Path("outputs") / f"task_{task_id}" / "result.json"
        if result_file.exists():
            try:
                with open(result_file, "r") as f:
                    result = json.load(f)
                    TASKS_DB[task_id] = result
            except:
                pass
                
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Task with ID {task_id} not found."
        )
        
    # Read updated status from worker output if it's pending
    if result.get("status") in ["pending", "processing"]:
        result_file = Path("outputs") / f"task_{task_id}" / "result.json"
        if result_file.exists():
            try:
                with open(result_file, "r") as f:
                    worker_result = json.load(f)
                    result.update(worker_result)
            except:
                pass

    return result

@router.get("/alerts", response_model=List[AlertResponse])
def get_alerts():
    """
    Retrieves all unacknowledged change detection alerts.
    """
    return [a for a in ALERTS_DB if not a.get("acknowledged", False)]