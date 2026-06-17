import os
import sys
import json
import yaml
import redis
import uuid
from typing import List
from pathlib import Path

from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from shapely.geometry import shape

# Add the project root to the python path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.schemas import (
    DetectionRequest, DetectionResponse, ResultResponse, AlertResponse,
    SceneSearchRequest, SceneSearchResponse, DownloadRequest,
    UserRegister, UserLogin, UserResponse
)
from database.models import get_db, AOI, DetectionResult, Alert, User
from backend.copernicus import search_scenes
from backend.storage import get_signed_url, file_exists
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter()

from backend.tiles import get_tile

@router.get("/tiles/{layer}/{z}/{x}/{y}")
def serve_tile(layer: str, z: int, x: int, y: int):
    return get_tile(layer, z, x, y)

@router.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user: UserRegister, db: Session = Depends(get_db)):
    """Registers a new user."""
    # Check if email or username exists
    existing_email = db.query(User).filter(User.email == user.email).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    existing_username = db.query(User).filter(User.username == user.username).first()
    if existing_username:
        raise HTTPException(status_code=400, detail="Username already taken")

    if len(user.password) < 6 or not any(c.isdigit() for c in user.password) or not any(c.isupper() for c in user.password):
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters with 1 number and 1 uppercase letter")

    hashed_pw = pwd_context.hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_pw
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return UserResponse(
        user_id=db_user.id,
        username=db_user.username,
        email=db_user.email
    )

@router.post("/auth/login", response_model=UserResponse)
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    """Authenticates a user and returns simple session info."""
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not pwd_context.verify(user.password, db_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    return UserResponse(
        user_id=db_user.id,
        username=db_user.username,
        email=db_user.email
    )

@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint to verify API availability."""
    return {"status": "ok", "service": "satellite-change-detector"}

@router.post("/detect-change", response_model=DetectionResponse, status_code=status.HTTP_202_ACCEPTED)
def detect_change(request: DetectionRequest, db: Session = Depends(get_db)):
    """
    Initiates a change detection task.
    Saves the AOI and task info to the PostgreSQL database and queues the processing task in Celery.
    """
    # 1. Create AOI geometry from request
    geom_dict = request.aoi.geom
    shapely_geom = shape(geom_dict)
    wkt_geom = f"SRID=4326;{shapely_geom.wkt}"
    
    db_aoi = AOI(
        name=request.aoi.name,
        description=request.aoi.description,
        geom=wkt_geom
    )
    db.add(db_aoi)
    db.commit()
    db.refresh(db_aoi)
    
    # 2. Create DetectionResult placeholder
    db_result = DetectionResult(
        aoi_id=db_aoi.id,
        status="pending",
        t1_date=request.t1_date,
        t2_date=request.t2_date
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
    
    # 3. Queue the Celery task
    from workers.tasks import run_detection_task
    task = run_detection_task.delay(
        result_id=str(db_result.id),
        t1_folder=request.t1_folder,
        t2_folder=request.t2_folder,
        aoi_geojson=request.aoi_geojson,
        recipient_email=request.recipient_email,
        alert_threshold_km2=request.alert_threshold_km2 if request.alert_threshold_km2 is not None else 1.0,
        t1_date=request.t1_date.isoformat() if request.t1_date else "2021-03-14",
        t2_date=request.t2_date.isoformat() if request.t2_date else "2023-03-14"
    )
    
    # 4. Update task_id in db
    db_result.task_id = task.id
    db.commit()

    return DetectionResponse(
        task_id=task.id,
        message="Change detection task queued successfully."
    )

@router.get("/results/{task_id}", response_model=ResultResponse)
def get_result(task_id: str, db: Session = Depends(get_db)):
    """
    Retrieves the status and output metadata of a change detection task using its task_id.
    """
    result = db.query(DetectionResult).filter(DetectionResult.task_id == task_id).first()
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Task with ID {task_id} not found."
        )

    result_response = ResultResponse.model_validate(result)

    if result.geojson_b2_key and file_exists(result.geojson_b2_key):
        result_response.geojson_url = get_signed_url(result.geojson_b2_key)
    if result.geotiff_b2_key and file_exists(result.geotiff_b2_key):
        result_response.geotiff_url = get_signed_url(result.geotiff_b2_key)

    return result_response

@router.get("/alerts", response_model=List[AlertResponse])
def get_alerts(db: Session = Depends(get_db)):
    """
    Retrieves all unacknowledged change detection alerts.
    """
    alerts = db.query(Alert).filter(Alert.acknowledged == False).all()
    return alerts

@router.post("/alerts/acknowledge-all")
def acknowledge_all_alerts(db: Session = Depends(get_db)):
    """
    Marks all unacknowledged alerts as acknowledged.
    """
    alerts = db.query(Alert).filter(Alert.acknowledged == False).all()
    for alert in alerts:
        alert.acknowledged = True
    db.commit()
    return {"message": f"{len(alerts)} alerts acknowledged."}

@router.post("/alerts/{id}/acknowledge", response_model=AlertResponse)
def acknowledge_alert(id: int, db: Session = Depends(get_db)):
    """
    Marks a specific alert as acknowledged.
    """
    alert = db.query(Alert).filter(Alert.id == id).first()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Alert with ID {id} not found."
        )
    alert.acknowledged = True
    db.commit()
    db.refresh(alert)
    return alert

@router.post("/scenes/search", response_model=SceneSearchResponse)
def search_scenes_endpoint(request: SceneSearchRequest):
    """
    Searches for Sentinel-2 scenes on Copernicus Data Space.
    """
    try:
        config_path = "configs/config.yaml"
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}

        results = search_scenes(
            tile_id=request.tile_id,
            date_from=request.date_from,
            date_to=request.date_to,
            max_cloud_cover=request.max_cloud_cover,
            config=config
        )
        return SceneSearchResponse(scenes=results, total=len(results))
    except Exception as e:
        return SceneSearchResponse(scenes=[], total=0)

@router.post("/scenes/download")
def download_scene_endpoint(request: DownloadRequest):
    """
    Queues a scene for download.
    """
    from workers.tasks import download_scene_task
    task = download_scene_task.delay(request.download_id, request.title, request.year)
    return {"task_id": task.id}

@router.get("/scenes/download/{task_id}")
def get_download_status(task_id: str):
    """
    Retrieves the status and progress of a scene download task.
    """
    config_path = "configs/config.yaml"
    redis_url = "redis://localhost:6379/0"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
            redis_url = config.get("celery", {}).get("broker_url", redis_url)
    
    r = redis.Redis.from_url(redis_url)
    key = f"dl_progress:{task_id}"
    
    data = r.get(key)
    if data:
        return json.loads(data)
    else:
        return {"status": "pending", "progress": 0}

@router.get("/timeseries/summary")
def get_timeseries_summary():
    """
    Reads the outputs/timeseries/ folder and builds a summary of changed areas.
    If outputs/timeseries/summary.json exists, returns its content.
    Otherwise, scans subdirectories (e.g. 2021_2022) for change_polygons.geojson,
    sums the area, and returns a dynamic summary.
    """
    timeseries_dir = Path("outputs/timeseries")
    
    # Return empty summary if the base directory doesn't exist
    if not timeseries_dir.exists():
        return {"pairs": [], "results": []}
        
    summary_path = timeseries_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Build dynamically if summary.json is missing
    results = []
    pairs = []
    
    # Look for subdirectories named like YYYY_YYYY
    for period_dir in timeseries_dir.iterdir():
        if period_dir.is_dir() and "_" in period_dir.name:
            parts = period_dir.name.split("_")
            if len(parts) == 2:
                y1, y2 = parts
                geojson_path = period_dir / "change_polygons.geojson"
                
                area_km2 = 0.0
                status = "failed"
                
                if geojson_path.exists():
                    try:
                        with open(geojson_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            features = data.get("features", [])
                            for feature in features:
                                props = feature.get("properties", {})
                                area_km2 += props.get("area_km2", 0.0)
                        status = "success"
                    except Exception as e:
                        print(f"Error reading geojson {geojson_path}: {e}")
                
                result_obj = {
                    "period": period_dir.name,
                    "y1": y1,
                    "y2": y2,
                    "changed_area_km2": area_km2,
                    "status": status
                }
                results.append(result_obj)
                
                pairs.append({
                    "label": f"{y1}→{y2}",
                    "changed_area_km2": area_km2,
                    "y1": y1 # Used for sorting
                })

    # Sort chronologically based on y1
    results.sort(key=lambda x: x["y1"])
    pairs.sort(key=lambda x: x["y1"])
    
    # Remove y1 from pairs after sorting to match the exact requested format
    for p in pairs:
        p.pop("y1", None)

    return {"pairs": pairs, "results": results}

@router.get("/timeseries/{period}/geojson")
def get_timeseries_geojson(period: str):
    """
    Reads and returns the outputs/timeseries/{period}/change_polygons.geojson file.
    """
    geojson_path = Path(f"outputs/timeseries/{period}/change_polygons.geojson")
    if not geojson_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"GeoJSON for period {period} not found."
        )
    with open(geojson_path, "r", encoding="utf-8") as f:
        return json.load(f)