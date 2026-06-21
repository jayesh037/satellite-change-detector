import os
import sys
import json
import yaml
import redis
import uuid
import random
import string
from datetime import datetime, timedelta, timezone
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
    UserRegister, UserLogin, UserResponse,
    ForgotPasswordRequest, ResetPasswordRequest
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


def _send_otp_email(recipient_email: str, otp: str) -> bool:
    """Send the password-reset OTP via SMTP using the existing alert SMTP config."""
    import smtplib
    import traceback
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from workers.alerts import load_alert_config
    try:
        config = load_alert_config()
        import os as _os
        _email_enabled = config.get("email_enabled", False) or _os.environ.get("ALERT_EMAIL_ENABLED", "").lower() == "true"
        if not _email_enabled:
            print("[reset-pw] Email disabled in config — OTP not sent.")
            return False

        html_body = f"""
        <div style="background:#0d1117;color:#c9d1d9;font-family:Arial,sans-serif;padding:20px;">
          <div style="max-width:480px;margin:0 auto;background:#161b22;border-radius:8px;border:1px solid #30363d;overflow:hidden;">
            <div style="padding:20px;border-bottom:1px solid #30363d;text-align:center;">
              <h2 style="margin:0;color:#fff;">🔑 Password Reset</h2>
            </div>
            <div style="padding:24px;">
              <p style="font-size:14px;line-height:1.6;">You requested a password reset for your Satellite Change Detector account.<br>
              Use the OTP below within <strong>15 minutes</strong>:</p>
              <div style="background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:20px;text-align:center;margin:20px 0;">
                <span style="font-size:36px;font-weight:bold;letter-spacing:8px;color:#58a6ff;font-family:monospace;">{otp}</span>
              </div>
              <p style="font-size:12px;color:#8b949e;">If you did not request this, you can safely ignore this email.</p>
            </div>
            <div style="background:#0d1117;padding:15px;border-top:1px solid #30363d;text-align:center;font-size:12px;color:#8b949e;">
              Satellite Change Detection System
            </div>
          </div>
        </div>
        """
        text_body = f"Your password reset OTP is: {otp}\nIt expires in 15 minutes."

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "🔑 Your Password Reset OTP"
        msg["From"] = f"{config.get('sender_name', 'Satellite Change Detector')} <{config.get('smtp_user', '')}>"
        msg["To"] = recipient_email
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        smtp_host = config.get("smtp_host", "smtp.gmail.com")
        smtp_user = config.get("smtp_user", "") or os.environ.get("SMTP_USER", "")
        smtp_password = config.get("smtp_password", "") or os.environ.get("SMTP_PASSWORD", "")
        # Use SSL on 465 (Render blocks 587)
        with smtplib.SMTP_SSL(smtp_host, 465) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, recipient_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"[reset-pw] Failed to send OTP email: {e}")
        traceback.print_exc()
        return False


@router.post("/auth/forgot-password")
def forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Generate a 6-digit OTP, store it with a 15-min expiry, and email it.
    Always returns the same message to avoid revealing whether an account exists."""
    GENERIC_RESPONSE = {"message": "If an account exists for that email, a reset link has been sent."}

    db_user = db.query(User).filter(User.email == request.email).first()
    if not db_user:
        return GENERIC_RESPONSE

    otp = "".join(random.choices(string.digits, k=6))
    expiry = datetime.now(timezone.utc) + timedelta(minutes=15)

    db_user.reset_otp = otp
    db_user.reset_otp_expiry = expiry
    db.commit()

    _send_otp_email(db_user.email, otp)
    return GENERIC_RESPONSE


@router.post("/auth/reset-password")
def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Verify OTP, update password, and clear OTP fields."""
    db_user = db.query(User).filter(User.email == request.email).first()
    if not db_user or not db_user.reset_otp or not db_user.reset_otp_expiry:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")

    now = datetime.now(timezone.utc)
    # Make reset_otp_expiry timezone-aware if stored without tz
    expiry = db_user.reset_otp_expiry
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)

    if expiry < now:
        raise HTTPException(status_code=400, detail="OTP has expired. Please request a new one.")

    if db_user.reset_otp != request.otp:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")

    new_pw = request.new_password
    if len(new_pw) < 6 or not any(c.isdigit() for c in new_pw) or not any(c.isupper() for c in new_pw):
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 6 characters with 1 number and 1 uppercase letter"
        )

    db_user.hashed_password = pwd_context.hash(new_pw)
    db_user.reset_otp = None
    db_user.reset_otp_expiry = None
    db.commit()

    return {"message": "Password reset successful"}

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
    
    # 3. Queue the Celery task via send_task (avoids importing torch on Render)
    from celery import Celery as _Celery
    import os as _os
    _redis_url = _os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    if _redis_url.startswith("rediss://") and "ssl_cert_reqs" not in _redis_url:
        _sep = "&" if "?" in _redis_url else "?"
        _redis_url = f"{_redis_url}{_sep}ssl_cert_reqs=CERT_NONE"
    _app = _Celery(broker=_redis_url, backend=_redis_url)
    import ssl as _ssl
    if _redis_url.startswith("rediss://"):
        _app.conf.broker_use_ssl = {"ssl_cert_reqs": _ssl.CERT_NONE}
        _app.conf.redis_backend_use_ssl = {"ssl_cert_reqs": _ssl.CERT_NONE}
    task = _app.send_task(
        "workers.tasks.run_detection_task",
        kwargs=dict(
            result_id=str(db_result.id),
            t1_folder=request.t1_folder,
            t2_folder=request.t2_folder,
            t1_scene_id=request.t1_scene_id,
            t2_scene_id=request.t2_scene_id,
            aoi_geojson=request.aoi_geojson,
            recipient_email=request.recipient_email,
            alert_threshold_km2=request.alert_threshold_km2 if request.alert_threshold_km2 is not None else 1.0,
            t1_date=request.t1_date.isoformat() if request.t1_date else "2021-03-14",
            t2_date=request.t2_date.isoformat() if request.t2_date else "2023-03-14"
        )
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

@router.get("/debug/env")
def debug_env():
    """Temporary debug endpoint to check if environment variables are set."""
    return {
        "copernicus_client_id": bool(os.environ.get("COPERNICUS_CLIENT_ID")),
        "copernicus_username": bool(os.environ.get("COPERNICUS_USERNAME")),
        "copernicus_password": bool(os.environ.get("COPERNICUS_PASSWORD")),
    }


@router.get("/timeseries/summary")
def get_timeseries_summary():
    """Build timeseries summary from B2 stored GeoJSON files."""
    import httpx as _httpx
    periods = [
        ("2019","2020"), ("2020","2021"), ("2021","2022"),
        ("2022","2023"), ("2023","2024"), ("2024","2025"), ("2025","2026")
    ]
    results = []
    pairs = []
    for y1, y2 in periods:
        period = f"{y1}_{y2}"
        b2_key = f"timeseries/{period}/change_polygons.geojson"
        area_km2 = 0.0
        status_val = "failed"
        try:
            signed_url = get_signed_url(b2_key)
            r = _httpx.get(signed_url, timeout=60)
            if r.status_code == 200:
                data = r.json()
                features = data.get("features", [])
                for feature in features:
                    props = feature.get("properties", {})
                    area_km2 += props.get("area_km2", 0.0)
                status_val = "success"
        except Exception as e:
            print(f"Timeseries {period} error: {e}")
        results.append({"period": period, "y1": y1, "y2": y2,
                        "changed_area_km2": area_km2, "status": status_val})
        pairs.append({"label": f"{y1}\u2192{y2}", "changed_area_km2": area_km2})
    return {"pairs": pairs, "results": results}

@router.get("/timeseries/{period}/geojson")
def get_timeseries_geojson(period: str):
    """Fetch timeseries GeoJSON from B2."""
    import httpx
    b2_key = f"timeseries/{period}/change_polygons.geojson"
    if not file_exists(b2_key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"GeoJSON for period {period} not found."
        )
    signed_url = get_signed_url(b2_key)
    r = httpx.get(signed_url, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=404, detail="Failed to fetch from B2")
    return r.json()

