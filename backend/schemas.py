from datetime import datetime, date
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class AOICreate(BaseModel):
    """Schema for creating a new Area of Interest."""
    name: str = Field(..., description="Name of the AOI")
    description: Optional[str] = Field(None, description="Optional description of the AOI")
    geom: Dict[str, Any] = Field(
        ..., 
        description="GeoJSON Polygon geometry for the AOI",
        example={"type": "Polygon", "coordinates": [[[0,0], [0,1], [1,1], [1,0], [0,0]]]}
    )


class DetectionRequest(BaseModel):
    """Schema for requesting a new change detection task."""
    aoi: AOICreate
    t1_folder: str = Field(..., description="Path to the T1 (before) Sentinel-2 images folder")
    t2_folder: str = Field(..., description="Path to the T2 (after) Sentinel-2 images folder")
    t1_date: Optional[date] = Field(None, description="Date of the T1 images")
    t2_date: Optional[date] = Field(None, description="Date of the T2 images")
    aoi_geojson: Optional[str] = None
    recipient_email: Optional[str] = None
    alert_threshold_km2: Optional[float] = 1.0


class DetectionResponse(BaseModel):
    """Schema for the response after submitting a detection task."""
    task_id: str = Field(..., description="The Celery task ID")
    message: str = Field(..., description="Status message")


class ResultResponse(BaseModel):
    """Schema for retrieving the results of a detection task."""
    task_id: str
    status: str
    changed_area_km2: Optional[float] = None
    geojson_path: Optional[str] = None
    geojson_url: Optional[str] = None
    geotiff_url: Optional[str] = None
    t1_date: Optional[date] = None
    t2_date: Optional[date] = None

    class Config:
        from_attributes = True


class AlertResponse(BaseModel):
    """Schema for returning alert details."""
    id: int
    result_id: int
    message: str
    triggered_at: datetime
    acknowledged: bool

    class Config:
        from_attributes = True

class SceneSearchRequest(BaseModel):
    tile_id: str = "T43PGQ"
    date_from: str          # "YYYY-MM-DD"
    date_to: str
    max_cloud_cover: float = 20.0

class SceneResult(BaseModel):
    scene_id: str
    title: str
    date: str
    cloud_cover: float
    tile: str
    download_id: str
    bbox: list[float]
    size_mb: float

class SceneSearchResponse(BaseModel):
    scenes: list[SceneResult]
    total: int

class DownloadRequest(BaseModel):
    download_id: str
    title: str
    year: str               # e.g. "2026"

class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    user_id: int
    username: str
    email: str

    class Config:
        from_attributes = True
