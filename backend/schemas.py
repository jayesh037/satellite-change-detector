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
