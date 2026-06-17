import os
import yaml
from datetime import datetime, date
from typing import List, Optional, Any

from sqlalchemy import String, Text, ForeignKey, Float, Boolean, Date, DateTime, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.sql import func
from geoalchemy2 import Geometry


class Base(DeclarativeBase):
    """Base class for SQLAlchemy declarative models."""
    pass

class User(Base):
    """
    User model.
    Stores simple user credentials for authentication and emailing.
    """
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )


class AOI(Base):
    """
    Area of Interest (AOI) model.
    Stores the spatial polygon bounding the user's requested region.
    """
    __tablename__ = "aois"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Using GeoAlchemy2 for PostGIS Geometry
    geom: Mapped[Any] = mapped_column(
        Geometry(geometry_type="POLYGON", srid=4326), 
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )

    # Relationships
    results: Mapped[List["DetectionResult"]] = relationship(
        back_populates="aoi", 
        cascade="all, delete-orphan"
    )


class DetectionResult(Base):
    """
    Detection Result model.
    Stores the status and output metadata of a satellite change detection run.
    """
    __tablename__ = "detection_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    aoi_id: Mapped[int] = mapped_column(
        ForeignKey("aois.id", ondelete="CASCADE"), 
        nullable=False
    )
    task_id: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    
    change_mask_path: Mapped[Optional[str]] = mapped_column(String(512))
    geojson_path: Mapped[Optional[str]] = mapped_column(String(512))
    geojson_b2_key: Mapped[Optional[str]] = mapped_column(String(512))
    geotiff_b2_key: Mapped[Optional[str]] = mapped_column(String(512))
    changed_area_km2: Mapped[Optional[float]] = mapped_column(Float)
    
    t1_date: Mapped[Optional[date]] = mapped_column(Date)
    t2_date: Mapped[Optional[date]] = mapped_column(Date)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )

    # Relationships
    aoi: Mapped["AOI"] = relationship(back_populates="results")
    alerts: Mapped[List["Alert"]] = relationship(
        back_populates="result", 
        cascade="all, delete-orphan"
    )


class Alert(Base):
    """
    Alert model.
    Records triggered alerts when significant change areas are detected.
    """
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    result_id: Mapped[int] = mapped_column(
        ForeignKey("detection_results.id", ondelete="CASCADE"), 
        nullable=False
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    acknowledged: Mapped[bool] = mapped_column(
        Boolean, 
        default=False, 
        server_default="false"
    )
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    result: Mapped["DetectionResult"] = relationship(back_populates="alerts")


def get_engine() -> Any:
    """
    Creates and returns the SQLAlchemy engine based on configuration.
    Falls back to a default localhost URL if the config is not present.
    """
    config_path = "configs/config.yaml"
    db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/satchange")  # Use cloud DB if available
    
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
                if "database" in config and "url" in config["database"]:
                    db_url = config["database"]["url"]
    except Exception as e:
        print(f"Failed to load config, using default database URL. Error: {e}")
        
    return create_engine(db_url, echo=False)


# Initialize global session maker
engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    Generator dependency to provide a clean database session for requests or tasks.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
