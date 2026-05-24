import json
from pathlib import Path
from typing import Any, Union, Dict

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom
from rasterio.crs import CRS
from shapely.geometry import shape
from pyproj import Geod


def save_geotiff(
    mask: np.ndarray, 
    transform: Any, 
    crs: CRS, 
    output_path: Union[str, Path]
) -> None:
    """
    Saves a binary mask as a georeferenced GeoTIFF.

    Args:
        mask (np.ndarray): The binary prediction mask of shape (H, W).
        transform (Any): Rasterio Affine transform.
        crs (CRS): Rasterio Coordinate Reference System.
        output_path (Union[str, Path]): Path where the GeoTIFF will be saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure mask is at least 2D
    if len(mask.shape) == 3:
        mask = mask.squeeze()
        
    # We save binary mask as uint8
    mask_uint8 = mask.astype(np.uint8)

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=mask_uint8.shape[0],
        width=mask_uint8.shape[1],
        count=1,
        dtype=mask_uint8.dtype,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(mask_uint8, 1)


def mask_to_geojson(
    mask: np.ndarray, 
    transform: Any, 
    crs: CRS, 
    output_path: Union[str, Path]
) -> None:
    """
    Vectorizes a binary mask to polygons, reprojects to EPSG:4326 if necessary,
    calculates the area (m^2 and km^2) for each polygon, and saves as GeoJSON.

    Args:
        mask (np.ndarray): Binary mask array (H, W).
        transform (Any): Rasterio Affine transform.
        crs (CRS): Original Rasterio Coordinate Reference System.
        output_path (Union[str, Path]): Path where the GeoJSON will be saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if len(mask.shape) == 3:
        mask = mask.squeeze()
        
    mask_uint8 = mask.astype(np.uint8)
    
    # Target CRS is WGS84 for GeoJSON
    target_crs = CRS.from_epsg(4326)
    
    features = []
    
    # Pyproj Geod for accurate area calculation on the WGS84 ellipsoid
    geod = Geod(ellps="WGS84")
    
    # Extract shapes where mask is 1 (change detected)
    for geom, value in shapes(mask_uint8, mask=(mask_uint8 == 1), transform=transform):
        # Reproject geometry if the source CRS is not already EPSG:4326
        if crs != target_crs:
            geom_4326 = transform_geom(crs, target_crs, geom)
        else:
            geom_4326 = geom
            
        # Calculate area using Shapely and Geod
        shapely_polygon = shape(geom_4326)
        
        # geometry_area_perimeter returns (area, perimeter)
        # Area might be negative based on vertex winding order
        area_m2, _ = geod.geometry_area_perimeter(shapely_polygon)
        area_m2 = abs(area_m2)
        area_km2 = area_m2 / 1_000_000.0
        
        # Construct GeoJSON Feature
        feature = {
            "type": "Feature",
            "geometry": geom_4326,
            "properties": {
                "area_m2": area_m2,
                "area_km2": area_km2
            }
        }
        features.append(feature)
        
    # Construct complete FeatureCollection
    geojson_collection = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_collection, f, indent=2)


def compute_total_change_area(geojson_path: Union[str, Path]) -> float:
    """
    Reads a GeoJSON file and computes the total changed area in square kilometers.

    Args:
        geojson_path (Union[str, Path]): Path to the GeoJSON file.

    Returns:
        float: Total changed area in km^2.
    """
    geojson_path = Path(geojson_path)
    
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found at {geojson_path}")
        
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    total_area_km2 = 0.0
    features = data.get("features", [])
    
    for feature in features:
        props = feature.get("properties", {})
        total_area_km2 += props.get("area_km2", 0.0)
        
    return total_area_km2
