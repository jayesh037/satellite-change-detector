import os
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional

import numpy as np
import rasterio
from rasterio.env import Env
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import from_bounds
import rasterio.mask
import rasterio.features
from rasterio.io import MemoryFile
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

# Platform check for JP2 drivers
with Env() as env:
    drivers = env.drivers()
    jp2_drivers = [d for d in drivers.keys() if 'JP2' in d]
    print(f"Available JP2 drivers: {jp2_drivers}")
    if 'JP2OpenJPEG' not in drivers:
        raise RuntimeError(
            "JP2OpenJPEG GDAL driver is required to natively read Sentinel-2 JP2 files but was not found. "
            "On Pop OS/Ubuntu, try installing: sudo apt-get install libopenjp2-7 gdal-bin libgdal-dev"
        )


def load_sentinel2_bands(folder_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any], CRS]:
    """
    Loads Sentinel-2 bands (B02, B03, B04, B08) from a specified folder,
    stacks them, and normalizes the values using 2nd to 98th percentile clipping.

    It uses rasterio to read the JP2 pixel data and extracts proper spatial metadata.
    It expects band files to be named like T43PGQ_XXXXXXXX_B02_10m.jp2.

    Args:
        folder_path (Union[str, Path]): Path to the folder containing the .jp2 files.

    Returns:
        Tuple[np.ndarray, Dict[str, Any], CRS]: 
            - Stacked and normalized numpy array of shape (H, W, 4).
            - Rasterio profile/metadata dictionary (transform, width, height, etc.).
            - Rasterio CRS object.

    Raises:
        FileNotFoundError: If any of the required bands cannot be found in the folder.
        ValueError: If multiple files match a single band.
    """
    folder_path = Path(folder_path)
    
    # Required bands and their corresponding expected file substring
    band_names = ['B02', 'B03', 'B04', 'B08']
    band_paths = {}
    
    # Locate files for each band
    for band in band_names:
        matches = list(folder_path.rglob(f"*_{band}_*.jp2"))
        if not matches:
            raise FileNotFoundError(f"Could not find Sentinel-2 band {band} in {folder_path}")
        if len(matches) > 1:
            # Handle potential metadata files or different processing levels, pick the first one roughly
            matches = [m for m in matches if not m.name.endswith('aux.jp2')]
        band_paths[band] = matches[0]

    bands_data = []
    meta = None
    crs = None

    # Read each band using rasterio
    for i, band in enumerate(band_names):
        band_path = str(band_paths[band])
        
        with rasterio.open(band_path) as src:
            data = src.read(1).astype(np.float32)
            if i == 0:
                meta = src.profile.copy()
                crs = src.crs
            bands_data.append(data)

    # Stack bands into (C, H, W)
    # where C=4 corresponding to B02, B03, B04, B08
    stacked_bands = np.stack(bands_data, axis=0)
    
    # Normalize using 2nd and 98th percentile clipping to ignore extreme outliers
    # We do this globally across the image to preserve relative band ratios.
    p2 = np.percentile(stacked_bands, 2)
    p98 = np.percentile(stacked_bands, 98)
    
    # Clip and normalize to [0, 1]
    stacked_bands = np.clip(stacked_bands, p2, p98)
    normalized_bands = (stacked_bands - p2) / (p98 - p2 + 1e-8)
    
    # Transpose to (H, W, 4)
    normalized_bands = np.transpose(normalized_bands, (1, 2, 0))

    # Update the metadata for the stacked image
    meta.update({
        "driver": "GTiff",
        "dtype": "float32",
        "count": 4
    })

    return normalized_bands, meta, crs


def align_images(
    img1: np.ndarray, meta1: Dict[str, Any], 
    img2: np.ndarray, meta2: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray, Dict[str, Any]]:
    """
    Aligns two images to the same Coordinate Reference System (CRS) and resolution.
    
    If the CRS, transform, or dimensions differ, img2 is reprojected to match img1.

    Args:
        img1 (np.ndarray): First image array of shape (H, W, C).
        meta1 (Dict[str, Any]): Metadata for the first image.
        img2 (np.ndarray): Second image array of shape (H, W, C).
        meta2 (Dict[str, Any]): Metadata for the second image.

    Returns:
        Tuple[np.ndarray, Dict[str, Any], np.ndarray, Dict[str, Any]]: 
            Aligned img1, meta1, aligned img2, meta2.
    """
    crs1 = meta1['crs']
    crs2 = meta2['crs']
    
    transform1 = meta1['transform']
    transform2 = meta2['transform']
    
    h1, w1 = meta1['height'], meta1['width']
    h2, w2 = meta2['height'], meta2['width']

    # Check if they are already perfectly aligned
    if (crs1 == crs2 and 
        transform1 == transform2 and 
        h1 == h2 and w1 == w2):
        return img1, meta1, img2, meta2

    # We need to reproject img2 to match img1
    # rasterio.warp.reproject expects data in (C, H, W)
    img1_chw = np.transpose(img1, (2, 0, 1))
    img2_chw = np.transpose(img2, (2, 0, 1))
    
    channels = img1_chw.shape[0]
    reprojected_img2 = np.zeros_like(img1_chw)

    reproject(
        source=img2_chw,
        destination=reprojected_img2,
        src_transform=transform2,
        src_crs=crs2,
        dst_transform=transform1,
        dst_crs=crs1,
        resampling=Resampling.bilinear
    )
    
    # Transpose back to (H, W, C)
    reprojected_img2_hwc = np.transpose(reprojected_img2, (1, 2, 0))
    
    # meta2 now matches meta1 spatially
    aligned_meta2 = meta2.copy()
    aligned_meta2.update({
        'crs': crs1,
        'transform': transform1,
        'width': w1,
        'height': h1
    })

    return img1, meta1, reprojected_img2_hwc, aligned_meta2


def compute_ndvi(bands: np.ndarray) -> np.ndarray:
    """
    Computes the Normalized Difference Vegetation Index (NDVI) from Sentinel-2 bands.

    The bands array is expected to be of shape (H, W, 4) representing:
    Index 0: B02 (Blue)
    Index 1: B03 (Green)
    Index 2: B04 (Red)
    Index 3: B08 (Near Infrared - NIR)

    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        bands (np.ndarray): Stacked bands array of shape (H, W, 4).

    Returns:
        np.ndarray: NDVI array of shape (H, W) with values roughly between -1 and 1.
    """
    if len(bands.shape) != 3 or bands.shape[2] != 4:
        raise ValueError(f"Expected bands array of shape (H, W, 4), got {bands.shape}")

    red = bands[:, :, 2]
    nir = bands[:, :, 3]

    # Calculate NDVI, adding epsilon to denominator to prevent division by zero
    epsilon = 1e-8
    ndvi = (nir - red) / (nir + red + epsilon)
    
    # Clip to valid NDVI range [-1, 1] just in case of edge values
    ndvi = np.clip(ndvi, -1.0, 1.0)
    
    return ndvi


def clip_image_to_aoi(
    image_array: np.ndarray,
    meta: Dict[str, Any],
    aoi_geojson_str: Optional[str]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Clip a (H, W, C) image array to the bounding box of an AOI polygon.
    
    Steps:
    1. Parse aoi_geojson_str as dict
    2. Extract geometry (type: Polygon or Feature with geometry)
    3. Create shapely polygon from coordinates
    4. Reproject polygon from EPSG:4326 to meta['crs'] using pyproj Transformer
       pyproj.Transformer.from_crs('EPSG:4326', meta['crs'], always_xy=True)
    5. Use rasterio.mask.mask() with [reprojected_polygon], crop=True, all_touched=True
       NOTE: rasterio.mask.mask expects (C, H, W) format — transpose before, transpose back after
    6. Update meta: width, height, transform from the mask output
    7. Return (clipped_array_HWC, updated_meta)
    
    If aoi_geojson_str is None or empty string: return (image_array, meta) unchanged.
    If reprojection fails: log warning and return original (graceful degradation).
    """
    if not aoi_geojson_str or not aoi_geojson_str.strip():
        return image_array, meta
        
    try:
        aoi_data = json.loads(aoi_geojson_str)
        # Extract geometry
        geom_dict = aoi_data.get('geometry', aoi_data) if aoi_data.get('type') == 'Feature' else aoi_data
        if 'features' in geom_dict: # FeatureCollection
            geom_dict = geom_dict['features'][0]['geometry']
            
        geom = shape(geom_dict)
        
        # Reproject from EPSG:4326 to target CRS
        transformer = Transformer.from_crs("EPSG:4326", meta['crs'], always_xy=True)
        reprojected_geom = shapely_transform(transformer.transform, geom)
        
        # Write array to memory file to use rasterio.mask
        # image_array is (H, W, C). rasterio expects (C, H, W)
        image_chw = np.transpose(image_array, (2, 0, 1))
        
        with MemoryFile() as memfile:
            # We must use proper dtype and count in the memory file based on the array
            temp_meta = meta.copy()
            temp_meta.update({
                "driver": "GTiff",
                "count": image_chw.shape[0],
                "dtype": str(image_chw.dtype)
            })
            with memfile.open(**temp_meta) as dataset:
                dataset.write(image_chw)
                
                # Apply mask
                out_image, out_transform = rasterio.mask.mask(
                    dataset,
                    [reprojected_geom],
                    crop=True,
                    all_touched=True
                )
                
        out_meta = meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        # Transpose back to (H, W, C)
        out_image_hwc = np.transpose(out_image, (1, 2, 0))
        return out_image_hwc, out_meta
        
    except Exception as e:
        print(f"Warning: Failed to clip image to AOI. Error: {e}. Proceeding with full image.")
        return image_array, meta


def create_aoi_pixel_mask(
    image_shape: tuple,
    meta: Dict[str, Any],
    aoi_geojson_str: Optional[str]
) -> np.ndarray:
    """
    Burn AOI polygon into a binary numpy array matching image_shape (H, W).
    Returns array with 1 inside AOI polygon, 0 outside.
    Uses rasterio.features.geometry_mask() with invert=True.
    Reprojects AOI to image CRS same as clip_image_to_aoi.
    If aoi_geojson_str is None: return np.ones(image_shape) — all pixels valid.
    """
    if not aoi_geojson_str or not aoi_geojson_str.strip():
        return np.ones(image_shape[:2], dtype=np.uint8)
        
    try:
        aoi_data = json.loads(aoi_geojson_str)
        geom_dict = aoi_data.get('geometry', aoi_data) if aoi_data.get('type') == 'Feature' else aoi_data
        if 'features' in geom_dict:
            geom_dict = geom_dict['features'][0]['geometry']
            
        geom = shape(geom_dict)
        
        transformer = Transformer.from_crs("EPSG:4326", meta['crs'], always_xy=True)
        reprojected_geom = shapely_transform(transformer.transform, geom)
        
        mask = rasterio.features.geometry_mask(
            [reprojected_geom],
            out_shape=image_shape[:2],
            transform=meta['transform'],
            invert=True,
            all_touched=True
        )
        return mask.astype(np.uint8)
        
    except Exception as e:
        print(f"Warning: Failed to create AOI pixel mask. Error: {e}. Returning all ones mask.")
        return np.ones(image_shape[:2], dtype=np.uint8)
