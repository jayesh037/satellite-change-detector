import os
from pathlib import Path
from typing import Tuple, Dict, Any, Union

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import from_bounds
import glymur


def load_sentinel2_bands(folder_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any], CRS]:
    """
    Loads Sentinel-2 bands (B02, B03, B04, B08) from a specified folder,
    stacks them, and normalizes the values using 2nd to 98th percentile clipping.

    It uses glymur to read the JP2 pixel data. It generates synthetic transform
    and CRS metadata based on standard Sentinel-2 T43PGQ bounds to avoid opening
    JP2 files with rasterio, which may fail. It expects band files to be named 
    like T43PGQ_XXXXXXXX_B02_10m.jp2.

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

    # Read each band purely using glymur
    for band in band_names:
        band_path = str(band_paths[band])
        
        # Read pixel data using glymur
        jp2 = glymur.Jp2k(band_path)
        data = jp2[:].astype(np.float32)
        
        # If the image has a single channel, it might be returned as (H, W, 1). We ensure it's (H, W)
        if data.ndim == 3 and data.shape[2] == 1:
            data = data.squeeze(-1)
            
        bands_data.append(data)

    # Stack bands into (C, H, W)
    # where C=4 corresponding to B02, B03, B04, B08
    stacked_bands = np.stack(bands_data, axis=0)
    _, h, w = stacked_bands.shape
    
    # Normalize using 2nd and 98th percentile clipping to ignore extreme outliers
    # We do this globally across the image to preserve relative band ratios.
    p2 = np.percentile(stacked_bands, 2)
    p98 = np.percentile(stacked_bands, 98)
    
    # Clip and normalize to [0, 1]
    stacked_bands = np.clip(stacked_bands, p2, p98)
    normalized_bands = (stacked_bands - p2) / (p98 - p2 + 1e-8)
    
    # Transpose to (H, W, 4)
    normalized_bands = np.transpose(normalized_bands, (1, 2, 0))

    # Generate synthetic metadata for Sentinel-2 T43PGQ approximate bounds
    # bbox: 78.0, 18.0, 79.0, 19.0 (west, south, east, north)
    transform = from_bounds(78.0, 18.0, 79.0, 19.0, w, h)
    crs = CRS.from_epsg(4326)
    
    # Create the metadata dictionary
    meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": None,
        "width": w,
        "height": h,
        "count": 4,
        "crs": crs,
        "transform": transform
    }

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
