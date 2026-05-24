import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Any, Union

def load_band(folder_path: Union[str, Path], band_name: str) -> Tuple[np.ndarray, Dict[str, Any], CRS]:
    """
    Searches R10m for B02/B03/B04/B08, R20m for B11/B12 and loads the specified band.
    """
    folder_path = Path(folder_path)
    
    # Sentinel-2 band to resolution mapping
    res_map = {
        'B02': 'R10m', 'B03': 'R10m', 'B04': 'R10m', 'B08': 'R10m',
        'B11': 'R20m', 'B12': 'R20m'
    }
    
    if band_name not in res_map:
        raise ValueError(f"Unsupported band: {band_name}")
        
    res_folder = res_map[band_name]
    
    # Try finding the specific resolution folder or just search globally
    search_dir = folder_path
    
    # In some SAFE structures, the resolution folder is nested deeper, but we'll try a global rglob first
    matches = list(search_dir.rglob(f"*_{band_name}_*.jp2"))
    
    # Fallback to TIF if JP2 not found
    if not matches:
        matches = list(search_dir.rglob(f"*_{band_name}_*.tif"))
        
    if not matches:
        raise FileNotFoundError(f"Could not find band {band_name} in {folder_path}")
        
    # Pick the first one that is not an auxiliary file
    valid_matches = [m for m in matches if not m.name.endswith('aux.jp2') and not m.name.endswith('aux.tif')]
    if not valid_matches:
        valid_matches = matches
        
    band_path = str(valid_matches[0])
    
    with rasterio.open(band_path) as src:
        data = src.read(1).astype(np.float32)
        meta = src.profile.copy()
        crs = src.crs
        
    return data, meta, crs

def resample_band_to_10m(band_array: np.ndarray, band_meta: Dict[str, Any], target_meta: Dict[str, Any]) -> np.ndarray:
    """
    Resamples a band (e.g., 20m) to match the target 10m metadata.
    """
    if band_meta['transform'] == target_meta['transform'] and band_meta['width'] == target_meta['width'] and band_meta['height'] == target_meta['height']:
        return band_array
        
    target_shape = (target_meta['height'], target_meta['width'])
    resampled = np.zeros(target_shape, dtype=np.float32)
    
    reproject(
        source=band_array,
        destination=resampled,
        src_transform=band_meta['transform'],
        src_crs=band_meta['crs'],
        dst_transform=target_meta['transform'],
        dst_crs=target_meta['crs'],
        resampling=Resampling.bilinear
    )
    return resampled

def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """ NDVI = (NIR - Red) / (NIR + Red) """
    epsilon = 1e-8
    ndvi = (nir - red) / (nir + red + epsilon)
    return np.clip(ndvi, -1.0, 1.0)

def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """ NDWI = (Green - NIR) / (Green + NIR) """
    epsilon = 1e-8
    ndwi = (green - nir) / (green + nir + epsilon)
    return np.clip(ndwi, -1.0, 1.0)

def compute_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """ NDBI = (SWIR - NIR) / (SWIR + NIR) """
    epsilon = 1e-8
    ndbi = (swir - nir) / (swir + nir + epsilon)
    return np.clip(ndbi, -1.0, 1.0)

def normalize_index(index: np.ndarray, vmin: float = -1.0, vmax: float = 1.0) -> np.ndarray:
    """ Normalizes an index array from [vmin, vmax] to [0, 255] uint8. """
    index_clipped = np.clip(index, vmin, vmax)
    normalized = (index_clipped - vmin) / (vmax - vmin)
    return (normalized * 255).astype(np.uint8)

def apply_colormap(uint8_array: np.ndarray, colormap_name: str) -> np.ndarray:
    """ Applies a matplotlib colormap to a uint8 array and returns RGBA uint8. """
    cmap = plt.get_cmap(colormap_name)
    # cmap returns RGBA floats [0, 1], convert to uint8
    rgba = cmap(uint8_array / 255.0)
    return (rgba * 255).astype(np.uint8)

def save_index_geotiff(array: np.ndarray, meta: Dict[str, Any], output_path: Union[str, Path], colormap: str = None) -> None:
    """ Saves the index array as a GeoTIFF. """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If a colormap is provided, we save as a 4-band RGBA image
    if colormap:
        uint8_arr = normalize_index(array)
        rgba_arr = apply_colormap(uint8_arr, colormap)
        # rgba_arr is (H, W, 4). rasterio wants (4, H, W)
        rgba_arr = np.transpose(rgba_arr, (2, 0, 1))
        
        out_meta = meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "count": 4,
            "dtype": "uint8",
            "photometric": "RGB",
            "alpha": "yes",
            "compress": "lzw"
        })
        
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(rgba_arr)
    else:
        # Save as single band float32
        out_meta = meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "count": 1,
            "dtype": "float32",
            "compress": "lzw"
        })
        
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(array, 1)
