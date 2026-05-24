import argparse
from pathlib import Path
import sys
import os

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.indices import (
    load_band,
    resample_band_to_10m,
    compute_ndvi,
    compute_ndwi,
    compute_ndbi,
    save_index_geotiff
)

def compute_and_save_indices(safe_root: str, year: str):
    """
    Computes NDVI, NDWI, and NDBI for a given SAFE root and saves them.
    """
    safe_path = Path(safe_root)
    if not safe_path.exists():
        print(f"Error: Path {safe_root} does not exist.")
        return

    print(f"Processing year {year} from {safe_root}")
    
    # Load 10m bands
    print("Loading 10m bands (B02, B03, B04, B08)...")
    b02_data, b02_meta, crs = load_band(safe_path, 'B02')
    b03_data, _, _ = load_band(safe_path, 'B03')
    b04_data, _, _ = load_band(safe_path, 'B04')
    b08_data, _, _ = load_band(safe_path, 'B08')
    
    # Target 10m metadata for resampling
    target_meta = b02_meta
    
    # Load 20m bands
    print("Loading 20m bands (B11, B12)...")
    b11_data, b11_meta, _ = load_band(safe_path, 'B11')
    b12_data, b12_meta, _ = load_band(safe_path, 'B12')
    
    # Resample 20m bands to 10m
    print("Resampling 20m bands to 10m...")
    b11_data_10m = resample_band_to_10m(b11_data, b11_meta, target_meta)
    
    # Compute indices
    print("Computing indices...")
    ndvi = compute_ndvi(b04_data, b08_data)
    ndwi = compute_ndwi(b03_data, b08_data)
    ndbi = compute_ndbi(b11_data_10m, b08_data)
    
    # Setup output paths
    out_dir = Path("outputs/indices")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # We will save single band float32 arrays and let rio-tiler apply coloring dynamically in the backend
    # This aligns better with the instructions since backend/tiles.py is already setup to handle single band index coloring
    print(f"Saving indices for {year}...")
    save_index_geotiff(ndvi, target_meta, out_dir / f"ndvi_{year}.tif")
    save_index_geotiff(ndwi, target_meta, out_dir / f"ndwi_{year}.tif")
    save_index_geotiff(ndbi, target_meta, out_dir / f"ndbi_{year}.tif")
    
    print(f"Finished processing {year}.")

def main():
    parser = argparse.ArgumentParser(description="Compute spectral indices from Sentinel-2 data.")
    parser.add_argument("--t1-safe-root", required=True, help="Path to T1 .SAFE folder")
    parser.add_argument("--t2-safe-root", required=True, help="Path to T2 .SAFE folder")
    
    args = parser.parse_args()
    
    # We attempt to extract the year from the path as a simple heuristic, or fallback to fixed names
    t1_path = Path(args.t1_safe_root)
    t2_path = Path(args.t2_safe_root)
    
    # Assuming path contains the year (e.g. data/ISRO/2021/...)
    t1_year = next((part for part in t1_path.parts if part in ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]), "2021")
    t2_year = next((part for part in t2_path.parts if part in ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]), "2023")
    
    compute_and_save_indices(args.t1_safe_root, t1_year)
    compute_and_save_indices(args.t2_safe_root, t2_year)
    
    print("\nAll indices computed successfully.")

if __name__ == "__main__":
    main()