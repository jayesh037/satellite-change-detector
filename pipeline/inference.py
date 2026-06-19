import os
import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional

import torch
import numpy as np
from tqdm import tqdm

# Add the project root to the python path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocess import load_sentinel2_bands, align_images, compute_ndvi, clip_image_to_aoi, create_aoi_pixel_mask
from pipeline.tiling import tile_image_pair
from pipeline.stitch import stitch_patches
from pipeline.postprocess import postprocess
from pipeline.gis import save_geotiff, mask_to_geojson, compute_total_change_area
from ml.model import ChangeFormer


def run_inference(
    t1_folder: Union[str, Path],
    t2_folder: Union[str, Path],
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path],
    patch_size: int = 256,
    overlap: int = 16,
    batch_size: int = 8,
    prob_threshold: float = 0.3,
    min_area: int = 500,
    ndvi_threshold: float = 0.15,
    aoi_geojson: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs the full end-to-end inference pipeline for satellite change detection.

    Steps:
    1. Loads Sentinel-2 bands from T1 and T2 folders.
    2. Aligns images to ensure spatial matching.
    3. Computes NDVI for both timestamps using all 4 channels (B02, B03, B04, B08).
    4. Slices the aligned images to keep only the RGB channels (B02, B03, B04).
    5. Tiles the RGB image pair into overlapping patches.
    6. Loads the trained Siamese UNet model (in_channels=3).
    7. Runs batch inference with mixed precision.
    8. Stitches the predicted patches back together using Gaussian blending.
    9. Post-processes the raw mask (threshold, noise removal, NDVI filtering).
    10. Saves outputs as GeoTIFF and GeoJSON.
    11. Computes the total changed area.

    Args:
        t1_folder (Union[str, Path]): Directory containing T1 Sentinel-2 jp2 bands.
        t2_folder (Union[str, Path]): Directory containing T2 Sentinel-2 jp2 bands.
        checkpoint_path (Union[str, Path]): Path to the trained model checkpoint (.pth).
        output_dir (Union[str, Path]): Directory to save the resulting GeoTIFF and GeoJSON.
        patch_size (int): Size of the inference patches. Defaults to 256.
        overlap (int): Overlap between adjacent patches. Defaults to 32.
        batch_size (int): Batch size for model inference. Defaults to 8.
        prob_threshold (float): Threshold to binarize predictions. Defaults to 0.3.
        min_area (int): Minimum pixel area for connected components. Defaults to 500.
        ndvi_threshold (float): Minimum NDVI difference to keep a change. Defaults to 0.15.
        aoi_geojson (Optional[str]): GeoJSON string representing the Area of Interest polygon.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'geotiff_path': Path to the saved GeoTIFF.
            - 'geojson_path': Path to the saved GeoJSON.
            - 'changed_area_km2': Total changed area in square kilometers.
    """
    t1_folder = Path(t1_folder)
    t2_folder = Path(t2_folder)
    output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading bands from {t1_folder} and {t2_folder}...")
    img1, meta1, crs1 = load_sentinel2_bands(t1_folder)
    img2, meta2, crs2 = load_sentinel2_bands(t2_folder)
    
    if aoi_geojson is not None and aoi_geojson.strip() != '':
        img1, meta1 = clip_image_to_aoi(img1, meta1, aoi_geojson)
        img2, meta2 = clip_image_to_aoi(img2, meta2, aoi_geojson)
        print(f"AOI clipping applied. Clipped image shape: {img1.shape}")
    else:
        print("No AOI provided — processing full tile")
        
    print("Aligning images...")
    img1_aligned, meta1_aligned, img2_aligned, meta2_aligned = align_images(img1, meta1, img2, meta2)
    
    image_shape = img1_aligned.shape
    transform = meta1_aligned['transform']
    crs = meta1_aligned['crs']
    
    print("Computing NDVI...")
    ndvi1 = compute_ndvi(img1_aligned)
    ndvi2 = compute_ndvi(img2_aligned)
    
    # Keep only the RGB channels (first 3 channels: B02, B03, B04) for tiling and inference
    img1_rgb = img1_aligned[:, :, :3]
    img2_rgb = img2_aligned[:, :, :3]
    
    print("Tiling images...")
    patches = tile_image_pair(img1_rgb, img2_rgb, patch_size=patch_size, overlap=overlap)
    
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # The model was trained on 3-channel RGB (LEVIR-CD). 
    # Sentinel-2 channels: B02(Blue)=0, B03(Green)=1, B04(Red)=2
    # If checkpoint_path is None, initialize with pretrained encoder weights
    pretrained = checkpoint_path is None
    model = ChangeFormer(in_channels=3, pretrained=pretrained).to(device)
    
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle the state dict cleanly whether it was saved directly or inside a dictionary mapping
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        # Strip '_orig_mod.' prefix for torch.compile compatibility
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        
    model.eval()
    
    stitched_patches = []
    print(f"Running inference on {len(patches)} patches...")
    
    # Process in batches
    for i in tqdm(range(0, len(patches), batch_size), desc="Inference"):
        batch_patches = patches[i:i + batch_size]
        
        # Convert patches to tensors: shape (B, C, H, W)
        t1_batch = torch.stack([
            torch.from_numpy(p["patch1"]).permute(2, 0, 1) for p in batch_patches
        ]).to(device)
        
        t2_batch = torch.stack([
            torch.from_numpy(p["patch2"]).permute(2, 0, 1) for p in batch_patches
        ]).to(device)
        
        # Extract RGB channels (Red=2, Green=1, Blue=0) to feed into the 3-channel pretrained model
        t1_rgb = t1_batch[:, [2, 1, 0], :, :]
        t2_rgb = t2_batch[:, [2, 1, 0], :, :]
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                pred_batch_logits = model(t1_rgb, t2_rgb)
                pred_batch = torch.sigmoid(pred_batch_logits)
                
        # Squeeze channel dim and move to CPU
        pred_batch_np = pred_batch.squeeze(1).cpu().numpy()
        
        for j, p in enumerate(batch_patches):
            stitched_patches.append({
                "pred": pred_batch_np[j],
                "row": p["row"],
                "col": p["col"],
                "y": p["y"],
                "x": p["x"]
            })
            
    print("Stitching patches...")
    raw_mask = stitch_patches(stitched_patches, image_shape, patch_size=patch_size, overlap=overlap)
    
    if aoi_geojson is not None and aoi_geojson.strip() != '':
        aoi_mask = create_aoi_pixel_mask(raw_mask.shape, meta1_aligned, aoi_geojson)
        raw_mask = raw_mask * aoi_mask
        
    print("Post-processing mask...")
    final_mask = postprocess(
        mask=raw_mask,
        ndvi_t1=ndvi1,
        ndvi_t2=ndvi2,
        prob_threshold=prob_threshold,
        min_area=min_area,
        ndvi_threshold=ndvi_threshold
    )
    
    print("Exporting GIS formats...")
    geotiff_path = output_dir / "change_mask.tif"
    geojson_path = output_dir / "change_polygons.geojson"
    
    save_geotiff(final_mask, transform, crs, geotiff_path)
    mask_to_geojson(final_mask, transform, crs, geojson_path)
    
    area_km2 = compute_total_change_area(geojson_path)
    print(f"Total changed area: {area_km2:.4f} km²")
    
    return {
        "geotiff_path": str(geotiff_path),
        "geojson_path": str(geojson_path),
        "changed_area_km2": area_km2
    }
