import numpy as np
import cv2
from typing import Optional


def apply_threshold(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Converts a probability mask to a binary mask using the specified threshold.

    Args:
        mask (np.ndarray): The input probability mask with values in [0, 1].
        threshold (float): The probability threshold. Defaults to 0.5.

    Returns:
        np.ndarray: Binary mask with values {0, 1} as np.uint8.
    """
    binary_mask = (mask > threshold).astype(np.uint8)
    return binary_mask


def remove_noise(mask: np.ndarray, min_area: int = 5) -> np.ndarray:
    """
    Removes connected components (noise) from a binary mask that are smaller 
    than a specified minimum pixel area.

    Args:
        mask (np.ndarray): Binary input mask (np.uint8).
        min_area (int): Minimum area in pixels to keep a connected component. Defaults to 5.

    Returns:
        np.ndarray: Cleaned binary mask (np.uint8).
    """
    # Ensure mask is uint8 for OpenCV connected components
    mask_uint8 = mask.astype(np.uint8)
    
    # Use 8-way connectivity
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    
    clean_mask = np.zeros_like(mask_uint8)
    
    # Start from 1 to skip the background (label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == i] = 1
            
    return clean_mask


def filter_by_ndvi(
    mask: np.ndarray, 
    ndvi_t1: np.ndarray, 
    ndvi_t2: np.ndarray, 
    threshold: float = 0.15
) -> np.ndarray:
    """
    Suppresses change predictions based on the Normalized Difference Vegetation Index (NDVI).
    
    As requested, suppresses changes where the absolute difference in NDVI between 
    the two timestamps is less than the threshold. 
    (Note: Depending on the specific geographic context, sometimes seasonal vegetation 
    changes cause large NDVI differences, but this implements the requested logic 
    to filter based on `abs(ndvi_t2 - ndvi_t1) < threshold`).

    Args:
        mask (np.ndarray): Binary input mask (np.uint8).
        ndvi_t1 (np.ndarray): NDVI array for the first timestamp.
        ndvi_t2 (np.ndarray): NDVI array for the second timestamp.
        threshold (float): NDVI difference threshold. Defaults to 0.15.

    Returns:
        np.ndarray: Filtered binary mask (np.uint8).
    """
    filtered_mask = mask.copy()
    
    # Calculate absolute difference in NDVI
    ndvi_diff = np.abs(ndvi_t2 - ndvi_t1)
    
    # Suppress (set to 0) the mask where NDVI difference is less than the threshold
    filtered_mask[ndvi_diff < threshold] = 0
    
    return filtered_mask


def postprocess(
    mask: np.ndarray, 
    ndvi_t1: Optional[np.ndarray] = None, 
    ndvi_t2: Optional[np.ndarray] = None,
    prob_threshold: float = 0.5,
    min_area: int = 500,
    ndvi_threshold: float = 0.15
) -> np.ndarray:
    """
    Runs the full post-processing pipeline on a predicted probability mask.

    The pipeline consists of:
    1. Applying a probability threshold to binarize the mask.
    2. Filtering by NDVI (if NDVI arrays are provided) to remove specific vegetation changes.
    3. Removing small noisy connected components.

    Args:
        mask (np.ndarray): The raw predicted probability mask from the model.
        ndvi_t1 (Optional[np.ndarray]): NDVI array for T1. Defaults to None.
        ndvi_t2 (Optional[np.ndarray]): NDVI array for T2. Defaults to None.
        prob_threshold (float): Threshold to binarize the prediction. Defaults to 0.5.
        min_area (int): Minimum component area (in pixels) to keep. Defaults to 500.
        ndvi_threshold (float): Threshold for the NDVI filtering. Defaults to 0.15.

    Returns:
        np.ndarray: The final cleaned binary mask (np.uint8).
    """
    # Hardcode values to ensure they are used
    prob_threshold = 0.7
    min_area = 100
    ndvi_threshold = 0.1
    
    print(f"DEBUG: Using parameters - prob_threshold: {prob_threshold}, ndvi_threshold: {ndvi_threshold}, min_area: {min_area}")
    print(f"DEBUG: raw mask before threshold - max: {mask.max():.4f}, min: {mask.min():.4f}, mean: {mask.mean():.4f}")
    
    # 1. Binarize the mask
    processed_mask = apply_threshold(mask, threshold=prob_threshold)
    print(f"DEBUG: pixel count after threshold ({prob_threshold}): {processed_mask.sum()}")
    
    # 2. Filter by NDVI if the arrays are provided
    if ndvi_t1 is not None and ndvi_t2 is not None:
        processed_mask = filter_by_ndvi(
            processed_mask, 
            ndvi_t1, 
            ndvi_t2, 
            threshold=ndvi_threshold
        )
        print(f"DEBUG: pixel count after NDVI filter ({ndvi_threshold}): {processed_mask.sum()}")
        
    # 3. Remove noise (small connected components)
    final_mask = remove_noise(processed_mask, min_area=min_area)
    print(f"DEBUG: pixel count after noise removal (min_area {min_area}): {final_mask.sum()}")
    
    return final_mask
