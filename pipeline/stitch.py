import numpy as np
from typing import List, Dict, Any, Tuple


def _get_gaussian_window(patch_size: int, sigma_scale: float = 6.0) -> np.ndarray:
    """
    Generates a 2D Gaussian window for blending patches smoothly.

    Args:
        patch_size (int): The size of the square patch.
        sigma_scale (float): Determines the standard deviation. A higher 
            value makes the Gaussian sharper (more concentrated in the center).

    Returns:
        np.ndarray: A 2D Gaussian mask of shape (patch_size, patch_size).
    """
    # Create coordinate grid centered at 0
    center = patch_size // 2
    x = np.arange(0, patch_size) - center
    y = np.arange(0, patch_size) - center
    xx, yy = np.meshgrid(x, y)
    
    # Calculate standard deviation
    sigma = patch_size / sigma_scale
    
    # Compute 2D Gaussian
    window = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return window.astype(np.float32)


def stitch_patches(
    patches: List[Dict[str, Any]], 
    image_shape: Tuple[int, ...], 
    patch_size: int = 256, 
    overlap: int = 32
) -> np.ndarray:
    """
    Reassembles a full prediction mask from overlapping patches using Gaussian blending.
    
    Overlapping regions are averaged based on a Gaussian weight map, giving higher 
    importance to the center of each patch and lower importance to the edges. This 
    helps reduce visible seam artifacts.

    Args:
        patches (List[Dict[str, Any]]): List of patch dictionaries. Each dict must contain:
            - 'pred': The predicted patch mask (np.ndarray of shape H, W).
            - 'y': Top-left Y coordinate of the patch in the original image.
            - 'x': Top-left X coordinate of the patch in the original image.
        image_shape (Tuple[int, ...]): Spatial dimensions (H, W) of the original image.
        patch_size (int): Size of the patches. Defaults to 256.
        overlap (int): Overlapping pixels used during tiling (unused directly but conceptually paired). Defaults to 32.

    Returns:
        np.ndarray: The reassembled prediction mask of shape (H, W).
    """
    h, w = image_shape[:2]
    
    # Accumulators for the final stitched image and the Gaussian weights
    stitched_image = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    
    gaussian_window = _get_gaussian_window(patch_size)
    
    for p in patches:
        pred = p["pred"]
        y = p["y"]
        x = p["x"]
        
        # In the rare case that the image was smaller than patch_size and was padded
        valid_h = min(patch_size, h - y)
        valid_w = min(patch_size, w - x)
        
        # Accumulate the weighted prediction and the weights
        stitched_image[y:y+valid_h, x:x+valid_w] += pred[:valid_h, :valid_w] * gaussian_window[:valid_h, :valid_w]
        weight_map[y:y+valid_h, x:x+valid_w] += gaussian_window[:valid_h, :valid_w]
        
    # Normalize by the accumulated weights to get the weighted average
    # Add epsilon to prevent division by zero in untouched areas
    stitched_image /= (weight_map + 1e-8)
    
    return stitched_image
