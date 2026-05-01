import numpy as np
from typing import List, Dict, Any


def tile_image(image: np.ndarray, patch_size: int = 256, overlap: int = 32) -> List[Dict[str, Any]]:
    """
    Tiles a single image into overlapping patches.
    
    If the image dimensions are not perfectly divisible by the stride, the last 
    patches are shifted to align with the bottom/right edges of the image to 
    ensure full coverage without out-of-bounds errors. If the image is smaller 
    than the patch size, it is zero-padded.

    Args:
        image (np.ndarray): The input image array of shape (H, W, C) or (H, W).
        patch_size (int): The size of the square patches (height and width). Defaults to 256.
        overlap (int): The number of overlapping pixels between patches. Defaults to 32.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing:
            - 'patch': The extracted image patch (np.ndarray).
            - 'row': Row index of the patch in the grid.
            - 'col': Column index of the patch in the grid.
            - 'y': Top-left Y coordinate of the patch in the original image.
            - 'x': Top-left X coordinate of the patch in the original image.
    """
    h, w = image.shape[:2]
    stride = patch_size - overlap
    
    # Ensure stride is valid
    if stride <= 0:
        raise ValueError("Overlap must be strictly less than patch_size.")
        
    patches = []
    
    # Calculate the grid size
    rows = (h - overlap) // stride + (1 if (h - overlap) % stride != 0 else 0)
    cols = (w - overlap) // stride + (1 if (w - overlap) % stride != 0 else 0)
    
    # Handle case where image is smaller than patch size
    if rows <= 0: rows = 1
    if cols <= 0: cols = 1
    
    for row in range(rows):
        for col in range(cols):
            y = row * stride
            x = col * stride
            
            # Adjust the last patches to not exceed image boundaries
            if y + patch_size > h:
                y = max(0, h - patch_size)
            if x + patch_size > w:
                x = max(0, w - patch_size)
                
            patch = image[y:y+patch_size, x:x+patch_size]
            
            # Pad the patch if the original image is smaller than patch_size
            pad_h = max(0, patch_size - patch.shape[0])
            pad_w = max(0, patch_size - patch.shape[1])
            
            if pad_h > 0 or pad_w > 0:
                if len(image.shape) == 3:
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                else:
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode='constant')
            
            patches.append({
                "patch": patch,
                "row": row,
                "col": col,
                "y": y,
                "x": x
            })
            
    return patches


def tile_image_pair(
    img1: np.ndarray, 
    img2: np.ndarray, 
    patch_size: int = 256, 
    overlap: int = 32
) -> List[Dict[str, Any]]:
    """
    Tiles a pair of aligned images into overlapping patches.
    
    It assumes both images have the exact same spatial dimensions (H, W).

    Args:
        img1 (np.ndarray): First input image.
        img2 (np.ndarray): Second input image.
        patch_size (int): Size of the patches. Defaults to 256.
        overlap (int): Overlap between patches. Defaults to 32.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing:
            - 'patch1': Patch from the first image.
            - 'patch2': Patch from the second image.
            - 'row': Row index of the patch.
            - 'col': Column index of the patch.
            - 'y': Top-left Y coordinate in the original images.
            - 'x': Top-left X coordinate in the original images.
    """
    if img1.shape[:2] != img2.shape[:2]:
        raise ValueError(f"Images must have same spatial dimensions. Got {img1.shape[:2]} and {img2.shape[:2]}")
        
    patches1 = tile_image(img1, patch_size, overlap)
    patches2 = tile_image(img2, patch_size, overlap)
    
    paired_patches = []
    for p1, p2 in zip(patches1, patches2):
        paired_patches.append({
            "patch1": p1["patch"],
            "patch2": p2["patch"],
            "row": p1["row"],
            "col": p1["col"],
            "y": p1["y"],
            "x": p1["x"]
        })
        
    return paired_patches
