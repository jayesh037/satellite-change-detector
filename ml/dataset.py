import os
from pathlib import Path
from typing import Dict, List, Optional
import yaml

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class LEVIRDataset(Dataset):
    """
    Dataset class for the LEVIR-CD change detection dataset.
    
    Reads pairs of images (t1 and t2) and their corresponding change mask (label)
    from the specified split directory. Applies data augmentations (flip, rotate,
    color jitter) for the training split and normalizes images to [0, 1].
    
    Attributes:
        split (str): The dataset split, typically 'train' or 'val'.
        config (Dict): Configuration dictionary loaded from config.yaml.
        base_dir (Path): Base directory for the dataset.
        split_dir (Path): Directory for the specific split.
        img_dir_a (Path): Directory containing 'A' images (t1).
        img_dir_b (Path): Directory containing 'B' images (t2).
        label_dir (Path): Directory containing 'label' images.
        filenames (List[str]): List of valid image filenames.
        transform (A.Compose): Albumentations composition for augmentations.
    """

    def __init__(self, split: str = "train", config_path: str = "configs/config.yaml") -> None:
        """
        Initializes the LEVIRDataset.

        Args:
            split (str): Dataset split ('train' or 'val'). Defaults to 'train'.
            config_path (str): Path to the YAML configuration file. Defaults to 'configs/config.yaml'.
        
        Raises:
            FileNotFoundError: If the configuration file cannot be read.
        """
        super().__init__()
        self.split = split
        
        # Load configuration
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            self.config = {}

        # Extract dataset parameters from config, with sensible defaults
        data_cfg = self.config.get("data", {})
        self.base_dir = Path(data_cfg.get("levir_root", "data/LEVIR-CD"))
        self.split_dir = self.base_dir / self.split
        
        self.img_dir_a = self.split_dir / "A"
        self.img_dir_b = self.split_dir / "B"
        self.label_dir = self.split_dir / "label"
        
        # Gather all filenames present in the 'A' directory
        self.filenames: List[str] = []
        if self.img_dir_a.exists():
            self.filenames = sorted([f.name for f in self.img_dir_a.glob("*.png")])

        # Define augmentations and normalization
        # We use 'additional_targets' to apply exact same spatial transforms to both t1 and t2
        additional_targets = {'image_t2': 'image'}
        
        if self.split == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                    # Normalize to [0, 1]
                    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
                    ToTensorV2(),
                ],
                additional_targets=additional_targets
            )
        else:
            self.transform = A.Compose(
                [
                    # Normalize to [0, 1]
                    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
                    ToTensorV2(),
                ],
                additional_targets=additional_targets
            )

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset split.
        
        Returns:
            int: Number of samples.
        """
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Loads, augments, and returns a specific sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 't1': Image tensor of shape (3, 256, 256).
                - 't2': Image tensor of shape (3, 256, 256).
                - 'label': Binary mask tensor of shape (1, 256, 256).
                
        Raises:
            FileNotFoundError: If any required image (t1, t2, label) is missing on disk.
        """
        filename = self.filenames[idx]
        
        path_a = self.img_dir_a / filename
        path_b = self.img_dir_b / filename
        path_label = self.label_dir / filename
        
        # Load A image (t1)
        img_t1 = cv2.imread(str(path_a))
        if img_t1 is None:
            raise FileNotFoundError(f"Failed to load image t1: {path_a}")
        img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)
        
        # Load B image (t2)
        img_t2 = cv2.imread(str(path_b))
        if img_t2 is None:
            raise FileNotFoundError(f"Failed to load image t2: {path_b}")
        img_t2 = cv2.cvtColor(img_t2, cv2.COLOR_BGR2RGB)
        
        # Load label (change mask)
        label = cv2.imread(str(path_label), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Failed to load label: {path_label}")
            
        # Convert grayscale label to binary float32 mask
        # Changes are represented by high values (usually 255)
        label = (label > 127).astype(np.float32)

        # Apply transformations (spatial augmentations + normalization)
        augmented = self.transform(image=img_t1, image_t2=img_t2, mask=label)
        
        img_t1_aug = augmented['image']
        img_t2_aug = augmented['image_t2']
        label_aug = augmented['mask']
        
        # albumentations.pytorch.ToTensorV2 converts 2D mask to (H, W).
        # PyTorch BCE loss and UNet expect (1, H, W).
        label_aug = label_aug.unsqueeze(0)

        return {
            "t1": img_t1_aug,
            "t2": img_t2_aug,
            "label": label_aug
        }


class OSCDDataset(Dataset):
    """
    Dataset class for the Onera Satellite Change Detection (OSCD) dataset.
    
    Reads Sentinel-2 image bands (B04, B03, B02 for RGB compatibility) from 
    data/OSCD/images/{city}/imgs_1 and imgs_2, and the binary change mask from
    data/OSCD/train_labels/{city}/cm/cm.png.
    
    Tiles each city image into 256x256 patches with 32px overlap and normalizes 
    each band using 2nd-98th percentile before applying Albumentations.
    """

    def __init__(self, split: str = "train", config_path: str = "configs/config.yaml") -> None:
        """
        Initializes the OSCDDataset.

        Args:
            split (str): Dataset split ('train' or 'val'). Defaults to 'train'.
            config_path (str): Path to the YAML configuration file. Defaults to 'configs/config.yaml'.
        """
        super().__init__()
        self.split = split
        self.patch_size = 256
        self.overlap = 32
        
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            self.config = {}

        data_cfg = self.config.get("data", {})
        self.base_dir = Path(data_cfg.get("oscd_root", "data/OSCD"))
        self.images_dir = self.base_dir / "images"
        self.labels_dir = self.base_dir / "train_labels"
        
        # Load train cities from train.txt
        train_txt = self.images_dir / "train.txt"
        if not train_txt.exists():
            raise FileNotFoundError(f"Missing {train_txt}. Cannot determine OSCD cities.")
            
        with open(train_txt, "r") as f:
            content = f.read().strip()
            # Handle both comma-separated and newline-separated formats just in case
            if "," in content:
                all_train_cities = [c.strip() for c in content.split(",") if c.strip()]
            else:
                all_train_cities = [c.strip() for c in content.splitlines() if c.strip()]
            
        # Include all cities for training, use last 2 for validation
        if self.split == "train":
            self.cities = all_train_cities
        elif self.split == "val":
            self.cities = all_train_cities[-2:]
        else:
            self.cities = []

        # Define augmentations and normalization (same as LEVIR)
        additional_targets = {'image_t2': 'image'}
        
        if self.split == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
                    ToTensorV2(),
                ],
                additional_targets=additional_targets
            )
        else:
            self.transform = A.Compose(
                [
                    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
                    ToTensorV2(),
                ],
                additional_targets=additional_targets
            )

        # Pre-load and tile all cities into memory to create the dataset index
        self.patches = []
        self._prepare_patches()

    def _prepare_patches(self):
        """Loads full images for all cities, tiles them, and stores the patches in memory."""
        import rasterio
        
        for city in self.cities:
            city_img_dir = self.images_dir / city
            dir_t1 = city_img_dir / "imgs_1"
            dir_t2 = city_img_dir / "imgs_2"
            
            # Label might be in cm/cm.png or just cm.png
            label_dir = self.labels_dir / city
            cm_path = label_dir / "cm" / "cm.png"
            if not cm_path.exists():
                cm_path = label_dir / "cm.png"
                if not cm_path.exists():
                    print(f"Warning: Label not found for city {city} at {label_dir}")
                    continue

            # Load full RGB images
            img_t1 = self._load_rgb_bands(dir_t1)
            img_t2 = self._load_rgb_bands(dir_t2)
            
            # Load full label mask
            label = cv2.imread(str(cm_path), cv2.IMREAD_GRAYSCALE)
            label = (label > 0).astype(np.float32)
            
            h, w = img_t1.shape[:2]
            stride = self.patch_size // 2
            
            # Tile the images
            for y in range(0, h - self.patch_size + 1, stride):
                for x in range(0, w - self.patch_size + 1, stride):
                    patch_t1 = img_t1[y:y+self.patch_size, x:x+self.patch_size]
                    patch_t2 = img_t2[y:y+self.patch_size, x:x+self.patch_size]
                    patch_label = label[y:y+self.patch_size, x:x+self.patch_size]
                    
                    self.patches.append({
                        "t1": patch_t1,
                        "t2": patch_t2,
                        "label": patch_label,
                        "city": city
                    })

    def _load_rgb_bands(self, img_dir: Path) -> np.ndarray:
        """
        Loads B04 (Red), B03 (Green), and B02 (Blue) bands from Sentinel-2 using rasterio,
        applies 2nd-98th percentile normalization to each band independently,
        and stacks them into an 8-bit RGB image.
        """
        import rasterio
        
        band_names = ['B04', 'B03', 'B02']
        bands = []
        
        for b_name in band_names:
            band_path = None
            # e.g. S2A_OPER_MSI_L1C_TL_SGS__20160915T155806_A006437_T31UFQ_B02.tif
            matches = list(img_dir.glob(f"*{b_name}.tif"))
            if matches:
                band_path = matches[0]
            else:
                 raise FileNotFoundError(f"Could not find band {b_name} in {img_dir}")
                 
            with rasterio.open(band_path) as src:
                b_img = src.read(1).astype(np.float32)
                
            # 2nd-98th Percentile Normalization
            p2, p98 = np.percentile(b_img, (2, 98))
            b_img = np.clip(b_img, p2, p98)
            b_img = (b_img - p2) / (p98 - p2 + 1e-8)
            b_img = (b_img * 255.0).astype(np.uint8)
                
            bands.append(b_img)
            
        # Stack as R, G, B
        rgb_img = np.stack(bands, axis=-1)
        return rgb_img

    def __len__(self) -> int:
        """Returns the total number of patches in the dataset split."""
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Applies augmentations and returns a specific 256x256 patch from the OSCD dataset.
        """
        patch_data = self.patches[idx]
        img_t1 = patch_data["t1"]
        img_t2 = patch_data["t2"]
        label = patch_data["label"]

        augmented = self.transform(image=img_t1, image_t2=img_t2, mask=label)
        
        img_t1_aug = augmented['image']
        img_t2_aug = augmented['image_t2']
        label_aug = augmented['mask']
        
        # albumentations.pytorch.ToTensorV2 converts 2D mask to (H, W).
        label_aug = label_aug.unsqueeze(0)

        return {
            "t1": img_t1_aug,
            "t2": img_t2_aug,
            "label": label_aug
        }


class PseudoLabelDataset(Dataset):
    """
    Dataset class for fine-tuning using pseudo-labels on Sentinel-2 data.
    
    Reads T1 and T2 images from the configured folders, reads the pseudo-label
    GeoTIFF, and extracts 256x256 patches. Patches where fewer than 10% of pixels
    are definitive (i.e. != -1) are skipped.
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        """
        Initializes the PseudoLabelDataset.

        Args:
            config_path (str): Path to the YAML configuration file. Defaults to 'configs/config.yaml'.
        """
        super().__init__()
        self.patch_size = 256
        self.overlap = 32
        
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            self.config = {}

        pseudo_cfg = self.config.get("pseudo", {})
        self.t1_folder = Path(pseudo_cfg.get("t1_folder", ""))
        self.t2_folder = Path(pseudo_cfg.get("t2_folder", ""))
        self.pseudo_label_path = Path(pseudo_cfg.get("pseudo_label_path", ""))
        
        if not self.t1_folder.exists() or not self.t2_folder.exists() or not self.pseudo_label_path.exists():
            raise FileNotFoundError("Missing T1 folder, T2 folder, or pseudo label path.")
            
        additional_targets = {'image_t2': 'image'}
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ],
            additional_targets=additional_targets
        )

        self.patches = []
        self._prepare_patches()

    def _prepare_patches(self):
        """Loads full images, tiles them, filters uncertain patches, and stores valid patches in memory."""
        import rasterio
        
        # load_sentinel2_bands returns normalized RGB (H, W, 4) and handles 2nd-98th percentile.
        # We need RGB only (first 3 channels). 
        # Note: the dataset returns float32 normalized [0, 1] arrays.
        img_t1_full, _, _ = load_sentinel2_bands(self.t1_folder)
        img_t2_full, _, _ = load_sentinel2_bands(self.t2_folder)
        
        img_t1 = img_t1_full[:, :, :3]
        img_t2 = img_t2_full[:, :, :3]
        
        # Load pseudo label map
        with rasterio.open(self.pseudo_label_path) as src:
            label = src.read(1).astype(np.float32)
            
        h, w = img_t1.shape[:2]
        stride = self.patch_size // 2
        
        for y in range(0, h - self.patch_size + 1, stride):
            for x in range(0, w - self.patch_size + 1, stride):
                patch_label = label[y:y+self.patch_size, x:x+self.patch_size]
                
                # Calculate certainty ratio: number of pixels != -1
                definitive_pixels = np.sum(patch_label != -1)
                total_pixels = self.patch_size * self.patch_size
                
                if (definitive_pixels / total_pixels) < 0.10:
                    continue
                    
                patch_t1 = img_t1[y:y+self.patch_size, x:x+self.patch_size]
                patch_t2 = img_t2[y:y+self.patch_size, x:x+self.patch_size]
                
                self.patches.append({
                    "t1": patch_t1,
                    "t2": patch_t2,
                    "label": patch_label
                })

    def __len__(self) -> int:
        """Returns the total number of valid pseudo-labeled patches."""
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Applies spatial augmentations to T1, T2, and Label patch.
        Creates a certainty mask where label != -1.
        Returns T1, T2, Label, and Mask tensors.
        """
        patch_data = self.patches[idx]
        img_t1 = patch_data["t1"]
        img_t2 = patch_data["t2"]
        label = patch_data["label"]

        augmented = self.transform(image=img_t1, image_t2=img_t2, mask=label)
        
        img_t1_aug = augmented['image']
        img_t2_aug = augmented['image_t2']
        label_aug = augmented['mask']
        
        label_aug = label_aug.unsqueeze(0)
        
        # Create mask: 1 where label != -1, 0 where label == -1
        certainty_mask = (label_aug != -1).float()
        
        # For loss compatibility, ensure label values at -1 are set to 0.
        # The masked loss will zero out the gradient anyway, but BCE requires target in [0, 1].
        safe_label = torch.where(label_aug == -1, torch.zeros_like(label_aug), label_aug)

        return {
            "t1": img_t1_aug,
            "t2": img_t2_aug,
            "label": safe_label,
            "mask": certainty_mask
        }

