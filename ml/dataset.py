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
