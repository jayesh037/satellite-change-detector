import os
import sys
from pathlib import Path
from typing import Tuple, Dict

# Add the project root to the python path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.dataset import LEVIRDataset
from ml.model import SiameseUNet


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    device: torch.device, 
    threshold: float = 0.5
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Evaluates the model on the provided dataloader, calculating global metrics.

    Args:
        model (torch.nn.Module): The trained Siamese UNet model.
        dataloader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device to compute on (CPU/GPU).
        threshold (float): Threshold to binarize model predictions. Defaults to 0.5.

    Returns:
        Tuple[np.ndarray, Dict[str, float]]: 
            - A 2x2 confusion matrix (numpy array).
            - A dictionary of calculated metrics.
    """
    model.eval()
    
    # Accumulate true positives, false positives, true negatives, false negatives
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        t1 = batch["t1"].to(device)
        t2 = batch["t2"].to(device)
        label = batch["label"].to(device)
        
        pred = model(t1, t2)
        
        pred_bin = (pred > threshold).bool()
        label_bin = (label > 0.5).bool()
        
        # Accumulate metrics
        tp += (pred_bin & label_bin).sum().item()
        fp += (pred_bin & ~label_bin).sum().item()
        tn += (~pred_bin & ~label_bin).sum().item()
        fn += (~pred_bin & label_bin).sum().item()

    # Calculate global metrics
    eps = 1e-6
    
    # Class 1 (Change) Metrics
    precision_1 = tp / (tp + fp + eps)
    recall_1 = tp / (tp + fn + eps)
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1 + eps)
    iou_1 = tp / (tp + fp + fn + eps)
    accuracy_1 = (tp + tn) / (tp + tn + fp + fn + eps)  # Global accuracy
    
    # Class 0 (No Change) Metrics
    precision_0 = tn / (tn + fn + eps)
    recall_0 = tn / (tn + fp + eps)
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0 + eps)
    iou_0 = tn / (tn + fp + fn + eps)

    metrics = {
        "Accuracy (Global)": accuracy_1,
        "Change (Class 1) - IoU": iou_1,
        "Change (Class 1) - F1": f1_1,
        "Change (Class 1) - Precision": precision_1,
        "Change (Class 1) - Recall": recall_1,
        "No Change (Class 0) - IoU": iou_0,
        "No Change (Class 0) - F1": f1_0,
        "No Change (Class 0) - Precision": precision_0,
        "No Change (Class 0) - Recall": recall_0,
    }
    
    # Confusion Matrix: [[TN, FP], [FN, TP]]
    conf_matrix = np.array([
        [tn, fp],
        [fn, tp]
    ])
    
    return conf_matrix, metrics


def save_confusion_matrix(conf_matrix: np.ndarray, save_path: Path) -> None:
    """
    Plots and saves the confusion matrix as an image.

    Args:
        conf_matrix (np.ndarray): The 2x2 confusion matrix.
        save_path (Path): Path to save the generated image.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=["No Change", "Change"],
        yticklabels=["No Change", "Change"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_sample_predictions(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    device: torch.device, 
    save_path: Path, 
    num_samples: int = 4
) -> None:
    """
    Saves a grid of sample predictions (T1, T2, Label, Prediction).

    Args:
        model (torch.nn.Module): The trained Siamese UNet model.
        dataloader (DataLoader): DataLoader for validation dataset.
        device (torch.device): The device to compute on.
        save_path (Path): Path to save the generated image.
        num_samples (int): Number of samples to plot. Defaults to 4.
    """
    model.eval()
    
    # Get a single batch
    batch = next(iter(dataloader))
    
    t1 = batch["t1"].to(device)
    t2 = batch["t2"].to(device)
    label = batch["label"].to(device)
    
    with torch.no_grad():
        pred = model(t1, t2)
        pred_bin = (pred > 0.5).float()
    
    # Move tensors to CPU and limit to num_samples
    t1 = t1.cpu()[:num_samples]
    t2 = t2.cpu()[:num_samples]
    label = label.cpu()[:num_samples]
    pred_bin = pred_bin.cpu()[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3 * num_samples))
    
    for i in range(num_samples):
        # Denormalize images for visualization (assuming mean=0, std=1 in dataset)
        # If mean/std were different, we would need to reverse the specific normalization.
        img_t1 = t1[i].permute(1, 2, 0).numpy()
        img_t2 = t2[i].permute(1, 2, 0).numpy()
        mask_label = label[i].squeeze(0).numpy()
        mask_pred = pred_bin[i].squeeze(0).numpy()
        
        # Clip to [0, 1] just in case
        img_t1 = np.clip(img_t1, 0, 1)
        img_t2 = np.clip(img_t2, 0, 1)
        
        axes[i, 0].imshow(img_t1)
        axes[i, 0].set_title(f"Sample {i+1}: T1")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(img_t2)
        axes[i, 1].set_title(f"Sample {i+1}: T2")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(mask_label, cmap="gray")
        axes[i, 2].set_title(f"Sample {i+1}: Ground Truth")
        axes[i, 2].axis("off")
        
        axes[i, 3].imshow(mask_pred, cmap="gray")
        axes[i, 3].set_title(f"Sample {i+1}: Prediction")
        axes[i, 3].axis("off")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    """
    Main evaluation script.
    
    Loads the best checkpoint, runs inference over the validation set,
    prints metrics, and saves confusion matrix and sample visualizations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    checkpoint_path = Path("checkpoints/best_model.pth")
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
        
    # Setup dataset and dataloader
    config_path = "configs/config.yaml"
    val_dataset = LEVIRDataset(split="val", config_path=config_path)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    
    # Setup model and load weights
    model = SiameseUNet(in_channels=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')} with Val IoU {checkpoint.get('val_iou', 'unknown'):.4f}")
    
    # Ensure outputs directory exists
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate globally
    conf_matrix, metrics = evaluate_model(model, val_loader, device)
    
    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Save visualizations
    print("\nSaving visualizations...")
    save_confusion_matrix(conf_matrix, outputs_dir / "confusion_matrix.png")
    save_sample_predictions(model, val_loader, device, outputs_dir / "samples.png", num_samples=4)
    print(f"Visualizations saved to {outputs_dir}/")


if __name__ == "__main__":
    main()
