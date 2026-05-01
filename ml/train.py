import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple

# Add the project root to the python path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import mlflow
from tqdm import tqdm

from ml.dataset import LEVIRDataset
from ml.model import SiameseUNet
from ml.losses import BCEDiceLoss


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float]:
    """
    Calculates the Intersection over Union (IoU) and F1 Score.

    Args:
        pred (torch.Tensor): Model predictions with values in [0, 1].
        target (torch.Tensor): Ground truth labels with values {0, 1}.
        threshold (float): Threshold to binarize predictions. Defaults to 0.5.

    Returns:
        Tuple[float, float]: A tuple containing (IoU, F1 Score).
    """
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0.5).float()

    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1 - target_bin)).sum()
    fn = ((1 - pred_bin) * target_bin).sum()

    # Add epsilon to prevent division by zero
    eps = 1e-6
    
    iou = (tp + eps) / (tp + fp + fn + eps)
    
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    return iou.item(), f1.item()


def train_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    scaler: GradScaler, 
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        scaler (GradScaler): Gradient scaler for mixed precision.
        device (torch.device): The device to compute on (CPU/GPU).

    Returns:
        Tuple[float, float, float]: Average Loss, IoU, and F1 Score for the epoch.
    """
    model.train()
    running_loss, running_iou, running_f1 = 0.0, 0.0, 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        t1 = batch["t1"].to(device)
        t2 = batch["t2"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            pred = model(t1, t2)
            loss = criterion(pred, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        iou, f1 = calculate_metrics(pred.detach(), label)
        
        running_loss += loss.item()
        running_iou += iou
        running_f1 += f1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{iou:.4f}"})

    n = len(dataloader)
    return running_loss / n, running_iou / n, running_f1 / n


@torch.no_grad()
def validate_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Validates the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to compute on (CPU/GPU).

    Returns:
        Tuple[float, float, float]: Average Loss, IoU, and F1 Score for the epoch.
    """
    model.eval()
    running_loss, running_iou, running_f1 = 0.0, 0.0, 0.0
    
    pbar = tqdm(dataloader, desc="Validation")
    for batch in pbar:
        t1 = batch["t1"].to(device)
        t2 = batch["t2"].to(device)
        label = batch["label"].to(device)

        with autocast():
            pred = model(t1, t2)
            loss = criterion(pred, label)

        iou, f1 = calculate_metrics(pred, label)
        
        running_loss += loss.item()
        running_iou += iou
        running_f1 += f1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{iou:.4f}"})

    n = len(dataloader)
    return running_loss / n, running_iou / n, running_f1 / n


def main() -> None:
    """
    Main training script.
    
    Loads configuration, sets up the datasets and model, and runs the training loop
    with mixed precision, learning rate scheduling, MLflow logging, and early stopping.
    """
    # Load Config
    config_path = "configs/config.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Warning: {config_path} not found. Using defaults.")
        config = {}

    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 8)
    num_epochs = train_cfg.get("epochs", 100)
    learning_rate = float(train_cfg.get("learning_rate", 1e-4))
    early_stopping_patience = train_cfg.get("early_stopping_patience", 10)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up datasets and dataloaders
    train_dataset = LEVIRDataset(split="train", config_path=config_path)
    val_dataset = LEVIRDataset(split="val", config_path=config_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss, optimizer
    model = SiameseUNet(in_channels=3).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler()

    # Checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.pth"

    # Training state variables
    best_val_iou = 0.0
    patience_counter = 0

    # Initialize MLflow
    mlflow.set_experiment("LEVIR_CD_Siamese_UNet")

    with mlflow.start_run():
        mlflow.log_params({
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "loss": "BCEDiceLoss"
        })

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train and Validate
            train_loss, train_iou, train_f1 = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
            val_loss, val_iou, val_f1 = validate_epoch(model, val_loader, criterion, device)
            
            # Step the scheduler
            scheduler.step()

            # Logging
            metrics = {
                "train_loss": train_loss,
                "train_iou": train_iou,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_f1": val_f1,
                "lr": scheduler.get_last_lr()[0]
            }
            mlflow.log_metrics(metrics, step=epoch)

            print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, F1: {val_f1:.4f}")

            # Best Model Saving & Early Stopping Check
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_iou': best_val_iou,
                }, best_model_path)
                print(f"--> Saved new best model to {best_model_path} (IoU: {best_val_iou:.4f})")
            else:
                patience_counter += 1
                print(f"--> No improvement in IoU. Patience: {patience_counter}/{early_stopping_patience}")

            if patience_counter >= early_stopping_patience:
                print("\nEarly stopping triggered. Halting training.")
                break

if __name__ == "__main__":
    main()
