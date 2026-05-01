import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss for image segmentation tasks.
    
    Computes the Dice coefficient loss, which is defined as:
    1 - (2 * intersection + smooth) / (prediction_sum + target_sum + smooth)
    
    This loss is useful for handling class imbalances in segmentation tasks.
    
    Attributes:
        smooth (float): A smoothing constant to prevent division by zero and
            stabilize training. Defaults to 1e-6.
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        """
        Initializes the DiceLoss module.
        
        Args:
            smooth (float): Smoothing factor. Defaults to 1e-6.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Dice Loss between predictions and targets.
        
        Args:
            pred (torch.Tensor): Predictions from the model, usually post-sigmoid,
                with shape (B, C, H, W).
            target (torch.Tensor): Ground truth labels with shape (B, C, H, W).
            
        Returns:
            torch.Tensor: The computed Dice loss scalar.
        """
        # Flatten predictions and targets for general batch computing
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        
        # Calculate the Dice coefficient
        dice_score = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy (BCE) and Dice Loss.
    
    This loss function leverages both BCE (good for pixel-wise classification)
    and Dice loss (good for addressing class imbalance and capturing structural
    similarity). It is a weighted sum of the two losses.
    
    Attributes:
        bce_weight (float): Weight for the BCE loss component.
        dice_weight (float): Weight for the Dice loss component.
        bce_loss (nn.BCEWithLogitsLoss): The Binary Cross Entropy with Logits loss module.
        dice_loss (DiceLoss): The Dice loss module.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6) -> None:
        """
        Initializes the BCEDiceLoss module.
        
        Args:
            bce_weight (float): Weight multiplier for BCE loss. Defaults to 0.5.
            dice_weight (float): Weight multiplier for Dice loss. Defaults to 0.5.
            smooth (float): Smoothing factor for the Dice loss calculation. Defaults to 1e-6.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        # We use BCEWithLogitsLoss as it is safer for mixed precision.
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the combined BCE and Dice Loss.
        
        Args:
            pred (torch.Tensor): Logit predictions from the model
                with shape (B, C, H, W).
            target (torch.Tensor): Ground truth labels with shape (B, C, H, W).
            
        Returns:
            torch.Tensor: The computed combined loss scalar.
        """
        # BCEWithLogitsLoss takes raw logits
        bce = self.bce_loss(pred, target)
        
        # Dice loss requires probabilities, so we apply sigmoid
        pred_probs = torch.sigmoid(pred)
        dice = self.dice_loss(pred_probs, target)
        
        return (self.bce_weight * bce) + (self.dice_weight * dice)
