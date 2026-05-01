import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SiameseUNet(nn.Module):
    """
    Siamese UNet model for change detection between two satellite images.

    This model uses a shared ResNet34 encoder (pretrained on ImageNet) to extract
    features from two input images (t1 and t2) independently. The absolute
    difference between the extracted features at each encoder stage is computed
    and then passed to a UNet decoder to generate a binary change mask.

    Attributes:
        in_channels (int): Number of input channels (e.g., 3 for RGB, 4 for multi-band).
        encoder (nn.Module): Shared ResNet34 encoder from segmentation_models_pytorch.
        decoder (nn.Module): UNet decoder from segmentation_models_pytorch.
        segmentation_head (nn.Module): Final convolution layers to produce the mask.
    """

    def __init__(self, in_channels: int = 3) -> None:
        """
        Initializes the SiameseUNet model.

        Args:
            in_channels (int): Number of input channels. Defaults to 3.
        """
        super().__init__()

        self.in_channels = in_channels

        # Instantiate a standard UNet to leverage its pre-configured components
        unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=1,
            activation=None  # We output raw logits to be used with BCEWithLogitsLoss
        )

        self.encoder = unet.encoder
        self.decoder = unet.decoder
        self.segmentation_head = unet.segmentation_head

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Siamese UNet.

        Args:
            t1 (torch.Tensor): First image tensor of shape (B, C, H, W).
            t2 (torch.Tensor): Second image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Predicted change mask logits of shape (B, 1, H, W).
        """
        # Extract hierarchical features from both images using the shared encoder
        features_t1 = self.encoder(t1)
        features_t2 = self.encoder(t2)

        # Compute the absolute difference between the features at each scale
        diff_features = [
            torch.abs(f1 - f2)
            for f1, f2 in zip(features_t1, features_t2)
        ]

        # Decode the combined feature differences
        # UnetDecoder.forward expects a list of features, so we pass it without unpacking
        decoder_output = self.decoder(diff_features)

        # Pass through the segmentation head to get logits
        logits = self.segmentation_head(decoder_output)

        # Return raw logits (BCEWithLogitsLoss applies sigmoid internally)
        return logits
