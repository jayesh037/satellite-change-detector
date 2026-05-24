import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class MLPDecoder(nn.Module):
    """
    Lightweight All-MLP decoder for Segformer-like architectures.
    Projects multi-scale features to a common embedding dimension,
    upsamples them to the largest feature map resolution, concatenates them,
    and produces the final segmentation mask.
    """
    def __init__(self, in_channels_list: list[int], embedding_dim: int = 256):
        super().__init__()
        
        # Linear projection layers for each feature scale
        self.projections = nn.ModuleList([
            nn.Conv2d(in_channels, embedding_dim, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(embedding_dim * len(in_channels_list), embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final segmentation head
        self.head = nn.Conv2d(embedding_dim, 1, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # features list corresponds to [c1, c2, c3, c4]
        # c1 has the largest spatial resolution
        target_size = features[0].shape[2:]
        
        projected_features = []
        for i, (feat, proj) in enumerate(zip(features, self.projections)):
            x = proj(feat)
            if i > 0:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            projected_features.append(x)
            
        # Concatenate all upsampled features
        x = torch.cat(projected_features, dim=1)
        
        # Fuse and predict
        x = self.fusion(x)
        x = self.head(x)
        
        return x


class ChangeFormer(nn.Module):
    """
    ChangeFormer architecture utilizing a shared EfficientNet-B0 encoder.
    
    Extracts features from t1 and t2, computes the absolute difference in the 
    feature space, and processes the fused features using an All-MLP decoder
    to predict a binary change mask.
    """
    def __init__(self, in_channels: int = 3, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        
        # Load EfficientNet-B0 encoder with pretrained weights
        self.encoder = timm.create_model(
            'efficientnet_b0', 
            pretrained=pretrained, 
            in_chans=in_channels, 
            features_only=True
        )
        
        # efficientnet_b0 feature dimensions: [16, 24, 40, 112, 320]
        encoder_channels = self.encoder.feature_info.channels()
        
        self.decoder = MLPDecoder(
            in_channels_list=encoder_channels, 
            embedding_dim=embedding_dim
        )
        
    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ChangeFormer.
        
        Args:
            t1 (torch.Tensor): First image tensor of shape (B, C, H, W).
            t2 (torch.Tensor): Second image tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Predicted change mask logits of shape (B, 1, H, W).
                          (BCEWithLogitsLoss applies sigmoid internally to obtain probabilities).
        """
        # Extract hierarchical features from both images
        features_t1 = self.encoder(t1)
        features_t2 = self.encoder(t2)
        
        # Compute absolute difference in transformer feature space
        diff_features = [
            torch.abs(f1 - f2) 
            for f1, f2 in zip(features_t1, features_t2)
        ]
        
        # Decode differences
        decoder_output = self.decoder(diff_features)
        
        # Upsample from decoder resolution (1/4 input resolution) to original input resolution
        input_size = t1.shape[2:]
        logits = F.interpolate(
            decoder_output, 
            size=input_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        return logits
