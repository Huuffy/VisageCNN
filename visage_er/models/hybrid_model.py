"""Hybrid emotion recognition model combining EfficientNet-B0 appearance features
with an MLP-based facial landmark coordinate encoder, fused via cross-attention.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
import logging

from ..config import Config


class FaceCropCNN(nn.Module):
    """EfficientNet-B0 backbone for extracting appearance features from 224×224 face crops.

    The first three feature blocks are frozen (low-level edge/texture patterns); blocks
    3–8 are fine-tuned to learn expression-specific appearance cues.

    Args:
        feature_dim: Output projection dimension. Default 256.
        pretrained: Whether to initialise with ImageNet weights. Default True.
    """

    def __init__(self, feature_dim: int = 256, pretrained: bool = True):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        self.projection = nn.Sequential(
            nn.Linear(1280, feature_dim),
            nn.GELU(),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.4),
        )

        for i, layer in enumerate(self.features):
            if i < 3:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract appearance features from a batch of face crops.

        Args:
            x: RGB face crop tensor of shape [B, 3, 224, 224], ImageNet-normalised.

        Returns:
            Feature tensor of shape [B, feature_dim].
        """
        features = self.features(x)
        features = self.avgpool(features)
        features = features.flatten(1)
        features = self.projection(features)
        return features


class CoordinateBranch(nn.Module):
    """MLP encoder that maps 1434 facial landmark coordinates to a compact feature vector.

    Args:
        input_size: Number of input coordinate features. Default 1434.
        feature_dim: Output feature dimension. Default 256.
    """

    def __init__(self, input_size: int = 1434, feature_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),

            nn.Linear(512, 384),
            nn.GELU(),
            nn.BatchNorm1d(384),
            nn.Dropout(0.25),

            nn.Linear(384, feature_dim),
            nn.GELU(),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode landmark coordinates into a fixed-length feature vector.

        Args:
            x: Coordinate tensor of shape [B, 1434].

        Returns:
            Feature tensor of shape [B, feature_dim].
        """
        return self.encoder(x)


class HybridEmotionNet(nn.Module):
    """Hybrid model that fuses facial appearance and geometry for emotion classification.

    Architecture:
        Face crop [B, 3, 224, 224]  → EfficientNet-B0 → [B, 256] appearance features
        Coordinates [B, 1434]       → MLP encoder     → [B, 256] geometry features
        Cross-attention → concat [B, 512] → fusion MLP → [B, num_classes]

    Args:
        num_classes: Number of emotion output classes. Default 7.
        coord_dim: Dimensionality of the landmark coordinate input. Default 1434.
        feature_dim: Shared feature dimension for both branches. Default 256.
        pretrained_cnn: Whether to use pretrained EfficientNet-B0 weights. Default True.
    """

    def __init__(self, num_classes: int = 7, coord_dim: int = 1434,
                 feature_dim: int = 256, pretrained_cnn: bool = True):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.cnn_branch = FaceCropCNN(feature_dim=feature_dim, pretrained=pretrained_cnn)
        self.coord_branch = CoordinateBranch(input_size=coord_dim, feature_dim=feature_dim)

        self.cross_attn_c2a = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=4, dropout=0.1, batch_first=True,
        )
        self.cross_attn_a2c = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=4, dropout=0.1, batch_first=True,
        )

        fused_dim = feature_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 384),
            nn.GELU(),
            nn.BatchNorm1d(384),
            nn.Dropout(0.35),

            nn.Linear(384, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Linear(128, num_classes)
        self._init_weights()

    def _init_weights(self):
        """Initialise fusion and classifier weights with Kaiming normal."""
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='relu')
        nn.init.zeros_(self.classifier.bias)

    def forward(self, coordinates: torch.Tensor,
                face_crop: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the hybrid model.

        Args:
            coordinates: Landmark coordinate tensor of shape [B, 1434].
            face_crop: Face crop tensor of shape [B, 3, 224, 224]. When None,
                the coordinate branch output is duplicated for fusion (coordinate-only mode).

        Returns:
            Class logits of shape [B, num_classes].
        """
        coord_features = self.coord_branch(coordinates)

        if face_crop is not None:
            cnn_features = self.cnn_branch(face_crop)

            coord_q = coord_features.unsqueeze(1)
            cnn_q = cnn_features.unsqueeze(1)

            attended_coords, _ = self.cross_attn_c2a(coord_q, cnn_q, cnn_q)
            attended_cnn, _ = self.cross_attn_a2c(cnn_q, coord_q, coord_q)

            fused = torch.cat([attended_coords.squeeze(1), attended_cnn.squeeze(1)], dim=1)
        else:
            fused = torch.cat([coord_features, coord_features], dim=1)

        features = self.fusion(fused)
        logits = self.classifier(features)
        return logits


def create_hybrid_model(pretrained_cnn: bool = True) -> HybridEmotionNet:
    """Instantiate and return a HybridEmotionNet on the configured device.

    Args:
        pretrained_cnn: Whether to initialise the CNN branch with ImageNet weights.

    Returns:
        HybridEmotionNet instance moved to Config.DEVICE.
    """
    model = HybridEmotionNet(
        num_classes=Config.NUM_CLASSES,
        coord_dim=Config.COORDINATE_DIM,
        feature_dim=256,
        pretrained_cnn=pretrained_cnn,
    )

    model = model.to(Config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f"HybridEmotionNet: {total_params:,} total params, {trainable_params:,} trainable")
    logging.info("CNN branch: EfficientNet-B0 (blocks 0-2 frozen, blocks 3-8 fine-tuned)")
    logging.info("Coordinate branch: MLP encoder (1434 -> 512 -> 384 -> 256)")
    logging.info("Fusion: bidirectional cross-attention + MLP (512 -> 384 -> 256 -> 128 -> 7)")

    return model
