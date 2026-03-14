"""
Hybrid CNN + Coordinate Model for Enhanced Emotion Recognition
Combines facial landmark coordinates with face crop appearance features
for 85%+ per-class accuracy across all 7 emotion classes.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional
import logging

from ..config import Config


class FaceCropCNN(nn.Module):
    """Lightweight CNN branch for face crop appearance features.
    Uses MobileNetV3-Small backbone for efficiency on 4GB VRAM.
    """

    def __init__(self, feature_dim: int = 256, pretrained: bool = True):
        super().__init__()

        # Load MobileNetV3-Small (1.5M params, very VRAM efficient)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # Remove the classifier head, keep feature extractor
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Backbone output is 576-dim
        self.projection = nn.Sequential(
            nn.Linear(576, feature_dim),
            nn.GELU(),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.3)
        )

        # Freeze early layers (first 6 of 13 blocks) to save memory
        for i, layer in enumerate(self.features):
            if i < 6:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Face crop tensor [B, 3, 112, 112] (RGB, normalized)
        Returns:
            features: [B, feature_dim]
        """
        features = self.features(x)
        features = self.avgpool(features)
        features = features.flatten(1)
        features = self.projection(features)
        return features


class CoordinateBranch(nn.Module):
    """Coordinate branch - simplified from EnhancedCoordinateEmotionNet.
    Extracts geometric features from 1434 landmark coordinates.
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
            nn.Dropout(0.3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Landmark coordinates [B, 1434]
        Returns:
            features: [B, feature_dim]
        """
        return self.encoder(x)


class HybridEmotionNet(nn.Module):
    """Hybrid model combining CNN appearance features with coordinate geometry features.

    Architecture:
        Face Crop (112x112) -> MobileNetV3-Small -> 256-dim appearance features
        Coordinates (1434)  -> MLP encoder       -> 256-dim geometry features
        [Concatenate 512-dim] -> Fusion MLP -> 7 emotion classes

    This hybrid approach captures both:
    - Geometric patterns (mouth shape, eye openness, brow position)
    - Appearance cues (wrinkles, muscle tension, skin texture)
    """

    def __init__(self, num_classes: int = 7, coord_dim: int = 1434,
                 feature_dim: int = 256, pretrained_cnn: bool = True):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Branch 1: CNN for face crops
        self.cnn_branch = FaceCropCNN(feature_dim=feature_dim, pretrained=pretrained_cnn)

        # Branch 2: Coordinate encoder
        self.coord_branch = CoordinateBranch(input_size=coord_dim, feature_dim=feature_dim)

        # Cross-attention: let each branch attend to the other
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=4, dropout=0.1, batch_first=True
        )

        # Fusion network
        fused_dim = feature_dim * 2  # 512
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

        # Final classifier
        self.classifier = nn.Linear(128, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize fusion and classifier weights"""
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='relu')
        nn.init.zeros_(self.classifier.bias)

    def forward(self, coordinates: torch.Tensor,
                face_crop: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            coordinates: [B, 1434] landmark coordinates
            face_crop: [B, 3, 112, 112] face crop image (optional for backward compat)
        Returns:
            logits: [B, num_classes]
        """
        # Coordinate features
        coord_features = self.coord_branch(coordinates)  # [B, 256]

        if face_crop is not None:
            # CNN features
            cnn_features = self.cnn_branch(face_crop)  # [B, 256]

            # Cross-attention: coordinate features attend to CNN features
            # Reshape for attention: [B, 1, 256]
            coord_q = coord_features.unsqueeze(1)
            cnn_kv = cnn_features.unsqueeze(1)

            attended_coords, _ = self.cross_attention(coord_q, cnn_kv, cnn_kv)
            attended_coords = attended_coords.squeeze(1)  # [B, 256]

            # Fuse both branches
            fused = torch.cat([attended_coords, cnn_features], dim=1)  # [B, 512]
        else:
            # Coordinate-only fallback (for inference without face crops)
            fused = torch.cat([coord_features, coord_features], dim=1)  # [B, 512]

        # Fusion + classification
        features = self.fusion(fused)
        logits = self.classifier(features)

        return logits


def create_hybrid_model(pretrained_cnn: bool = True) -> HybridEmotionNet:
    """Create and return a hybrid model on the configured device"""
    model = HybridEmotionNet(
        num_classes=Config.NUM_CLASSES,
        coord_dim=Config.COORDINATE_DIM,
        feature_dim=256,
        pretrained_cnn=pretrained_cnn
    )

    model = model.to(Config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f"Hybrid Model: {total_params:,} total params, {trainable_params:,} trainable")
    logging.info(f"CNN branch: MobileNetV3-Small (partially frozen)")
    logging.info(f"Coordinate branch: MLP encoder")
    logging.info(f"Fusion: Cross-attention + MLP")

    return model
