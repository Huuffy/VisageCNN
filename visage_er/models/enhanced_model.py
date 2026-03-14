import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from ..config import Config

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism for facial landmarks"""

    def __init__(self, embed_dim, num_heads=8, dropout=0.05):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # Generate queries, keys, values
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention weights
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )

        return self.out_proj(attention_output)

class GeometricFeatureExtractor(nn.Module):
    """Extract geometric features from facial landmarks"""

    def __init__(self, num_landmarks=478):
        super(GeometricFeatureExtractor, self).__init__()
        self.num_landmarks = num_landmarks

        # Define key facial regions (indices for important landmarks)
        self.eye_region = list(range(0, 50))
        self.mouth_region = list(range(50, 120))
        self.nose_region = list(range(120, 180))
        self.eyebrow_region = list(range(180, 220))
        self.face_contour = list(range(220, 300))

    def extract_distances(self, landmarks):
        """Extract pairwise distances between key landmarks"""
        # Reshape landmarks to (batch_size, num_landmarks, 3)
        landmarks = landmarks.view(-1, self.num_landmarks, 3)

        # Calculate distances between specific landmark pairs
        eye_center_left = torch.mean(landmarks[:, self.eye_region[:25], :], dim=1)
        eye_center_right = torch.mean(landmarks[:, self.eye_region[25:], :], dim=1)
        mouth_center = torch.mean(landmarks[:, self.mouth_region, :], dim=1)
        nose_tip = landmarks[:, 1, :]  # Nose tip landmark

        # Calculate key distances
        eye_distance = torch.norm(eye_center_left - eye_center_right, dim=1)
        mouth_width = torch.norm(landmarks[:, 60, :] - landmarks[:, 90, :], dim=1)
        face_height = torch.norm(landmarks[:, 10, :] - landmarks[:, 152, :], dim=1)

        return torch.stack([eye_distance, mouth_width, face_height], dim=1)

    def extract_angles(self, landmarks):
        """Extract angular features from facial landmarks"""
        landmarks = landmarks.view(-1, self.num_landmarks, 3)

        # Calculate angles between key facial features
        # Eye angle (tilt)
        eye_left = landmarks[:, 33, :2]  # Left eye corner
        eye_right = landmarks[:, 133, :2]  # Right eye corner
        eye_angle = torch.atan2(eye_right[:, 1] - eye_left[:, 1],
                               eye_right[:, 0] - eye_left[:, 0])

        # Mouth angle
        mouth_left = landmarks[:, 61, :2]  # Left mouth corner
        mouth_right = landmarks[:, 291, :2]  # Right mouth corner
        mouth_angle = torch.atan2(mouth_right[:, 1] - mouth_left[:, 1],
                                 mouth_right[:, 0] - mouth_left[:, 0])

        return torch.stack([eye_angle, mouth_angle], dim=1)

    def forward(self, landmarks):
        """Extract comprehensive geometric features"""
        distances = self.extract_distances(landmarks)
        angles = self.extract_angles(landmarks)

        return torch.cat([distances, angles], dim=1)

class AdvancedResidualBlock(nn.Module):
    """Advanced residual block with attention and normalization"""

    def __init__(self, in_features, out_features, dropout_rate=0.05, use_attention=True):
        super(AdvancedResidualBlock, self).__init__()

        # Main path
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(out_features, out_features // 4),
                nn.ReLU(),
                nn.Linear(out_features // 4, out_features),
                nn.Sigmoid()
            )

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Skip connection
        self.skip_connection = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

        # Activation
        self.activation = nn.GELU()  # GELU activation for better performance

    def forward(self, x):
        identity = self.skip_connection(x)

        # Main path
        out = self.activation(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))

        # Apply attention if enabled
        if self.use_attention:
            attention_weights = self.attention(out)
            out = out * attention_weights

        # Residual connection
        out += identity
        out = self.activation(out)

        return out

class EnhancedCoordinateEmotionNet(nn.Module):
    """Enhanced coordinate-based emotion recognition network with advanced features"""

    def __init__(self, input_size=Config.FEATURE_SIZE, num_classes=Config.NUM_CLASSES):
        super(EnhancedCoordinateEmotionNet, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.num_landmarks = Config.NUM_LANDMARKS

        # Input processing
        self.input_projection = nn.Linear(input_size, 512)
        self.input_bn = nn.BatchNorm1d(512)
        self.input_dropout = nn.Dropout(0.2)

        # Geometric feature extraction
        self.geometric_extractor = GeometricFeatureExtractor(self.num_landmarks)
        self.geometric_projection = nn.Linear(5, 64)  # 5 geometric features

        # Landmark reshaping for attention
        self.landmark_embed = nn.Linear(3, 64)  # Project each landmark to 64 dims

        # Multi-head self-attention for spatial relationships
        self.spatial_attention = MultiHeadSelfAttention(64, num_heads=8, dropout=0.2)

        # Advanced residual blocks
        self.residual_blocks = nn.ModuleList([
            AdvancedResidualBlock(512 + 64 + 64, 1024, dropout_rate=0.4, use_attention=True),
            AdvancedResidualBlock(1024, 768, dropout_rate=0.3, use_attention=True),
            AdvancedResidualBlock(768, 512, dropout_rate=0.3, use_attention=True),
            AdvancedResidualBlock(512, 256, dropout_rate=0.2, use_attention=False),
        ])

        # Global feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )

        # Emotion-specific expert networks
        self.emotion_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32)
            ) for _ in range(num_classes)
        ])

        # Final classification with uncertainty estimation
        self.classifier = nn.Sequential(
            nn.Linear(32 * num_classes, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

        # Uncertainty estimation branch
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights with advanced techniques"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for linear layers
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x, return_uncertainty=False):
        """Enhanced forward pass with multiple feature streams"""
        batch_size = x.size(0)

        # Main coordinate processing
        coord_features = self.input_dropout(torch.relu(self.input_bn(self.input_projection(x))))

        # Extract geometric features
        geometric_features = self.geometric_extractor(x)
        geometric_features = torch.relu(self.geometric_projection(geometric_features))

        # Process landmarks for spatial attention
        landmarks = x.view(batch_size, self.num_landmarks, 3)
        landmark_embeds = self.landmark_embed(landmarks)  # (batch_size, num_landmarks, 64)

        # Apply spatial attention
        spatial_features = self.spatial_attention(landmark_embeds)
        spatial_features = torch.mean(spatial_features, dim=1)  # Global average pooling

        # Concatenate all features
        fused_features = torch.cat([coord_features, geometric_features, spatial_features], dim=1)

        # Pass through residual blocks
        for block in self.residual_blocks:
            fused_features = block(fused_features)

        # Global feature fusion
        global_features = self.feature_fusion(fused_features)

        # Emotion-specific expert processing
        expert_outputs = []
        for expert in self.emotion_experts:
            expert_output = expert(global_features)
            expert_outputs.append(expert_output)

        # Concatenate expert outputs
        expert_features = torch.cat(expert_outputs, dim=1)

        # Final classification
        logits = self.classifier(expert_features)

        if return_uncertainty:
            uncertainty = self.uncertainty_head(global_features)
            return logits, uncertainty

        return logits

    def get_attention_weights(self, x):
        """Extract attention weights for interpretability"""
        batch_size = x.size(0)
        landmarks = x.view(batch_size, self.num_landmarks, 3)
        landmark_embeds = self.landmark_embed(landmarks)

        # Get attention weights from spatial attention
        attention_module = self.spatial_attention
        Q = attention_module.query(landmark_embeds)
        K = attention_module.key(landmark_embeds)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(attention_module.head_dim)
        attention_weights = F.softmax(scores, dim=-1)

        return attention_weights

    def get_feature_importance(self):
        """Get feature importance scores"""
        # Get weights from first layer
        first_layer_weights = self.input_projection.weight.data.abs().mean(dim=0)

        # Normalize to get importance scores
        importance_scores = first_layer_weights / first_layer_weights.sum()

        return importance_scores.cpu().numpy()

# Enhanced loss functions
class AdaptiveFocalLoss(nn.Module):
    """Fixed adaptive focal loss implementation"""

    def __init__(self, num_classes=7, alpha=1.0, gamma=2.0, device=None):
        super(AdaptiveFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma

        # Fix: Ensure alpha is properly initialized as a tensor
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = torch.ones(num_classes, dtype=torch.float32) * alpha

        if device:
            self.alpha = self.alpha.to(device)

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        """Fixed forward pass with proper tensor handling"""
        # Ensure inputs and targets are on the same device
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        # Calculate cross entropy loss
        ce_loss = self.ce_loss(inputs, targets)

        # Calculate probabilities
        pt = torch.exp(-ce_loss)

        # Fix: Handle tensor indexing properly
        if targets.dim() == 0:  # Single scalar target
            targets = targets.unsqueeze(0)

        # Ensure targets are within valid range
        targets = torch.clamp(targets, 0, self.num_classes - 1)

        # Get alpha values for targets
        try:
            alpha_t = self.alpha[targets]
        except IndexError:
            # Fallback: use uniform alpha
            alpha_t = torch.ones_like(targets, dtype=torch.float32, device=inputs.device)

        # Calculate focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


class UncertaintyLoss(nn.Module):
    """Uncertainty-aware loss function"""

    def __init__(self, base_loss_fn):
        super(UncertaintyLoss, self).__init__()
        self.base_loss = base_loss_fn

    def forward(self, predictions, targets, uncertainty=None):
        base_loss = self.base_loss(predictions, targets)

        if uncertainty is not None:
            # Higher uncertainty should lead to higher loss
            uncertainty_penalty = torch.mean(uncertainty * base_loss)
            # Encourage confidence when predictions are correct
            correct_mask = (predictions.argmax(dim=1) == targets).float()
            confidence_reward = torch.mean((1 - uncertainty) * correct_mask)

            return base_loss + uncertainty_penalty - 0.1 * confidence_reward

        return base_loss

# Model utilities
class EnhancedModelUtils:
    """Enhanced utilities for the improved model"""

    @staticmethod
    def count_parameters(model):
        """Count total trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def save_enhanced_model(model, optimizer, scheduler, epoch, loss, metrics, path):
        """Save enhanced model with additional metadata"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'metrics': metrics,
            'model_architecture': {
                'input_size': Config.FEATURE_SIZE,
                'num_classes': Config.NUM_CLASSES,
                'num_landmarks': Config.NUM_LANDMARKS,
                'hidden_layers': Config.HIDDEN_LAYERS,
                'dropout_rates': Config.DROPOUT_RATES
            },
            'enhanced_features': {
                'spatial_attention': True,
                'geometric_features': True,
                'emotion_experts': True,
                'uncertainty_estimation': True
            }
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load_enhanced_model(path, device=Config.DEVICE):
        """Load enhanced model checkpoint"""
        checkpoint = torch.load(path, map_location=device)

        # Create enhanced model
        model = EnhancedCoordinateEmotionNet(
            input_size=checkpoint['model_architecture']['input_size'],
            num_classes=checkpoint['model_architecture']['num_classes']
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model, checkpoint

    @staticmethod
    def get_enhanced_model_summary(model):
        """Get detailed model summary"""
        total_params = EnhancedModelUtils.count_parameters(model)

        summary = f"""
Enhanced VisageCNN Model Summary
================================
Architecture: EnhancedCoordinateEmotionNet
Input Features: {Config.FEATURE_SIZE} (from {Config.NUM_LANDMARKS} landmarks)
Output Classes: {Config.NUM_CLASSES}
Total Parameters: {total_params:,}

Enhanced Features:
✓ Multi-head spatial attention
✓ Geometric feature extraction
✓ Advanced residual blocks with attention
✓ Emotion-specific expert networks
✓ Uncertainty estimation
✓ Adaptive focal loss support

Training Device: {Config.DEVICE}
        """

        return summary.strip()

def create_enhanced_model():
    """Create and return the enhanced emotion recognition model"""
    model = EnhancedCoordinateEmotionNet()
    model.to(Config.DEVICE)

    print(EnhancedModelUtils.get_enhanced_model_summary(model))

    return model

# Compatibility aliases
CoordinateEmotionNet = EnhancedCoordinateEmotionNet
create_model = create_enhanced_model
ModelUtils = EnhancedModelUtils
FocalLoss = AdaptiveFocalLoss
