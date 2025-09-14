"""
CNN Model for CAPTCHA Classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class CaptchaCNN(nn.Module):
    """CNN model for CAPTCHA text classification"""

    def __init__(self, num_classes: int = 100, dropout: float = 0.5):
        """
        Initialize the CNN model

        Args:
            num_classes: Number of word classes
            dropout: Dropout probability
        """
        super(CaptchaCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 12))

        self.fc1 = nn.Linear(256 * 4 * 12, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = F.relu(self.bn4(self.conv4(x)))

        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn6(self.fc2(x)))
        x = self.dropout(x)

        x = self.fc3(x)

        return x

class ImprovedCaptchaCNN(nn.Module):
    """Improved CNN with residual connections and attention"""

    def __init__(self, num_classes: int = 100, dropout: float = 0.5):
        super(ImprovedCaptchaCNN, self).__init__()

        self.conv_init = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn_init = nn.BatchNorm2d(64)
        self.pool_init = nn.MaxPool2d(3, stride=2, padding=1)

        self.res_block1 = ResidualBlock(64, 128)
        self.res_block2 = ResidualBlock(128, 256)
        self.res_block3 = ResidualBlock(256, 512)

        self.attention = SpatialAttention(512)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = F.relu(self.bn_init(self.conv_init(x)))
        x = self.pool_init(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = self.attention(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x

class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""

    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        residual = self.shortcut(residual)
        out += residual
        out = F.relu(out)

        return out

class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""

    def __init__(self, in_channels: int):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.sigmoid(self.conv(x))
        return x * attention

class LightweightCaptchaCNN(nn.Module):
    """Lightweight model for faster training"""

    def __init__(self, num_classes: int = 100, dropout: float = 0.3):
        super(LightweightCaptchaCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 25))
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 25, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def create_model(model_type: str = 'standard',
                num_classes: int = 100,
                dropout: float = 0.5) -> nn.Module:
    """
    Factory function to create models

    Args:
        model_type: 'standard', 'improved', or 'lightweight'
        num_classes: Number of output classes
        dropout: Dropout probability

    Returns:
        PyTorch model
    """
    if model_type == 'standard':
        return CaptchaCNN(num_classes, dropout)
    elif model_type == 'improved':
        return ImprovedCaptchaCNN(num_classes, dropout)
    elif model_type == 'lightweight':
        return LightweightCaptchaCNN(num_classes, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":

    batch_size = 4
    num_classes = 100

    x = torch.randn(batch_size, 3, 64, 200)

    print("Testing models...")
    for model_type in ['lightweight', 'standard', 'improved']:
        print(f"\n{model_type.capitalize()} Model:")
        model = create_model(model_type, num_classes)

        output = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")