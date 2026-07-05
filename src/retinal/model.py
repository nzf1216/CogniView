import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from config import NUM_CLASSES


class RetinaClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        in_features = (
            self.backbone.classifier[1]
            .in_features
        )

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(
                in_features,
                NUM_CLASSES
            )
        )

    def forward(self, x):

        return self.backbone(x)