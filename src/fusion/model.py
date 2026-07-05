"""
model.py

Adaptive multimodal temporal fusion model for CogniView.

Current inputs
--------------
✓ Eye Tracking

Future inputs
-------------
✓ Retina Embedding
✓ PLR Features

Outputs
-------
Risk Class
Confidence
Feature Importance

Author
------
CogniView Research Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("cogniview.fusion.model")


# ---------------------------------------------------------
# Risk Classes
# ---------------------------------------------------------

class RiskClass(Enum):
    LOW = 0
    MODERATE = 1
    HIGH = 2


# ---------------------------------------------------------
# Current Eye Tracking Feature Vector
#
# When PLR is implemented simply append new names.
# ---------------------------------------------------------

FUSION_FEATURE_NAMES = [

    "ear",

    "blink_count",

    "blink_rate",

    "fixation_stability",

    "fixation_count",

    "saccade_count",

    "smooth_pursuit_count",

    "left_iris_x",
    "left_iris_y",

    "right_iris_x",
    "right_iris_y",

]

NUM_FEATURES = len(FUSION_FEATURE_NAMES)


# ---------------------------------------------------------
# Model Output
# ---------------------------------------------------------

@dataclass
class FusionPrediction:

    logits: torch.Tensor

    probabilities: torch.Tensor

    predicted_class: RiskClass

    confidence: float


# ---------------------------------------------------------
# Temporal CNN Block
# ---------------------------------------------------------

class TemporalConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.20,
    ):

        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(

            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
            ),

            nn.BatchNorm1d(out_channels),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

        )

    def forward(self, x):

        return self.block(x)


# ---------------------------------------------------------
# Attention Layer
# ---------------------------------------------------------

class TemporalAttention(nn.Module):

    def __init__(self, channels: int):

        super().__init__()

        self.score = nn.Linear(channels, 1)

    def forward(self, x):

        #
        # x
        #
        # batch
        # sequence
        # channels
        #

        weights = self.score(x)

        weights = torch.softmax(weights, dim=1)

        context = torch.sum(weights * x, dim=1)

        return context, weights
    # ---------------------------------------------------------
# Adaptive Fusion Network
# ---------------------------------------------------------

class AdaptiveFusionModel(nn.Module):
    """
    Input shape:
        (batch, sequence_length, NUM_FEATURES)

    Output:
        logits (batch, 3)
    """

    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        hidden_size: int = 64,
        num_classes: int = 3,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        #
        # Temporal CNN
        #

        self.conv1 = TemporalConvBlock(
            in_channels=num_features,
            out_channels=64,
        )

        self.conv2 = TemporalConvBlock(
            in_channels=64,
            out_channels=128,
        )

        self.conv3 = TemporalConvBlock(
            in_channels=128,
            out_channels=hidden_size,
        )

        #
        # Attention
        #

        self.attention = TemporalAttention(hidden_size)

        #
        # Classifier
        #

        self.classifier = nn.Sequential(

            nn.Linear(hidden_size, 128),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(128, 64),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(64, num_classes),

        )

        logger.info(
            "AdaptiveFusionModel initialized "
            "(features=%d classes=%d)",
            num_features,
            num_classes,
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x

        Shape

        (batch, sequence, features)
        """

        if x.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "(batch, sequence, features)"
            )

        #
        # Conv1D expects
        #
        # (batch, channels, sequence)
        #

        x = x.transpose(1, 2)

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        #
        # Attention expects
        #
        # (batch, sequence, channels)
        #

        x = x.transpose(1, 2)

        context, attention_weights = self.attention(x)

        logits = self.classifier(context)

        return logits, attention_weights

    @torch.no_grad()
    def predict(self, x):
        """
        Returns

        FusionPrediction
        """

        self.eval()

        logits, attention = self.forward(x)

        probabilities = F.softmax(logits, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

        return (
            logits,
            probabilities,
            prediction,
            confidence,
            attention,
        )
    # ---------------------------------------------------------
# Feature Vectorization
# ---------------------------------------------------------

def vectorize(features) -> torch.Tensor:
    """
    Convert EyeTrackingFeatures into a tensor.

    Returns
    -------
    Tensor shape:
        (NUM_FEATURES,)
    """

    left_x, left_y = (0.0, 0.0)
    right_x, right_y = (0.0, 0.0)

    if features.left_iris is not None:
        left_x, left_y = features.left_iris

    if features.right_iris is not None:
        right_x, right_y = features.right_iris

    vector = torch.tensor(
        [
            features.ear,
            features.blink_count,
            features.blink_rate,
            features.fixation_stability,
            features.fixation_count,
            features.saccade_count,
            features.smooth_pursuit_count,
            left_x,
            left_y,
            right_x,
            right_y,
        ],
        dtype=torch.float32,
    )

    return vector


# ---------------------------------------------------------
# Inference Wrapper
# ---------------------------------------------------------

class FusionEngine:

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ):

        self.device = torch.device(
            device if device else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model = AdaptiveFusionModel()

        self.model.to(self.device)

        self.model.eval()

        if checkpoint is not None:

            state = torch.load(
                checkpoint,
                map_location=self.device,
            )

            self.model.load_state_dict(state)

            logger.info(
                "Loaded checkpoint: %s",
                checkpoint,
            )

        else:

            logger.warning(
                "Running FusionModel with random weights. "
                "Train the model before clinical inference."
            )

    @torch.no_grad()
    def predict_sequence(
        self,
        sequence: torch.Tensor,
    ) -> FusionPrediction:
        """
        Parameters
        ----------
        sequence

        Shape

        (sequence, features)

        or

        (1, sequence, features)
        """

        if sequence.ndim == 2:
            sequence = sequence.unsqueeze(0)

        sequence = sequence.to(self.device)

        logits, probabilities, prediction, confidence, _ = \
            self.model.predict(sequence)

        return FusionPrediction(
            logits=logits.cpu(),
            probabilities=probabilities.cpu(),
            predicted_class=RiskClass(
                prediction.item()
            ),
            confidence=float(
                confidence.item()
            ),
        )


# ---------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------

def _demo():

    model = AdaptiveFusionModel()

    dummy = torch.randn(
        1,
        30,
        NUM_FEATURES,
    )

    logits, attention = model(dummy)

    print()

    print("Input :", dummy.shape)

    print("Logits :", logits.shape)

    print("Attention :", attention.shape)

    print()

    engine = FusionEngine()

    prediction = engine.predict_sequence(dummy)

    print("Prediction")

    print("----------------")

    print(
        "Risk:",
        prediction.predicted_class.name,
    )

    print(
        "Confidence:",
        f"{prediction.confidence:.4f}",
    )

    print(
        "Probabilities:",
        prediction.probabilities,
    )


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO
    )

    _demo()