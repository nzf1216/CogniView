"""
train.py

Training pipeline for CogniView Adaptive Fusion Model.

Features
--------
✓ GPU / CPU Auto Selection
✓ Mixed Precision Training
✓ Early Stopping
✓ Best Model Saving
✓ LR Scheduler
✓ Validation
✓ Precision
✓ Recall
✓ F1
✓ TensorBoard
✓ CSV Logs

Author
------
CogniView Research Team
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.fusion.dataset import (
    create_dataloader,
)

from src.fusion.model import (
    AdaptiveFusionModel,
)

logger = logging.getLogger("cogniview.fusion.train")


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TRAIN_CSV = "data/fusion/train.csv"

VAL_CSV = "data/fusion/val.csv"

CHECKPOINT_DIR = Path("models/fusion")

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_FILE = (
    CHECKPOINT_DIR /
    "fusion_model.pth"
)

LOG_DIR = Path("results/logs")

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NUM_EPOCHS = 100

BATCH_SIZE = 32

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

SEQUENCE_LENGTH = 30

PATIENCE = 15


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

logger.info(
    "Using device: %s",
    DEVICE,
)


# ---------------------------------------------------------
# TensorBoard
# ---------------------------------------------------------

writer = SummaryWriter(
    LOG_DIR
)


# ---------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------

class EarlyStopping:

    def __init__(

        self,

        patience=PATIENCE,

    ):

        self.patience = patience

        self.counter = 0

        self.best_loss = float("inf")

        self.stop = False

    def update(

        self,

        loss,

    ):

        if loss < self.best_loss:

            self.best_loss = loss

            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.stop = True
# ---------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------

train_loader = create_dataloader(
    csv_file=TRAIN_CSV,
    sequence_length=SEQUENCE_LENGTH,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = create_dataloader(
    csv_file=VAL_CSV,
    sequence_length=SEQUENCE_LENGTH,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = AdaptiveFusionModel()

model.to(DEVICE)


# ---------------------------------------------------------
# Loss
# ---------------------------------------------------------

criterion = nn.CrossEntropyLoss()


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ---------------------------------------------------------
# LR Scheduler
# ---------------------------------------------------------

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
)


# ---------------------------------------------------------
# Mixed Precision
# ---------------------------------------------------------

scaler = torch.cuda.amp.GradScaler(
    enabled=torch.cuda.is_available()
)


# ---------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------

early_stopping = EarlyStopping()


# ---------------------------------------------------------
# Training Epoch
# ---------------------------------------------------------

def train_epoch():

    model.train()

    total_loss = 0.0

    predictions = []

    labels = []

    for x, y in train_loader:

        x = x.to(DEVICE)

        y = y.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(
            enabled=torch.cuda.is_available()
        ):

            logits, _ = model(x)

            loss = criterion(
                logits,
                y,
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item()

        pred = torch.argmax(
            logits,
            dim=1,
        )

        predictions.extend(
            pred.cpu().numpy()
        )

        labels.extend(
            y.cpu().numpy()
        )

    loss = total_loss / len(train_loader)

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return (
        loss,
        accuracy,
        precision,
        recall,
        f1,
    )
# ---------------------------------------------------------
# Validation Epoch
# ---------------------------------------------------------

@torch.no_grad()
def validate_epoch():

    model.eval()

    total_loss = 0.0

    predictions = []

    labels = []

    for x, y in val_loader:

        x = x.to(DEVICE)

        y = y.to(DEVICE)

        logits, _ = model(x)

        loss = criterion(
            logits,
            y,
        )

        total_loss += loss.item()

        pred = torch.argmax(
            logits,
            dim=1,
        )

        predictions.extend(
            pred.cpu().numpy()
        )

        labels.extend(
            y.cpu().numpy()
        )

    loss = total_loss / len(val_loader)

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return (
        loss,
        accuracy,
        precision,
        recall,
        f1,
    )


# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------

def train():

    best_loss = float("inf")

    logger.info("Starting Fusion Model Training")

    for epoch in range(NUM_EPOCHS):

        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch()

        val_loss, val_acc, val_prec, val_rec, val_f1 = validate_epoch()

        scheduler.step(val_loss)

        writer.add_scalar(
            "Loss/Train",
            train_loss,
            epoch,
        )

        writer.add_scalar(
            "Loss/Validation",
            val_loss,
            epoch,
        )

        writer.add_scalar(
            "Accuracy/Train",
            train_acc,
            epoch,
        )

        writer.add_scalar(
            "Accuracy/Validation",
            val_acc,
            epoch,
        )

        logger.info(
            "Epoch [%d/%d] | "
            "Train Loss %.4f | "
            "Val Loss %.4f | "
            "Train Acc %.4f | "
            "Val Acc %.4f",
            epoch + 1,
            NUM_EPOCHS,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
        )

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                model.state_dict(),
                CHECKPOINT_FILE,
            )

            logger.info(
                "Best model saved -> %s",
                CHECKPOINT_FILE,
            )

        early_stopping.update(
            val_loss,
        )

        if early_stopping.stop:

            logger.info(
                "Early stopping triggered."
            )

            break

    writer.close()

    logger.info("Training Complete.")
# ---------------------------------------------------------
# Resume Training
# ---------------------------------------------------------

def load_checkpoint():

    if CHECKPOINT_FILE.exists():

        logger.info(
            "Loading checkpoint: %s",
            CHECKPOINT_FILE,
        )

        model.load_state_dict(
            torch.load(
                CHECKPOINT_FILE,
                map_location=DEVICE,
            )
        )

        return True

    logger.info(
        "No checkpoint found. Starting fresh training."
    )

    return False


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s | %(levelname)s | %(message)s",

    )

    logger.info("------------------------------------")
    logger.info("CogniView Fusion Training")
    logger.info("------------------------------------")

    logger.info("Device : %s", DEVICE)

    logger.info("Training CSV : %s", TRAIN_CSV)

    logger.info("Validation CSV : %s", VAL_CSV)

    load_checkpoint()

    train()

    logger.info("------------------------------------")
    logger.info("Training Finished")
    logger.info("------------------------------------")


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":

    main()