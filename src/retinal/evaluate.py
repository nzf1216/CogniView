import torch
import numpy as np

from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from dataset import RetinaDataset
from transforms import get_valid_transforms
from model import RetinaClassifier

from config import (
    DATA_DIR,
    BATCH_SIZE,
    CHECKPOINT_PATH
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using Device: {device}")


# ==========================
# DATASET
# ==========================

full_dataset = RetinaDataset(
    csv_file=DATA_DIR / "retina" / "train.csv",
    image_dir=DATA_DIR / "retina" / "train_images",
    transform=get_valid_transforms()
)

train_size = int(
    0.8 * len(full_dataset)
)

val_size = (
    len(full_dataset)
    - train_size
)

_, val_dataset = random_split(
    full_dataset,
    [train_size, val_size]
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    f"Validation Samples: {len(val_dataset)}"
)


# ==========================
# MODEL
# ==========================

model = RetinaClassifier()

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()

print("Model Loaded Successfully")


# ==========================
# EVALUATION
# ==========================

all_labels = []
all_preds = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        preds = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_preds.extend(
            preds.cpu().numpy()
        )


all_labels = np.array(
    all_labels
)

all_preds = np.array(
    all_preds
)


accuracy = accuracy_score(
    all_labels,
    all_preds
)

precision = precision_score(
    all_labels,
    all_preds,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_preds,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_preds,
    average="weighted",
    zero_division=0
)


print("\n========== RESULTS ==========")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        all_labels,
        all_preds
    )
)

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_preds,
        zero_division=0
    )
)