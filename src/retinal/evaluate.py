import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize

from dataset import RetinaDataset
from transforms import get_valid_transforms
from model import RetinaClassifier

from config import (
    DATA_DIR,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    NUM_CLASSES,
    RANDOM_SEED,
    METRICS_DIR,
    PLOTS_DIR
)


torch.manual_seed(RANDOM_SEED)

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

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# NOTE: uses the same RANDOM_SEED as this file's generator. If train.py's
# split was not seeded the same way, this val set may overlap with what
# the model actually trained on. Worth double-checking / re-training with
# a fixed seed + saved split indices for a fully rigorous separation.
generator = torch.Generator().manual_seed(RANDOM_SEED)
_, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Validation Samples: {len(val_dataset)}")


# ==========================
# MODEL
# ==========================

model = RetinaClassifier()
model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=device)
)
model.to(device)
model.eval()

print("Model Loaded Successfully")


# ==========================
# EVALUATION
# ==========================

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)


accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
report_text = classification_report(all_labels, all_preds, zero_division=0)

# Multi-class ROC-AUC (one-vs-rest, macro average -- standard for a 5-class
# ordinal diagnosis task like this)
labels_binarized = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
try:
    roc_auc_macro = roc_auc_score(
        labels_binarized, all_probs, average="macro", multi_class="ovr"
    )
except ValueError as e:
    roc_auc_macro = None
    print(f"Could not compute macro ROC-AUC: {e}")


print("\n========== RESULTS ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
if roc_auc_macro is not None:
    print(f"ROC-AUC (macro, OvR): {roc_auc_macro:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report_text)


# ==========================
# SAVE METRICS (JSON)
# ==========================

metrics_out = {
    "accuracy": accuracy,
    "precision_weighted": precision,
    "recall_weighted": recall,
    "f1_weighted": f1,
    "roc_auc_macro_ovr": roc_auc_macro,
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "val_samples": len(val_dataset),
}

metrics_path = METRICS_DIR / "retina_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics_out, f, indent=2)
print(f"\nSaved metrics -> {metrics_path}")


# ==========================
# SAVE CONFUSION MATRIX PLOT
# ==========================

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Retina Classifier - Confusion Matrix")
plt.tight_layout()
cm_path = PLOTS_DIR / "retina_confusion_matrix.png"
plt.savefig(cm_path, dpi=200)
plt.close()
print(f"Saved confusion matrix plot -> {cm_path}")


# ==========================
# SAVE ROC CURVES (per class + macro)
# ==========================

plt.figure(figsize=(7, 6))
for class_idx in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(
        labels_binarized[:, class_idx], all_probs[:, class_idx]
    )
    class_auc = roc_auc_score(
        labels_binarized[:, class_idx], all_probs[:, class_idx]
    )
    plt.plot(fpr, tpr, label=f"Class {class_idx} (AUC={class_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
title_auc = f"{roc_auc_macro:.3f}" if roc_auc_macro is not None else "N/A"
plt.title(f"Retina Classifier - ROC Curves (Macro AUC={title_auc})")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
roc_path = PLOTS_DIR / "retina_roc_curve.png"
plt.savefig(roc_path, dpi=200)
plt.close()
print(f"Saved ROC curve plot -> {roc_path}")

print("\nDone. Upload the two PNGs from results/plots/ and paste the printed")
print("metrics/JSON back to continue.")
