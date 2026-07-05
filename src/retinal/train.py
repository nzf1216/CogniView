import time

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    random_split
)

from dataset import RetinaDataset
from transforms import get_train_transforms
from model import RetinaClassifier

from config import (
    DATA_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
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
    transform=get_train_transforms()
)

print(
    f"Original Dataset Size: {len(full_dataset)}"
)


# ==========================
# TRAIN / VALIDATION SPLIT
# ==========================

train_size = int(
    0.8 * len(full_dataset)
)

val_size = (
    len(full_dataset)
    - train_size
)

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size]
)

print(
    f"Train Samples: {len(train_dataset)}"
)

print(
    f"Validation Samples: {len(val_dataset)}"
)


# ==========================
# DATALOADERS
# ==========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    f"Train Batches: {len(train_loader)}"
)

print(
    f"Validation Batches: {len(val_loader)}"
)


# ==========================
# MODEL
# ==========================

model = RetinaClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


best_loss = float("inf")


# ==========================
# TRAINING LOOP
# ==========================

for epoch in range(NUM_EPOCHS):

    print(
        f"\nStarting Epoch {epoch+1}"
    )

    epoch_start = time.time()

    model.train()

    train_loss = 0

    for batch_idx, (
        images,
        labels
    ) in enumerate(train_loader):

        if batch_idx % 20 == 0:

            print(
                f"Batch {batch_idx+1}/{len(train_loader)}"
            )

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = (
        train_loss /
        len(train_loader)
    )

    print(
        f"\nEpoch {epoch+1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss: {avg_train_loss:.4f}"
    )

    print(
        f"Epoch Time: "
        f"{time.time()-epoch_start:.2f} sec"
    )

    if avg_train_loss < best_loss:

        best_loss = avg_train_loss

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )

        print(
            "Best model saved."
        )


print(
    "\nTraining Complete."
)