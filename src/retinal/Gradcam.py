"""
Grad-CAM heatmap generation for the retina EfficientNet-B0 classifier.

Usage:
    python gradcam.py

Picks a handful of validation images (one per available class where
possible), runs Grad-CAM on the last conv block of EfficientNet-B0, and
saves side-by-side (original | heatmap overlay) figures to
results/gradcam/.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from model import RetinaClassifier
from transforms import get_valid_transforms

from config import (
    DATA_DIR,
    CHECKPOINT_PATH,
    GRADCAM_DIR,
    IMAGE_SIZE,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RetinaClassifier()
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()

# Last conv block of EfficientNet-B0 backbone -- good default target layer
# for Grad-CAM on this architecture.
target_layer = model.backbone.features[-1]

activations = {}
gradients = {}


def forward_hook(module, input, output):
    activations["value"] = output.detach()


def backward_hook(module, grad_input, grad_output):
    gradients["value"] = grad_output[0].detach()


target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


def generate_gradcam(input_tensor, class_idx):
    input_tensor = input_tensor.unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    output = model(input_tensor)
    model.zero_grad()

    score = output[0, class_idx]
    score.backward()

    acts = activations["value"][0]       # (C, H, W)
    grads = gradients["value"][0]        # (C, H, W)

    weights = grads.mean(dim=(1, 2))     # (C,)
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=device)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = F.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cam.cpu().numpy()
    cam = np.uint8(255 * cam)
    cam = Image.fromarray(cam).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    predicted_class = output.argmax(dim=1).item()
    return np.array(cam), predicted_class


def unnormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    return np.clip(img, 0, 1)


def main():
    df = pd.read_csv(DATA_DIR / "retina" / "train.csv")
    image_dir = DATA_DIR / "retina" / "train_images"

    # Grab up to one example per class, 5 total (matches NUM_CLASSES).
    samples = (
        df.groupby("diagnosis")
        .head(1)
        .reset_index(drop=True)
    )

    transform = get_valid_transforms()

    for _, row in samples.iterrows():
        image_id = row["id_code"]
        true_label = int(row["diagnosis"])

        image_path = image_dir / f"{image_id}.png"
        image = np.array(Image.open(image_path).convert("RGB"))

        transformed = transform(image=image)["image"]

        cam, pred_class = generate_gradcam(transformed, class_idx=true_label)
        display_img = unnormalize(transformed)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(display_img)
        axes[0].set_title(f"Original - {image_id}\nTrue: {true_label}")
        axes[0].axis("off")

        axes[1].imshow(display_img)
        axes[1].imshow(cam, cmap="jet", alpha=0.45)
        axes[1].set_title(f"Grad-CAM\nPredicted: {pred_class}")
        axes[1].axis("off")

        plt.tight_layout()
        out_path = GRADCAM_DIR / f"gradcam_{image_id}_true{true_label}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved -> {out_path}")

    print("\nDone. Upload the PNGs from results/gradcam/ back to continue.")


if __name__ == "__main__":
    main()