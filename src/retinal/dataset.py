from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class RetinaDataset(Dataset):

    def __init__(
        self,
        csv_file,
        image_dir,
        transform=None
    ):

        self.data = pd.read_csv(csv_file)

        self.image_dir = Path(image_dir)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image_id = row["id_code"]

        label = int(row["diagnosis"])

        image_path = (
            self.image_dir /
            f"{image_id}.png"
        )

        # Load image and convert to NumPy array
        image = np.array(
            Image.open(image_path).convert("RGB")
        )

        if self.transform:
            image = self.transform(
                image=image
            )["image"]

        return image, label