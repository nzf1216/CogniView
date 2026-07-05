"""
dataset.py

PyTorch Dataset implementation for CogniView multimodal fusion.

Current Inputs
--------------
✓ Eye Tracking Features

Future Inputs
-------------
✓ Retina Embeddings
✓ PLR Features

Outputs
-------
Temporal feature sequences + labels

Author
------
CogniView Research Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

logger = logging.getLogger("cogniview.fusion.dataset")


# ---------------------------------------------------------
# Feature Order
# ---------------------------------------------------------

FEATURE_COLUMNS = [

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


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class FusionDataset(Dataset):

    """
    Temporal sequence dataset.

    Every sample returned is

    sequence_length × num_features

    together with one label.
    """

    def __init__(

        self,

        csv_file: str,

        sequence_length: int = 30,

        feature_columns: Optional[List[str]] = None,

        label_column: str = "label",

        transform=None,

    ):

        self.csv_file = Path(csv_file)

        if not self.csv_file.exists():

            raise FileNotFoundError(

                f"{self.csv_file} not found."

            )

        self.sequence_length = sequence_length

        self.transform = transform

        self.feature_columns = (

            feature_columns

            if feature_columns

            else FEATURE_COLUMNS

        )

        self.label_column = label_column

        logger.info(

            "Loading dataset: %s",

            self.csv_file,

        )

        self.data = pd.read_csv(

            self.csv_file

        )

        self._validate()

        self.features = self.data[

            self.feature_columns

        ].astype(

            np.float32

        ).values

        self.labels = self.data[

            self.label_column

        ].astype(

            np.int64

        ).values

        self.indices = self._build_indices()

        logger.info(

            "Dataset loaded."

        )

        logger.info(

            "Rows : %d",

            len(self.data),

        )

        logger.info(

            "Sequences : %d",

            len(self.indices),

        )


    # -----------------------------------------------------

    # Validation

    # -----------------------------------------------------

    def _validate(self):

        missing = [

            c

            for c in self.feature_columns

            if c not in self.data.columns

        ]

        if missing:

            raise ValueError(

                f"Missing feature columns: {missing}"

            )

        if self.label_column not in self.data.columns:

            raise ValueError(

                f"Missing label column '{self.label_column}'"

            )
    # -----------------------------------------------------
    # Sequence Generation
    # -----------------------------------------------------

    def _build_indices(self):

        indices = []

        total = len(self.features)

        if total < self.sequence_length:
            raise ValueError(
                "Dataset contains fewer rows than sequence_length."
            )

        for start in range(
            total - self.sequence_length + 1
        ):

            end = start + self.sequence_length

            indices.append(
                (start, end)
            )

        return indices

    # -----------------------------------------------------

    def __len__(self):

        return len(
            self.indices
        )

    # -----------------------------------------------------

    def __getitem__(self, index):

        start, end = self.indices[index]

        x = self.features[
            start:end
        ]

        #
        # Use the last frame's label
        #

        y = self.labels[
            end - 1
        ]

        if self.transform is not None:

            x = self.transform(x)

        x = torch.tensor(
            x,
            dtype=torch.float32,
        )

        y = torch.tensor(
            y,
            dtype=torch.long,
        )

        return x, y

    # -----------------------------------------------------

    @property
    def num_features(self):

        return len(
            self.feature_columns
        )

    @property
    def num_classes(self):

        return len(
            np.unique(
                self.labels
            )
        )

    @property
    def sequence_count(self):

        return len(
            self.indices
        )

    # -----------------------------------------------------

    def summary(self):

        logger.info("")

        logger.info(
            "Fusion Dataset Summary"
        )

        logger.info(
            "----------------------"
        )

        logger.info(
            "Rows            : %d",
            len(self.data),
        )

        logger.info(
            "Sequences       : %d",
            self.sequence_count,
        )

        logger.info(
            "Features        : %d",
            self.num_features,
        )

        logger.info(
            "Classes         : %d",
            self.num_classes,
        )

        logger.info(
            "Sequence Length : %d",
            self.sequence_length,
        )

        logger.info("")
# -----------------------------------------------------
# DataLoader Helper
# -----------------------------------------------------

from torch.utils.data import DataLoader


def create_dataloader(
    csv_file: str,
    sequence_length: int = 30,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
):

    dataset = FusionDataset(
        csv_file=csv_file,
        sequence_length=sequence_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return loader


# -----------------------------------------------------
# Smoke Test
# -----------------------------------------------------

def _demo():

    import tempfile

    rows = 120

    dummy = pd.DataFrame({

        "ear": np.random.rand(rows),

        "blink_count": np.random.randint(0, 10, rows),

        "blink_rate": np.random.rand(rows) * 20,

        "fixation_stability": np.random.rand(rows),

        "fixation_count": np.random.randint(0, 30, rows),

        "saccade_count": np.random.randint(0, 20, rows),

        "smooth_pursuit_count": np.random.randint(0, 10, rows),

        "left_iris_x": np.random.rand(rows),

        "left_iris_y": np.random.rand(rows),

        "right_iris_x": np.random.rand(rows),

        "right_iris_y": np.random.rand(rows),

        "label": np.random.randint(0, 3, rows),

    })

    with tempfile.NamedTemporaryFile(
        suffix=".csv",
        delete=False,
    ) as f:

        dummy.to_csv(
            f.name,
            index=False,
        )

        dataset = FusionDataset(
            f.name,
            sequence_length=30,
        )

        dataset.summary()

        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
        )

        x, y = next(iter(loader))

        print()

        print("Input Shape :", x.shape)

        print("Label Shape :", y.shape)

        print("Features :", dataset.num_features)

        print("Classes :", dataset.num_classes)

        print("Sequences :", dataset.sequence_count)

        print()


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
    )

    _demo()