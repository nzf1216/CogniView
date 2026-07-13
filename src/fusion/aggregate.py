"""
aggregate.py

Builds data/fusion/train.csv and data/fusion/val.csv from the raw
per-trial recordings in data/fusion/sessions/trial*.csv.

Why this exists
----------------
FusionDataset (see dataset.py) builds sliding windows over whatever CSV
it's given, in raw row order. Recordings can't just be concatenated
naively: a window built from the tail of one trial + the head of the
next would mix two different recording sessions (and, if their labels
differ, two different classes) into one training example.

This script:
  1. Reads every trial CSV under data/fusion/sessions/.
  2. Tags every row with a session_id (the trial filename stem), so
     FusionDataset's session-boundary guard can refuse to build any
     window that spans two sessions.
  3. Splits by WHOLE SESSION (never by row) into train/val, stratified
     by session-level label so val isn't accidentally all-resting or
     all-reading.
  4. Writes data/fusion/train.csv and data/fusion/val.csv.

Usage
-----
    python -m src.fusion.aggregate
    python -m src.fusion.aggregate --val-sessions-per-class 1 --seed 42
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("cogniview.fusion.aggregate")

SESSIONS_DIR = Path("data/fusion/sessions")
OUT_DIR = Path("data/fusion")
TRAIN_OUT = OUT_DIR / "train.csv"
VAL_OUT = OUT_DIR / "val.csv"

LABEL_COLUMN = "label"
SESSION_COLUMN = "session_id"


def load_sessions(sessions_dir: Path) -> dict[str, pd.DataFrame]:
    """Load every trial*.csv into a dict keyed by session_id."""

    paths = sorted(glob.glob(str(sessions_dir / "trial*.csv")))

    if not paths:
        raise FileNotFoundError(
            f"No trial*.csv files found under {sessions_dir}"
        )

    sessions: dict[str, pd.DataFrame] = {}

    for p in paths:
        path = Path(p)
        session_id = path.stem  # e.g. "trial06"

        df = pd.read_csv(path)

        if LABEL_COLUMN not in df.columns:
            raise ValueError(f"{path} has no '{LABEL_COLUMN}' column")

        labels = df[LABEL_COLUMN].unique()
        if len(labels) != 1:
            raise ValueError(
                f"{path} contains mixed labels {labels} — a single "
                "recording session should have one label throughout."
            )

        df[SESSION_COLUMN] = session_id
        sessions[session_id] = df

    return sessions


def split_sessions(
    sessions: dict[str, pd.DataFrame],
    val_sessions_per_class: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Pick whole sessions for val, stratified by session-level label."""

    rng = np.random.RandomState(seed)

    by_label: dict[int, list[str]] = {}
    for session_id, df in sessions.items():
        label = int(df[LABEL_COLUMN].iloc[0])
        by_label.setdefault(label, []).append(session_id)

    val_ids: list[str] = []

    for label, ids in sorted(by_label.items()):
        ids = sorted(ids)
        rng.shuffle(ids)

        n_val = val_sessions_per_class
        if n_val >= len(ids):
            raise ValueError(
                f"Only {len(ids)} session(s) recorded for label {label}; "
                f"can't hold out {n_val} for val and still have any "
                "left to train on. Record more trials for this class "
                "or lower --val-sessions-per-class."
            )

        val_ids.extend(ids[:n_val])

    train_ids = sorted(set(sessions.keys()) - set(val_ids))

    return train_ids, sorted(val_ids)


def build_and_write(
    sessions_dir: Path = SESSIONS_DIR,
    val_sessions_per_class: int = 1,
    seed: int = 42,
) -> None:

    sessions = load_sessions(sessions_dir)

    logger.info("Loaded %d session(s): %s", len(sessions), sorted(sessions))

    train_ids, val_ids = split_sessions(
        sessions, val_sessions_per_class, seed
    )

    logger.info("Train sessions (%d): %s", len(train_ids), train_ids)
    logger.info("Val sessions   (%d): %s", len(val_ids), val_ids)

    train_df = pd.concat(
        [sessions[s] for s in train_ids], ignore_index=True
    )
    val_df = pd.concat(
        [sessions[s] for s in val_ids], ignore_index=True
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)

    logger.info(
        "Wrote %s (%d rows, %d sessions) and %s (%d rows, %d sessions)",
        TRAIN_OUT, len(train_df), len(train_ids),
        VAL_OUT, len(val_df), len(val_ids),
    )

    for name, ids in (("train", train_ids), ("val", val_ids)):
        label_counts = {
            label: sum(
                1 for s in ids if int(sessions[s][LABEL_COLUMN].iloc[0]) == label
            )
            for label in sorted({
                int(df[LABEL_COLUMN].iloc[0]) for df in sessions.values()
            })
        }
        logger.info("%s sessions by label: %s", name, label_counts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate per-trial session CSVs into train/val CSVs, "
        "splitting by whole session (never by row)."
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=SESSIONS_DIR,
        help="Directory containing trial*.csv files.",
    )
    parser.add_argument(
        "--val-sessions-per-class",
        type=int,
        default=1,
        help="How many whole sessions per label to hold out for val.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for picking which sessions go to val.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()

    build_and_write(
        sessions_dir=args.sessions_dir,
        val_sessions_per_class=args.val_sessions_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
