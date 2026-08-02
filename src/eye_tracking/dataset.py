import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class ADEyeTrackingDataset(Dataset):
    """
    Deployable dataset reader with built-in data type enforcement and padding safety.
    """
    def __init__(self, data_dir: str, max_seq_len: int = 200):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.metadata_path = os.path.join(data_dir, "metadata.csv")
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file missing in directory: {data_dir}")
            
        self.metadata = pd.read_csv(self.metadata_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        subject_id = row["subject_id"]
        label = int(row["clinical_group"])
        
        seq_file = os.path.join(self.data_dir, "sequences", f"{subject_id}.csv")
        df = pd.read_csv(seq_file)
        
        features = df[["saccadic_latency", "antisaccade_error", "fixation_duration", "saccade_amplitude"]].values.astype(np.float32)
        
        # Normalize features per sequence
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        seq_len = len(features)
        if seq_len < self.max_seq_len:
            # Zero-pad sequences if shorter than standard window
            padding = np.zeros((self.max_seq_len - seq_len, features.shape[1]), dtype=np.float32)
            features = np.vstack([features, padding])
        elif seq_len > self.max_seq_len:
            # Truncate to standard window length
            features = features[:self.max_seq_len, :]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)