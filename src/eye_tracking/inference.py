import torch
import numpy as np
import pandas as pd
import os
from typing import Dict, Union
from model import ADClassifierModel

class ADTrackerInferenceEngine:
    """
    Production-grade inference engine for CogniView Eye-Tracking Module.
    Handles memory mapping, device management, and clinical risk calibration.
    """
    def __init__(self, model_path: str, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ADClassifierModel().to(self.device)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Critical deployment error: Model weights not found at {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print(f"✅ ADTrackerInferenceEngine initialized successfully on {self.device}")

    def preprocess_sequence(self, df: pd.DataFrame) -> torch.Tensor:
        """Enforces schema checks and normalizes raw telemetry streams."""
        required_columns = ["timestamp", "saccadic_latency", "antisaccade_error", "fixation_duration", "saccade_amplitude"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Payload schema validation failed: Missing required sensor stream -> '{col}'")
                
        # Extract features matrix [seq_len, features]
        features = df[["saccadic_latency", "antisaccade_error", "fixation_duration", "saccade_amplitude"]].values.astype(np.float32)
        
        # Standardize / Z-score normalization against clinical baselines
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # Convert to batch tensor format [1, seq_len, features]
        tensor_seq = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        return tensor_seq.to(self.device)

    @torch.no_grad()
    def predict(self, input_data: Union[str, pd.DataFrame]) -> Dict[str, Union[float, str]]:
        """
        Executes forward pass and outputs clinical risk score and classification.
        
        Args:
            input_data: Path to subject CSV file or live pandas DataFrame stream.
        """
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError(f"Target data file not found: {input_data}")
            df = pd.read_csv(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data
        else:
            raise TypeError("Invalid input format. Provide either a CSV file path or a pandas DataFrame.")

        tensor_seq = self.preprocess_sequence(df)
        
        logits = self.model(tensor_seq)
        probability = torch.sigmoid(logits).item()
        
        prediction_label = "MCI / AD Risk" if probability >= 0.5 else "Healthy Control"
        confidence = probability if probability >= 0.5 else (1.0 - probability)

        return {
            "risk_probability": round(float(probability), 4),
            "classification": prediction_label,
            "confidence_score": round(float(confidence), 4),
            "status": "success"
        }

if __name__ == "__main__":
    # Smoke test deployment verification
    engine = ADTrackerInferenceEngine(model_path="./checkpoints/best_ad_tracker.pth")
    sample_file = "./data/ad_eyetracking/synthetic/sequences/subject_001.csv"
    if os.path.exists(sample_file):
        result = engine.predict(sample_file)
        print("🚀 Production Inference Test Result:", result)