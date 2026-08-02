import os
import numpy as np
import pandas as pd
import argparse

def generate_realistic_ad_data(output_dir, num_subjects=100, seq_length=200):
    os.makedirs(os.path.join(output_dir, "sequences"), exist_ok=True)
    metadata = []

    np.random.seed(42)
    for subj_id in range(1, num_subjects + 1):
        # Balanced ratio (50% Healthy Control, 50% MCI/AD Risk)
        label = 0 if subj_id <= num_subjects // 2 else 1
        
        # Enhanced clinical separation between Healthy Controls and MCI/AD Risk
        lat_mean = np.random.normal(210 if label == 0 else 380, 20)  # Distinct Saccadic latency (ms)
        err_rate = np.random.beta(1, 30) if label == 0 else np.random.beta(8, 5)   # Distinct Antisaccade error probability
        fix_dur = np.random.normal(200 if label == 0 else 320, 25)   # Distinct Fixation duration (ms)
        
        timestamps = np.arange(0, seq_length * 10, 10) # 100Hz eye-tracking stream
        
        # Introduce realistic temporal correlation (Autoregressive/Pink noise simulation)
        noise_lat = np.cumsum(np.random.normal(0, 4, seq_length)) * 0.1
        noise_fix = np.cumsum(np.random.normal(0, 5, seq_length)) * 0.1
        
        latency_series = (np.random.normal(lat_mean, 25, seq_length) + noise_lat).clip(min=100)
        
        # Clipped error rate for Bernoulli trials
        clipped_err_rate = np.clip(err_rate, 0.0, 1.0)
        error_flag_series = np.random.binomial(1, clipped_err_rate, seq_length)
        
        fixation_duration_series = (np.random.normal(fix_dur, 30, seq_length) + noise_fix).clip(min=70)
        saccade_amplitude = np.random.normal(4.5, 1.0, seq_length).clip(min=0.2) # degrees
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "saccadic_latency": latency_series,
            "antisaccade_error": error_flag_series,
            "fixation_duration": fixation_duration_series,
            "saccade_amplitude": saccade_amplitude
        })
        
        filename = f"subject_{subj_id:03d}.csv"
        df.to_csv(os.path.join(output_dir, "sequences", filename), index=False)
        metadata.append({"subject_id": f"subject_{subj_id:03d}", "clinical_group": label})
        
    pd.DataFrame(metadata).to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    print(f"✅ Generated {num_subjects} high-contrast clinical tracking profiles in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/ad_eyetracking/synthetic")
    parser.add_argument("--num_subjects", type=int, default=250)
    args = parser.parse_args()
    generate_realistic_ad_data(args.output_dir, num_subjects=args.num_subjects)