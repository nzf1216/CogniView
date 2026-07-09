"""
plr_aggregate.py
Aggregates multiple PLR trial recordings (saved by plr_test.py in
results/plr_raw/) into a single averaged, noise-reduced PLR curve.

Why this exists:
    A single webcam-based trial has a large noise floor relative to the true
    pupillary response (no infrared illumination, limited resolution, frame-
    to-frame contour jitter). Averaging multiple trials aligned to the same
    flash timing cancels out random per-trial noise and reveals the
    underlying response - this is standard practice in pupillometry
    research, not specific to this project.

Usage:
    Run plr_test.py several times first (5+ recommended), then:
        python plr_aggregate.py
"""

import glob
import json
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plr_test import moving_average, extract_features  # reuse existing logic

from config import (
    BASELINE_DURATION_S,
    FLASH_DURATION_S,
    TOTAL_DURATION_S,
    SMOOTHING_WINDOW,
    RAW_DIR,
    PLOTS_DIR,
    METRICS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cogniview.plr.aggregate")

COMMON_TIME_STEP_S = 0.05


def load_trial(csv_path: str) -> tuple:
    df = pd.read_csv(csv_path)
    if "left_pupil_iris_ratio" not in df.columns:
        return None, None
    times = df["timestamp_s"].to_numpy(dtype=float)
    left = pd.to_numeric(df["left_pupil_iris_ratio"], errors="coerce").to_numpy()
    right = pd.to_numeric(df["right_pupil_iris_ratio"], errors="coerce").to_numpy()
    diam = np.nanmean(np.vstack([left, right]), axis=0) * 100.0
    return times, diam


def resample_to_grid(times: np.ndarray, diam: np.ndarray, grid: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(diam)
    if valid.sum() < 2:
        return np.full_like(grid, np.nan)
    # Only interpolate within the observed time range; outside it, leave NaN
    # rather than extrapolating a guess.
    result = np.interp(grid, times[valid], diam[valid], left=np.nan, right=np.nan)
    return result


def main():
    csv_paths = sorted(glob.glob(str(RAW_DIR / "plr_session_*.csv")))
    if len(csv_paths) == 0:
        logger.error("No trial CSVs found in %s. Run plr_test.py first.", RAW_DIR)
        return

    logger.info("Found %d trial(s): %s", len(csv_paths), [p.split('/')[-1] for p in csv_paths])

    grid = np.arange(0, TOTAL_DURATION_S, COMMON_TIME_STEP_S)
    all_trials = []
    used_paths = []

    for path in csv_paths:
        times, diam = load_trial(path)
        if times is None:
            logger.warning(
                "%s: old CSV format (pre pupil:iris ratio) - skipping.",
                path.split("/")[-1]
            )
            continue
        resampled = resample_to_grid(times, diam, grid)
        valid_ratio = np.mean(~np.isnan(resampled))
        logger.info("%s: %.0f%% valid coverage on common grid", path.split("/")[-1], valid_ratio * 100)
        all_trials.append(resampled)
        used_paths.append(path)

    if len(all_trials) == 0:
        logger.error("No trials in the new ratio-based format found. Run plr_test.py again first.")
        return

    csv_paths = used_paths
    trial_matrix = np.vstack(all_trials)  # shape: (n_trials, n_timepoints)

    n_valid_per_point = np.sum(~np.isnan(trial_matrix), axis=0)
    mean_curve = np.nanmean(trial_matrix, axis=0)
    std_curve = np.nanstd(trial_matrix, axis=0)

    # Only trust grid points where at least half the trials contributed data
    min_trials_needed = max(2, len(csv_paths) // 2)
    reliable = n_valid_per_point >= min_trials_needed
    mean_curve_reliable = np.where(reliable, mean_curve, np.nan)

    smoothed = moving_average(mean_curve_reliable, SMOOTHING_WINDOW)

    features = extract_features(grid, smoothed)

    print(f"\n========== AVERAGED PLR FEATURES (n={len(csv_paths)} trials) ==========")
    for k, v in features.items():
        print(f"{k}: {v}")

    metrics_path = METRICS_DIR / "plr_features_averaged.json"
    with open(metrics_path, "w") as f:
        json.dump({"n_trials": len(csv_paths), "features": features}, f, indent=2)
    logger.info("Saved averaged features -> %s", metrics_path)

    # Plot: individual trials (thin, transparent) + mean +/- std band + final smoothed mean
    plt.figure(figsize=(10, 6))
    for i, trial in enumerate(all_trials):
        plt.plot(grid, trial, color="gray", alpha=0.25, linewidth=1,
                  label="Individual trials" if i == 0 else None)

    plt.fill_between(
        grid, mean_curve_reliable - std_curve, mean_curve_reliable + std_curve,
        color="tab:blue", alpha=0.2, label="±1 SD across trials"
    )
    plt.plot(grid, smoothed, color="tab:blue", linewidth=2.5, label=f"Mean (n={len(csv_paths)}, smoothed)")

    plt.axvline(BASELINE_DURATION_S, color="orange", linestyle="--", label="Flash onset")
    plt.axvline(BASELINE_DURATION_S + FLASH_DURATION_S, color="gray", linestyle=":", label="Flash end")

    if "min_diameter_px" in features and "time_to_min_constriction_s" in features:
        t_min = BASELINE_DURATION_S + features["time_to_min_constriction_s"]
        plt.scatter([t_min], [features["min_diameter_px"]], color="red", zorder=5, label="Peak constriction")

    plt.xlabel("Time (s)")
    plt.ylabel("Pupil:Iris diameter ratio (%)")
    plt.title(f"Averaged Pupillary Light Reflex Curve (n={len(csv_paths)} trials, distance-invariant ratio)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plot_path = PLOTS_DIR / "plr_curve_averaged.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logger.info("Saved averaged PLR curve plot -> %s", plot_path)

    print("\nDone. Upload results/plots/plr_curve_averaged.png and paste the")
    print("printed averaged features back to continue.")


if __name__ == "__main__":
    main()
