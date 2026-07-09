"""
plr_test.py
Pupillary Light Reflex (PLR) capture and feature-extraction pipeline.

Method (low-cost, no dedicated hardware required):
    - The laptop/monitor screen itself is used as the light stimulus.
    - A dim baseline period is shown, then the screen flashes bright white,
      then returns dim while the pupil re-dilates.
    - The webcam (positioned above the screen, as normal) records the eye
      throughout. Per-frame iris landmarks (MediaPipe, via the existing
      eye_tracking.landmarks module) give a pixel-based pupil diameter
      proxy for every frame.
    - After the recording, the diameter-vs-time curve is smoothed and used
      to extract standard PLR features reported in the pupillometry
      literature: baseline diameter, minimum (peak constriction) diameter,
      constriction amplitude, constriction latency, and recovery time.

IMPORTANT — framing for the paper:
    This produces a per-session feature extraction, not a trained AD/healthy
    classifier (no labeled patient dataset exists publicly for that). Report
    this as a feasibility/proof-of-concept pipeline, comparing extracted
    values against published clinical reference ranges/directions, not as
    diagnostic accuracy.

Usage:
    python plr_test.py
    (keep your face still, look at the screen, don't cover the webcam)

Author: CogniView PLR Module
"""

import os
import sys
import csv
import json
import time
import logging
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reuse the existing, already-working eye_tracking landmark extractor
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eye_tracking")
)
from landmarks import LandmarkExtractor, open_webcam, EyeLandmarks  # noqa: E402

from config import (
    SETTLE_DURATION_S,
    BASELINE_DURATION_S,
    FLASH_DURATION_S,
    RECOVERY_DURATION_S,
    TOTAL_DURATION_S,
    CONSTRICTION_SEARCH_WINDOW_S,
    DIM_COLOR_BGR,
    FLASH_COLOR_BGR,
    STIMULUS_WINDOW_SIZE,
    SMOOTHING_WINDOW,
    MIN_VALID_FRAME_RATIO,
    RAW_DIR,
    PLOTS_DIR,
    METRICS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cogniview.plr")


@dataclass
class FrameRecord:
    timestamp_s: float
    stimulus_state: str  # "baseline" | "flash" | "recovery"
    left_pupil_iris_ratio: Optional[float]
    right_pupil_iris_ratio: Optional[float]
    left_iris_diameter_px: Optional[float]
    right_iris_diameter_px: Optional[float]


def iris_diameter_px(iris_points: np.ndarray, frame_w: int, frame_h: int) -> Optional[float]:
    """
    Estimate the OUTER IRIS diameter in pixels from the 5 MediaPipe iris
    points. NOTE: the iris (colored ring) does not change size with light -
    only the pupil (dark center) does. This function is kept only as a
    reference/sanity value, NOT used for the actual PLR measurement below.
    """
    if iris_points is None or len(iris_points) < 5:
        return None

    pts_px = np.array([[p[0] * frame_w, p[1] * frame_h] for p in iris_points])
    center, right, top, left, bottom = pts_px
    horiz = np.linalg.norm(right - left)
    vert = np.linalg.norm(top - bottom)
    return float((horiz + vert) / 2.0)


def estimate_pupil_and_iris_diameter_px(
    frame_bgr: np.ndarray,
    iris_points: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> tuple:
    """
    Returns (pupil_diameter_px, iris_diameter_px) for one eye, or (None, None)
    if detection fails. The iris diameter is returned alongside the pupil
    diameter so callers can compute a pupil:iris RATIO - a head-distance-
    invariant measure. Raw pixel diameters alone are contaminated by any
    change in distance to the camera (e.g. an unconscious lean toward/away
    from the screen when startled by the flash), which the ratio cancels out
    since both pupil and iris scale together with distance.
    """
    if iris_points is None or len(iris_points) < 5:
        return None, None

    pts_px = np.array([[p[0] * frame_w, p[1] * frame_h] for p in iris_points])
    center, right, top, left, bottom = pts_px
    outer_iris_diameter = float((np.linalg.norm(right - left) + np.linalg.norm(top - bottom)) / 2.0)
    iris_radius_px = outer_iris_diameter / 2.0
    if iris_radius_px < 3:
        return None, None

    cx, cy = center
    crop_r = int(iris_radius_px * 1.4)
    x0, x1 = int(max(cx - crop_r, 0)), int(min(cx + crop_r, frame_w))
    y0, y1 = int(max(cy - crop_r, 0)), int(min(cy + crop_r, frame_h))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None, outer_iris_diameter

    crop = frame_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Upsample before detection: at typical webcam resolution/distance, the
    # pupil may only span ~15-20px, so a 2-3px detection wobble reads as a
    # 15-20% "change" - the exact noise magnitude we've been seeing. A 3x
    # upsample gives the detector sub-pixel-equivalent precision to work
    # with. All measurements below are taken in upsampled coordinates, then
    # divided back down by UPSAMPLE at the end.
    UPSAMPLE = 3
    gray = cv2.resize(gray, None, fx=UPSAMPLE, fy=UPSAMPLE, interpolation=cv2.INTER_CUBIC)
    iris_radius_px_up = iris_radius_px * UPSAMPLE
    crop_center = np.array([(x1 - x0) / 2.0, (y1 - y0) / 2.0]) * UPSAMPLE

    # CLAHE normalizes local contrast so the pupil/iris boundary is more
    # consistent across different ambient lighting / flash conditions,
    # instead of relying on a raw global brightness threshold.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.medianBlur(gray, 5)

    # Physically constrain the search: the pupil is never larger than the
    # iris, and rarely smaller than ~15% of it. Bounds computed in the
    # upsampled coordinate system to match the resized crop.
    min_r = max(2, int(iris_radius_px_up * 0.15))
    max_r = int(iris_radius_px_up * 0.95)
    if max_r <= min_r:
        return None, outer_iris_diameter

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=iris_radius_px_up,
        param1=60,
        param2=15,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is not None:
        circles = circles[0]
        best = min(circles, key=lambda c: np.linalg.norm(np.array([c[0], c[1]]) - crop_center))
        _, _, radius = best
        return float(radius * 2.0) / UPSAMPLE, outer_iris_diameter

    # Fallback: Hough found nothing (common on lower webcam image quality).
    # Use darkest-blob contour detection instead, but keep it physically
    # bounded to [min_r, max_r] so it can't return an implausible size like
    # the unbounded version did before.
    thresh_val = np.percentile(gray, 20)
    _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, outer_iris_diameter

    best_c = None
    best_score = -1.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 4:
            continue
        (cx_c, cy_c), radius = cv2.minEnclosingCircle(c)
        if radius < min_r or radius > max_r:
            continue
        dist = np.linalg.norm(np.array([cx_c, cy_c]) - crop_center)
        circularity = area / (np.pi * radius * radius + 1e-6)
        score = circularity - 0.02 * dist
        if score > best_score:
            best_score = score
            best_c = radius

    if best_c is None:
        return None, outer_iris_diameter

    return float(best_c * 2.0) / UPSAMPLE, outer_iris_diameter


class TemporalOutlierFilter:
    """
    Rejects single-frame pupil readings that deviate too much from a short
    rolling history - a real pupil cannot physically jump by e.g. 30% in
    1/30th of a second, so a reading that far from recent history is almost
    certainly a detection error, not a real change. This directly targets
    frame-to-frame jitter without needing to know in advance what a "real"
    response should look like (the allowed deviation is generous enough to
    still let a genuine fast constriction through over a few frames).
    """
    def __init__(self, history_len: int = 5, max_relative_jump: float = 0.25):
        self.history: List[float] = []
        self.history_len = history_len
        self.max_relative_jump = max_relative_jump

    def filter(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if len(self.history) < 3:
            # Not enough history yet to judge - accept and build up history.
            self.history.append(value)
            self.history = self.history[-self.history_len:]
            return value

        reference = float(np.median(self.history))
        if reference > 0 and abs(value - reference) / reference > self.max_relative_jump:
            # Implausible jump - reject this reading, don't update history
            # with it (so a run of bad frames doesn't drag the reference
            # away from the true value).
            return None

        self.history.append(value)
        self.history = self.history[-self.history_len:]
        return value


def get_stimulus_state(elapsed_s: float) -> str:
    if elapsed_s < BASELINE_DURATION_S:
        return "baseline"
    elif elapsed_s < BASELINE_DURATION_S + FLASH_DURATION_S:
        return "flash"
    else:
        return "recovery"


def run_capture() -> List[FrameRecord]:
    cap = open_webcam()

    # Request the highest resolution the webcam supports. Most webcams
    # default to a lower resolution (often 640x480) unless asked otherwise;
    # more pixels across the eye directly reduces pupil-size measurement
    # noise, since the current bottleneck is too few pixels spanning the
    # pupil to measure it precisely.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.info("Requested 1920x1080, webcam granted %dx%d", int(actual_w), int(actual_h))

    extractor = LandmarkExtractor()
    extractor.initialize()

    stim_w, stim_h = STIMULUS_WINDOW_SIZE
    cv2.namedWindow("PLR Stimulus - Look at this window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PLR Stimulus - Look at this window", stim_w, stim_h)

    records: List[FrameRecord] = []
    left_filter = TemporalOutlierFilter()
    right_filter = TemporalOutlierFilter()

    logger.info(
        "Settle period: %.1fs (not recorded) - get into position now.",
        SETTLE_DURATION_S
    )
    settle_start = time.time()
    while time.time() - settle_start < SETTLE_DURATION_S:
        remaining = SETTLE_DURATION_S - (time.time() - settle_start)
        settle_frame = np.full((stim_h, stim_w, 3), DIM_COLOR_BGR, dtype=np.uint8)
        cv2.putText(
            settle_frame, f"GET READY  {remaining:0.1f}s",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2
        )
        cv2.imshow("PLR Stimulus - Look at this window", settle_frame)
        cap.read()  # keep camera buffer fresh, discard frames during settle
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            extractor.close()
            cv2.destroyAllWindows()
            return records

    start_time = time.time()

    logger.info(
        "Starting PLR capture (recording): %.1fs baseline, %.1fs flash, %.1fs recovery. "
        "Look steadily at the stimulus window, try not to blink near the flash.",
        BASELINE_DURATION_S, FLASH_DURATION_S, RECOVERY_DURATION_S
    )

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > TOTAL_DURATION_S:
                break

            state = get_stimulus_state(elapsed)
            color = FLASH_COLOR_BGR if state == "flash" else DIM_COLOR_BGR
            stim_frame = np.full((stim_h, stim_w, 3), color, dtype=np.uint8)
            cv2.putText(
                stim_frame, f"{state.upper()}  t={elapsed:0.2f}s",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0) if state == "flash" else (200, 200, 200), 2
            )
            cv2.imshow("PLR Stimulus - Look at this window", stim_frame)

            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame grab failed; skipping.")
                cv2.waitKey(1)
                continue

            frame = cv2.flip(frame, 1)
            timestamp_ms = int(elapsed * 1000)
            landmarks = extractor.process(frame, timestamp_ms)

            left_pupil_d, left_iris_d = (None, None)
            right_pupil_d, right_iris_d = (None, None)
            if landmarks.detected and len(landmarks.left_iris_points) == 5:
                left_pupil_d, left_iris_d = estimate_pupil_and_iris_diameter_px(
                    frame, landmarks.left_iris_points, landmarks.frame_width, landmarks.frame_height
                )
            if landmarks.detected and len(landmarks.right_iris_points) == 5:
                right_pupil_d, right_iris_d = estimate_pupil_and_iris_diameter_px(
                    frame, landmarks.right_iris_points, landmarks.frame_width, landmarks.frame_height
                )

            left_ratio = (
                left_pupil_d / left_iris_d
                if left_pupil_d is not None and left_iris_d not in (None, 0)
                else None
            )
            right_ratio = (
                right_pupil_d / right_iris_d
                if right_pupil_d is not None and right_iris_d not in (None, 0)
                else None
            )
            left_ratio = left_filter.filter(left_ratio)
            right_ratio = right_filter.filter(right_ratio)

            records.append(FrameRecord(elapsed, state, left_ratio, right_ratio, left_iris_d, right_iris_d))

            # Debug preview: shows the webcam feed with detected iris centers
            # marked, so you can visually confirm tracking is holding up
            # (especially right around the flash).
            debug_frame = frame.copy()
            status_color = (0, 200, 0) if landmarks.detected else (0, 0, 255)
            status_text = "TRACKING OK" if landmarks.detected else "LOST"
            cv2.putText(debug_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            if landmarks.detected:
                if landmarks.left_iris_center and left_pupil_d:
                    px, py = landmarks.to_pixel(landmarks.left_iris_center)
                    cv2.circle(debug_frame, (px, py), int(left_pupil_d / 2), (0, 255, 255), 1)
                    cv2.circle(debug_frame, (px, py), 2, (0, 0, 255), -1)
                if landmarks.right_iris_center and right_pupil_d:
                    px, py = landmarks.to_pixel(landmarks.right_iris_center)
                    cv2.circle(debug_frame, (px, py), int(right_pupil_d / 2), (0, 255, 255), 1)
                    cv2.circle(debug_frame, (px, py), 2, (0, 0, 255), -1)
            cv2.imshow("Debug - Webcam Feed (for troubleshooting only)", debug_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Capture aborted by user.")
                break
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()

    return records


def save_raw_csv(records: List[FrameRecord], path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_s", "stimulus_state",
            "left_pupil_iris_ratio", "right_pupil_iris_ratio",
            "left_iris_diameter_px", "right_iris_diameter_px"
        ])
        for r in records:
            writer.writerow([
                f"{r.timestamp_s:.4f}", r.stimulus_state,
                r.left_pupil_iris_ratio, r.right_pupil_iris_ratio,
                r.left_iris_diameter_px, r.right_iris_diameter_px
            ])


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window) / window
    # 'same' mode keeps array length aligned with timestamps
    return np.convolve(x, kernel, mode="same")


def extract_features(times: np.ndarray, diam: np.ndarray) -> dict:
    """
    Extract standard PLR features from a smoothed diameter-vs-time curve.
    All feature names/directions follow conventions used in the clinical
    pupillometry literature (e.g. constriction amplitude, latency, recovery
    time) referenced in the paper's related-work section.
    """
    flash_onset = BASELINE_DURATION_S

    baseline_mask = times < flash_onset
    baseline_vals = diam[baseline_mask]
    baseline_vals = baseline_vals[~np.isnan(baseline_vals)]
    baseline_diameter = float(np.mean(baseline_vals)) if len(baseline_vals) else float("nan")

    # Full post-flash signal (used later for recovery tracking)
    post_flash_mask = times >= flash_onset
    post_times_all = times[post_flash_mask]
    post_diam_all = diam[post_flash_mask]
    post_valid = ~np.isnan(post_diam_all)
    post_times = post_times_all[post_valid]
    post_diam = post_diam_all[post_valid]

    # Constrained window for finding the actual constriction peak - real PLR
    # constriction happens quickly after light onset. Searching the full
    # recovery period risks locking onto a blink or an end-of-recording
    # tracking-loss artifact instead of the genuine response.
    search_mask = (times >= flash_onset) & (times <= flash_onset + CONSTRICTION_SEARCH_WINDOW_S)
    search_times_all = times[search_mask]
    search_diam_all = diam[search_mask]
    search_valid = ~np.isnan(search_diam_all)
    search_times = search_times_all[search_valid]
    search_diam = search_diam_all[search_valid]

    if np.isnan(baseline_diameter):
        return {"error": "no valid baseline data - tracking was lost throughout the baseline period"}

    # Require at least half the search window to have real (non-gap) data,
    # otherwise the "peak" would be based on too little real signal to trust.
    expected_search_frames = max(1, len(times[search_mask]))
    if len(search_diam) < 0.5 * expected_search_frames or len(search_diam) == 0:
        return {
            "error": (
                "eye tracking was lost for most of the window right after the "
                "flash - likely glare/auto-exposure reaction to the bright "
                "flash. Retry: dim the room slightly less, or reduce flash "
                "brightness, or move closer to the webcam."
            ),
            "search_window_detection_rate_pct": round(100 * len(search_diam) / expected_search_frames, 1)
        }

    min_idx_local = int(np.argmin(search_diam))
    min_diameter = float(search_diam[min_idx_local])
    time_to_min_s = float(search_times[min_idx_local] - flash_onset)
    # Index of the same point within the full post-flash arrays, needed below
    min_idx = int(np.argmin(np.abs(post_times - search_times[min_idx_local])))

    constriction_amplitude_px = baseline_diameter - min_diameter
    constriction_amplitude_pct = (
        (constriction_amplitude_px / baseline_diameter) * 100.0
        if baseline_diameter > 0 else float("nan")
    )

    # Latency: first time point after flash onset where diameter drops below
    # baseline - 10% of the eventual amplitude (a standard threshold-based
    # latency definition).
    threshold = baseline_diameter - 0.1 * constriction_amplitude_px
    below_threshold = np.where(post_diam < threshold)[0]
    latency_s = float(post_times[below_threshold[0]] - flash_onset) if len(below_threshold) else None

    # Recovery time: time from peak constriction to 75% recovery back
    # toward baseline.
    recovery_target = min_diameter + 0.75 * constriction_amplitude_px
    after_min_times = post_times[min_idx:]
    after_min_diam = post_diam[min_idx:]
    recovered = np.where(after_min_diam >= recovery_target)[0]
    recovery_time_s = (
        float(after_min_times[recovered[0]] - after_min_times[0]) if len(recovered) else None
    )

    return {
        "baseline_diameter_px": baseline_diameter,
        "min_diameter_px": min_diameter,
        "time_to_min_constriction_s": time_to_min_s,
        "constriction_amplitude_px": constriction_amplitude_px,
        "constriction_amplitude_pct": constriction_amplitude_pct,
        "constriction_latency_s": latency_s,
        "recovery_time_75pct_s": recovery_time_s,
    }


def main():
    records = run_capture()

    if len(records) == 0:
        logger.error("No frames captured. Check webcam connection and try again.")
        return

    raw_csv_path = RAW_DIR / f"plr_session_{int(time.time())}.csv"
    save_raw_csv(records, raw_csv_path)
    logger.info("Saved raw session data -> %s", raw_csv_path)

    times = np.array([r.timestamp_s for r in records])
    left = np.array([
        r.left_pupil_iris_ratio if r.left_pupil_iris_ratio is not None else np.nan for r in records
    ])
    right = np.array([
        r.right_pupil_iris_ratio if r.right_pupil_iris_ratio is not None else np.nan for r in records
    ])

    valid_mask = ~np.isnan(left) | ~np.isnan(right)
    valid_ratio = np.mean(valid_mask)
    if valid_ratio < MIN_VALID_FRAME_RATIO:
        logger.warning(
            "Only %.0f%% of frames had a valid iris detection overall.",
            valid_ratio * 100
        )

    # Report detection rate per phase - this is the actually useful diagnostic,
    # since dropout specifically around the flash silently corrupts the result.
    states = np.array([r.stimulus_state for r in records])
    for phase in ("baseline", "flash", "recovery"):
        phase_mask = states == phase
        if phase_mask.any():
            phase_valid = np.mean(valid_mask[phase_mask])
            logger.info("Detection rate during %s: %.0f%%", phase, phase_valid * 100)

    # Average both eyes where available, otherwise fall back to whichever is present.
    # Scaled x100 so values read as a percentage (pupil is typically 20-80% of iris
    # diameter), which is easier to reason about than a raw 0-1 ratio.
    diam = np.nanmean(np.vstack([left, right]), axis=0) * 100.0

    # Only bridge SHORT dropouts (a few frames - e.g. a blink or brief tracking
    # blip). Long gaps must NOT be smoothed into a fake straight line - that
    # misrepresents missing data as a real reading. Long gaps are left as NaN
    # and excluded from feature extraction and plotting.
    nan_mask = np.isnan(diam)
    max_gap_frames = 6  # ~0.2-0.3s at typical webcam frame rates
    diam_filled = diam.copy()
    if nan_mask.any() and not nan_mask.all():
        gap_start = None
        for i in range(len(nan_mask)):
            if nan_mask[i] and gap_start is None:
                gap_start = i
            elif not nan_mask[i] and gap_start is not None:
                gap_len = i - gap_start
                if gap_len <= max_gap_frames:
                    diam_filled[gap_start:i] = np.interp(
                        times[gap_start:i], times[~nan_mask], diam[~nan_mask]
                    )
                gap_start = None
    diam = diam_filled
    remaining_nan_ratio = np.mean(np.isnan(diam))
    if remaining_nan_ratio > 0:
        logger.warning(
            "%.0f%% of the session has no usable data even after gap-filling "
            "(long tracking dropouts were left as gaps rather than faked).",
            remaining_nan_ratio * 100
        )
    diam_smooth = moving_average(diam, SMOOTHING_WINDOW)

    features = extract_features(times, diam_smooth)

    print("\n========== PLR FEATURES ==========")
    for k, v in features.items():
        print(f"{k}: {v}")

    metrics_path = METRICS_DIR / "plr_features.json"
    with open(metrics_path, "w") as f:
        json.dump(features, f, indent=2)
    logger.info("Saved features -> %s", metrics_path)

    # Plot
    plt.figure(figsize=(9, 5))
    plt.plot(times, diam, alpha=0.3, color="gray", label="Raw (avg. both eyes)")
    plt.plot(times, diam_smooth, color="tab:blue", linewidth=2, label="Smoothed")
    plt.axvline(BASELINE_DURATION_S, color="orange", linestyle="--", label="Flash onset")
    plt.axvline(BASELINE_DURATION_S + FLASH_DURATION_S, color="gray", linestyle=":", label="Flash end")
    if "min_diameter_px" in features and "time_to_min_constriction_s" in features:
        t_min = BASELINE_DURATION_S + features["time_to_min_constriction_s"]
        plt.scatter([t_min], [features["min_diameter_px"]], color="red", zorder=5, label="Peak constriction")
    plt.xlabel("Time (s)")
    plt.ylabel("Pupil:Iris diameter ratio (%)")
    plt.title("Pupillary Light Reflex Curve (distance-invariant ratio)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plot_path = PLOTS_DIR / "plr_curve.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logger.info("Saved PLR curve plot -> %s", plot_path)

    print("\nDone. Upload results/plots/plr_curve.png and paste the printed")
    print("features (or results/metrics/plr_features.json) back to continue.")


if __name__ == "__main__":
    main()