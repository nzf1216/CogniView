"""
record_session.py
CLI tool: record an eye-tracking session and export it as a CSV matching the
fusion module's expected schema (src/fusion/dataset.py:FEATURE_COLUMNS).

Single Responsibility:
    Drive CameraSource -> EyeTrackingExtractor -> CSV, for one labeled
    recording session. This is the missing link between the eye-tracking
    extractors (already complete) and FusionDataset (which needs real CSVs
    to train on).

Usage:
    # Webcam, 30 seconds, label=0 (e.g. "control")
    python -m src.eye_tracking.record_session --source webcam --camera-index 0 \
        --duration 30 --label 0 --out data/fusion/sessions/subject01_trial1.csv

    # Phone camera (e.g. IP Webcam app)
    python -m src.eye_tracking.record_session --source phone \
        --stream-url http://192.168.1.23:8080/video --duration 30 --label 1 \
        --out data/fusion/sessions/subject02_trial1.csv

    # Pre-recorded video file (also how this script is smoke-tested without
    # physical hardware)
    python -m src.eye_tracking.record_session --source video_file \
        --path sample.mp4 --label 0 --out out.csv

Author: CogniView Eye Tracking Module
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Optional

from src.camera.camera_manager import CameraManager
from src.camera.base_camera import CameraSource
from src.eye_tracking.extractor import EyeTrackingExtractor, EyeTrackingFeatures

logger = logging.getLogger("cogniview.eye_tracking.record_session")

# Must stay in sync with src/fusion/dataset.py:FEATURE_COLUMNS.
CSV_COLUMNS = [
    "timestamp_ms",
    "detected",
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
    "gaze_direction",
    "label",
]


def _row_from_features(features: EyeTrackingFeatures, label: int) -> dict:
    left_x, left_y = (features.left_iris if features.left_iris else (0.0, 0.0))
    right_x, right_y = (features.right_iris if features.right_iris else (0.0, 0.0))
    return {
        "timestamp_ms": features.timestamp_ms,
        "detected": int(features.detected),
        "ear": features.ear,
        "blink_count": features.blink_count,
        "blink_rate": features.blink_rate,
        "fixation_stability": features.fixation_stability,
        "fixation_count": features.fixation_count,
        "saccade_count": features.saccade_count,
        "smooth_pursuit_count": features.smooth_pursuit_count,
        "left_iris_x": left_x,
        "left_iris_y": left_y,
        "right_iris_x": right_x,
        "right_iris_y": right_y,
        "gaze_direction": features.gaze_direction,
        "label": label,
    }


def record_session(
    camera: CameraSource,
    out_path: str,
    label: int,
    duration_s: Optional[float] = None,
    max_frames: Optional[int] = None,
    require_detection: bool = False,
) -> int:
    """
    Run camera -> EyeTrackingExtractor -> CSV until duration_s / max_frames is
    reached or the camera runs out of frames.

    Args:
        camera: an opened-or-openable CameraSource (this function will call
            .open() if it hasn't been opened yet).
        out_path: destination CSV path. Parent directories are created.
        label: integer class label attached to every row of this session
            (fusion training reads one label per row; a whole session is
            normally one label).
        duration_s: stop after this many seconds of session time. None = no
            time limit (rely on max_frames or end-of-stream instead).
        max_frames: stop after this many frames, regardless of duration_s.
        require_detection: if True, frames with no detected face are skipped
            entirely rather than written as a "carried forward" row.

    Returns:
        Number of rows written.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not camera.is_opened:
        camera.open()

    extractor = EyeTrackingExtractor()
    extractor.initialize()

    rows_written = 0
    last_timestamp_ms = 0
    session_start = time.time()

    try:
        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()

            while True:
                if duration_s is not None and (time.time() - session_start) >= duration_s:
                    logger.info("Reached duration limit (%.1fs); stopping.", duration_s)
                    break
                if max_frames is not None and rows_written >= max_frames:
                    logger.info("Reached max_frames limit (%d); stopping.", max_frames)
                    break

                ret, frame, timestamp_ms = camera.read()
                if not ret:
                    if not camera.is_opened:
                        logger.info("Camera closed / end of stream; stopping.")
                        break
                    # Transient read failure (e.g. a dropped Wi-Fi frame from
                    # a phone camera) -- skip this iteration, keep going.
                    continue

                features = extractor.process(frame, timestamp_ms)
                last_timestamp_ms = timestamp_ms

                if require_detection and not features.detected:
                    continue

                writer.writerow(_row_from_features(features, label))
                rows_written += 1

        extractor.finalize(last_timestamp_ms)
    finally:
        extractor.close()
        camera.release()

    logger.info("Session complete: %d rows written to %s", rows_written, out_file)
    return rows_written


def _build_camera_from_args(args: argparse.Namespace) -> CameraSource:
    if args.source == "webcam":
        return CameraManager.create("webcam", camera_index=args.camera_index)
    elif args.source == "phone":
        if not args.stream_url:
            raise SystemExit("--stream-url is required when --source phone is used.")
        return CameraManager.create("phone", stream_url=args.stream_url)
    elif args.source == "video_file":
        if not args.path:
            raise SystemExit("--path is required when --source video_file is used.")
        return CameraManager.create("video_file", path=args.path)
    else:
        raise SystemExit(f"Unknown --source '{args.source}'.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a CogniView eye-tracking session to CSV.")
    parser.add_argument("--source", choices=["webcam", "phone", "video_file"], required=True)
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--stream-url", type=str, default=None, help="Phone camera stream URL.")
    parser.add_argument("--path", type=str, default=None, help="Path to a video file (source=video_file).")
    parser.add_argument("--out", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--label", type=int, required=True, help="Integer class label for this session.")
    parser.add_argument("--duration", type=float, default=None, help="Session duration in seconds.")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to record.")
    parser.add_argument(
        "--require-detection", action="store_true",
        help="Drop frames where no face was detected instead of recording them.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = _parse_args()
    camera = _build_camera_from_args(args)
    record_session(
        camera=camera,
        out_path=args.out,
        label=args.label,
        duration_s=args.duration,
        max_frames=args.max_frames,
        require_detection=args.require_detection,
    )


if __name__ == "__main__":
    main()
