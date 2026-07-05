"""
extractor.py
Master orchestrator for the CogniView eye-tracking pipeline — wires together
LandmarkExtractor, BlinkDetector, and GazeTracker into a single per-frame
feature vector consumed by the fusion module.

Single Responsibility:
    Own the lifecycle of the three underlying components (landmark
    extraction, blink detection, gaze/fixation/saccade tracking) and, for
    each frame, combine their outputs into one immutable EyeTrackingFeatures
    record. Downstream modules (fusion, PLR, XAI) depend only on this
    dataclass and never touch MediaPipe / OpenCV directly.

Author: CogniView Eye Tracking Module
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.eye_tracking.landmarks import (
    FrameLandmarks,
    LandmarkExtractor,
    open_webcam,
)
from src.eye_tracking.blink import (
    BlinkDetector,
    compute_average_ear,
    EAR_LOW,
    EAR_HIGH,
    MIN_BLINK_DURATION_MS,
    MAX_BLINK_DURATION_MS,
)
from src.eye_tracking.gaze import (
    GazeTracker,
    GazeDirection,
    SACCADE_VELOCITY_THRESHOLD,
    SMOOTH_PURSUIT_VELOCITY_THRESHOLD,
    MIN_FIXATION_DURATION_MS,
)

logger = logging.getLogger("cogniview.eye_tracking.extractor")


# --------------------------------------------------------------------------- #
# Unified feature record
# --------------------------------------------------------------------------- #

@dataclass
class EyeTrackingFeatures:
    """
    Unified, per-frame eye-tracking feature vector.

    This is the sole contract between the eye-tracking package and everything
    downstream (fusion/model.py, plr/extractor.py, the FastAPI backend). Every
    field has a well-defined value even on frames where the face wasn't
    detected, so consumers never need to special-case FrameLandmarks, EAR
    smoothing state, or gaze-tracker internals.
    """

    timestamp_ms: int
    detected: bool

    # Blink (from BlinkDetector / BlinkStats)
    ear: float
    blink_count: int
    blink_rate: float
    blink_just_occurred: bool = False

    # Gaze / fixation / saccade (from GazeTracker / GazeStats)
    gaze_direction: str = GazeDirection.CENTER.value
    fixation_stability: float = 0.0
    fixation_count: int = 0
    saccade_count: int = 0
    smooth_pursuit_count: int = 0

    # Raw normalized iris positions, kept alongside the discrete gaze label
    # since fusion models may want spatial detail rather than just LEFT/RIGHT/etc.
    left_iris: Optional[Tuple[float, float]] = None
    right_iris: Optional[Tuple[float, float]] = None


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

class EyeTrackingExtractor:
    """
    Master orchestrator that combines LandmarkExtractor, BlinkDetector, and
    GazeTracker into a single per-frame call, producing one EyeTrackingFeatures
    record per frame.

    Usage:
        extractor = EyeTrackingExtractor()
        extractor.initialize()
        for frame, timestamp_ms in video_stream:
            features = extractor.process(frame, timestamp_ms)
        extractor.finalize(last_timestamp_ms)
        extractor.close()

    Or as a context manager:
        with EyeTrackingExtractor() as extractor:
            features = extractor.process(frame, timestamp_ms)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        ear_low: float = EAR_LOW,
        ear_high: float = EAR_HIGH,
        min_blink_duration_ms: int = MIN_BLINK_DURATION_MS,
        max_blink_duration_ms: int = MAX_BLINK_DURATION_MS,
        saccade_velocity_threshold: float = SACCADE_VELOCITY_THRESHOLD,
        smooth_pursuit_velocity_threshold: float = SMOOTH_PURSUIT_VELOCITY_THRESHOLD,
        min_fixation_duration_ms: int = MIN_FIXATION_DURATION_MS,
    ):
        self._landmark_extractor = LandmarkExtractor(
            model_path=model_path,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode="VIDEO",
        )
        self._blink_detector = BlinkDetector(
            ear_low=ear_low,
            ear_high=ear_high,
            min_blink_duration_ms=min_blink_duration_ms,
            max_blink_duration_ms=max_blink_duration_ms,
        )
        self._gaze_tracker = GazeTracker(
            saccade_velocity_threshold=saccade_velocity_threshold,
            smooth_pursuit_velocity_threshold=smooth_pursuit_velocity_threshold,
            min_fixation_duration_ms=min_fixation_duration_ms,
        )

        self._initialized = False
        self._last_timestamp_ms: Optional[int] = None

    def initialize(self) -> None:
        """Initialize the underlying MediaPipe FaceLandmarker. Idempotent."""
        if self._initialized:
            logger.warning("EyeTrackingExtractor already initialized; skipping.")
            return
        self._landmark_extractor.initialize()
        self._initialized = True
        logger.info("EyeTrackingExtractor initialized.")

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> EyeTrackingFeatures:
        """
        Run the full pipeline on a single BGR frame and return one fused
        feature record.

        Args:
            frame_bgr: OpenCV frame in BGR format.
            timestamp_ms: Monotonically increasing timestamp in milliseconds,
                          forwarded unchanged to LandmarkExtractor.process()
                          (VIDEO mode requires strictly increasing timestamps).

        Returns:
            EyeTrackingFeatures for this frame. `detected=False` frames still
            return a valid record — blink/gaze counters simply carry forward
            unchanged from prior frames, and ear/iris fields fall back to
            neutral defaults rather than raising.
        """
        if not self._initialized:
            raise RuntimeError("EyeTrackingExtractor.initialize() must be called before process().")

        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            logger.debug(
                "Non-increasing timestamp (%d <= %d) — MediaPipe VIDEO mode requires "
                "strictly increasing timestamps; this frame may be rejected upstream.",
                timestamp_ms, self._last_timestamp_ms,
            )
        self._last_timestamp_ms = timestamp_ms

        frame_landmarks = self._landmark_extractor.process(frame_bgr, timestamp_ms)
        return self._fuse(frame_landmarks)

    def _fuse(self, frame_landmarks: FrameLandmarks) -> EyeTrackingFeatures:
        """Feed one frame's landmarks through blink + gaze, then merge the results."""
        avg_ear = compute_average_ear(frame_landmarks)
        blink_event = self._blink_detector.update(avg_ear, frame_landmarks.timestamp_ms)
        gaze_sample = self._gaze_tracker.update(frame_landmarks)

        blink_stats = self._blink_detector.stats
        gaze_stats = self._gaze_tracker.stats

        direction = gaze_sample.direction.value if gaze_sample else GazeDirection.CENTER.value

        return EyeTrackingFeatures(
            timestamp_ms=frame_landmarks.timestamp_ms,
            detected=frame_landmarks.detected,
            ear=avg_ear if avg_ear is not None else 0.0,
            blink_count=blink_stats.blink_count,
            blink_rate=blink_stats.blink_rate_per_min,
            blink_just_occurred=blink_event is not None,
            gaze_direction=direction,
            fixation_stability=gaze_stats.fixation_stability,
            fixation_count=gaze_stats.fixation_count,
            saccade_count=gaze_stats.saccade_count,
            smooth_pursuit_count=gaze_stats.smooth_pursuit_count,
            left_iris=frame_landmarks.left_iris_center,
            right_iris=frame_landmarks.right_iris_center,
        )

    def finalize(self, final_timestamp_ms: int) -> None:
        """
        Close out any in-progress gaze segment (fixation/saccade/pursuit).
        Call once at the end of a recording session — without this, the last
        segment never crosses its classification boundary and is silently
        dropped from fixation_count/saccade_count/smooth_pursuit_count.
        """
        self._gaze_tracker.finalize(final_timestamp_ms)

    def reset(self) -> None:
        """
        Reset blink and gaze state/statistics for a new session, without
        tearing down or reinitializing the underlying MediaPipe model.
        """
        self._blink_detector.reset()
        self._gaze_tracker.reset()
        self._last_timestamp_ms = None

    def close(self) -> None:
        """Release the underlying MediaPipe FaceLandmarker resources."""
        self._landmark_extractor.close()
        self._initialized = False
        logger.info("EyeTrackingExtractor closed.")

    def __enter__(self) -> "EyeTrackingExtractor":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Standalone smoke test (run: python -m src.eye_tracking.extractor)
# --------------------------------------------------------------------------- #

def _run_live_demo() -> None:
    """Quick manual test: opens the webcam and overlays fused features live."""
    cap = open_webcam()
    extractor = EyeTrackingExtractor()
    extractor.initialize()

    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame grab failed; skipping.")
                continue

            frame = cv2.flip(frame, 1)
            timestamp_ms = int((time.time() - start_time) * 1000)
            features = extractor.process(frame, timestamp_ms)

            cv2.putText(
                frame,
                f"EAR:{features.ear:.3f} Blinks:{features.blink_count} "
                f"Rate:{features.blink_rate:.1f}/min Gaze:{features.gaze_direction}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            cv2.putText(
                frame,
                f"Fix:{features.fixation_count} Sac:{features.saccade_count} "
                f"Pur:{features.smooth_pursuit_count} Stability:{features.fixation_stability:.4f}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )
            if features.blink_just_occurred:
                cv2.putText(frame, "BLINK", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("CogniView - Eye Tracking Extractor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        extractor.finalize(int((time.time() - start_time) * 1000))
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_live_demo()