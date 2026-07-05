"""
blink.py
EAR-based blink detection with hysteresis, built on top of landmarks.py.

Single Responsibility:
    Compute Eye Aspect Ratio (EAR) from FrameLandmarks and detect blinks
    using a hysteresis state machine (rather than a single-threshold +
    cooldown), so that noisy EAR values near the threshold don't cause
    false double-counts, and blink duration is validated to reject both
    detection glitches (too short) and prolonged eye closure (too long).

Author: CogniView Eye Tracking Module
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import numpy as np

from src.eye_tracking.landmarks import FrameLandmarks

logger = logging.getLogger("cogniview.eye_tracking.blink")


# --------------------------------------------------------------------------- #
# Tunable constants
# --------------------------------------------------------------------------- #

# Hysteresis thresholds: EAR must drop below EAR_LOW to start closing,
# and rise above EAR_HIGH to be considered fully open again.
# EAR_HIGH > EAR_LOW creates a "dead zone" that prevents jitter-triggered
# double counts when EAR hovers near a single threshold.
EAR_LOW = 0.19
EAR_HIGH = 0.23

# Blink duration bounds, in milliseconds, used to distinguish a genuine
# blink from a detection glitch (too short) or a sustained eye closure /
# gaze-away event (too long).
MIN_BLINK_DURATION_MS = 60
MAX_BLINK_DURATION_MS = 400

# Smoothing window (frames) for EAR, to reduce landmark jitter noise.
EAR_SMOOTHING_WINDOW = 3


# --------------------------------------------------------------------------- #
# EAR computation
# --------------------------------------------------------------------------- #

def compute_ear(eye_points: np.ndarray) -> float:
    """
    Compute Eye Aspect Ratio from the standard 6-point EAR landmark set,
    in the order: [p1 (left corner), p2, p3, p4 (right corner), p5, p6]
    following the Soukupová & Čech (2016) convention:

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        eye_points: (6, 2) array of normalized (x, y) coordinates, matching
                    EyeLandmarks.LEFT_EAR_POINTS / RIGHT_EAR_POINTS ordering.

    Returns:
        EAR value as a float. Returns 0.0 if input is malformed (e.g. eye
        not detected), which naturally reads as "closed" downstream — callers
        should check `detected` on FrameLandmarks before trusting this.
    """
    if eye_points is None or eye_points.shape[0] != 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal < 1e-6:
        return 0.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def compute_average_ear(frame_landmarks: FrameLandmarks) -> Optional[float]:
    """
    Compute the average EAR across both eyes for a given frame.

    Returns:
        Average EAR, or None if the face was not detected in this frame.
    """
    if not frame_landmarks.detected:
        return None

    left_ear = compute_ear(frame_landmarks.left_ear_points)
    right_ear = compute_ear(frame_landmarks.right_ear_points)
    return (left_ear + right_ear) / 2.0


# --------------------------------------------------------------------------- #
# Blink state machine
# --------------------------------------------------------------------------- #

class EyeState(Enum):
    OPEN = auto()
    CLOSING = auto()   # EAR dropped below EAR_LOW, timing the closure
    OPENING = auto()   # EAR rose back above EAR_LOW but not yet above EAR_HIGH


@dataclass
class BlinkEvent:
    """A single completed, validated blink."""
    start_timestamp_ms: int
    end_timestamp_ms: int
    duration_ms: int
    min_ear: float


@dataclass
class BlinkStats:
    """Running statistics exposed to downstream modules (e.g. fusion)."""
    blink_count: int = 0
    blink_rate_per_min: float = 0.0
    last_blink: Optional[BlinkEvent] = None
    all_blinks: List[BlinkEvent] = field(default_factory=list)


class BlinkDetector:
    """
    Hysteresis-based blink detector.

    Usage:
        detector = BlinkDetector()
        for frame_landmarks in stream:
            avg_ear = compute_average_ear(frame_landmarks)
            event = detector.update(avg_ear, frame_landmarks.timestamp_ms)
            if event:
                print(f"Blink #{detector.stats.blink_count}, "
                      f"duration={event.duration_ms}ms")
    """

    def __init__(
        self,
        ear_low: float = EAR_LOW,
        ear_high: float = EAR_HIGH,
        min_blink_duration_ms: int = MIN_BLINK_DURATION_MS,
        max_blink_duration_ms: int = MAX_BLINK_DURATION_MS,
        smoothing_window: int = EAR_SMOOTHING_WINDOW,
    ):
        if ear_high <= ear_low:
            raise ValueError("ear_high must be greater than ear_low to create a valid hysteresis band.")

        self.ear_low = ear_low
        self.ear_high = ear_high
        self.min_blink_duration_ms = min_blink_duration_ms
        self.max_blink_duration_ms = max_blink_duration_ms
        self.smoothing_window = max(1, smoothing_window)

        self._state = EyeState.OPEN
        self._closure_start_ms: Optional[int] = None
        self._min_ear_during_closure = float("inf")
        self._ear_history: List[float] = []

        self._session_start_ms: Optional[int] = None
        self.stats = BlinkStats()

    def _smoothed_ear(self, raw_ear: float) -> float:
        self._ear_history.append(raw_ear)
        if len(self._ear_history) > self.smoothing_window:
            self._ear_history.pop(0)
        return sum(self._ear_history) / len(self._ear_history)

    def update(self, avg_ear: Optional[float], timestamp_ms: int) -> Optional[BlinkEvent]:
        """
        Feed one frame's average EAR into the state machine.

        Args:
            avg_ear: Average EAR for this frame, or None if no face was
                     detected (treated as a dropped frame — does not reset
                     an in-progress closure, to tolerate brief detection
                     flicker during a real blink).
            timestamp_ms: Frame timestamp in milliseconds (must be
                          monotonically increasing, matching FrameLandmarks).

        Returns:
            A BlinkEvent if a validated blink completed on this frame,
            otherwise None.
        """
        if self._session_start_ms is None:
            self._session_start_ms = timestamp_ms

        if avg_ear is None:
            # No face detected this frame — don't feed into smoothing/state,
            # just wait for the next valid frame.
            return None

        ear = self._smoothed_ear(avg_ear)
        completed_blink: Optional[BlinkEvent] = None

        if self._state == EyeState.OPEN:
            if ear < self.ear_low:
                self._state = EyeState.CLOSING
                self._closure_start_ms = timestamp_ms
                self._min_ear_during_closure = ear

        elif self._state == EyeState.CLOSING:
            self._min_ear_during_closure = min(self._min_ear_during_closure, ear)
            if ear >= self.ear_low:
                # Started reopening, but not yet confirmed fully open.
                self._state = EyeState.OPENING

        elif self._state == EyeState.OPENING:
            self._min_ear_during_closure = min(self._min_ear_during_closure, ear)
            if ear >= self.ear_high:
                # Fully reopened — validate and close out the blink.
                completed_blink = self._finalize_closure(timestamp_ms)
                self._state = EyeState.OPEN
            elif ear < self.ear_low:
                # Dipped back down before fully reopening — still closing.
                self._state = EyeState.CLOSING

        return completed_blink

    def _finalize_closure(self, end_timestamp_ms: int) -> Optional[BlinkEvent]:
        """Validate closure duration and record a blink if it qualifies."""
        if self._closure_start_ms is None:
            return None

        duration_ms = end_timestamp_ms - self._closure_start_ms

        is_valid = self.min_blink_duration_ms <= duration_ms <= self.max_blink_duration_ms
        event = None

        if is_valid:
            event = BlinkEvent(
                start_timestamp_ms=self._closure_start_ms,
                end_timestamp_ms=end_timestamp_ms,
                duration_ms=duration_ms,
                min_ear=self._min_ear_during_closure,
            )
            self.stats.blink_count += 1
            self.stats.last_blink = event
            self.stats.all_blinks.append(event)
            self._update_blink_rate(end_timestamp_ms)
        else:
            logger.debug(
                "Rejected closure: duration=%dms outside [%d, %d]ms bounds.",
                duration_ms, self.min_blink_duration_ms, self.max_blink_duration_ms,
            )

        self._closure_start_ms = None
        self._min_ear_during_closure = float("inf")
        return event

    def _update_blink_rate(self, current_timestamp_ms: int) -> None:
        """Update blinks-per-minute based on elapsed session time."""
        if self._session_start_ms is None:
            return
        elapsed_minutes = (current_timestamp_ms - self._session_start_ms) / 60000.0
        if elapsed_minutes > 0:
            self.stats.blink_rate_per_min = self.stats.blink_count / elapsed_minutes

    def reset(self) -> None:
        """Reset all state and statistics (e.g. for a new session/recording)."""
        self._state = EyeState.OPEN
        self._closure_start_ms = None
        self._min_ear_during_closure = float("inf")
        self._ear_history.clear()
        self._session_start_ms = None
        self.stats = BlinkStats()


# --------------------------------------------------------------------------- #
# Standalone smoke test (run: python -m src.eye_tracking.blink)
# --------------------------------------------------------------------------- #

def _run_live_demo() -> None:
    import time
    import cv2
    from src.eye_tracking.landmarks import LandmarkExtractor, open_webcam

    cap = open_webcam()
    extractor = LandmarkExtractor()
    extractor.initialize()
    detector = BlinkDetector()

    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            timestamp_ms = int((time.time() - start_time) * 1000)
            frame_landmarks = extractor.process(frame, timestamp_ms)
            avg_ear = compute_average_ear(frame_landmarks)

            event = detector.update(avg_ear, timestamp_ms)
            if event:
                logger.info(
                    "Blink #%d detected — duration=%dms, min_ear=%.3f",
                    detector.stats.blink_count, event.duration_ms, event.min_ear,
                )

            ear_display = f"{avg_ear:.3f}" if avg_ear is not None else "N/A"
            cv2.putText(
                frame, f"EAR: {ear_display}  Blinks: {detector.stats.blink_count}  "
                       f"Rate: {detector.stats.blink_rate_per_min:.1f}/min",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.imshow("CogniView - Blink Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_live_demo()