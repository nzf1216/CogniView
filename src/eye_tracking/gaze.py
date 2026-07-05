"""
gaze.py
Gaze direction, fixation, saccade, and smooth pursuit estimation, built on
top of landmarks.py's iris and eye-corner landmarks.

Single Responsibility:
    Convert per-frame iris/eye landmarks into (a) a discrete gaze direction
    label and (b) temporal eye-movement events (fixation / saccade / smooth
    pursuit) using a velocity-threshold (I-VT) classifier — the standard
    approach in eye-tracking literature, needed because saccadic/fixation
    abnormalities are frame-to-frame temporal phenomena, not single-frame ones.

Author: CogniView Eye Tracking Module
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, List, Optional, Tuple

import numpy as np

from src.eye_tracking.landmarks import EyeLandmarks, FrameLandmarks

logger = logging.getLogger("cogniview.eye_tracking.gaze")


# --------------------------------------------------------------------------- #
# Tunable constants
# --------------------------------------------------------------------------- #

# Horizontal/vertical ratio cutoffs for discrete direction labeling.
# ratio = (iris_pos - eye_min) / eye_range, so 0.5 == dead center.
HORIZONTAL_LEFT_CUTOFF = 0.40
HORIZONTAL_RIGHT_CUTOFF = 0.60
VERTICAL_UP_CUTOFF = 0.40
VERTICAL_DOWN_CUTOFF = 0.60

# I-VT (Velocity-Threshold Identification) classifier parameters.
# Velocity is in normalized-coordinate units per second (eye-widths/sec
# after normalization), since we don't have a physical calibration without
# a known screen distance.
SACCADE_VELOCITY_THRESHOLD = 0.15   # units/sec — above this => saccade
SMOOTH_PURSUIT_VELOCITY_THRESHOLD = 0.02  # units/sec — between this and
                                           # SACCADE threshold => pursuit
MIN_FIXATION_DURATION_MS = 100
VELOCITY_SMOOTHING_WINDOW = 3


class GazeDirection(Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    CENTER = "CENTER"


class EyeMovementType(Enum):
    FIXATION = auto()
    SACCADE = auto()
    SMOOTH_PURSUIT = auto()
    UNDETERMINED = auto()  # not enough history yet


@dataclass
class GazeSample:
    timestamp_ms: int
    iris_center: Tuple[float, float]   # normalized (x, y), averaged both eyes
    direction: GazeDirection


@dataclass
class EyeMovementEvent:
    movement_type: EyeMovementType
    start_timestamp_ms: int
    end_timestamp_ms: int
    duration_ms: int
    mean_velocity: float


@dataclass
class GazeStats:
    fixation_count: int = 0
    saccade_count: int = 0
    smooth_pursuit_count: int = 0
    fixation_stability: float = 0.0   # lower = more stable; std-dev of iris pos during fixations
    events: List[EyeMovementEvent] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Single-frame gaze direction (spatial only)
# --------------------------------------------------------------------------- #

def _axis_ratio(pos: float, low_bound: float, high_bound: float) -> Optional[float]:
    span = high_bound - low_bound
    if span == 0:
        return None
    return (pos - low_bound) / span


def get_gaze_direction(
    iris_pts: np.ndarray,
    eye_pts: np.ndarray,
    eye_top_idx_local: int = 0,
    eye_bottom_idx_local: int = 0,
) -> GazeDirection:
    """
    Single-frame spatial gaze direction from iris position relative to the
    eye's bounding extent (not fixed indices — uses min/max of the contour
    so it works regardless of point ordering).

    Args:
        iris_pts: (N, 2) normalized iris landmark points.
        eye_pts:  (N, 2) normalized eye contour landmark points (e.g.
                  FrameLandmarks.left_eye_points, from EyeLandmarks.LEFT_EYE_CONTOUR).

    Returns:
        GazeDirection enum value. Returns CENTER if inputs are degenerate.
    """
    if iris_pts is None or eye_pts is None or len(iris_pts) == 0 or len(eye_pts) == 0:
        return GazeDirection.CENTER

    iris_cx = float(np.mean(iris_pts[:, 0]))
    iris_cy = float(np.mean(iris_pts[:, 1]))

    eye_left = float(np.min(eye_pts[:, 0]))
    eye_right = float(np.max(eye_pts[:, 0]))
    eye_top = float(np.min(eye_pts[:, 1]))
    eye_bottom = float(np.max(eye_pts[:, 1]))

    h_ratio = _axis_ratio(iris_cx, eye_left, eye_right)
    v_ratio = _axis_ratio(iris_cy, eye_top, eye_bottom)

    if h_ratio is None or v_ratio is None:
        return GazeDirection.CENTER

    # Horizontal takes priority over vertical when both are off-center,
    # since horizontal saccades dominate typical screening tasks.
    if h_ratio < HORIZONTAL_LEFT_CUTOFF:
        return GazeDirection.LEFT
    if h_ratio > HORIZONTAL_RIGHT_CUTOFF:
        return GazeDirection.RIGHT
    if v_ratio < VERTICAL_UP_CUTOFF:
        return GazeDirection.UP
    if v_ratio > VERTICAL_DOWN_CUTOFF:
        return GazeDirection.DOWN
    return GazeDirection.CENTER


def get_combined_iris_center(frame_landmarks: FrameLandmarks) -> Optional[Tuple[float, float]]:
    """Average both eyes' iris centers into a single normalized (x, y) point."""
    if frame_landmarks.left_iris_center is None or frame_landmarks.right_iris_center is None:
        return None
    lx, ly = frame_landmarks.left_iris_center
    rx, ry = frame_landmarks.right_iris_center
    return ((lx + rx) / 2.0, (ly + ry) / 2.0)


# --------------------------------------------------------------------------- #
# Temporal gaze tracking: fixations, saccades, smooth pursuit (I-VT)
# --------------------------------------------------------------------------- #

class GazeTracker:
    """
    Stateful tracker that consumes per-frame iris positions and classifies
    eye movement into fixation / saccade / smooth pursuit segments using
    an I-VT (velocity-threshold) algorithm, plus computes fixation stability.

    Usage:
        tracker = GazeTracker()
        for frame_landmarks in stream:
            sample = tracker.update(frame_landmarks)
            # sample.direction is the discrete label for this frame
            # tracker.stats accumulates fixation/saccade/pursuit counts
    """

    def __init__(
        self,
        saccade_velocity_threshold: float = SACCADE_VELOCITY_THRESHOLD,
        smooth_pursuit_velocity_threshold: float = SMOOTH_PURSUIT_VELOCITY_THRESHOLD,
        min_fixation_duration_ms: int = MIN_FIXATION_DURATION_MS,
        velocity_smoothing_window: int = VELOCITY_SMOOTHING_WINDOW,
    ):
        if saccade_velocity_threshold <= smooth_pursuit_velocity_threshold:
            raise ValueError(
                "saccade_velocity_threshold must exceed smooth_pursuit_velocity_threshold."
            )

        self.saccade_velocity_threshold = saccade_velocity_threshold
        self.smooth_pursuit_velocity_threshold = smooth_pursuit_velocity_threshold
        self.min_fixation_duration_ms = min_fixation_duration_ms

        self._prev_point: Optional[Tuple[float, float]] = None
        self._prev_timestamp_ms: Optional[int] = None
        self._velocity_history: Deque[float] = deque(maxlen=velocity_smoothing_window)

        self._current_segment_type: EyeMovementType = EyeMovementType.UNDETERMINED
        self._segment_start_ms: Optional[int] = None
        self._segment_velocities: List[float] = []
        self._segment_points: List[Tuple[float, float]] = []

        self.stats = GazeStats()
        self._fixation_positions_for_stability: List[Tuple[float, float]] = []

    def update(self, frame_landmarks: FrameLandmarks) -> Optional[GazeSample]:
        """
        Feed one frame's landmarks into the tracker.

        Returns:
            GazeSample for this frame, or None if the face/iris wasn't
            detected (gap in tracking — resets velocity computation but
            not accumulated stats).
        """
        if not frame_landmarks.detected:
            self._prev_point = None
            self._prev_timestamp_ms = None
            return None

        iris_center = get_combined_iris_center(frame_landmarks)
        if iris_center is None:
            return None

        direction = self._frame_direction(frame_landmarks)
        timestamp_ms = frame_landmarks.timestamp_ms
        sample = GazeSample(timestamp_ms=timestamp_ms, iris_center=iris_center, direction=direction)

        velocity = self._compute_velocity(iris_center, timestamp_ms)
        if velocity is not None:
            self._classify(velocity, iris_center, timestamp_ms)

        self._prev_point = iris_center
        self._prev_timestamp_ms = timestamp_ms
        return sample

    @staticmethod
    def _frame_direction(frame_landmarks: FrameLandmarks) -> GazeDirection:
        left_dir = get_gaze_direction(frame_landmarks.left_iris_points, frame_landmarks.left_eye_points)
        right_dir = get_gaze_direction(frame_landmarks.right_iris_points, frame_landmarks.right_eye_points)
        # Prefer agreement between both eyes; fall back to left eye if they disagree
        # (left eye chosen arbitrarily but consistently, since disagreement usually
        # means one eye's landmarks are noisier at the frame boundary).
        return left_dir if left_dir == right_dir else left_dir

    def _compute_velocity(
        self, current_point: Tuple[float, float], timestamp_ms: int
    ) -> Optional[float]:
        if self._prev_point is None or self._prev_timestamp_ms is None:
            return None

        dt_s = (timestamp_ms - self._prev_timestamp_ms) / 1000.0
        if dt_s <= 0:
            return None

        dx = current_point[0] - self._prev_point[0]
        dy = current_point[1] - self._prev_point[1]
        distance = float(np.hypot(dx, dy))
        raw_velocity = distance / dt_s

        self._velocity_history.append(raw_velocity)
        return sum(self._velocity_history) / len(self._velocity_history)

    def _classify(self, velocity: float, point: Tuple[float, float], timestamp_ms: int) -> None:
        if velocity > self.saccade_velocity_threshold:
            movement_type = EyeMovementType.SACCADE
        elif velocity > self.smooth_pursuit_velocity_threshold:
            movement_type = EyeMovementType.SMOOTH_PURSUIT
        else:
            movement_type = EyeMovementType.FIXATION

        if self._current_segment_type == EyeMovementType.UNDETERMINED:
            self._start_segment(movement_type, timestamp_ms)
        elif movement_type != self._current_segment_type:
            self._close_segment(timestamp_ms)
            self._start_segment(movement_type, timestamp_ms)

        self._segment_velocities.append(velocity)
        self._segment_points.append(point)

    def _start_segment(self, movement_type: EyeMovementType, timestamp_ms: int) -> None:
        self._current_segment_type = movement_type
        self._segment_start_ms = timestamp_ms
        self._segment_velocities = []
        self._segment_points = []

    def _close_segment(self, end_timestamp_ms: int) -> None:
        if self._segment_start_ms is None:
            return

        duration_ms = end_timestamp_ms - self._segment_start_ms
        mean_velocity = (
            sum(self._segment_velocities) / len(self._segment_velocities)
            if self._segment_velocities else 0.0
        )

        # Fixations shorter than the minimum duration are likely noise/transition
        # artifacts, not genuine fixations — drop them rather than counting them.
        if self._current_segment_type == EyeMovementType.FIXATION and duration_ms < self.min_fixation_duration_ms:
            return

        event = EyeMovementEvent(
            movement_type=self._current_segment_type,
            start_timestamp_ms=self._segment_start_ms,
            end_timestamp_ms=end_timestamp_ms,
            duration_ms=duration_ms,
            mean_velocity=mean_velocity,
        )
        self.stats.events.append(event)

        if self._current_segment_type == EyeMovementType.FIXATION:
            self.stats.fixation_count += 1
            self._fixation_positions_for_stability.extend(self._segment_points)
            self._update_fixation_stability()
        elif self._current_segment_type == EyeMovementType.SACCADE:
            self.stats.saccade_count += 1
        elif self._current_segment_type == EyeMovementType.SMOOTH_PURSUIT:
            self.stats.smooth_pursuit_count += 1

    def _update_fixation_stability(self) -> None:
        """
        Fixation stability = mean of (x, y) standard deviations across all
        recorded fixation points. Lower values indicate steadier gaze —
        reduced fixation stability is a documented early cognitive-decline
        marker, so this is exposed directly for the fusion module.
        """
        if len(self._fixation_positions_for_stability) < 2:
            return
        pts = np.array(self._fixation_positions_for_stability)
        std_x, std_y = np.std(pts[:, 0]), np.std(pts[:, 1])
        self.stats.fixation_stability = float((std_x + std_y) / 2.0)

    def finalize(self, final_timestamp_ms: int) -> None:
        """Close out any in-progress segment. Call at the end of a recording session."""
        if self._current_segment_type != EyeMovementType.UNDETERMINED:
            self._close_segment(final_timestamp_ms)

    def reset(self) -> None:
        """Reset all tracking state and statistics (e.g. for a new session)."""
        self._prev_point = None
        self._prev_timestamp_ms = None
        self._velocity_history.clear()
        self._current_segment_type = EyeMovementType.UNDETERMINED
        self._segment_start_ms = None
        self._segment_velocities = []
        self._segment_points = []
        self._fixation_positions_for_stability = []
        self.stats = GazeStats()


# --------------------------------------------------------------------------- #
# Standalone smoke test (run: python -m src.eye_tracking.gaze)
# --------------------------------------------------------------------------- #

def _run_live_demo() -> None:
    import time
    import cv2
    from src.eye_tracking.landmarks import LandmarkExtractor, open_webcam

    cap = open_webcam()
    extractor = LandmarkExtractor()
    extractor.initialize()
    tracker = GazeTracker()

    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            timestamp_ms = int((time.time() - start_time) * 1000)
            frame_landmarks = extractor.process(frame, timestamp_ms)
            sample = tracker.update(frame_landmarks)

            direction_text = sample.direction.value if sample else "N/A"
            cv2.putText(
                frame,
                f"Gaze: {direction_text}  Fix: {tracker.stats.fixation_count}  "
                f"Sac: {tracker.stats.saccade_count}  Pur: {tracker.stats.smooth_pursuit_count}  "
                f"Stability: {tracker.stats.fixation_stability:.4f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )
            cv2.imshow("CogniView - Gaze Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.finalize(int((time.time() - start_time) * 1000))
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_live_demo()