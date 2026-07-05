"""
landmarks.py
Core face/eye landmark extraction module using MediaPipe Tasks API (FaceLandmarker).

Single Responsibility:
    Initialize the MediaPipe FaceLandmarker, manage webcam capture (with
    Windows-specific fixes), and expose per-frame facial landmark data
    (including eye and iris landmarks) to downstream modules (blink.py, gaze.py).

Author: CogniView Eye Tracking Module
"""

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

logger = logging.getLogger("cogniview.eye_tracking.landmarks")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# --------------------------------------------------------------------------- #
# Landmark index constants (MediaPipe FaceMesh 478-point topology)
# --------------------------------------------------------------------------- #

class EyeLandmarks:
    """MediaPipe canonical face mesh indices relevant to eye tracking."""

    # Left eye (subject's left, camera's right in mirrored view)
    LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                         173, 157, 158, 159, 160, 161, 246]
    LEFT_EYE_TOP = [159, 158, 157, 173]
    LEFT_EYE_BOTTOM = [145, 153, 154, 155]
    LEFT_EYE_LEFT_CORNER = 33
    LEFT_EYE_RIGHT_CORNER = 133
    LEFT_EYE_VERTICAL_TOP = 159
    LEFT_EYE_VERTICAL_BOTTOM = 145
    LEFT_EYE_HORIZONTAL_LEFT = 33
    LEFT_EYE_HORIZONTAL_RIGHT = 133

    # Right eye
    RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249,
                          263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_TOP = [386, 387, 388, 466]
    RIGHT_EYE_BOTTOM = [374, 373, 380, 381]
    RIGHT_EYE_LEFT_CORNER = 362
    RIGHT_EYE_RIGHT_CORNER = 263
    RIGHT_EYE_VERTICAL_TOP = 386
    RIGHT_EYE_VERTICAL_BOTTOM = 374
    RIGHT_EYE_HORIZONTAL_LEFT = 362
    RIGHT_EYE_HORIZONTAL_RIGHT = 263

    # EAR (Eye Aspect Ratio) 6-point sets, standard Soukupová & Čech convention
    LEFT_EAR_POINTS = [33, 160, 158, 133, 153, 144]
    RIGHT_EAR_POINTS = [362, 385, 387, 263, 373, 380]

    # Iris landmarks (require the 478-point refined model — indices 468-477)
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass
class FrameLandmarks:
    """Structured landmark data extracted from a single frame."""

    timestamp_ms: int
    frame_width: int
    frame_height: int
    detected: bool = False

    all_landmarks: List[Tuple[float, float, float]] = field(default_factory=list)  # normalized (x, y, z)

    left_eye_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    right_eye_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    left_ear_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    right_ear_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    left_iris_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    right_iris_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    left_iris_center: Optional[Tuple[float, float]] = None
    right_iris_center: Optional[Tuple[float, float]] = None

    def to_pixel(self, point: Tuple[float, float]) -> Tuple[int, int]:
        """Convert a normalized (x, y) point to pixel coordinates for this frame."""
        return int(point[0] * self.frame_width), int(point[1] * self.frame_height)


# --------------------------------------------------------------------------- #
# Model path resolution (fixes the Windows path-mismatch bug)
# --------------------------------------------------------------------------- #

def resolve_model_path(model_filename: str = "face_landmarker.task") -> str:
    """
    Resolve the absolute path to the FaceLandmarker model file.

    Searches, in order:
        1. COGNIVIEW_MODEL_PATH environment variable (if set)
        2. <project_root>/models/eye_tracking/<model_filename>
        3. Current working directory

    Raises:
        FileNotFoundError: if the model cannot be located.
    """
    env_path = os.environ.get("COGNIVIEW_MODEL_PATH")
    if env_path and os.path.isfile(env_path):
        logger.info("Using model path from COGNIVIEW_MODEL_PATH: %s", env_path)
        return os.path.abspath(env_path)

    module_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = .../CogniView  (src/eye_tracking/landmarks.py -> up 2 levels)
    project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
    candidate = os.path.join(project_root, "models", "eye_tracking", model_filename)

    if os.path.isfile(candidate):
        logger.info("Resolved model path: %s", candidate)
        return candidate

    # Fallback: current working directory
    cwd_candidate = os.path.join(os.getcwd(), "models", "eye_tracking", model_filename)
    if os.path.isfile(cwd_candidate):
        logger.info("Resolved model path (cwd fallback): %s", cwd_candidate)
        return cwd_candidate

    raise FileNotFoundError(
        f"Could not locate '{model_filename}'. Checked:\n"
        f"  - {candidate}\n"
        f"  - {cwd_candidate}\n"
        f"Set COGNIVIEW_MODEL_PATH env var or place the model at "
        f"models/eye_tracking/{model_filename}."
    )


# --------------------------------------------------------------------------- #
# Webcam capture with Windows-specific fixes
# --------------------------------------------------------------------------- #

def open_webcam(
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    warmup_attempts: int = 10,
    warmup_delay_s: float = 0.1,
) -> cv2.VideoCapture:
    """
    Open a webcam capture device, using the DirectShow backend on Windows to
    avoid slow/failed MSMF initialization, with a warm-up retry loop to handle
    cameras that return empty frames for the first few reads.

    Raises:
        RuntimeError: if the camera cannot be opened or warmed up.
    """
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Failed to open webcam at index {camera_index} "
            f"(backend={'CAP_DSHOW' if backend == cv2.CAP_DSHOW else 'CAP_ANY'})."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Warm-up loop: some webcams (esp. on Windows/DSHOW) return ret=False or
    # black frames for the first N reads immediately after opening.
    for attempt in range(1, warmup_attempts + 1):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            logger.info("Webcam warmed up after %d attempt(s).", attempt)
            return cap
        logger.debug("Webcam warm-up attempt %d/%d failed, retrying...", attempt, warmup_attempts)
        time.sleep(warmup_delay_s)

    cap.release()
    raise RuntimeError(
        f"Webcam opened but failed to produce a valid frame after "
        f"{warmup_attempts} warm-up attempts."
    )


# --------------------------------------------------------------------------- #
# FaceLandmarker wrapper
# --------------------------------------------------------------------------- #

class LandmarkExtractor:
    """
    Wraps MediaPipe's FaceLandmarker (Tasks API) for real-time video use.

    Usage:
        extractor = LandmarkExtractor()
        extractor.initialize()
        frame_landmarks = extractor.process(frame_bgr, timestamp_ms)
        extractor.close()

    Or as a context manager:
        with LandmarkExtractor() as extractor:
            frame_landmarks = extractor.process(frame_bgr, timestamp_ms)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        running_mode: str = "VIDEO",
    ):
        self._model_path = model_path or resolve_model_path()
        self._num_faces = num_faces
        self._min_face_detection_confidence = min_face_detection_confidence
        self._min_face_presence_confidence = min_face_presence_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._running_mode_str = running_mode.upper()
        self._landmarker: Optional[mp_vision.FaceLandmarker] = None
        self._initialized = False

    def initialize(self) -> None:
        """Build the FaceLandmarker instance. Must be called before process()."""
        if self._initialized:
            logger.warning("LandmarkExtractor already initialized; skipping.")
            return

        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(f"FaceLandmarker model not found at: {self._model_path}")

        running_mode_map = {
            "IMAGE": mp_vision.RunningMode.IMAGE,
            "VIDEO": mp_vision.RunningMode.VIDEO,
            "LIVE_STREAM": mp_vision.RunningMode.LIVE_STREAM,
        }
        if self._running_mode_str not in running_mode_map:
            raise ValueError(f"Invalid running_mode: {self._running_mode_str}")

        base_options = mp_python.BaseOptions(model_asset_path=self._model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode_map[self._running_mode_str],
            num_faces=self._num_faces,
            min_face_detection_confidence=self._min_face_detection_confidence,
            min_face_presence_confidence=self._min_face_presence_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._initialized = True
        logger.info(
            "FaceLandmarker initialized (mode=%s, model=%s).",
            self._running_mode_str, self._model_path
        )

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FrameLandmarks:
        """
        Run landmark detection on a single BGR frame.

        Args:
            frame_bgr: OpenCV frame in BGR format.
            timestamp_ms: Monotonically increasing timestamp in milliseconds
                          (required for VIDEO mode; must strictly increase
                          between calls).

        Returns:
            FrameLandmarks with detected=False if no face was found.
        """
        if not self._initialized or self._landmarker is None:
            raise RuntimeError("LandmarkExtractor.initialize() must be called before process().")

        h, w = frame_bgr.shape[:2]
        result = FrameLandmarks(timestamp_ms=timestamp_ms, frame_width=w, frame_height=h)

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if self._running_mode_str == "VIDEO":
            detection_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        elif self._running_mode_str == "IMAGE":
            detection_result = self._landmarker.detect(mp_image)
        else:
            raise RuntimeError("LIVE_STREAM mode requires detect_async(); not implemented here.")

        if not detection_result.face_landmarks:
            return result

        face_landmarks = detection_result.face_landmarks[0]
        result.detected = True
        result.all_landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks]

        result.left_eye_points = self._extract_points(face_landmarks, EyeLandmarks.LEFT_EYE_CONTOUR)
        result.right_eye_points = self._extract_points(face_landmarks, EyeLandmarks.RIGHT_EYE_CONTOUR)
        result.left_ear_points = self._extract_points(face_landmarks, EyeLandmarks.LEFT_EAR_POINTS)
        result.right_ear_points = self._extract_points(face_landmarks, EyeLandmarks.RIGHT_EAR_POINTS)

        # Iris landmarks only exist if the model outputs 478 points (with refinement).
        if len(face_landmarks) >= 478:
            result.left_iris_points = self._extract_points(face_landmarks, EyeLandmarks.LEFT_IRIS)
            result.right_iris_points = self._extract_points(face_landmarks, EyeLandmarks.RIGHT_IRIS)
            lm_l = face_landmarks[EyeLandmarks.LEFT_IRIS_CENTER]
            lm_r = face_landmarks[EyeLandmarks.RIGHT_IRIS_CENTER]
            result.left_iris_center = (lm_l.x, lm_l.y)
            result.right_iris_center = (lm_r.x, lm_r.y)

        return result

    @staticmethod
    def _extract_points(face_landmarks, indices: List[int]) -> np.ndarray:
        """Extract normalized (x, y) coordinates for the given landmark indices."""
        return np.array([[face_landmarks[i].x, face_landmarks[i].y] for i in indices], dtype=np.float32)

    def close(self) -> None:
        """Release the FaceLandmarker resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        self._initialized = False
        logger.info("LandmarkExtractor closed.")

    def __enter__(self) -> "LandmarkExtractor":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Standalone smoke test (run: python -m src.eye_tracking.landmarks)
# --------------------------------------------------------------------------- #

def _run_live_demo() -> None:
    """Quick manual test: opens the webcam and draws detected eye/iris landmarks."""
    cap = open_webcam()
    extractor = LandmarkExtractor()
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
            landmarks = extractor.process(frame, timestamp_ms)

            if landmarks.detected:
                for pt in landmarks.left_eye_points:
                    cv2.circle(frame, landmarks.to_pixel(pt), 1, (0, 255, 0), -1)
                for pt in landmarks.right_eye_points:
                    cv2.circle(frame, landmarks.to_pixel(pt), 1, (0, 255, 0), -1)
                if landmarks.left_iris_center:
                    cv2.circle(frame, landmarks.to_pixel(landmarks.left_iris_center), 2, (0, 0, 255), -1)
                if landmarks.right_iris_center:
                    cv2.circle(frame, landmarks.to_pixel(landmarks.right_iris_center), 2, (0, 0, 255), -1)

            cv2.imshow("CogniView - Landmark Extraction", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_live_demo()