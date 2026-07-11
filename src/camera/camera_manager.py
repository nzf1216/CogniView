"""
camera_manager.py
Factory / lifecycle manager for CogniView camera sources.

Single Responsibility:
    Turn a small config (source type + connection details) into a ready-to-use
    CameraSource, so callers (record_session.py, live demo scripts, the
    eventual FastAPI backend) never construct WebcamCamera / PhoneCamera
    directly and don't need to know about backend-specific arguments.

Also provides VideoFileCamera, a CameraSource over a pre-recorded video file.
This isn't a "real" acquisition backend, but it's what makes the rest of the
pipeline testable without physical hardware (CI, sandboxes, unit tests) --
recording sessions, feature extraction, and CSV export all run through the
exact same CameraSource interface regardless of where frames come from.

Author: CogniView Camera Module
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from src.camera.base_camera import CameraSource
from src.camera.webcam_camera import WebcamCamera
from src.camera.phone_camera import PhoneCamera

logger = logging.getLogger("cogniview.camera.manager")


class VideoFileCamera(CameraSource):
    """
    CameraSource backed by a video file on disk.

    Timestamps are derived from the file's own frame rate (frame_index / fps),
    not wall-clock time, so playback speed doesn't affect the timing fed to
    downstream MediaPipe / blink / gaze processing.
    """

    def __init__(self, path: str, width: int = 640, height: int = 480):
        super().__init__(width=width, height=height)
        self.path = path
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index = 0
        self._fps: float = 30.0

    def open(self) -> None:
        if self._opened:
            logger.warning("VideoFileCamera already open; skipping.")
            return
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open video file: {self.path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 0 else 30.0
        self._cap = cap
        self._opened = True
        self._frame_index = 0
        logger.info("Opened video file '%s' at %.2f fps.", self.path, self._fps)

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        if not self._opened or self._cap is None:
            raise RuntimeError("VideoFileCamera.open() must be called before read().")
        ret, frame = self._cap.read()
        if not ret or frame is None or frame.size == 0:
            # End of file (VideoCapture keeps reporting ret=False forever
            # after this point) -- mark closed so callers relying on
            # is_opened to detect end-of-stream actually stop.
            self._opened = False
            return False, None, 0
        timestamp_ms = int(round(self._frame_index * (1000.0 / self._fps)))
        self._frame_index += 1
        return True, frame, timestamp_ms

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._opened = False

    @property
    def is_opened(self) -> bool:
        return self._opened and self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> Optional[float]:
        return self._fps


class CameraManager:
    """
    Constructs a CameraSource from a small, serializable config so callers
    don't need to import backend-specific classes directly.

    Example:
        cam = CameraManager.create(source_type="webcam", camera_index=0)
        cam = CameraManager.create(source_type="phone", stream_url="http://192.168.1.23:8080/video")
        cam = CameraManager.create(source_type="video_file", path="session.mp4")
    """

    SOURCE_TYPES = ("webcam", "phone", "video_file")

    @staticmethod
    def create(source_type: str, **kwargs) -> CameraSource:
        if source_type == "webcam":
            return WebcamCamera(
                camera_index=kwargs.get("camera_index", 0),
                width=kwargs.get("width", 640),
                height=kwargs.get("height", 480),
                warmup_attempts=kwargs.get("warmup_attempts", 10),
                warmup_delay_s=kwargs.get("warmup_delay_s", 0.1),
                mirror=kwargs.get("mirror", True),
            )
        elif source_type == "phone":
            stream_url = kwargs.get("stream_url")
            if not stream_url:
                raise ValueError("source_type='phone' requires a 'stream_url' argument.")
            return PhoneCamera(
                stream_url=stream_url,
                width=kwargs.get("width", 640),
                height=kwargs.get("height", 480),
                connect_timeout_s=kwargs.get("connect_timeout_s", 5.0),
                max_read_failures=kwargs.get("max_read_failures", 5),
                max_reconnect_attempts=kwargs.get("max_reconnect_attempts", 3),
                reconnect_delay_s=kwargs.get("reconnect_delay_s", 1.0),
                mirror=kwargs.get("mirror", False),
            )
        elif source_type == "video_file":
            path = kwargs.get("path")
            if not path:
                raise ValueError("source_type='video_file' requires a 'path' argument.")
            return VideoFileCamera(
                path=path,
                width=kwargs.get("width", 640),
                height=kwargs.get("height", 480),
            )
        else:
            raise ValueError(
                f"Unknown source_type '{source_type}'. Must be one of {CameraManager.SOURCE_TYPES}."
            )
