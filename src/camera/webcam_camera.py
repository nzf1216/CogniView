"""
webcam_camera.py
CameraSource implementation for a locally attached webcam.

Single Responsibility:
    Wrap cv2.VideoCapture for a physical webcam, handling backend selection
    (DirectShow on Windows to avoid slow/failed MSMF init), a warm-up retry
    loop for cameras that return empty frames right after opening, and
    monotonically increasing millisecond timestamps derived from wall-clock
    time (so the same recording pipeline works whether frames come from a
    webcam or a phone stream).

Author: CogniView Camera Module
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from src.camera.base_camera import CameraSource

logger = logging.getLogger("cogniview.camera.webcam")


class WebcamCamera(CameraSource):
    """
    Local webcam camera source.

    Usage:
        with WebcamCamera(camera_index=0) as cam:
            ret, frame, timestamp_ms = cam.read()
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        warmup_attempts: int = 10,
        warmup_delay_s: float = 0.1,
        mirror: bool = True,
    ):
        super().__init__(width=width, height=height)
        self.camera_index = camera_index
        self.warmup_attempts = warmup_attempts
        self.warmup_delay_s = warmup_delay_s
        self.mirror = mirror

        self._cap: Optional[cv2.VideoCapture] = None
        self._start_time: Optional[float] = None
        self._last_timestamp_ms: int = -1

    def open(self) -> None:
        if self._opened:
            logger.warning("WebcamCamera already open; skipping.")
            return

        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.camera_index, backend)

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Failed to open webcam at index {self.camera_index} "
                f"(backend={'CAP_DSHOW' if backend == cv2.CAP_DSHOW else 'CAP_ANY'})."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)

        for attempt in range(1, self.warmup_attempts + 1):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                logger.info("Webcam warmed up after %d attempt(s).", attempt)
                self._cap = cap
                self._opened = True
                self._start_time = time.time()
                self._last_timestamp_ms = -1
                return
            logger.debug("Webcam warm-up attempt %d/%d failed, retrying...", attempt, self.warmup_attempts)
            time.sleep(self.warmup_delay_s)

        cap.release()
        raise RuntimeError(
            f"Webcam opened but failed to produce a valid frame after "
            f"{self.warmup_attempts} warm-up attempts."
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        if not self._opened or self._cap is None:
            raise RuntimeError("WebcamCamera.open() must be called before read().")

        ret, frame = self._cap.read()
        if not ret or frame is None or frame.size == 0:
            return False, None, 0

        if self.mirror:
            frame = cv2.flip(frame, 1)

        timestamp_ms = int((time.time() - self._start_time) * 1000)
        # Guard against duplicate/non-increasing timestamps (can happen on
        # very fast loops with coarse clock resolution) -- MediaPipe VIDEO
        # mode requires strictly increasing timestamps per frame.
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

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
        if self._cap is None:
            return None
        value = self._cap.get(cv2.CAP_PROP_FPS)
        return value if value and value > 0 else None
