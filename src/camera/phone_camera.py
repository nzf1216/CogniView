"""
phone_camera.py
CameraSource implementation for a phone acting as an IP/network camera.

Single Responsibility:
    Connect to an MJPEG/RTSP stream served by a phone camera app (e.g.
    "IP Webcam" on Android, or an equivalent iOS app / a small companion app
    bundled with CogniView) and expose it through the same CameraSource
    interface as WebcamCamera.

Why this needs more than cv2.VideoCapture(url):
    Wi-Fi phone streams are much less reliable than a local webcam -- frames
    get dropped, the connection can stall or drop entirely, and MJPEG/RTSP
    backends sometimes need a moment to recover. This class adds bounded
    reconnect attempts around VideoCapture so a single missed frame (or a
    brief Wi-Fi hiccup) doesn't take down an entire recording session.

Author: CogniView Camera Module
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from src.camera.base_camera import CameraSource

logger = logging.getLogger("cogniview.camera.phone")


class PhoneCamera(CameraSource):
    """
    Network camera source for a phone streaming over Wi-Fi.

    Args:
        stream_url: Full URL of the MJPEG/RTSP stream, e.g.
            "http://192.168.1.23:8080/video" (IP Webcam default) or
            "rtsp://192.168.1.23:8554/live".
        max_read_failures: consecutive failed reads tolerated before
            attempting a full reconnect.
        max_reconnect_attempts: reconnect attempts before read() gives up
            and returns (False, None, 0) for good.

    Usage:
        with PhoneCamera("http://192.168.1.23:8080/video") as cam:
            ret, frame, timestamp_ms = cam.read()
    """

    def __init__(
        self,
        stream_url: str,
        width: int = 640,
        height: int = 480,
        connect_timeout_s: float = 5.0,
        max_read_failures: int = 5,
        max_reconnect_attempts: int = 3,
        reconnect_delay_s: float = 1.0,
        mirror: bool = False,
    ):
        super().__init__(width=width, height=height)
        self.stream_url = stream_url
        self.connect_timeout_s = connect_timeout_s
        self.max_read_failures = max_read_failures
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay_s = reconnect_delay_s
        self.mirror = mirror

        self._cap: Optional[cv2.VideoCapture] = None
        self._start_time: Optional[float] = None
        self._last_timestamp_ms: int = -1
        self._consecutive_failures = 0

    def open(self) -> None:
        if self._opened:
            logger.warning("PhoneCamera already open; skipping.")
            return
        self._cap = self._connect()
        self._opened = True
        self._start_time = time.time()
        self._last_timestamp_ms = -1
        self._consecutive_failures = 0

    def _connect(self) -> cv2.VideoCapture:
        logger.info("Connecting to phone camera stream: %s", self.stream_url)
        cap = cv2.VideoCapture(self.stream_url)
        deadline = time.time() + self.connect_timeout_s
        while time.time() < deadline:
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
                logger.info("Connected to phone camera stream.")
                return cap
            time.sleep(0.1)
        cap.release()
        raise RuntimeError(
            f"Failed to connect to phone camera stream at '{self.stream_url}' "
            f"within {self.connect_timeout_s}s. Check that the phone app is "
            f"running and both devices are on the same network."
        )

    def _reconnect(self) -> bool:
        """Attempt to re-establish the stream after repeated read failures."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        for attempt in range(1, self.max_reconnect_attempts + 1):
            logger.warning(
                "Phone camera reconnect attempt %d/%d...",
                attempt, self.max_reconnect_attempts,
            )
            try:
                self._cap = self._connect()
                self._consecutive_failures = 0
                return True
            except RuntimeError as exc:
                logger.warning("Reconnect attempt %d failed: %s", attempt, exc)
                time.sleep(self.reconnect_delay_s)

        logger.error("Phone camera reconnect exhausted after %d attempts.", self.max_reconnect_attempts)
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        if not self._opened or self._cap is None:
            raise RuntimeError("PhoneCamera.open() must be called before read().")

        ret, frame = self._cap.read()

        if not ret or frame is None or frame.size == 0:
            self._consecutive_failures += 1
            logger.debug(
                "Phone camera read failed (%d/%d consecutive).",
                self._consecutive_failures, self.max_read_failures,
            )
            if self._consecutive_failures >= self.max_read_failures:
                if not self._reconnect():
                    self._opened = False
                    return False, None, 0
            return False, None, 0

        self._consecutive_failures = 0

        if self.mirror:
            frame = cv2.flip(frame, 1)

        timestamp_ms = int((time.time() - self._start_time) * 1000)
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
