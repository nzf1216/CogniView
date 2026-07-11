"""
base_camera.py
Abstract interface for all CogniView camera sources.

Single Responsibility:
    Define the contract that every camera backend (webcam, phone/IP camera,
    pre-recorded video file) must implement, so that
    src/eye_tracking/record_session.py and any live-demo script can consume
    frames without caring where they came from.

Design notes:
    - Timestamps are the camera's job, not the caller's. Different sources
      have very different timing characteristics (a local webcam is close to
      wall-clock; a phone camera streamed over Wi-Fi has variable latency and
      can drop frames), so each backend is responsible for producing a
      monotonically increasing millisecond timestamp for every frame it
      returns. MediaPipe's VIDEO running mode requires strictly increasing
      timestamps, so `read()` must never go backwards.
    - `read()` returns a simple (ret, frame, timestamp_ms) tuple rather than
      raising on a dropped frame, mirroring cv2.VideoCapture.read() semantics
      so existing call sites are easy to port.

Author: CogniView Camera Module
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("cogniview.camera.base")


class CameraSource(ABC):
    """
    Abstract base class for a source of BGR video frames with timestamps.

    Usage:
        with SomeCameraSource(...) as cam:
            while True:
                ret, frame, timestamp_ms = cam.read()
                if not ret:
                    break
                ...

    Subclasses must implement `open`, `read`, `release`, and `is_opened`.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.requested_width = width
        self.requested_height = height
        self._opened = False

    # --------------------------------------------------------------- #
    # Required interface
    # --------------------------------------------------------------- #

    @abstractmethod
    def open(self) -> None:
        """
        Acquire the underlying capture device / stream.

        Raises:
            RuntimeError: if the source cannot be opened or fails warm-up.
        """
        raise NotImplementedError

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """
        Grab the next frame.

        Returns:
            (ret, frame, timestamp_ms)
              ret: False if no frame could be produced (end of stream, or a
                   transient failure the caller should treat as "skip this
                   iteration").
              frame: BGR np.ndarray, or None if ret is False.
              timestamp_ms: monotonically increasing integer millisecond
                   timestamp. Meaningless (0) if ret is False.
        """
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        """Release any underlying resources. Safe to call multiple times."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_opened(self) -> bool:
        """Whether the source is currently open and ready to be read from."""
        raise NotImplementedError

    # --------------------------------------------------------------- #
    # Shared helpers
    # --------------------------------------------------------------- #

    @property
    def fps(self) -> Optional[float]:
        """
        Nominal frames-per-second of the source, if known. Returns None when
        the source doesn't expose a reliable value (e.g. a live phone stream
        whose effective rate depends on network conditions) -- callers should
        fall back to measuring the achieved rate themselves rather than
        trusting this blindly.
        """
        return None

    def __enter__(self) -> "CameraSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
