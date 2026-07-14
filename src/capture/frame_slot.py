"""Slot de um frame só. Frame velho é descartado, nunca enfileirado:
em vídeo ao vivo, atraso acumulado é pior que frame perdido."""
from __future__ import annotations

import threading

from src.core.types import Frame


class LatestFrameSlot:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._dropped = 0

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def put(self, frame: Frame) -> None:
        with self._lock:
            if self._frame is not None:
                self._dropped += 1
            self._frame = frame

    def get(self) -> Frame | None:
        with self._lock:
            f, self._frame = self._frame, None
            return f

    def peek(self) -> Frame | None:
        with self._lock:
            return self._frame
