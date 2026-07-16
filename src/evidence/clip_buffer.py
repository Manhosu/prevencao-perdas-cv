"""Buffer circular de frames por câmera.

O clipe de evidência precisa dos segundos ANTES do alerta (o gesto inteiro),
mas só sabemos do alerta depois que ele acontece. Guardamos uma janela curta
dos frames recentes; quando dispara, o "antes" já está na mão. Limitado por
tempo — a memória não cresce."""
from __future__ import annotations

import threading
from collections import deque

import numpy as np


class ClipBuffer:
    def __init__(self, seconds: float, fps_hint: float = 5.0) -> None:
        self.seconds = seconds
        # teto de itens com folga, para o deque nunca crescer sem limite
        maxlen = max(2, int(seconds * fps_hint * 2) + 2)
        self._buf: deque[tuple[float, np.ndarray]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, frame_bgr: np.ndarray, ts: float) -> None:
        # copia: a thread de captura reusa o array, e sem cópia o clipe sairia
        # todo com a mesma imagem
        with self._lock:
            self._buf.append((ts, frame_bgr.copy()))
            self._drop_old()

    def _drop_old(self) -> None:
        if not self._buf:
            return
        cutoff = self._buf[-1][0] - self.seconds
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def frames_between(self, t0: float, t1: float) -> list[tuple[float, np.ndarray]]:
        with self._lock:
            return [(ts, img) for ts, img in self._buf if t0 <= ts <= t1]

    @property
    def newest_ts(self) -> float | None:
        with self._lock:
            return self._buf[-1][0] if self._buf else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
