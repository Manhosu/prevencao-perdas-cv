"""Escolhe qual câmera o próximo worker atende.

Câmera com pessoa recente recebe mais fatias de CPU; câmera de corredor vazio
é visitada com menos frequência. É isso que faz 10 câmeras numa loja calma
custarem menos que 3 câmeras num sábado lotado."""
from __future__ import annotations

import threading


class Scheduler:
    def __init__(
        self,
        camera_names: list[str],
        active_boost: float = 3.0,
        active_window: float = 5.0,
    ) -> None:
        self.cameras = list(camera_names)
        self.active_boost = active_boost
        self.active_window = active_window
        self._lock = threading.Lock()
        self._last_activity: dict[str, float] = {}
        # -inf, não 0.0: câmera nunca atendida tem fome infinita. Com 0.0, todas
        # empatariam no primeiro ciclo e a mesma câmera seria escolhida sempre.
        self._last_served: dict[str, float] = {c: float("-inf") for c in self.cameras}

    def mark_activity(self, camera: str, ts: float) -> None:
        """Chamado quando o worker encontrou pessoa nesta câmera."""
        with self._lock:
            self._last_activity[camera] = ts

    def mark_served(self, camera: str, now: float) -> None:
        with self._lock:
            self._last_served[camera] = now

    def next_camera(self, now: float) -> str | None:
        """A câmera com maior "fome": tempo desde a última vez que foi
        atendida, multiplicado pelo boost se houve pessoa recentemente."""
        with self._lock:
            if not self.cameras:
                return None
            best, best_score = None, float("-inf")
            for cam in self.cameras:
                starving = now - self._last_served.get(cam, 0.0)
                active = now - self._last_activity.get(cam, float("-inf"))
                weight = self.active_boost if active <= self.active_window else 1.0
                score = starving * weight
                if score > best_score:
                    best, best_score = cam, score
            return best
