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
        # Câmeras atualmente reivindicadas por algum worker (entre next_camera()
        # e o release() correspondente). É isso que impede que a MESMA câmera
        # seja entregue a dois workers ao mesmo tempo — o cenário fatal era
        # 1 câmera + workers=2 (o Tracker por câmera não é thread-safe: dois
        # workers chamando update() concorrentemente corrompiam o estado).
        self._in_flight: set[str] = set()

    def mark_activity(self, camera: str, ts: float) -> None:
        """Chamado quando o worker encontrou pessoa nesta câmera."""
        with self._lock:
            self._last_activity[camera] = ts

    def mark_served(self, camera: str, now: float) -> None:
        with self._lock:
            self._last_served[camera] = now

    def next_camera(self, now: float) -> str | None:
        """A câmera com maior "fome" (tempo desde a última vez que foi
        atendida, multiplicado pelo boost se houve pessoa recentemente) entre
        as que NÃO estão em processamento por outro worker agora.

        A escolha e a reivindicação (marcar in-flight) acontecem atomicamente
        sob o mesmo lock: se todas as câmeras candidatas já estiverem
        in-flight, devolve None (o worker chamador dorme e tenta de novo —
        "nada para fazer agora", igual a slot vazio). O chamador DEVE liberar
        a câmera com `release()` depois de processá-la (ou de descobrir que
        não havia frame novo), mesmo em caso de exceção."""
        with self._lock:
            if not self.cameras:
                return None
            best, best_score = None, float("-inf")
            for cam in self.cameras:
                if cam in self._in_flight:
                    continue
                starving = now - self._last_served.get(cam, 0.0)
                active = now - self._last_activity.get(cam, float("-inf"))
                weight = self.active_boost if active <= self.active_window else 1.0
                score = starving * weight
                if score > best_score:
                    best, best_score = cam, score
            if best is not None:
                self._in_flight.add(best)
            return best

    def release(self, camera: str) -> None:
        """Devolve a câmera reivindicada por `next_camera()`, tornando-a
        elegível de novo para o próximo worker que chamar `next_camera()`."""
        with self._lock:
            self._in_flight.discard(camera)
