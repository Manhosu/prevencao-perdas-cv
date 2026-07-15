"""Pool pequeno de workers de inferência.

Não rodar pose em 5 streams ao mesmo tempo: com 1-2 workers, o hardware fraco
degrada suavemente (o FPS efetivo cai) em vez de travar."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from src.capture.frame_slot import LatestFrameSlot
from src.core.types import Frame
from src.inference.scheduler import Scheduler

log = logging.getLogger(__name__)

IDLE_SLEEP = 0.02  # nada para processar: não queimar CPU em busy-wait


class WorkerPool:
    def __init__(
        self,
        slots: dict[str, LatestFrameSlot],
        scheduler: Scheduler,
        process: Callable[[Frame], bool],
        workers: int = 2,
    ) -> None:
        self.slots = slots
        self.scheduler = scheduler
        self.process = process
        self.workers = max(1, workers)
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._processed = 0

    @property
    def processed(self) -> int:
        with self._lock:
            return self._processed

    def start(self) -> None:
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._run, name=f"infer-{i}", daemon=True)
            for i in range(self.workers)
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)
        self._threads = []

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            cam = self.scheduler.next_camera(now)
            if cam is None:
                self._stop.wait(IDLE_SLEEP)
                continue

            # A partir daqui a câmera está reivindicada (in-flight) no
            # Scheduler: nenhum outro worker vai recebê-la até o release()
            # abaixo. Isso é o que impede dois workers chamarem
            # Tracker.update() da mesma câmera ao mesmo tempo — por isso o
            # release precisa acontecer SEMPRE, mesmo se o slot estiver vazio
            # ou se process() lançar uma exceção.
            try:
                frame = self.slots[cam].get()
                self.scheduler.mark_served(cam, now)
                if frame is None:
                    self._stop.wait(IDLE_SLEEP)
                    continue

                try:
                    had_person = self.process(frame)
                    if had_person:
                        self.scheduler.mark_activity(cam, now)
                except Exception:
                    # Um erro de inferência num frame não pode derrubar o worker —
                    # o sistema roda 24/7 numa loja, sem ninguém olhando.
                    log.exception("erro processando frame da câmera '%s'", cam)
                finally:
                    with self._lock:
                        self._processed += 1
            finally:
                self.scheduler.release(cam)
