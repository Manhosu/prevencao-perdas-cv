"""Watchdog: vigia o heartbeat de cada câmera.

É o pior tipo de falha — o DVR reinicia, um cabo solta, a câmera para de
mandar frame, e o sistema segue "rodando" sem vigiar nada. Sem este aviso, a
loja acha que está protegida e não está."""
from __future__ import annotations

import logging
import threading
import time

from src.config.settings import WatchdogConfig
from src.core.types import CameraState
from src.storage.db import Database

log = logging.getLogger(__name__)

CHECK_INTERVAL = 2.0


class Watchdog:
    def __init__(self, threads: dict, db: Database, cfg: WatchdogConfig,
                 alert_queue=None, clock=time.monotonic) -> None:
        self.threads = threads
        self.db = db
        self.cfg = cfg
        self.alert_queue = alert_queue
        self.clock = clock
        self.states: dict[str, CameraState] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def check_once(self) -> None:
        agora = self.clock()
        for nome, cam in self.threads.items():
            last = getattr(cam, "last_frame_ts", None)
            vivo = last is not None and (agora - last) <= self.cfg.offline_after_seconds
            novo = CameraState.ONLINE if vivo else CameraState.OFFLINE
            anterior = self.states.get(nome)
            if novo == anterior:
                continue  # sem mudanca: nao registra nem avisa de novo

            self.states[nome] = novo
            try:
                self.db.upsert_camera_status(nome, novo, str(last) if last else None)
            except Exception:
                log.exception("falha ao registrar status da camera '%s'", nome)

            # primeira leitura: registra o estado sem alarmar quem acabou de subir
            if anterior is None and novo == CameraState.ONLINE:
                continue
            self._avisa(nome, novo, anterior)

    def _avisa(self, nome: str, novo: CameraState, anterior) -> None:
        if not self.cfg.notify or self.alert_queue is None:
            return
        if novo == CameraState.OFFLINE:
            texto = (f"🔴 Câmera '{nome}' está OFFLINE — o sistema parou de vigiar "
                     f"esta câmera. Verifique o DVR, o cabo ou a rede.")
        else:
            texto = f"🟢 Câmera '{nome}' voltou ao normal."
        try:
            self.alert_queue.enqueue_system(texto)
        except Exception:
            log.exception("falha ao enfileirar aviso de camera")

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.check_once()
            except Exception:
                log.exception("erro no watchdog")  # nunca derruba a thread
            self._stop.wait(CHECK_INTERVAL)
