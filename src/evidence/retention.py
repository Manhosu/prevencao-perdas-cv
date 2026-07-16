"""Purga periódica da evidência antiga, para o disco da loja não encher.

É rotina de manutenção: nunca pode derrubar o sistema. Qualquer erro é logado
e engolido — a próxima passada tenta de novo."""
from __future__ import annotations

import logging
import threading
import time

from src.config.settings import EvidenceConfig
from src.storage.db import Database

log = logging.getLogger(__name__)


class RetentionJob:
    def __init__(self, db: Database, cfg: EvidenceConfig,
                 interval_seconds: float = 6 * 3600) -> None:
        self.db = db
        self.cfg = cfg
        self.interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_once(self) -> int:
        try:
            removidos = self.db.purge_older_than(self.cfg.retention_days)
            if removidos:
                log.info("limpeza: %d arquivo(s) de evidencia com mais de %d dias removido(s)",
                         len(removidos), self.cfg.retention_days)
            return len(removidos)
        except Exception:
            log.exception("falha na limpeza de evidencias (tentara de novo depois)")
            return 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="retencao", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.interval)
