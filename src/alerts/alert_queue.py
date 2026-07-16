"""Fila de alertas em thread própria.

O Telegram é lento e cai. Se o envio fosse síncrono no pipeline, um timeout de
20s travaria a inferência de TODAS as câmeras. A fila isola isso: o pipeline só
enfileira e segue. Retry com backoff, rate-limit local (o Telegram bloqueia
acima de ~20 msg/min) e — importante — o evento já está no banco, então uma
falha de envio nunca perde a evidência."""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class _Item:
    event_id: int | None
    image_path: Path | None
    clip_path: Path | None
    caption: str
    text: str | None = None
    tentativas: int = 0


class AlertQueue:
    def __init__(self, sender, db, rate_limit_per_min: int = 15,
                 max_retries: int = 3, backoff_base: float = 1.0) -> None:
        self.sender = sender
        self.db = db
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._min_interval = 60.0 / max(1, rate_limit_per_min)
        self._q: queue.Queue[_Item] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_send = 0.0
        self._lock = threading.Lock()
        self._sent = 0
        self._failed = 0

    @property
    def sent_count(self) -> int:
        with self._lock:
            return self._sent

    @property
    def failed_count(self) -> int:
        with self._lock:
            return self._failed

    @property
    def pending(self) -> int:
        # unfinished_tasks (nao qsize!) conta itens ainda em processamento:
        # qsize() cai para 0 assim que a thread da fila faz o get() do item,
        # mesmo que ele ainda esteja esperando rate-limit/backoff ou seja
        # reenfileirado para nova tentativa. unfinished_tasks so decresce
        # quando task_done() e chamado, refletindo o trabalho de fato.
        return self._q.unfinished_tasks

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="alertas", daemon=True)
        self._thread.start()

    def stop(self, drain_seconds: float = 3.0) -> None:
        fim = time.monotonic() + drain_seconds
        while self._q.unfinished_tasks > 0 and time.monotonic() < fim:
            time.sleep(0.02)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def enqueue(self, event_id, image_path, clip_path, caption: str) -> None:
        self._q.put(_Item(event_id, Path(image_path) if image_path else None,
                          Path(clip_path) if clip_path else None, caption))

    def enqueue_system(self, text: str) -> None:
        self._q.put(_Item(None, None, None, "", text=text))

    def _respeita_rate_limit(self) -> None:
        espera = self._min_interval - (time.monotonic() - self._last_send)
        if espera > 0:
            self._stop.wait(espera)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._processa(item)
            except Exception:
                # nenhuma falha de envio pode derrubar a thread de alertas
                log.exception("erro inesperado ao processar alerta")
            finally:
                self._q.task_done()

    def _processa(self, item: _Item) -> None:
        self._respeita_rate_limit()
        ok = False
        try:
            if item.text is not None:
                ok = bool(self.sender.send_message(item.text))
            elif item.image_path is not None:
                ok = bool(self.sender.send_photo(item.image_path, item.caption))
            elif item.clip_path is not None:
                ok = bool(self.sender.send_video(item.clip_path, item.caption))
        except Exception as e:
            log.warning("envio falhou: %s", e)
            ok = False
        self._last_send = time.monotonic()

        if ok:
            with self._lock:
                self._sent += 1
            if item.event_id is not None:
                try:
                    self.db.mark_sent(item.event_id)
                except Exception:
                    log.exception("falha ao marcar evento %s como enviado", item.event_id)
            return

        item.tentativas += 1
        if item.tentativas <= self.max_retries:
            self._stop.wait(self.backoff_base * (2 ** (item.tentativas - 1)))
            self._q.put(item)  # re-enfileira para nova tentativa
        else:
            with self._lock:
                self._failed += 1
            log.warning("desisti de enviar o alerta apos %s tentativas "
                        "(o evento continua salvo no banco)", item.tentativas)
