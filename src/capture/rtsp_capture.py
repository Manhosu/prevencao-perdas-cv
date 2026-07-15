"""Captura RTSP: uma thread por câmera. Reconexão automática com backoff —
é a falha nº 1 em campo (DVR reinicia, cabo solta, rede oscila) e o sistema
tem que voltar sozinho, sem ninguém perceber que caiu."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import cv2

from src.capture.frame_slot import LatestFrameSlot
from src.config.settings import CameraConfig
from src.core.types import CameraState, Frame

log = logging.getLogger(__name__)

MAX_READ_FAILURES = 5  # leituras falhas seguidas antes de considerar o stream morto


def _open_rtsp(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    # buffer mínimo: queremos o frame mais novo, não a fila do driver
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Timeouts de abertura/leitura: sem eles, uma conexão TCP que não fecha
    # limpo (cabo meio solto, switch congelado) faz cap.read() travar para
    # sempre — sem lançar exceção, sem devolver ok=False — e a reconexão
    # nunca dispara. É o pior tipo de falha de campo: parece que está tudo
    # funcionando. Só afetam o backend FFMPEG real; os dublês de teste não
    # usam cv2.VideoCapture, então não mudam de comportamento.
    open_timeout = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    if open_timeout is not None:
        cap.set(open_timeout, 10000)  # 10s para abrir a conexão
    read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
    if read_timeout is not None:
        cap.set(read_timeout, 10000)  # 10s para uma leitura individual
    return cap


class CameraThread:
    def __init__(
        self,
        camera: CameraConfig,
        slot: LatestFrameSlot,
        backoff_max: float = 30.0,
        open_capture: Callable[[str], object] | None = None,
    ) -> None:
        self.camera = camera
        self.slot = slot
        self.backoff_max = backoff_max
        self._open = open_capture or _open_rtsp
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state = CameraState.OFFLINE
        self._last_frame_ts: float | None = None
        self._seq = 0
        self._fps_window: list[float] = []

    # --- estado observável ---

    @property
    def state(self) -> CameraState:
        with self._lock:
            return self._state

    @property
    def last_frame_ts(self) -> float | None:
        with self._lock:
            return self._last_frame_ts

    @property
    def effective_fps(self) -> float:
        with self._lock:
            if len(self._fps_window) < 2:
                return 0.0
            span = self._fps_window[-1] - self._fps_window[0]
            return (len(self._fps_window) - 1) / span if span > 0 else 0.0

    def _set_state(self, state: CameraState) -> None:
        with self._lock:
            if self._state != state:
                log.info("câmera '%s': %s", self.camera.name, state.value)
                self._state = state

    # --- ciclo de vida ---

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"capture-{self.camera.name}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        self._set_state(CameraState.OFFLINE)

    def _next_backoff(self, current: float) -> float:
        return min(self.backoff_max, max(1.0, current * 2))

    # --- chamadas de risco (I/O do driver) ---
    #
    # Uma câmera nunca pode derrubar a thread de captura: o sistema roda 24/7
    # sem supervisão, e um driver instável (frame corrompido, exceção do
    # FFmpeg) não pode tirar a loja de vigilância. Em vez de um único
    # try/except largo em torno do ciclo inteiro (abrir → ler → liberar) —
    # que fazia cap.release() ser chamado duas vezes quando o release() do
    # caminho normal lançava exceção, já que o except externo via cap is not
    # None e liberava de novo — cada chamada de risco é protegida
    # individualmente, no mesmo padrão usado no resto do projeto: envolver só
    # a chamada de risco, não o corpo inteiro do laço.

    def _safe_open(self) -> object | None:
        cap = None
        opened = False
        try:
            cap = self._open(self.camera.rtsp_url)
            opened = cap is not None and cap.isOpened()
        except Exception:
            log.exception("câmera '%s': erro ao abrir captura", self.camera.name)
        if not opened:
            self._safe_release(cap)
            return None
        return cap

    def _safe_read(self, cap: object) -> tuple[bool, object | None]:
        try:
            return cap.read()
        except Exception:
            # Uma exceção em read() conta como falha de leitura, igual a
            # ok=False: respeita a tolerância de MAX_READ_FAILURES em vez de
            # reconectar de imediato por causa de um glitch pontual do
            # decodificador (ex.: um frame corrompido isolado).
            log.exception("câmera '%s': erro ao ler frame", self.camera.name)
            return False, None

    def _safe_release(self, cap: object | None) -> None:
        if cap is None:
            return
        try:
            cap.release()
        except Exception:
            log.exception(
                "câmera '%s': falha ao liberar captura", self.camera.name
            )

    # --- laço principal ---

    def _run(self) -> None:
        backoff = 0.0
        interval = 1.0 / max(0.1, self.camera.target_fps)

        while not self._stop.is_set():
            cap = self._safe_open()
            if cap is None:
                # _safe_open já liberou a captura (se alguma foi criada) antes
                # de devolver None — nenhum release adicional é necessário aqui.
                self._set_state(CameraState.RECONNECTING)
                backoff = self._next_backoff(backoff)
                log.warning(
                    "câmera '%s': falha ao conectar, nova tentativa em %.0fs",
                    self.camera.name, backoff,
                )
                self._stop.wait(backoff)
                continue

            backoff = 0.0
            failures = 0
            next_sample = time.monotonic()

            while not self._stop.is_set():
                ok, image = self._safe_read(cap)
                if not ok or image is None:
                    failures += 1
                    if failures >= MAX_READ_FAILURES:
                        log.warning("câmera '%s': stream morreu", self.camera.name)
                        break
                    time.sleep(0.05)
                    continue

                failures = 0
                now = time.monotonic()
                if now < next_sample:
                    continue  # amostragem: descarta o frame sem processar

                next_sample = now + interval
                self._seq += 1
                self.slot.put(Frame(self.camera.name, image, now, self._seq))
                with self._lock:
                    self._last_frame_ts = now
                    self._fps_window.append(now)
                    if len(self._fps_window) > 20:
                        self._fps_window.pop(0)
                self._set_state(CameraState.ONLINE)

            # Único ponto de release do caminho "conexão abriu e leu": este
            # cap só existe aqui porque _safe_open devolveu um objeto válido,
            # e cada iteração do laço externo cria um cap novo — logo, cada
            # VideoCapture passa por exatamente uma chamada de release().
            self._safe_release(cap)
            if not self._stop.is_set():
                self._set_state(CameraState.RECONNECTING)
                backoff = self._next_backoff(backoff)
                self._stop.wait(backoff)
