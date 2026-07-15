"""Reproduz a corrida Critical: o Scheduler pode entregar a MESMA câmera a
dois workers ao mesmo tempo. Enquanto o worker A está "processando" (aqui,
dormindo um pouco para simular o custo real de detect+pose) o frame N de uma
câmera, a captura já reabasteceu o slot e o worker B pega o frame N+1 da
MESMA câmera — os dois chamam `Tracker.update()` concorrentemente no mesmo
objeto, que não é thread-safe.

Este teste roda o WorkerPool e o Scheduler REAIS (não dublês) contra um
Tracker REAL por câmera — exatamente a composição de `Pipeline`, só que sem
o InferenceEngine (que não é o que está sendo testado aqui).

Sem a correção (reivindicação de câmera no Scheduler), este teste falha de
forma consistente: ou por exceção capturada dentro de `Tracker.update()`
(tipicamente IndexError, vindo da lista `self._tracks` sendo reatribuída por
uma thread enquanto outra ainda indexa nela), ou por `track_id` duplicado
para a mesma câmera (corrida no contador `self._next_id`), ou pelo contador
de concorrência por câmera acusando mais de 1 worker simultâneo na mesma
câmera.
"""
from __future__ import annotations

import sys
import threading
import time
from itertools import count

import numpy as np
import pytest

from src.capture.frame_slot import LatestFrameSlot
from src.core.types import BBox, Frame, PersonDetection
from src.detection.tracker import Tracker
from src.inference.scheduler import Scheduler
from src.inference.worker_pool import WorkerPool

# Cada frame recebe uma posição bem distante de qualquer outra (offset
# proporcional a um contador global e nunca reaproveitado): num sistema
# correto, isso GARANTE que o Tracker nunca "casa" uma detecção com a
# anterior — toda chamada bem-sucedida a update() cria uma track nova, com um
# track_id novo. Se algum track_id se repetir para a mesma câmera ao longo do
# teste, é porque o contador `_next_id` (ou a lista `_tracks`) foi corrompido
# por acesso concorrente — não porque duas detecções genuinamente "casaram".
_SEQ = count()

# Reduz o intervalo de troca de contexto do GIL para maximizar a chance de
# interleaving de bytecode entre as threads durante a janela de corrida —
# sem isso, o bug ainda ocorre (confirmado por execução), mas de forma menos
# previsível em máquinas rápidas/pouco carregadas.
_SWITCH_INTERVAL = 1e-6

# Simula o custo de detect+pose (~150ms no bug real) que dá tempo da captura
# reabastecer o slot antes do worker terminar. Um valor menor já é suficiente
# aqui porque reduzimos o switch interval do GIL acima; mantém o teste
# rápido (a suíte "not slow" deve rodar em segundos, não minutos).
_FAKE_INFERENCE_SECONDS = 0.008


def _frame(cam: str, seq: int) -> Frame:
    return Frame(cam, np.zeros((8, 8, 3), dtype=np.uint8), ts=float(seq), seq=seq)


class _RaceHarness:
    """Um Tracker real por câmera + instrumentação para detectar corrida:
    exceções engolidas, track_id duplicado, e concorrência por câmera acima
    de 1."""

    def __init__(self, cameras: list[str]) -> None:
        self.trackers = {c: Tracker(max_lost_seconds=2.0) for c in cameras}
        self._lock = threading.Lock()
        self.exceptions: list[BaseException] = []
        self.seen_track_ids: dict[str, list[int]] = {c: [] for c in cameras}
        self._active: dict[str, int] = {c: 0 for c in cameras}
        self.max_active_per_camera: dict[str, int] = {c: 0 for c in cameras}
        self.max_active_total = 0

    def process(self, frame: Frame) -> bool:
        cam = frame.camera_name
        with self._lock:
            self._active[cam] += 1
            self.max_active_per_camera[cam] = max(
                self.max_active_per_camera[cam], self._active[cam]
            )
            total = sum(self._active.values())
            self.max_active_total = max(self.max_active_total, total)
        try:
            # Simula detect+pose: é durante esta espera que a CameraThread
            # real reabastece o slot e um segundo worker pode pegar o
            # próximo frame da mesma câmera.
            time.sleep(_FAKE_INFERENCE_SECONDS)
            offset = next(_SEQ) * 100_000.0
            person = PersonDetection(bbox=BBox(offset, 0, offset + 50, 150), conf=0.9)
            tracked = self.trackers[cam].update([person], frame.ts)
            for p in tracked:
                self.seen_track_ids[cam].append(p.track_id)
            return True
        except BaseException as exc:  # noqa: BLE001 — instrumentação de teste
            with self._lock:
                self.exceptions.append(exc)
            raise
        finally:
            with self._lock:
                self._active[cam] -= 1

    def assert_no_corruption(self) -> None:
        assert self.exceptions == [], (
            f"Tracker.update() lançou {len(self.exceptions)} exceção(ões) sob "
            f"concorrência (deveria ser impossível com 1 worker por câmera "
            f"por vez): {self.exceptions[:3]!r}"
        )
        for cam, ids in self.seen_track_ids.items():
            dupes = {i for i in ids if ids.count(i) > 1}
            assert not dupes, (
                f"câmera '{cam}': track_id duplicado {dupes} — corrida no "
                f"contador _next_id ou na lista _tracks do Tracker"
            )
        for cam, m in self.max_active_per_camera.items():
            assert m <= 1, (
                f"câmera '{cam}' foi processada por até {m} workers "
                f"simultaneamente — deveria ser no máximo 1 por vez"
            )


def _run_scenario(cameras: list[str], workers: int, seconds: float) -> _RaceHarness:
    harness = _RaceHarness(cameras)
    slots = {c: LatestFrameSlot() for c in cameras}
    scheduler = Scheduler(cameras)
    pool = WorkerPool(slots, scheduler, harness.process, workers=workers)

    stop_feeding = threading.Event()

    def _feed(cam: str) -> None:
        seq = 0
        while not stop_feeding.is_set():
            slots[cam].put(_frame(cam, seq))
            seq += 1
            time.sleep(0.002)  # bem mais rápido que _FAKE_INFERENCE_SECONDS

    feeders = [threading.Thread(target=_feed, args=(c,), daemon=True) for c in cameras]

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(_SWITCH_INTERVAL)
    try:
        for f in feeders:
            f.start()
        pool.start()
        time.sleep(seconds)
    finally:
        stop_feeding.set()
        pool.stop()
        for f in feeders:
            f.join(timeout=2)
        sys.setswitchinterval(old_interval)

    return harness


def test_one_camera_two_workers_never_race_the_same_tracker():
    """O cenário fatal do bug: 1 câmera + workers=2 — exatamente o que
    config/config.example.json entrega."""
    harness = _run_scenario(["cam1"], workers=2, seconds=2.0)
    assert sum(len(v) for v in harness.seen_track_ids.values()) > 0, (
        "teste não processou nenhum frame — ajuste os tempos do cenário"
    )
    harness.assert_no_corruption()


def test_three_cameras_four_workers_never_race_the_same_tracker():
    """Mais câmeras que o caso mínimo: garante que a exclusividade é por
    câmera (cada uma no máximo 1 worker), não uma serialização global
    acidental."""
    cams = ["cam1", "cam2", "cam3"]
    harness = _run_scenario(cams, workers=4, seconds=2.0)
    assert sum(len(v) for v in harness.seen_track_ids.values()) > 0, (
        "teste não processou nenhum frame — ajuste os tempos do cenário"
    )
    assert harness.max_active_total <= len(cams), (
        f"{harness.max_active_total} workers processando concorrentemente "
        f"com apenas {len(cams)} câmeras distintas"
    )
    harness.assert_no_corruption()
