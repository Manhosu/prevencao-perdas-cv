import logging
import math

import numpy as np
import pytest

from src.config.settings import DetectionConfig
from src.core.types import BBox, KP, PersonDetection
from src.tools.calibrate import SweepRow, best_row, format_table, sweep


class ScriptedEngine:
    """Mesmo dublê de tests/tools/test_replay.py: pose() lê uma posição de
    punho por chamada de uma lista pré-roteirizada (script), fixa por vídeo.
    Aqui uma única instância é reaproveitada entre as duas chamadas de
    `replay()` que o sweep faz (clipe de ocultação, depois clipe normal), por
    isso o script combinado precisa ter exatamente um item por frame
    processado ao longo das duas passagens, na ordem em que elas acontecem."""

    def __init__(self, script):
        self.script = script
        self.i = 0

    def detect(self, image):
        return [PersonDetection(bbox=BBox(80, 60, 120, 300), conf=0.9)], []

    def pose(self, image, boxes):
        wx, wy, wc = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        kp = np.zeros((17, 3), dtype=np.float32)
        for name, xy in (("left_shoulder", (90, 100)), ("right_shoulder", (110, 100)),
                         ("left_hip", (92, 200)), ("right_hip", (108, 200)),
                         ("nose", (100, 80)), ("left_eye", (96, 78)), ("right_eye", (104, 78))):
            kp[KP[name]] = [xy[0], xy[1], 0.9]
        kp[KP["right_wrist"]] = [wx, wy, wc]
        return [kp]

    def warmup(self):
        pass


def _make_video(path, n=15, size=(200, 360)):
    import cv2
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
    for _ in range(n):
        vw.write(np.zeros((h, w, 3), dtype=np.uint8))
    vw.release()


def test_best_row_respects_false_ceiling():
    rows = [
        SweepRow(threshold=0.5, dwell_seconds=1.0, detected=9, total_conceal=10, false_per_hour=12.0),
        SweepRow(threshold=0.6, dwell_seconds=1.2, detected=8, total_conceal=10, false_per_hour=4.0),
        SweepRow(threshold=0.7, dwell_seconds=1.5, detected=5, total_conceal=10, false_per_hour=1.0),
    ]
    # teto de 5 falsos/hora → a de threshold 0.5 (12/h) é descartada
    best = best_row(rows, max_false_per_hour=5.0)
    assert best.threshold == 0.6
    assert best.detected == 8


def test_best_row_returns_least_false_when_none_meet_ceiling():
    rows = [
        SweepRow(threshold=0.5, dwell_seconds=1.0, detected=10, total_conceal=10, false_per_hour=30.0),
        SweepRow(threshold=0.9, dwell_seconds=2.0, detected=3, total_conceal=10, false_per_hour=20.0),
    ]
    best = best_row(rows, max_false_per_hour=5.0)
    assert best.false_per_hour == 20.0  # a menos ruim


def test_best_row_ties_broken_by_lower_false_per_hour():
    """Duas linhas com o mesmo `detected`: desempata pela de menor falso/hora."""
    rows = [
        SweepRow(threshold=0.5, dwell_seconds=1.0, detected=8, total_conceal=10, false_per_hour=4.0),
        SweepRow(threshold=0.6, dwell_seconds=1.2, detected=8, total_conceal=10, false_per_hour=2.0),
    ]
    best = best_row(rows, max_false_per_hour=5.0)
    assert best.threshold == 0.6
    assert best.false_per_hour == 2.0


def test_best_row_excludes_nan_from_ceiling_even_with_more_detections():
    """Pasta normal vazia para uma combinação -> false_per_hour = NaN. Mesmo
    com mais detecções, essa linha não pode ser escolhida como 'respeita o
    teto' (NaN não é medido, não é uma promessa de zero falsos)."""
    rows = [
        SweepRow(threshold=0.5, dwell_seconds=1.0, detected=9, total_conceal=10, false_per_hour=float("nan")),
        SweepRow(threshold=0.6, dwell_seconds=1.2, detected=7, total_conceal=10, false_per_hour=3.0),
    ]
    best = best_row(rows, max_false_per_hour=5.0)
    assert best.threshold == 0.6
    assert best.detected == 7


def test_best_row_all_nan_is_deterministic():
    """Se nenhuma linha tem falsos/hora medido, o fallback não pode quebrar
    (NaN <= NaN é False) e deve devolver algo determinístico."""
    rows = [
        SweepRow(threshold=0.5, dwell_seconds=1.0, detected=9, total_conceal=10, false_per_hour=float("nan")),
        SweepRow(threshold=0.6, dwell_seconds=1.2, detected=7, total_conceal=10, false_per_hour=float("nan")),
    ]
    best = best_row(rows, max_false_per_hour=5.0)
    assert best.threshold == 0.5  # primeira da lista, escolha estável
    assert math.isnan(best.false_per_hour)


def test_format_table_shows_not_measured_for_nan_and_never_prints_nan():
    rows = [
        SweepRow(threshold=0.6, dwell_seconds=1.2, detected=8, total_conceal=10, false_per_hour=float("nan")),
    ]
    table = format_table(rows)
    assert "nao medido" in table
    assert "nan" not in table.lower()


def test_format_table_header_and_rows_stay_aligned():
    rows = [
        SweepRow(threshold=0.6, dwell_seconds=1.2, detected=8, total_conceal=10, false_per_hour=4.5),
        SweepRow(threshold=0.7, dwell_seconds=1.5, detected=10, total_conceal=10, false_per_hour=12.34),
        SweepRow(threshold=0.5, dwell_seconds=1.0, detected=3, total_conceal=10, false_per_hour=float("nan")),
    ]
    lines = format_table(rows).split("\n")
    header, separator, *data_lines = lines
    assert len(separator) == len(header)
    assert all(len(line) == len(header) for line in data_lines)


@pytest.mark.slow
def test_sweep_counts_detection_in_conceal_and_false_in_normal(tmp_path):
    """Monta 1 clipe rotulado em cada pasta (ocultacao/ dispara, normal/ não)
    e confirma que o sweep monta as pastas certas e conta detecção x falso."""
    conceal_dir = tmp_path / "ocultacao"
    normal_dir = tmp_path / "normal"
    conceal_dir.mkdir()
    normal_dir.mkdir()

    # mesmo gesto canônico de tests/tools/test_replay.py: reach fora da zona,
    # depois punho entra na zona 'waist' e a confiança cai (vanish sustenta o
    # score) — comprovadamente dispara evento de ocultação.
    conceal_script = [(200, 90, 0.9)] * 3 + [(130, 205, 0.9)] * 2 + [(130, 205, 0.05)] * 10
    # punho parado fora de qualquer zona: nunca dispara.
    normal_script = [(160, 120, 0.9)] * 12

    _make_video(conceal_dir / "clip_01.mp4", n=len(conceal_script))
    _make_video(normal_dir / "clip_01.mp4", n=len(normal_script))

    # instância única: o índice do script avança ao longo das DUAS chamadas
    # de replay() que o sweep faz (ocultação primeiro, depois normal).
    engine = ScriptedEngine(conceal_script + normal_script)

    rows = sweep(conceal_dir, normal_dir, engine, grid=[(0.6, 1.2)],
                base_cfg=DetectionConfig(), every=1)

    assert len(rows) == 1
    row = rows[0]
    assert row.threshold == 0.6
    assert row.dwell_seconds == 1.2
    assert row.total_conceal == 1
    assert row.detected == 1  # o clipe de ocultação disparou
    assert row.false_per_hour == 0.0  # o clipe normal não gerou nenhum evento


@pytest.mark.slow
def test_sweep_marks_nan_and_warns_when_normal_folder_has_no_clips(tmp_path, caplog):
    """Pasta normal/ vazia (sem clipes) -> false_per_hour tem que virar NaN,
    não 0.0 (0.0 seria uma promessa falsa de 'zero falsos'), e um aviso em
    português precisa avisar que a medição não rolou."""
    conceal_dir = tmp_path / "ocultacao"
    normal_dir = tmp_path / "normal"
    conceal_dir.mkdir()
    normal_dir.mkdir()  # normal/ fica vazia de propósito

    conceal_script = [(200, 90, 0.9)] * 3 + [(130, 205, 0.9)] * 2 + [(130, 205, 0.05)] * 10
    _make_video(conceal_dir / "clip_01.mp4", n=len(conceal_script))

    engine = ScriptedEngine(conceal_script)

    with caplog.at_level(logging.WARNING):
        rows = sweep(conceal_dir, normal_dir, engine, grid=[(0.6, 1.2)],
                    base_cfg=DetectionConfig(), every=1)

    assert len(rows) == 1
    assert math.isnan(rows[0].false_per_hour)
    assert any("falsos/hora" in rec.message for rec in caplog.records)
