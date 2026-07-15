from pathlib import Path

import numpy as np
import pytest

from src.config.settings import DetectionConfig
from src.core.types import BBox, KP, PersonDetection
from src.tools.replay import replay


class ScriptedEngine:
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


def test_replay_detects_event_and_writes_outputs(tmp_path):
    video = tmp_path / "in.mp4"
    _make_video(video, n=15)
    # Posições calculadas no referencial do corpo desta pose (ombros em
    # y=100, quadris em y=200, escala=100px) — mesmo gesto canônico de
    # tests/detection/test_concealment.py::test_conceal_gesture_fires_event:
    # (200, 90) é reach (braço esticado para a prateleira, fora das zonas),
    # (130, 205) cai dentro da zona 'waist' (bolso); a confiança cai depois
    # que o punho entra no bolso e "some" — é o vanish que sustenta o score.
    script = [(200, 90, 0.9)] * 3 + [(130, 205, 0.9)] * 2 + [(130, 205, 0.05)] * 10
    out_csv = tmp_path / "out.csv"
    out_video = tmp_path / "out.mp4"

    summary = replay(video, DetectionConfig(), ScriptedEngine(script),
                     out_video=out_video, out_csv=out_csv)

    assert summary.frames == 15
    assert summary.frames_with_person == 15
    assert len(summary.events) >= 1
    assert out_csv.exists()
    assert out_video.exists()
    # o CSV tem cabeçalho + uma linha por frame
    lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("frame,ts,")
    assert len(lines) == 16


def test_replay_no_event_for_normal(tmp_path):
    video = tmp_path / "in.mp4"
    _make_video(video, n=12)
    script = [(160, 120, 0.9)] * 12
    summary = replay(video, DetectionConfig(), ScriptedEngine(script))
    assert summary.events == []
