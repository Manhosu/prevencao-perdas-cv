from pathlib import Path

import numpy as np
import pytest

from src.config.settings import InferenceConfig
from src.core.types import BBox
from src.inference.engine import InferenceEngine, export_openvino


class FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = np.array(xyxy, dtype=np.float32)
        self.cls = np.array(cls, dtype=np.float32)
        self.conf = np.array(conf, dtype=np.float32)

    def __len__(self):
        return len(self.cls)


class FakeResult:
    def __init__(self, boxes=None, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


class FakeKeypoints:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)  # (n, 17, 3)


def test_detect_splits_persons_and_bags(monkeypatch):
    cfg = InferenceConfig(device="cpu", detect_bags=True)
    eng = InferenceEngine(cfg)

    # COCO: 0=person, 24=backpack, 26=handbag, 2=car (deve ser ignorado)
    boxes = FakeBoxes(
        xyxy=[[10, 10, 60, 200], [70, 20, 100, 60], [0, 0, 5, 5]],
        cls=[0, 24, 2],
        conf=[0.9, 0.7, 0.95],
    )
    monkeypatch.setattr(eng, "_person_model", lambda *a, **k: [FakeResult(boxes=boxes)])

    persons, objects = eng.detect(np.zeros((240, 320, 3), dtype=np.uint8))

    assert len(persons) == 1
    assert persons[0].bbox == BBox(10, 10, 60, 200)
    assert persons[0].conf == pytest.approx(0.9)
    assert len(objects) == 1
    assert objects[0].label == "backpack"


def test_detect_ignores_bags_when_disabled(monkeypatch):
    eng = InferenceEngine(InferenceConfig(device="cpu", detect_bags=False))
    boxes = FakeBoxes(xyxy=[[70, 20, 100, 60]], cls=[24], conf=[0.7])
    monkeypatch.setattr(eng, "_person_model", lambda *a, **k: [FakeResult(boxes=boxes)])

    persons, objects = eng.detect(np.zeros((240, 320, 3), dtype=np.uint8))

    assert persons == []
    assert objects == []


def test_pose_on_crop_returns_keypoints_in_full_frame_coords(monkeypatch):
    """A pose roda no recorte da pessoa (resolução efetiva muito maior em
    pessoa pequena), mas devolve coordenadas do frame completo."""
    eng = InferenceEngine(InferenceConfig(device="cpu", pose_on_crop=True))

    kp_local = np.zeros((1, 17, 3), dtype=np.float32)
    kp_local[0, 9] = [10.0, 20.0, 0.9]  # punho esquerdo a (10,20) DENTRO do recorte
    monkeypatch.setattr(
        eng, "_pose_model", lambda *a, **k: [FakeResult(keypoints=FakeKeypoints(kp_local))]
    )

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    box = BBox(100, 50, 200, 250)  # recorte expandido começa antes de (100,50)
    kps = eng.pose(image, [box])

    assert len(kps) == 1
    exp = box.expand(0.1).clip(640, 480)
    assert kps[0][9][0] == pytest.approx(10.0 + exp.x1)
    assert kps[0][9][1] == pytest.approx(20.0 + exp.y1)
    assert kps[0][9][2] == pytest.approx(0.9)


def test_pose_returns_zeros_when_model_finds_nothing(monkeypatch):
    eng = InferenceEngine(InferenceConfig(device="cpu"))
    monkeypatch.setattr(
        eng, "_pose_model", lambda *a, **k: [FakeResult(keypoints=None)]
    )
    kps = eng.pose(np.zeros((480, 640, 3), dtype=np.uint8), [BBox(0, 0, 100, 200)])
    assert len(kps) == 1
    assert kps[0].shape == (17, 3)
    assert kps[0][:, 2].sum() == 0.0  # confiança zero = "não sei"


def test_pose_skips_degenerate_box(monkeypatch):
    eng = InferenceEngine(InferenceConfig(device="cpu"))
    called = []
    monkeypatch.setattr(
        eng, "_pose_model", lambda *a, **k: called.append(1) or [FakeResult()]
    )
    kps = eng.pose(np.zeros((480, 640, 3), dtype=np.uint8), [BBox(10, 10, 10, 10)])
    assert kps[0][:, 2].sum() == 0.0
    assert not called, "não deve chamar o modelo para caixa degenerada"


@pytest.mark.slow
def test_real_model_detects_person_in_recorded_clip():
    """Roda o modelo de verdade sobre material próprio.
    Grave antes: python dev/record_clips.py --label normal --seconds 5"""
    import cv2

    clips = sorted(Path("dev/videos/normal").glob("*.mp4"))
    if not clips:
        pytest.skip("sem material: rode dev/record_clips.py --label normal --seconds 5")

    cap = cv2.VideoCapture(str(clips[0]))
    ok, frame = cap.read()
    cap.release()
    assert ok

    eng = InferenceEngine(InferenceConfig(device="cpu"))
    eng.warmup()
    persons, _ = eng.detect(frame)
    assert len(persons) >= 1, "não detectou pessoa no clipe gravado"

    kps = eng.pose(frame, [p.bbox for p in persons])
    assert kps[0].shape == (17, 3)
    assert kps[0][:, 2].max() > 0.3, "keypoints sem confiança nenhuma"


@pytest.mark.slow
def test_export_openvino_is_cached(tmp_path):
    out = export_openvino(Path("models/yolo11n.pt"))
    assert out.exists()
    mtime = out.stat().st_mtime
    again = export_openvino(Path("models/yolo11n.pt"))
    assert again == out
    assert out.stat().st_mtime == mtime, "re-exportou um modelo já exportado"
