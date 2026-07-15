import time
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
    Grave antes: python dev/record_clips.py --label normal --seconds 5

    Só deve FALHAR se existir material com pessoa e, mesmo assim, a
    detecção/pose falhar. `dev/videos/normal/` também pode conter só clipes
    placeholder (ex.: `synthetic_smoke.mp4`, sem ninguém detectável) — nesse
    caso o teste dá skip com uma mensagem clara, em vez de falhar, para que a
    suíte `-m slow` feche verde sem exigir material gravado de verdade."""
    import cv2

    clips = sorted(Path("dev/videos/normal").glob("*.mp4"))
    if not clips:
        pytest.skip("sem material: rode dev/record_clips.py --label normal --seconds 5")

    eng = InferenceEngine(InferenceConfig(device="cpu"))
    eng.warmup()

    # Varre todos os clipes disponíveis (não só o primeiro em ordem
    # alfabética) e, em cada um, alguns frames — a pessoa pode não estar
    # presente logo no primeiro frame do clipe.
    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        try:
            for _ in range(60):  # teto de frames por clipe; não precisa ler o vídeo inteiro
                ok, frame = cap.read()
                if not ok:
                    break
                persons, _ = eng.detect(frame)
                if persons:
                    kps = eng.pose(frame, [p.bbox for p in persons])
                    assert kps[0].shape == (17, 3)
                    assert kps[0][:, 2].max() > 0.3, "keypoints sem confiança nenhuma"
                    return
        finally:
            cap.release()

    pytest.skip(
        "grave material real com pessoa via dev/record_clips.py: nenhum "
        "clipe em dev/videos/normal teve pessoa detectada pelo YOLO "
        f"(clipes testados: {[c.name for c in clips]})"
    )


@pytest.mark.slow
def test_openvino_pose_survives_repeated_calls():
    """Regressão do bug crítico da Task 7: export_openvino() sem
    dynamic=True exporta o modelo de pose com entrada estática 640x640.
    pose() sempre roda o recorte da pessoa com imgsz=POSE_INPUT (320) — o
    confirmado ao vivo pelo revisor foi que a 1a chamada passa (o
    ultralytics silenciosamente sobrescreve 320 pelo 640 do modelo
    estático, na resolução errada) mas da 2a chamada em diante estoura
    RuntimeError de shape incompatível (modelo=[1,3,640,640] vs
    tensor=(1,3,320,320)). Como warmup() só chama pose() uma vez, esse bug
    dava falso verde no boot do sistema e só quebrava no primeiro frame
    real com pessoa na loja — device="openvino" é o default de
    InferenceConfig, então isso afeta a instalação padrão.

    Este teste precisa dos modelos reais em models/ (o ultralytics baixa
    sozinho na 1a chamada, como os demais testes @pytest.mark.slow deste
    arquivo já dependem implicitamente)."""
    eng = InferenceEngine(InferenceConfig(device="openvino"))

    t0 = time.perf_counter()
    eng.warmup()
    warmup_ms = (time.perf_counter() - t0) * 1000
    print(f"\n[openvino] warmup: {warmup_ms:.1f} ms")

    # Recorte "real" (não mockado) passando pelo modelo de pose de verdade,
    # em imgsz=POSE_INPUT (320) — igual ao que acontece em produção.
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    box = BBox(50, 50, 300, 400)

    pose_ms: list[float] = []
    for i in range(1, 4):
        t0 = time.perf_counter()
        kps = eng.pose(image, [box])
        pose_ms.append((time.perf_counter() - t0) * 1000)
        assert len(kps) == 1, f"chamada {i}: esperava 1 resultado de pose"
        assert kps[0].shape == (17, 3), (
            f"chamada {i}: forma da saída de pose deveria ser (17,3), veio "
            f"{kps[0].shape}"
        )
    print(f"[openvino] pose() x3: {[f'{m:.1f}' for m in pose_ms]} ms")

    # Mesmo mecanismo (export dinâmico) precisa servir os dois caminhos:
    # detect() roda em detect_size (640) sobre o frame inteiro.
    t0 = time.perf_counter()
    persons, _ = eng.detect(image)
    detect_ms = (time.perf_counter() - t0) * 1000
    print(f"[openvino] detect(): {detect_ms:.1f} ms")
    assert isinstance(persons, list)


@pytest.mark.slow
def test_export_openvino_is_cached(tmp_path):
    out = export_openvino(Path("models/yolo11n.pt"))
    assert out.exists()
    mtime = out.stat().st_mtime
    again = export_openvino(Path("models/yolo11n.pt"))
    assert again == out
    assert out.stat().st_mtime == mtime, "re-exportou um modelo já exportado"
