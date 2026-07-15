import numpy as np

from src.config.settings import AppConfig, CameraConfig, StoreConfig
from src.core.types import BBox, Frame, KP, PersonDetection
from src.pipeline import Pipeline


class ScriptedEngine:
    """Engine dublê: devolve uma pessoa cujo punho segue um roteiro."""

    def __init__(self, wrist_script):
        self.wrist_script = wrist_script
        self.i = 0

    def detect(self, image):
        return [PersonDetection(bbox=BBox(80, 60, 120, 300), conf=0.9)], []

    def pose(self, image, boxes):
        wx, wy, wc = self.wrist_script[min(self.i, len(self.wrist_script) - 1)]
        self.i += 1
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[KP["left_shoulder"]] = [90, 100, 0.9]
        kp[KP["right_shoulder"]] = [110, 100, 0.9]
        kp[KP["left_hip"]] = [92, 200, 0.9]
        kp[KP["right_hip"]] = [108, 200, 0.9]
        kp[KP["nose"]] = [100, 80, 0.9]
        kp[KP["left_eye"]] = [96, 78, 0.9]
        kp[KP["right_eye"]] = [104, 78, 0.9]
        kp[KP["right_wrist"]] = [wx, wy, wc]
        return [kp]

    def warmup(self):
        pass


def _cfg():
    return AppConfig(store=StoreConfig(id="l", name="L"),
                     cameras=[CameraConfig(name="cam1", rtsp_url="rtsp://x", target_fps=5, zones=[])])


def test_pipeline_emits_concealment_event():
    # (108, 210) do brief cai em x_n=0.08, abaixo do limite inferior de
    # waist_x (0.10) da geometria default — nunca classifica como zona
    # nenhuma (mesmo problema documentado na Task 3 do Plano 2: posição
    # sintética incoerente com a geometria). Ajustado para (130, 190),
    # que cai em x_n=0.30/y_n=0.10 -> zona "waist" de fato.
    script = [(160, 130, 0.9)] * 3 + [(130, 190, 0.9)] * 12
    p = Pipeline(_cfg(), ScriptedEngine(script))
    all_events = []
    t = 0.0
    for _ in range(len(script)):
        r = p.process_frame(Frame("cam1", np.zeros((360, 200, 3), np.uint8), t, 1))
        all_events.extend(r.events)
        t += 0.2
    assert len(all_events) >= 1
    assert all_events[0].zone in ("waist", "torso")


def test_pipeline_no_event_for_normal_movement():
    script = [(160, 120, 0.9)] * 15  # mão sempre longe do corpo
    p = Pipeline(_cfg(), ScriptedEngine(script))
    t = 0.0
    events = []
    for _ in range(len(script)):
        r = p.process_frame(Frame("cam1", np.zeros((360, 200, 3), np.uint8), t, 1))
        events.extend(r.events)
        t += 0.2
    assert events == []
