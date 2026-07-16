import time
from datetime import datetime

import numpy as np

from src.alerts.alert_queue import AlertQueue
from src.config.settings import AppConfig, CameraConfig, EvidenceConfig, StoreConfig
from src.core.types import BBox, Frame, KP, PersonDetection
from src.evidence.recorder import EvidenceRecorder
from src.pipeline import Pipeline
from src.storage.db import Database


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
        for nome, xy in (("left_shoulder", (90, 100)), ("right_shoulder", (110, 100)),
                         ("left_hip", (92, 200)), ("right_hip", (108, 200)),
                         ("nose", (100, 80)), ("left_eye", (96, 78)), ("right_eye", (104, 78))):
            kp[KP[nome]] = [xy[0], xy[1], 0.9]
        kp[KP["right_wrist"]] = [wx, wy, wc]
        return [kp]

    def warmup(self):
        pass


class FakeSender:
    configured = True

    def __init__(self):
        self.fotos = []

    def send_photo(self, path, caption):
        self.fotos.append((str(path), caption))
        return True

    def send_video(self, path, caption):
        return True

    def send_message(self, text):
        return True


def test_evento_vira_evidencia_e_alerta(tmp_path):
    """Ponta a ponta: gesto de ocultacao -> evento -> foto salva + linha no
    banco + alerta enfileirado e enviado."""
    db = Database(tmp_path / "app.db")
    db.init_schema()
    store = StoreConfig(id="l1", name="Mercado Teste")
    rec = EvidenceRecorder(db, EvidenceConfig(dir=str(tmp_path / "ev")), store)
    sender = FakeSender()
    fila = AlertQueue(sender, db, rate_limit_per_min=600)
    fila.start()

    cfg = AppConfig(store=store,
                    cameras=[CameraConfig(name="cam1", rtsp_url="rtsp://x",
                                          target_fps=5, zones=[])])
    # gesto: vem do reach (200,90) e some na cintura (130,205)
    script = [(200, 90, 0.9)] * 3 + [(130, 205, 0.9)] * 2 + [(130, 205, 0.05)] * 10
    p = Pipeline(cfg, ScriptedEngine(script))

    def on_result(result, frame):
        for ev in result.events:
            eid = rec.record(ev, result.camera_name, frame.image,
                             clip_buffer=p.clip_buffers.get(result.camera_name))
            row = db.list_events(limit=1)[0]
            fila.enqueue(eid, row["image_path"], row["clip_path"],
                         f"{store.name} / {result.camera_name}")

    p.on_result = on_result
    t = 0.0
    for _ in range(len(script)):
        p.process_frame(Frame("cam1", np.zeros((360, 200, 3), np.uint8), t, 1))
        t += 0.2

    fim = time.monotonic() + 3
    while fila.pending > 0 and time.monotonic() < fim:
        time.sleep(0.02)
    fila.stop()

    linhas = db.list_events(limit=10)
    assert len(linhas) >= 1, "o evento nao virou registro"
    assert linhas[0]["image_path"], "a foto da evidencia nao foi salva"
    assert linhas[0]["sent_telegram"] == 1, "o alerta nao foi marcado como enviado"
    assert sender.fotos, "o Telegram nao recebeu a foto"
    db.close()


def test_pipeline_alimenta_clip_buffer(tmp_path):
    cfg = AppConfig(store=StoreConfig(id="l", name="L"),
                    cameras=[CameraConfig(name="cam1", rtsp_url="rtsp://x",
                                          target_fps=5, zones=[])])
    p = Pipeline(cfg, ScriptedEngine([(160, 120, 0.9)] * 5))
    for i in range(5):
        p.process_frame(Frame("cam1", np.zeros((360, 200, 3), np.uint8), i * 0.2, i))
    assert len(p.clip_buffers["cam1"]) == 5
