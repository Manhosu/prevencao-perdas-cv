"""Modo caixa no pipeline: câmera em modo pânico usa PanicDetector, dispara no
gesto de mãos acima da cabeça e NÃO emite evento de ocultação — a garantia de
"desativar os outros avisos no caixa". Cobre também o override por câmera
(caixa em pânico enquanto o resto segue em ocultação)."""
import numpy as np

from src.config.settings import (AppConfig, CameraConfig, DetectionConfig,
                                  StoreConfig)
from src.core.types import BBox, Frame, KP, PersonDetection
from src.detection.concealment import ConcealmentAnalyzer
from src.detection.panic import PanicDetector
from src.pipeline import Pipeline


class MaosAcimaEngine:
    """Engine dublê: 1 pessoa com AS DUAS mãos acima do nariz (pose de assalto)."""

    def detect(self, image):
        return [PersonDetection(bbox=BBox(80, 60, 120, 300), conf=0.9)], []

    def pose(self, image, boxes):
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[KP["nose"]] = [100, 100, 0.9]
        kp[KP["left_wrist"]] = [90, 50, 0.9]    # y menor = mais alto que o nariz
        kp[KP["right_wrist"]] = [110, 50, 0.9]
        return [kp]

    def warmup(self):
        pass


def _run(cfg, n=16):
    p = Pipeline(cfg, MaosAcimaEngine())
    evs = []
    t = 0.0
    for _ in range(n):  # ~3s a 5fps: cobre o hold de 2s com folga
        r = p.process_frame(Frame("cam1", np.zeros((360, 200, 3), np.uint8), t, 1))
        evs.extend(r.events)
        t += 0.2
    return p, evs


def _cfg_panico():
    return AppConfig(
        store=StoreConfig(id="l", name="L"),
        detection=DetectionConfig(mode="panico"),
        cameras=[CameraConfig(name="cam1", rtsp_url="rtsp://x", target_fps=5, zones=[])],
    )


def test_camera_em_modo_panico_usa_panic_detector_e_dispara():
    p, evs = _run(_cfg_panico())
    assert isinstance(p._analyzers["cam1"], PanicDetector)
    assert len(evs) >= 1
    assert all(getattr(e, "kind", None) == "panico" for e in evs)


def test_camera_panico_nao_emite_evento_de_ocultacao():
    """Mãos acima da cabeça em modo pânico produz SÓ evento de pânico — nunca
    de ocultação. É a prova de que ligar o caixa desliga os avisos de furto
    daquela câmera (não há como sair um alerta de 'ocultação de produto')."""
    _, evs = _run(_cfg_panico())
    assert evs
    assert all(e.kind == "panico" and e.zone == "panico" for e in evs)


def test_override_por_camera_liga_panico_so_naquela_camera():
    """Global em ocultação; a câmera do caixa liga pânico por override. Cada
    câmera recebe o detector certo — permite caixa em pânico e prateleira em
    ocultação na mesma instalação."""
    cfg = AppConfig(
        store=StoreConfig(id="l", name="L"),
        detection=DetectionConfig(mode="ocultacao"),
        cameras=[
            CameraConfig(name="cam1", rtsp_url="rtsp://x", target_fps=5, zones=[],
                         overrides={"mode": "panico"}),
            CameraConfig(name="prateleira", rtsp_url="rtsp://y", target_fps=5, zones=[]),
        ],
    )
    p = Pipeline(cfg, MaosAcimaEngine())
    assert isinstance(p._analyzers["cam1"], PanicDetector)
    assert isinstance(p._analyzers["prateleira"], ConcealmentAnalyzer)
