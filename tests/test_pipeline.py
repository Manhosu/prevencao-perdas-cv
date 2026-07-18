import numpy as np
import pytest

from src.config.settings import AppConfig, CameraConfig, StoreConfig
from src.core.types import BBox, Frame, ObjectDetection, PersonDetection
from src.pipeline import Pipeline


class FakeEngine:
    """Engine dublê: devolve o que dissermos, sem carregar modelo."""

    def __init__(self, persons=None, objects=None):
        self.persons = persons or []
        self.objects = objects or []
        self.pose_calls = 0

    def detect(self, image):
        return list(self.persons), list(self.objects)

    def pose(self, image, boxes):
        self.pose_calls += 1
        out = []
        for _ in boxes:
            kp = np.zeros((17, 3), dtype=np.float32)
            kp[:, 2] = 0.8
            out.append(kp)
        return out

    def warmup(self):
        pass


def _cfg(zones):
    return AppConfig(
        store=StoreConfig(id="l", name="L"),
        cameras=[
            CameraConfig(name="cam1", rtsp_url="rtsp://x", target_fps=5, zones=zones)
        ],
    )


def _frame():
    return Frame("cam1", np.zeros((500, 1000, 3), dtype=np.uint8), ts=1.0, seq=1)


def test_no_person_skips_pose():
    """O gate é o que faz o sistema rodar em PC fraco: sem pessoa, sem pose."""
    engine = FakeEngine(persons=[])
    p = Pipeline(_cfg([]), engine)
    result = p.process_frame(_frame())
    assert result.had_person is False
    assert result.persons == []
    assert engine.pose_calls == 0


def test_person_outside_zone_skips_pose():
    zone = [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]  # metade direita
    engine = FakeEngine(
        persons=[PersonDetection(bbox=BBox(100, 100, 200, 400), conf=0.9)]  # pés em x=150
    )
    p = Pipeline(_cfg([zone]), engine)
    result = p.process_frame(_frame())
    assert result.had_person is False
    assert engine.pose_calls == 0


def test_person_inside_zone_gets_pose_and_track_id():
    zone = [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]
    engine = FakeEngine(
        persons=[PersonDetection(bbox=BBox(700, 100, 800, 400), conf=0.9)]
    )
    p = Pipeline(_cfg([zone]), engine)
    result = p.process_frame(_frame())
    assert result.had_person is True
    assert len(result.persons) == 1
    assert result.persons[0].person.track_id == 1
    assert result.persons[0].keypoints.shape == (17, 3)
    assert engine.pose_calls == 1


def test_track_id_is_stable_across_frames():
    engine = FakeEngine(
        persons=[PersonDetection(bbox=BBox(700, 100, 800, 400), conf=0.9)]
    )
    p = Pipeline(_cfg([]), engine)
    first = p.process_frame(_frame())
    second = p.process_frame(Frame("cam1", np.zeros((500, 1000, 3), np.uint8), 1.2, 2))
    assert first.persons[0].person.track_id == second.persons[0].person.track_id


def test_bags_are_passed_through():
    engine = FakeEngine(
        persons=[PersonDetection(bbox=BBox(700, 100, 800, 400), conf=0.9)],
        objects=[ObjectDetection(label="backpack", bbox=BBox(750, 150, 790, 220), conf=0.7)],
    )
    p = Pipeline(_cfg([]), engine)
    result = p.process_frame(_frame())
    assert result.objects[0].label == "backpack"


def test_invalidate_gate_faz_zona_nova_valer_no_proximo_frame():
    """Sem `invalidate_gate`, o `PersonGate` fica cacheado do 1º frame pra
    sempre: salvar uma zona nova na UI não muda nada no monitoramento em
    execução até reiniciar o processo — o bug que a revisão pegou."""
    zona_ampla = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]  # cobre o quadro inteiro
    engine = FakeEngine(
        persons=[PersonDetection(bbox=BBox(700, 100, 800, 400), conf=0.9)]  # pés em x=75%
    )
    cfg = _cfg([zona_ampla])
    p = Pipeline(cfg, engine)

    primeiro = p.process_frame(_frame())
    assert primeiro.had_person is True  # zona ampla cobre a pessoa -> gate criado e cacheado

    # simula o "Salvar zonas" da UI: muta a MESMA CameraConfig que o pipeline
    # enxerga, para uma zona que EXCLUI a pessoa (metade esquerda; pessoa em x=75%)
    zona_estreita = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]
    cfg.cameras[0].zones = [zona_estreita]

    p.invalidate_gate("cam1")

    segundo = p.process_frame(Frame("cam1", np.zeros((500, 1000, 3), np.uint8), 1.2, 2))
    assert segundo.had_person is False


def test_status_reports_every_camera():
    p = Pipeline(_cfg([]), FakeEngine())
    st = p.status()
    assert "cam1" in st
    assert st["cam1"]["state"] == "offline"
    assert st["cam1"]["fps"] == 0.0
