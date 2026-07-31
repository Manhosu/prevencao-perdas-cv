"""Detector de pânico: as duas mãos acima da cabeça por 2+ segundos → alerta.

É o "botão de emergência virtual" do caixa (pivô do projeto 29/jul): num
assalto, o caixa levanta as mãos (forçado ou pra sinalizar) e o dono é avisado
sem pegar o celular. Diferente do furto, é um gesto CLARO e sem ambiguidade —
por isso a pose acerta de forma confiável, inclusive de câmera de teto.
"""
import numpy as np

from src.core.types import BBox, KP, PersonDetection, PersonPose
from src.detection.panic import PanicConfig, PanicDetector


def _pose(track_id: int, maos_acima: bool, conf: float = 0.9) -> PersonPose:
    """Pose sintética. Nariz em y=100. Mãos acima (y=50) ou abaixo (y=200) dele."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[KP["nose"]] = [100, 100, conf]
    kp[KP["left_shoulder"]] = [90, 130, conf]
    kp[KP["right_shoulder"]] = [110, 130, conf]
    y = 50 if maos_acima else 200
    kp[KP["left_wrist"]] = [90, y, conf]
    kp[KP["right_wrist"]] = [110, y, conf]
    p = PersonDetection(bbox=BBox(80, 80, 120, 300), conf=0.9, track_id=track_id)
    return PersonPose(person=p, keypoints=kp)


def _rodar(det, maos_acima, de, ate, passo=0.2, tid=1):
    """Alimenta frames de `de` a `ate` segundos; devolve todos os eventos."""
    evs = []
    t = de
    while t < ate:
        evs += det.update([_pose(tid, maos_acima)], round(t, 3))
        t += passo
    return evs


def _cfg(**kw):
    base = dict(hold_seconds=2.0, cooldown_seconds=30.0)
    base.update(kw)
    return PanicConfig(**base)


def test_maos_acima_por_2s_dispara():
    det = PanicDetector(_cfg())
    evs = _rodar(det, True, 0.0, 3.0)
    assert len(evs) == 1
    assert evs[0].track_id == 1


def test_maos_acima_menos_de_2s_nao_dispara():
    det = PanicDetector(_cfg())
    evs = _rodar(det, True, 0.0, 1.5)  # só 1.5s
    assert evs == []


def test_maos_abaixo_nunca_dispara():
    det = PanicDetector(_cfg())
    evs = _rodar(det, False, 0.0, 5.0)
    assert evs == []


def test_uma_mao_so_nao_dispara():
    """Uma mão pra cima (acenar, alcançar prateleira) NÃO é pânico."""
    det = PanicDetector(_cfg())
    kp = np.zeros((17, 3), np.float32)
    kp[KP["nose"]] = [100, 100, 0.9]
    kp[KP["left_wrist"]] = [90, 50, 0.9]     # esquerda acima
    kp[KP["right_wrist"]] = [110, 200, 0.9]  # direita abaixo
    p = PersonDetection(bbox=BBox(80, 80, 120, 300), conf=0.9, track_id=1)
    evs = []
    t = 0.0
    while t < 5.0:
        evs += det.update([PersonPose(person=p, keypoints=kp)], round(t, 3))
        t += 0.2
    assert evs == []


def test_baixar_as_maos_zera_o_cronometro():
    """Ergueu 1.5s, baixou, ergueu de novo — não soma; precisa de 2s contínuos."""
    det = PanicDetector(_cfg())
    assert _rodar(det, True, 0.0, 1.5) == []       # 1.5s acima
    assert _rodar(det, False, 1.5, 2.5) == []      # baixou 1s (zera)
    evs = _rodar(det, True, 2.5, 4.0)              # só 1.5s de novo
    assert evs == []


def test_cooldown_nao_redispara_em_seguida():
    """Depois de disparar, fica em cooldown — não spamma o dono."""
    det = PanicDetector(_cfg(cooldown_seconds=30.0))
    evs = _rodar(det, True, 0.0, 10.0)  # mãos acima 10s seguidos
    assert len(evs) == 1  # dispara 1 vez, não a cada frame


def test_maos_com_baixa_confianca_nao_dispara():
    """Pose ruim (punhos com conf baixa) não gera pânico fantasma."""
    det = PanicDetector(_cfg())
    evs = _rodar(det, True, 0.0, 5.0)  # conf ok -> dispara
    det2 = PanicDetector(_cfg())
    evs2 = []
    t = 0.0
    while t < 5.0:
        evs2 += det2.update([_pose(1, True, conf=0.1)], round(t, 3))
        t += 0.2
    assert len(evs) == 1 and evs2 == []
