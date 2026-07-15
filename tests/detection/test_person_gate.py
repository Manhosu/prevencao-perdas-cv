from src.core.types import BBox, PersonDetection
from src.detection.person_gate import PersonGate

FRAME = (1000, 500)  # largura, altura


def _person(x1, y1, x2, y2) -> PersonDetection:
    return PersonDetection(bbox=BBox(x1, y1, x2, y2), conf=0.9)


def test_no_zones_means_whole_frame():
    gate = PersonGate(zones=[], frame_size=FRAME)
    assert gate.contains(_person(0, 0, 10, 10))
    assert gate.contains(_person(900, 400, 990, 490))


def test_person_inside_polygon():
    # metade direita do quadro
    zone = [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]
    gate = PersonGate(zones=[zone], frame_size=FRAME)
    # foot_point = (750, 400) → dentro
    assert gate.contains(_person(700, 100, 800, 400))


def test_person_outside_polygon():
    zone = [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]
    gate = PersonGate(zones=[zone], frame_size=FRAME)
    # foot_point = (150, 400) → fora
    assert not gate.contains(_person(100, 100, 200, 400))


def test_uses_foot_point_not_center():
    """Pessoa alta cujo CENTRO cai fora da zona mas cujos PÉS caem dentro
    deve contar: quem define a posição é onde a pessoa pisa."""
    zone = [(0.0, 0.7), (1.0, 0.7), (1.0, 1.0), (0.0, 1.0)]  # faixa inferior
    gate = PersonGate(zones=[zone], frame_size=FRAME)
    p = _person(400, 100, 500, 400)  # centro y=250 (fora), pés y=400 (dentro)
    assert gate.contains(p)


def test_multiple_zones_are_or():
    left = [(0.0, 0.0), (0.2, 0.0), (0.2, 1.0), (0.0, 1.0)]
    right = [(0.8, 0.0), (1.0, 0.0), (1.0, 1.0), (0.8, 1.0)]
    gate = PersonGate(zones=[left, right], frame_size=FRAME)
    assert gate.contains(_person(50, 100, 150, 400))    # pés em x=100 → zona esquerda
    assert gate.contains(_person(880, 100, 920, 400))   # pés em x=900 → zona direita
    assert not gate.contains(_person(450, 100, 550, 400))  # meio → fora


def test_filter_keeps_only_people_inside():
    zone = [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]
    gate = PersonGate(zones=[zone], frame_size=FRAME)
    dentro = _person(700, 100, 800, 400)
    fora = _person(100, 100, 200, 400)
    assert gate.filter([dentro, fora]) == [dentro]
