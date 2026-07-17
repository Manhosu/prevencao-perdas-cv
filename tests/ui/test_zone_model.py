import pytest

from src.ui.zone_model import ZoneModel


def test_starts_empty_and_covers_whole_frame():
    m = ZoneModel()
    assert m.to_config() == []
    assert m.covers_whole_frame is True


def test_add_points_and_finish_polygon():
    m = ZoneModel()
    m.add_point(0.1, 0.1)
    m.add_point(0.9, 0.1)
    m.add_point(0.9, 0.9)
    m.finish_polygon()
    cfg = m.to_config()
    assert len(cfg) == 1
    assert len(cfg[0]) == 3
    assert m.covers_whole_frame is False


def test_polygon_with_less_than_three_points_is_discarded():
    m = ZoneModel()
    m.add_point(0.1, 0.1)
    m.add_point(0.5, 0.5)
    m.finish_polygon()
    assert m.to_config() == []  # 2 pontos nao formam area


def test_pixel_conversion_roundtrip():
    m = ZoneModel()
    x_n, y_n = m.from_pixels(320, 120, w=640, h=480)
    assert x_n == pytest.approx(0.5)
    assert y_n == pytest.approx(0.25)
    x, y = m.to_pixels(x_n, y_n, w=640, h=480)
    assert (round(x), round(y)) == (320, 120)


def test_hit_test_finds_vertex():
    m = ZoneModel([[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8)]])
    assert m.hit_test(0.21, 0.21, tol=0.03) == (0, 0)
    assert m.hit_test(0.79, 0.21, tol=0.03) == (0, 1)
    assert m.hit_test(0.5, 0.5, tol=0.03) is None


def test_move_point_updates_and_clamps():
    m = ZoneModel([[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8)]])
    m.move_point(0, 0, 0.35, 0.4)
    assert m.to_config()[0][0] == (0.35, 0.4)
    m.move_point(0, 0, 1.5, -0.3)  # fora do quadro
    assert m.to_config()[0][0] == (1.0, 0.0)  # clampado


def test_remove_point_and_polygon():
    m = ZoneModel([[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]])
    m.remove_point(0, 0)
    assert len(m.to_config()[0]) == 3
    m.remove_polygon(0)
    assert m.to_config() == []


def test_removing_point_below_three_drops_the_polygon():
    m = ZoneModel([[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]])
    m.remove_point(0, 0)  # sobra 2 -> nao e mais area valida
    assert m.to_config() == []


def test_multiple_polygons():
    m = ZoneModel()
    for p in [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3)]:
        m.add_point(*p)
    m.finish_polygon()
    for p in [(0.6, 0.6), (0.9, 0.6), (0.9, 0.9)]:
        m.add_point(*p)
    m.finish_polygon()
    assert len(m.to_config()) == 2


def test_config_roundtrip_matches_person_gate_contract():
    """O formato tem que ser exatamente o que o PersonGate ja consome."""
    from src.core.types import BBox, PersonDetection
    from src.detection.person_gate import PersonGate

    m = ZoneModel()
    for p in [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]:
        m.add_point(*p)
    m.finish_polygon()

    gate = PersonGate(m.to_config(), frame_size=(1000, 500))
    dentro = PersonDetection(bbox=BBox(700, 100, 800, 400), conf=0.9)   # pes em x=750
    fora = PersonDetection(bbox=BBox(100, 100, 200, 400), conf=0.9)     # pes em x=150
    assert gate.contains(dentro) and not gate.contains(fora)
