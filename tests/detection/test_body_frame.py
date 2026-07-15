import numpy as np
import pytest

from src.config.settings import Guards, Geometry
from src.core.types import BBox, KP
from src.detection.body_frame import BodyFrame, classify_zone, in_reach

G = Guards()
GEO = Geometry()


def _kp(**pts):
    """Constrói (17,3) com confiança 0.9 nos keypoints dados, 0 no resto.
    pts: nome_coco -> (x, y)."""
    a = np.zeros((17, 3), dtype=np.float32)
    for name, (x, y) in pts.items():
        a[KP[name]] = [x, y, 0.9]
    return a


def _upright():
    """Pessoa em pé: ombros em y=100, quadris em y=200 (y cresce p/ baixo na imagem).
    Corpo vertical, tronco de 100px."""
    return _kp(
        left_shoulder=(90, 100), right_shoulder=(110, 100),
        left_hip=(92, 200), right_hip=(108, 200),
        nose=(100, 80), left_eye=(96, 78), right_eye=(104, 78),
    )


def test_builds_frame_from_upright_person():
    bf = BodyFrame.from_keypoints(_upright(), BBox(80, 60, 120, 300), G)
    assert bf is not None
    assert bf.hip_mid == pytest.approx((100, 200), abs=1)
    assert bf.shoulder_mid == pytest.approx((100, 100), abs=1)
    assert bf.scale == pytest.approx(100, abs=2)  # ||ombro-quadril||
    assert bf.quality > 0.8


def test_wrist_at_hip_line_is_yn_zero():
    bf = BodyFrame.from_keypoints(_upright(), BBox(80, 60, 120, 300), G)
    x_n, y_n = bf.to_body_coords((100, 200))  # exatamente no hip_mid
    assert x_n == pytest.approx(0, abs=0.05)
    assert y_n == pytest.approx(0, abs=0.05)


def test_wrist_at_shoulder_line_is_yn_one():
    bf = BodyFrame.from_keypoints(_upright(), BBox(80, 60, 120, 300), G)
    _, y_n = bf.to_body_coords((100, 100))  # na linha do ombro
    assert y_n == pytest.approx(1.0, abs=0.05)


def test_wrist_below_hip_is_negative_yn():
    bf = BodyFrame.from_keypoints(_upright(), BBox(80, 60, 120, 300), G)
    _, y_n = bf.to_body_coords((100, 250))  # abaixo do quadril (bolso/coxa)
    assert y_n < 0


def test_lateral_offset_is_xn():
    bf = BodyFrame.from_keypoints(_upright(), BBox(80, 60, 120, 300), G)
    x_n, _ = bf.to_body_coords((150, 200))  # 50px à direita do eixo, S=100
    assert abs(x_n) == pytest.approx(0.5, abs=0.05)


def test_scale_normalizes_distance():
    """Pessoa 2x mais longe (metade do tamanho) → mesmas coords de corpo."""
    near = BodyFrame.from_keypoints(_upright(), BBox(80, 60, 120, 300), G)
    far = _kp(
        left_shoulder=(45, 50), right_shoulder=(55, 50),
        left_hip=(46, 100), right_hip=(54, 100),
        nose=(50, 40), left_eye=(48, 39), right_eye=(52, 39),
    )
    bf_far = BodyFrame.from_keypoints(far, BBox(40, 30, 60, 150), G)
    # punho "no bolso": near (100,250) dy=+50 sobre S=100 → yn=-0.5
    # far equivalente (50,125) dy=+25 sobre S=50 → yn=-0.5
    _, yn_near = near.to_body_coords((100, 250))
    _, yn_far = bf_far.to_body_coords((50, 125))
    assert yn_near == pytest.approx(yn_far, abs=0.05)


def test_fallback_scale_when_hips_missing():
    kp = _kp(left_shoulder=(90, 100), right_shoulder=(110, 100), nose=(100, 80))
    bf = BodyFrame.from_keypoints(kp, BBox(80, 60, 120, 300), G)  # bbox altura 240
    assert bf is not None
    assert bf.scale == pytest.approx(0.55 * 240, abs=1)


def test_returns_none_when_no_torso_and_no_bbox():
    kp = np.zeros((17, 3), dtype=np.float32)
    assert BodyFrame.from_keypoints(kp, BBox(0, 0, 0, 0), G) is None


def test_facing_back_when_face_not_visible():
    kp = _upright()
    kp[KP["nose"]] = [0, 0, 0.0]
    kp[KP["left_eye"]] = [0, 0, 0.0]
    kp[KP["right_eye"]] = [0, 0, 0.0]
    bf = BodyFrame.from_keypoints(kp, BBox(80, 60, 120, 300), G)
    assert bf.facing_back is True


def test_facing_front_when_face_visible():
    bf = BodyFrame.from_keypoints(_upright(), BBox(80, 60, 120, 300), G)
    assert bf.facing_back is False


def test_classify_zone_waist():
    # waist_y [-0.45,0.25], waist_x [0.10,0.85]
    assert classify_zone(0.4, -0.1, GEO, facing_back=False) == "waist"


def test_classify_zone_torso():
    # torso_y [0.15,0.85], torso_x_max 0.55 — precisa NÃO cair em waist antes
    assert classify_zone(0.2, 0.5, GEO, facing_back=False) == "torso"


def test_classify_zone_back_waist_only_when_facing_back():
    assert classify_zone(0.4, -0.1, GEO, facing_back=True) == "back_waist"


def test_classify_zone_none_when_far_from_body():
    assert classify_zone(1.2, 0.5, GEO, facing_back=False) is None


def test_in_reach_arm_extended():
    # reach: y_n > 0.9 OU |x_n| > 0.95
    assert in_reach(1.1, 0.5, GEO) is True
    assert in_reach(0.2, 0.95, GEO) is True
    assert in_reach(0.3, 0.2, GEO) is False
