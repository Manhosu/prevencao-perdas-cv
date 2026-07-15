import numpy as np
import pytest

from src.core.types import BBox, CameraState, Frame, PersonDetection, PersonPose, KP


def test_bbox_geometry():
    b = BBox(10, 20, 110, 220)
    assert b.width == 100
    assert b.height == 200
    assert b.center == (60, 120)
    assert b.foot_point == (60, 220)


def test_bbox_contains():
    b = BBox(0, 0, 10, 10)
    assert b.contains(5, 5)
    assert not b.contains(11, 5)


def test_bbox_expand_grows_around_center():
    b = BBox(10, 10, 20, 20).expand(0.2)
    assert b.x1 == pytest.approx(9.0)
    assert b.x2 == pytest.approx(21.0)
    assert b.y1 == pytest.approx(9.0)
    assert b.y2 == pytest.approx(21.0)


def test_frame_holds_image_and_sequence():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    f = Frame(camera_name="cam1", image=img, ts=1.0, seq=7)
    assert f.camera_name == "cam1"
    assert f.seq == 7
    assert f.image.shape == (4, 4, 3)


def test_person_pose_keypoints_shape():
    kps = np.zeros((17, 3), dtype=np.float32)
    p = PersonPose(person=PersonDetection(bbox=BBox(0, 0, 1, 1), conf=0.9), keypoints=kps)
    assert p.keypoints.shape == (17, 3)
    assert p.person.track_id is None


def test_keypoint_index_map_is_coco17():
    assert KP["left_wrist"] == 9
    assert KP["right_wrist"] == 10
    assert KP["left_hip"] == 11
    assert KP["right_hip"] == 12
    assert KP["left_shoulder"] == 5
    assert len(KP) == 17


def test_camera_state_values():
    assert CameraState.ONLINE.value == "online"
    assert CameraState.OFFLINE.value == "offline"
    assert CameraState.RECONNECTING.value == "reconnecting"


def test_bbox_clip_keeps_box_inside_frame():
    # caixa que estoura as duas bordas do frame de 100x50
    b = BBox(-10, -5, 130, 80).clip(100, 50)
    assert (b.x1, b.y1, b.x2, b.y2) == (0.0, 0.0, 100.0, 50.0)


def test_bbox_clip_leaves_inner_box_untouched():
    b = BBox(10, 10, 40, 30).clip(100, 50)
    assert (b.x1, b.y1, b.x2, b.y2) == (10.0, 10.0, 40.0, 30.0)
