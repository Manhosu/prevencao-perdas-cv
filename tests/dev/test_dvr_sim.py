import cv2
import pytest

from dev.dvr_sim import DvrSim
from dev.make_sample_video import synthetic_video


@pytest.mark.slow
@pytest.mark.rtsp
def test_dvr_sim_serves_rtsp_frames(tmp_path):
    video = tmp_path / "cam1.mp4"
    synthetic_video(video, seconds=3, fps=10, size=(640, 360))

    with DvrSim({"ch1": video}) as sim:
        cap = cv2.VideoCapture(sim.url("ch1"), cv2.CAP_FFMPEG)
        assert cap.isOpened(), "não abriu o stream RTSP do DVR simulado"
        ok, frame = cap.read()
        cap.release()

    assert ok
    assert frame.shape[:2] == (360, 640)


def test_url_format():
    sim = DvrSim({}, port=9554)
    assert sim.url("ch3") == "rtsp://127.0.0.1:9554/ch3"


def test_synthetic_video_has_expected_length(tmp_path):
    p = tmp_path / "v.mp4"
    synthetic_video(p, seconds=2, fps=10, size=(320, 240))
    cap = cv2.VideoCapture(str(p))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert 15 <= n <= 25  # ~20 frames, tolerando o encoder
