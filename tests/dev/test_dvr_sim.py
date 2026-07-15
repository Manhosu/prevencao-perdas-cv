import time

import cv2
import numpy as np
import pytest

from dev.dvr_sim import DvrSim
from dev.make_sample_video import synthetic_video
from dev.record_clips import _next_path, record


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


@pytest.mark.slow
@pytest.mark.rtsp
def test_dvr_sim_kill_stream_then_restore_stream(tmp_path):
    """A razão de existir do DvrSim: derrubar e restaurar um canal para
    testar a reconexão, sem precisar ir a campo esperar o DVR do cliente
    reiniciar sozinho."""
    video = tmp_path / "cam1.mp4"
    synthetic_video(video, seconds=3, fps=10, size=(640, 360))

    # Timeout curto e limitado só para a checagem "deve falhar" logo depois
    # do kill_stream — sem publicador, não há por que esperar mais que
    # isso. As aberturas que devem ter sucesso usam o mesmo padrão sem
    # timeout explícito já validado em test_dvr_sim_serves_rtsp_frames
    # (o timeout global do pytest, 120s, já limita um hang de verdade).
    fail_fast_params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000,
    ]

    with DvrSim({"ch1": video}, port=8664) as sim:
        cap = cv2.VideoCapture(sim.url("ch1"), cv2.CAP_FFMPEG)
        assert cap.isOpened(), "não abriu o stream RTSP antes do kill_stream"
        ok, _frame = cap.read()
        cap.release()
        assert ok, "não leu frame antes do kill_stream"

        sim.kill_stream("ch1")
        time.sleep(2.0)  # dá tempo do publicador (ffmpeg) realmente morrer

        cap = cv2.VideoCapture(sim.url("ch1"), cv2.CAP_FFMPEG, fail_fast_params)
        opened_after_kill = cap.isOpened()
        ok_after_kill = False
        if opened_after_kill:
            ok_after_kill, _frame = cap.read()
        cap.release()
        assert not ok_after_kill, (
            "a leitura ainda funcionou depois de kill_stream — o canal "
            "deveria ter caído"
        )

        sim.restore_stream("ch1")
        time.sleep(3.0)  # o ffmpeg leva alguns segundos para republicar

        cap = cv2.VideoCapture(sim.url("ch1"), cv2.CAP_FFMPEG)
        assert cap.isOpened(), "não reabriu o stream depois de restore_stream"
        ok, frame = cap.read()
        cap.release()

    assert ok, "não leu frame depois de restore_stream"
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


class _FakeCapture:
    """Fonte de vídeo falsa que entrega frames num ritmo controlado (via
    `time.sleep`), para testar `record()` sem precisar de webcam física."""

    def __init__(self, width: int = 64, height: int = 48, frame_interval: float = 0.05):
        self._opened = True
        self._width = width
        self._height = height
        self._interval = frame_interval
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        time.sleep(self._interval)
        return True, self._frame.copy()

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self._width
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._height
        return 0.0

    def release(self) -> None:
        self._opened = False


def test_record_writes_the_real_measured_fps_not_the_requested_one(tmp_path):
    frame_interval = 0.05  # fonte falsa entrega ~20 frames por segundo real
    seconds = 1.0
    bogus_requested_fps = 5  # bem diferente do ritmo real da fonte falsa

    path = record(
        "normal",
        seconds=seconds,
        camera=0,
        fps=bogus_requested_fps,
        capture_factory=lambda: _FakeCapture(frame_interval=frame_interval),
        out_dir=tmp_path,
        show_preview=False,
    )

    cap = cv2.VideoCapture(str(path))
    written_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert frame_count > 0
    # prova que o FPS gravado não é simplesmente o valor mentiroso pedido
    assert written_fps > bogus_requested_fps * 2

    duration = frame_count / written_fps
    assert duration == pytest.approx(seconds, rel=0.10)


def test_next_path_uses_max_existing_index_plus_one(tmp_path):
    sub = tmp_path / "ocultacao"
    sub.mkdir()
    (sub / "bolso_01.mp4").write_bytes(b"x")
    (sub / "bolso_03.mp4").write_bytes(b"x")  # simula curadoria que apagou o 02

    path = _next_path("bolso", base=tmp_path)

    assert path.name == "bolso_04.mp4"
    assert not path.exists()


def test_next_path_never_overwrites_existing_file(tmp_path):
    sub = tmp_path / "normal"
    sub.mkdir()
    (sub / "normal_01.mp4").write_bytes(b"x")

    path = _next_path("normal", base=tmp_path)

    assert path.name != "normal_01.mp4"
    assert not path.exists()


def test_dvr_sim_rtp_port_is_even_even_with_odd_rtsp_port(tmp_path):
    """RTP exige porta par. Se a porta RTSP for ímpar, o DVRSim deve
    derivar uma porta RTP par mesmo assim, evitando 'ERR RTP port must
    be even' do MediaMTX."""
    from pathlib import Path

    video = tmp_path / "cam1.mp4"
    synthetic_video(video, seconds=1, fps=10, size=(320, 240))

    # Porto RTSP ímpar proposital
    sim = DvrSim({"ch1": video}, port=19555)
    sim.start()

    # Inspeciona o YAML gerado para verificar que a porta RTP é par
    cfg_path = Path("dev/bin") / f"mediamtx-{sim.port}.yml"
    cfg_text = cfg_path.read_text(encoding="utf-8")

    # Extrai os valores de porta do YAML
    import re

    rtp_match = re.search(r"rtpAddress: :(\d+)", cfg_text)
    rtcp_match = re.search(r"rtcpAddress: :(\d+)", cfg_text)
    assert rtp_match, "não achou rtpAddress no YAML"
    assert rtcp_match, "não achou rtcpAddress no YAML"

    rtp_port = int(rtp_match.group(1))
    rtcp_port = int(rtcp_match.group(1))

    assert rtp_port % 2 == 0, (
        f"RTP port {rtp_port} é ímpar, mas RTP exige porta par "
        "(MediaMTX morre com 'ERR RTP port must be even')"
    )
    assert rtcp_port == rtp_port + 1, (
        f"RTCP port {rtcp_port} deveria ser {rtp_port + 1} "
        "(a porta ímpar seguinte após RTP)"
    )

    sim.stop()
