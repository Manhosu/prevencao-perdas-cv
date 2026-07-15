import time

import numpy as np
import pytest

from src.capture.frame_slot import LatestFrameSlot
from src.capture.rtsp_capture import CameraThread
from src.config.settings import CameraConfig
from src.core.types import CameraState


class FakeCapture:
    """Dublê de cv2.VideoCapture com falha programável."""

    def __init__(self, fail_after: int | None = None, open_ok: bool = True):
        self.fail_after = fail_after
        self._open = open_ok
        self.reads = 0
        self.released = False

    def isOpened(self):  # noqa: N802 (assinatura do OpenCV)
        return self._open

    def read(self):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        return True, np.zeros((360, 640, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def _cam(**kw) -> CameraConfig:
    return CameraConfig(
        name="cam1", rtsp_url="rtsp://fake/ch1", target_fps=kw.pop("target_fps", 20), **kw
    )


def test_publishes_frames_and_goes_online():
    slot = LatestFrameSlot()
    cap = FakeCapture()
    t = CameraThread(_cam(), slot, open_capture=lambda url: cap)
    t.start()
    try:
        deadline = time.monotonic() + 3
        while slot.peek() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert slot.peek() is not None
        assert t.state == CameraState.ONLINE
        assert t.last_frame_ts is not None
    finally:
        t.stop()
    assert cap.released


def test_samples_at_target_fps():
    slot = LatestFrameSlot()
    t = CameraThread(_cam(target_fps=5), slot, open_capture=lambda url: FakeCapture())
    t.start()
    seqs = []
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            f = slot.get()
            if f:
                seqs.append(f.seq)
            time.sleep(0.01)
    finally:
        t.stop()
    # 5 fps por ~2s: aceita folga de agendamento do Windows
    assert 6 <= len(seqs) <= 14


def test_reconnects_after_stream_dies():
    slot = LatestFrameSlot()
    caps: list[FakeCapture] = []

    def factory(url):
        # o primeiro morre depois de 3 leituras; o segundo é saudável
        cap = FakeCapture(fail_after=3 if not caps else None)
        caps.append(cap)
        return cap

    t = CameraThread(_cam(), slot, backoff_max=0.2, open_capture=factory)
    t.start()
    try:
        deadline = time.monotonic() + 5
        while len(caps) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(caps) >= 2, "não reconectou depois da queda do stream"
        assert caps[0].released, "não liberou a captura morta"

        deadline = time.monotonic() + 3
        while t.state != CameraState.ONLINE and time.monotonic() < deadline:
            time.sleep(0.05)
        assert t.state == CameraState.ONLINE
    finally:
        t.stop()


def test_state_is_reconnecting_while_open_fails():
    slot = LatestFrameSlot()
    t = CameraThread(
        _cam(), slot, backoff_max=0.2, open_capture=lambda url: FakeCapture(open_ok=False)
    )
    t.start()
    try:
        deadline = time.monotonic() + 3
        while t.state != CameraState.RECONNECTING and time.monotonic() < deadline:
            time.sleep(0.05)
        assert t.state == CameraState.RECONNECTING
        assert slot.peek() is None
    finally:
        t.stop()


def test_backoff_is_capped():
    t = CameraThread(_cam(), LatestFrameSlot(), backoff_max=5.0, open_capture=lambda u: None)
    assert t._next_backoff(0.0) == 1.0
    assert t._next_backoff(1.0) == 2.0
    assert t._next_backoff(4.0) == 5.0
    assert t._next_backoff(5.0) == 5.0


@pytest.mark.slow
@pytest.mark.rtsp
def test_captures_from_simulated_dvr(tmp_path):
    from dev.dvr_sim import DvrSim
    from dev.make_sample_video import synthetic_video

    video = tmp_path / "ch1.mp4"
    synthetic_video(video, seconds=5, fps=10)

    with DvrSim({"ch1": video}) as sim:
        slot = LatestFrameSlot()
        cam = CameraConfig(name="cam1", rtsp_url=sim.url("ch1"), target_fps=5)
        t = CameraThread(cam, slot)
        t.start()
        try:
            # A conexão fria via backend FFmpeg do OpenCV contra este DVR
            # simulado leva, de forma consistente e repetida (medido: ~29s
            # em duas execuções), bem mais que os 15s originalmente
            # cogitados para o primeiro frame — provável tempo de espera
            # pelo próximo keyframe decodificável. Não é flakiness: o
            # deadline abaixo tem folga generosa sobre o valor medido, sem
            # apertar o suficiente para mascarar uma reconexão realmente
            # quebrada.
            deadline = time.monotonic() + 40
            while slot.peek() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            assert slot.peek() is not None, "não recebeu frame do DVR simulado"

            # derruba o canal: o sistema tem que perceber e voltar sozinho
            sim.kill_stream("ch1")
            deadline = time.monotonic() + 20
            while t.state == CameraState.ONLINE and time.monotonic() < deadline:
                time.sleep(0.2)
            assert t.state != CameraState.ONLINE

            sim.restore_stream("ch1")
            # Mesma latência de conexão fria observada acima se repete aqui
            # (medido: ~28s); 40s dá folga equivalente.
            deadline = time.monotonic() + 40
            while t.state != CameraState.ONLINE and time.monotonic() < deadline:
                time.sleep(0.2)
            assert t.state == CameraState.ONLINE, "não reconectou após o canal voltar"
        finally:
            t.stop()
