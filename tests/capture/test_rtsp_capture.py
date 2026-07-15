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


class CountingReleaseCapture(FakeCapture):
    """Dublê que conta quantas vezes release() é chamado (em vez de só um
    booleano), para provar liberação única, não apenas "liberou alguma vez"."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.release_calls = 0

    def release(self):
        self.release_calls += 1
        super().release()


class ExplodingReleaseCapture(FakeCapture):
    """Dublê cujo release() sempre lança, simulando uma falha do driver ao
    liberar a captura no caminho normal (o cenário que expunha o bug do
    release() duplicado)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.release_calls = 0

    def release(self):
        self.release_calls += 1
        raise RuntimeError("falha simulada ao liberar captura")


class ExplodingReadCapture(FakeCapture):
    """Dublê cujo read() lança uma vez e depois se comporta normalmente,
    para provar que a exceção conta como falha de leitura (tolerada por
    MAX_READ_FAILURES) em vez de forçar reconexão imediata."""

    def __init__(self, raise_on_read: int, **kw):
        super().__init__(**kw)
        self.raise_on_read = raise_on_read

    def read(self):
        self.reads += 1
        if self.reads == self.raise_on_read:
            raise RuntimeError("falha simulada de decodificação")
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        return True, np.zeros((360, 640, 3), dtype=np.uint8)


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


def test_release_called_exactly_once_on_normal_reconnect():
    """Regressão: o try/except largo envolvia o ciclo inteiro (abrir → ler →
    release do caminho normal), então uma falha nesse release() caía no
    except externo, que chamava cap.release() de novo (release_calls=2).
    Aqui o release() do caminho normal nunca falha — serve para provar que a
    correção não introduziu uma liberação dupla nem no caminho feliz."""
    slot = LatestFrameSlot()
    caps: list[CountingReleaseCapture] = []

    def factory(url):
        # o primeiro morre depois de 2 leituras; o segundo é saudável
        cap = CountingReleaseCapture(fail_after=2 if not caps else None)
        caps.append(cap)
        return cap

    t = CameraThread(_cam(), slot, backoff_max=0.1, open_capture=factory)
    t.start()
    try:
        deadline = time.monotonic() + 5
        while len(caps) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(caps) >= 2, "não reconectou depois da queda do stream"
        assert caps[0].release_calls == 1, (
            f"release() chamado {caps[0].release_calls}x, esperado exatamente 1x"
        )

        deadline = time.monotonic() + 3
        while t.state != CameraState.ONLINE and time.monotonic() < deadline:
            time.sleep(0.05)
        assert t.state == CameraState.ONLINE, "thread não sobreviveu/reconectou"
    finally:
        t.stop()


def test_release_exception_does_not_cause_double_release():
    """Cobre o bug relatado na revisão: release() do caminho normal lança
    exceção. Antes da correção, o except externo via `cap is not None` e
    chamava cap.release() de novo (2 chamadas). Depois da correção, cada
    VideoCapture é liberado exatamente uma vez mesmo quando release() falha,
    e a thread continua viva e reconectando com a próxima captura."""
    slot = LatestFrameSlot()
    caps: list[FakeCapture] = []

    def factory(url):
        if not caps:
            cap = ExplodingReleaseCapture(fail_after=2)
        else:
            cap = FakeCapture()  # captura seguinte é saudável e não lança
        caps.append(cap)
        return cap

    t = CameraThread(_cam(), slot, backoff_max=0.1, open_capture=factory)
    t.start()
    try:
        deadline = time.monotonic() + 5
        while len(caps) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(caps) >= 2, "não reconectou depois da falha no release()"
        assert caps[0].release_calls == 1, (
            f"release() chamado {caps[0].release_calls}x, esperado exatamente 1x "
            "mesmo com exceção"
        )

        # a thread precisa continuar viva e voltar a ficar ONLINE com a
        # próxima captura, mesmo depois da exceção no release() da anterior
        deadline = time.monotonic() + 3
        while t.state != CameraState.ONLINE and time.monotonic() < deadline:
            time.sleep(0.05)
        assert t.state == CameraState.ONLINE, "thread não sobreviveu à exceção no release()"
    finally:
        t.stop()


def test_read_exception_is_tolerated_like_ok_false():
    """Uma exceção isolada em read() deve contar como falha de leitura (o
    mesmo contador de ok=False), não forçar reconexão imediata — tolera um
    glitch pontual do decodificador até MAX_READ_FAILURES."""
    slot = LatestFrameSlot()
    cap = ExplodingReadCapture(raise_on_read=2)
    t = CameraThread(_cam(), slot, backoff_max=0.1, open_capture=lambda url: cap)
    t.start()
    try:
        deadline = time.monotonic() + 3
        while slot.peek() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert slot.peek() is not None, "não recebeu frames após a exceção isolada em read()"
        assert t.state == CameraState.ONLINE
        # a exceção não derrubou a captura: ainda é a mesma instância (não
        # houve reconexão/reabertura por causa de um único read() falho)
        assert cap.released is False
    finally:
        t.stop()


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
