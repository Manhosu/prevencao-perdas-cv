import numpy as np

from src.evidence.clip_buffer import ClipBuffer


def _img(v):
    return np.full((4, 4, 3), v, dtype=np.uint8)


def test_keeps_only_the_window():
    buf = ClipBuffer(seconds=2.0, fps_hint=5)
    for i in range(20):  # 4s a 5fps
        buf.add(_img(i), ts=i * 0.2)
    # janela de 2s termina no ts mais novo (3.8) -> mantem >= 1.8
    assert all(ts >= 1.8 - 1e-6 for ts, _ in buf.frames_between(0, 99))
    # 19 * 0.2 nao bate exatamente com o literal 3.8 em ponto flutuante
    # (erro de arredondamento do IEEE-754, nao do buffer) -> comparar com folga
    assert abs(buf.newest_ts - 3.8) < 1e-6


def test_frames_between_returns_window_in_order():
    buf = ClipBuffer(seconds=10.0, fps_hint=5)
    for i in range(10):
        buf.add(_img(i), ts=i * 0.2)
    got = buf.frames_between(0.4, 1.0)
    assert [round(ts, 2) for ts, _ in got] == [0.4, 0.6, 0.8, 1.0]
    assert got[0][1][0, 0, 0] == 2  # o frame de ts=0.4 e o i=2


def test_empty_buffer():
    buf = ClipBuffer(seconds=2.0, fps_hint=5)
    assert buf.newest_ts is None
    assert buf.frames_between(0, 1) == []
    assert len(buf) == 0


def test_add_copies_frame_so_caller_can_reuse_buffer():
    """A thread de captura reusa o array do frame; o buffer PRECISA copiar,
    senão o clipe sai todo com a mesma imagem."""
    buf = ClipBuffer(seconds=2.0, fps_hint=5)
    img = _img(1)
    buf.add(img, ts=0.0)
    img[:] = 99  # o chamador mexe no array depois
    _, guardado = buf.frames_between(0, 1)[0]
    assert guardado[0, 0, 0] == 1


def test_memory_is_bounded():
    buf = ClipBuffer(seconds=1.0, fps_hint=5)
    for i in range(1000):
        buf.add(_img(i % 255), ts=i * 0.2)
    assert len(buf) <= 8  # ~1s a 5fps, com folga
