import threading

import numpy as np

from src.capture.frame_slot import LatestFrameSlot
from src.core.types import Frame


def _frame(seq: int) -> Frame:
    return Frame("cam1", np.zeros((2, 2, 3), dtype=np.uint8), ts=float(seq), seq=seq)


def test_get_returns_none_when_empty():
    assert LatestFrameSlot().get() is None


def test_put_then_get():
    slot = LatestFrameSlot()
    slot.put(_frame(1))
    assert slot.get().seq == 1


def test_get_consumes_the_frame():
    slot = LatestFrameSlot()
    slot.put(_frame(1))
    slot.get()
    assert slot.get() is None


def test_second_put_overwrites_and_counts_drop():
    slot = LatestFrameSlot()
    slot.put(_frame(1))
    slot.put(_frame(2))
    assert slot.get().seq == 2  # o frame velho foi descartado, não enfileirado
    assert slot.dropped == 1


def test_peek_does_not_consume():
    slot = LatestFrameSlot()
    slot.put(_frame(5))
    assert slot.peek().seq == 5
    assert slot.get().seq == 5


def test_thread_safety_under_concurrent_writes():
    slot = LatestFrameSlot()

    def writer(start: int):
        for i in range(start, start + 200):
            slot.put(_frame(i))

    threads = [threading.Thread(target=writer, args=(i * 1000,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert slot.get() is not None
    assert slot.dropped == 800 - 1
