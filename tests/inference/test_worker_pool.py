import threading
import time

import numpy as np

from src.capture.frame_slot import LatestFrameSlot
from src.core.types import Frame
from src.inference.scheduler import Scheduler
from src.inference.worker_pool import WorkerPool


def _frame(cam: str, seq: int) -> Frame:
    return Frame(cam, np.zeros((8, 8, 3), dtype=np.uint8), ts=float(seq), seq=seq)


def test_processes_frames_from_all_cameras():
    slots = {"a": LatestFrameSlot(), "b": LatestFrameSlot()}
    seen: list[str] = []
    lock = threading.Lock()

    def process(frame: Frame) -> bool:
        with lock:
            seen.append(frame.camera_name)
        return False

    pool = WorkerPool(slots, Scheduler(list(slots)), process, workers=2)
    pool.start()
    try:
        for i in range(20):
            slots["a"].put(_frame("a", i))
            slots["b"].put(_frame("b", i))
            time.sleep(0.02)
        deadline = time.monotonic() + 3
        while pool.processed < 4 and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        pool.stop()

    assert "a" in seen and "b" in seen
    assert pool.processed >= 4


def test_empty_slots_do_not_spin_the_cpu():
    slots = {"a": LatestFrameSlot()}
    pool = WorkerPool(slots, Scheduler(["a"]), lambda f: False, workers=1)
    pool.start()
    time.sleep(0.3)
    pool.stop()
    assert pool.processed == 0  # nada para processar, e nada quebrou


def test_process_exception_does_not_kill_the_worker():
    slots = {"a": LatestFrameSlot()}
    calls = {"n": 0}

    def process(frame: Frame) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("erro de inferência")
        return False

    pool = WorkerPool(slots, Scheduler(["a"]), process, workers=1)
    pool.start()
    try:
        deadline = time.monotonic() + 3
        while calls["n"] < 2 and time.monotonic() < deadline:
            slots["a"].put(_frame("a", calls["n"]))
            time.sleep(0.05)
    finally:
        pool.stop()
    assert calls["n"] >= 2, "o worker morreu na primeira exceção"
