import numpy as np
import pytest

from src.core.types import BBox, PersonDetection
from src.tools.benchmark import BenchmarkReport, BenchmarkRow, benchmark


class SlowFakeEngine:
    """Cada detect custa ~5ms; cada pose custa ~10ms por pessoa."""

    def __init__(self, people: int):
        self.people = people

    def detect(self, image):
        _ = np.sum(image[:64, :64])  # trabalho de verdade, curto
        boxes = [
            PersonDetection(bbox=BBox(10 * i, 10, 60 + 10 * i, 200), conf=0.9)
            for i in range(self.people)
        ]
        return boxes, []

    def pose(self, image, boxes):
        out = []
        for _ in boxes:
            kp = np.zeros((17, 3), dtype=np.float32)
            kp[:, 2] = 0.9
            out.append(kp)
        return out

    def warmup(self):
        pass


def test_benchmark_produces_a_row_per_scenario():
    report = benchmark(
        SlowFakeEngine(people=1),
        scenarios=[(1, 1), (4, 1)],
        seconds_per_scenario=0.3,
    )
    assert len(report.rows) == 2
    assert report.rows[0].cameras == 1
    assert report.rows[1].cameras == 4
    assert report.rows[0].fps_sustained > 0


def test_more_cameras_lowers_fps_per_camera():
    report = benchmark(
        SlowFakeEngine(people=1), scenarios=[(1, 1), (8, 1)], seconds_per_scenario=0.4
    )
    assert report.rows[1].fps_sustained < report.rows[0].fps_sustained


def test_recommend_picks_the_largest_camera_count_above_min_fps():
    report = BenchmarkReport(
        rows=[
            BenchmarkRow(cameras=1, people_per_frame=1, fps_sustained=20.0, cpu_percent=30),
            BenchmarkRow(cameras=5, people_per_frame=1, fps_sustained=8.0, cpu_percent=60),
            BenchmarkRow(cameras=8, people_per_frame=1, fps_sustained=5.5, cpu_percent=80),
            BenchmarkRow(cameras=12, people_per_frame=1, fps_sustained=2.0, cpu_percent=98),
        ],
        cpu_name="Intel i5",
        cores=4,
        ram_gb=8.0,
    )
    texto = report.recommend(min_fps=5.0)
    assert "8 câmeras" in texto
    assert "12" not in texto


def test_recommend_warns_when_nothing_meets_min_fps():
    report = BenchmarkReport(
        rows=[BenchmarkRow(cameras=1, people_per_frame=1, fps_sustained=1.0, cpu_percent=99)],
        cpu_name="Celeron",
        cores=2,
        ram_gb=4.0,
    )
    texto = report.recommend(min_fps=5.0)
    assert "não" in texto.lower()


def test_as_text_mentions_hardware_and_rows():
    report = BenchmarkReport(
        rows=[BenchmarkRow(cameras=5, people_per_frame=2, fps_sustained=6.0, cpu_percent=70)],
        cpu_name="Intel i5-8250U",
        cores=4,
        ram_gb=8.0,
    )
    t = report.as_text()
    assert "Intel i5-8250U" in t
    assert "5" in t and "6.0" in t
