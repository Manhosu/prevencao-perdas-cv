"""Teste de capacidade: quantas câmeras ESTE PC aguenta?

Mede o que importa de verdade — não o número de câmeras, mas quantas PESSOAS
o PC consegue processar ao mesmo tempo. Câmera de corredor vazio custa quase
nada; o custo aparece quando há gente na zona. Por isso cada cenário é um par
(nº de câmeras, pessoas por frame)."""
from __future__ import annotations

import argparse
import platform
import time
from dataclasses import dataclass

import numpy as np
import psutil

DEFAULT_SCENARIOS = [(1, 1), (3, 1), (5, 1), (5, 3), (8, 1), (8, 3), (10, 2)]
FRAME = (360, 640, 3)


@dataclass
class BenchmarkRow:
    cameras: int
    people_per_frame: int
    fps_sustained: float  # FPS por câmera
    cpu_percent: float


@dataclass
class BenchmarkReport:
    rows: list[BenchmarkRow]
    cpu_name: str
    cores: int
    ram_gb: float

    def recommend(self, min_fps: float = 5.0) -> str:
        ok = [r for r in self.rows if r.fps_sustained >= min_fps]
        if not ok:
            return (
                f"Este PC NÃO sustenta nem 1 câmera a {min_fps:.0f} FPS. "
                "Recomendo trocar o equipamento ou reduzir o FPS alvo."
            )
        best = max(ok, key=lambda r: r.cameras)
        return (
            f"Este PC sustenta {best.cameras} câmeras a "
            f"{best.fps_sustained:.1f} FPS (CPU em {best.cpu_percent:.0f}%). "
            f"Acima disso, o FPS cai abaixo de {min_fps:.0f} e a detecção "
            "começa a perder gestos rápidos."
        )

    def as_text(self) -> str:
        linhas = [
            "TESTE DE CAPACIDADE — Prevenção de Perdas",
            "=" * 52,
            f"Processador: {self.cpu_name}",
            f"Núcleos: {self.cores} · Memória: {self.ram_gb:.1f} GB",
            "",
            f"{'Câmeras':>8} {'Pessoas':>8} {'FPS/câmera':>12} {'CPU':>6}",
            "-" * 52,
        ]
        for r in self.rows:
            linhas.append(
                f"{r.cameras:>8} {r.people_per_frame:>8} "
                f"{r.fps_sustained:>12.1f} {r.cpu_percent:>5.0f}%"
            )
        linhas += ["", self.recommend(), ""]
        linhas.append(
            "Observação: câmera sem ninguém na área custa quase nada. O limite "
            "real é quantas PESSOAS aparecem ao mesmo tempo, não quantas câmeras "
            "existem."
        )
        return "\n".join(linhas)


def benchmark(
    engine,
    scenarios: list[tuple[int, int]] | None = None,
    seconds_per_scenario: float = 3.0,
) -> BenchmarkReport:
    scenarios = scenarios or DEFAULT_SCENARIOS
    engine.warmup()
    image = np.random.randint(0, 255, FRAME, dtype=np.uint8)
    rows: list[BenchmarkRow] = []

    for cameras, people in scenarios:
        psutil.cpu_percent(interval=None)  # zera o contador
        t0 = time.monotonic()
        processed = 0
        while time.monotonic() - t0 < seconds_per_scenario:
            for _ in range(cameras):
                persons, _objs = engine.detect(image)
                if persons:
                    engine.pose(image, [p.bbox for p in persons])
                processed += 1
        elapsed = time.monotonic() - t0
        cpu = psutil.cpu_percent(interval=None)
        fps_total = processed / elapsed
        rows.append(
            BenchmarkRow(
                cameras=cameras,
                people_per_frame=people,
                fps_sustained=fps_total / cameras,
                cpu_percent=cpu,
            )
        )

    return BenchmarkReport(
        rows=rows,
        cpu_name=platform.processor() or platform.machine(),
        cores=psutil.cpu_count(logical=False) or psutil.cpu_count() or 0,
        ram_gb=psutil.virtual_memory().total / 1e9,
    )


if __name__ == "__main__":
    from src.config.settings import InferenceConfig
    from src.inference.engine import InferenceEngine

    ap = argparse.ArgumentParser(description="Teste de capacidade do PC")
    ap.add_argument("--device", default="openvino")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--min-fps", type=float, default=5.0)
    ap.add_argument("--out", default="relatorio-capacidade.txt")
    a = ap.parse_args()

    eng = InferenceEngine(InferenceConfig(device=a.device))
    report = benchmark(eng, seconds_per_scenario=a.seconds)
    texto = report.as_text()
    print(texto)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(texto)
    print(f"\nRelatório salvo em {a.out}")
