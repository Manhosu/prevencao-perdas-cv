"""Sweep de calibração (spec §8): varre combinações de parâmetros sobre uma
pasta de clipes ROTULADOS e mede detecção × falso-positivo.

    videos/ocultacao/  -> clipes onde DEVE disparar
    videos/normal/     -> movimento comum onde NÃO deve disparar

Produz a tabela que responde 'quantos alertas falsos por dia a equipe aguenta':
escolhe-se a linha que respeita o teto do cliente e maximiza a detecção."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import DetectionConfig
from src.tools.replay import replay


@dataclass
class SweepRow:
    threshold: float
    dwell_seconds: float
    detected: int
    total_conceal: int
    false_per_hour: float


def _clips(d: Path) -> list[Path]:
    return sorted(p for p in Path(d).glob("*.mp4"))


def sweep(conceal_dir, normal_dir, engine, grid, base_cfg: DetectionConfig, every=2) -> list[SweepRow]:
    conceal = _clips(conceal_dir)
    normal = _clips(normal_dir)
    rows: list[SweepRow] = []

    for threshold, dwell in grid:
        cfg = copy.deepcopy(base_cfg)
        cfg.threshold = threshold
        cfg.dwell_seconds = dwell

        detected = 0
        for clip in conceal:
            s = replay(clip, cfg, engine, every=every)
            if s.events:
                detected += 1

        false_events = 0
        normal_seconds = 0.0
        for clip in normal:
            s = replay(clip, cfg, engine, every=every)
            false_events += len(s.events)
            if s.csv_rows:
                normal_seconds += s.csv_rows[-1]["ts"]
        fph = (false_events / normal_seconds * 3600.0) if normal_seconds > 0 else 0.0

        rows.append(SweepRow(threshold, dwell, detected, len(conceal), round(fph, 2)))
    return rows


def best_row(rows: list[SweepRow], max_false_per_hour: float) -> SweepRow:
    ok = [r for r in rows if r.false_per_hour <= max_false_per_hour]
    if ok:
        return max(ok, key=lambda r: (r.detected, -r.false_per_hour))
    return min(rows, key=lambda r: r.false_per_hour)


def format_table(rows: list[SweepRow]) -> str:
    out = [f"{'limiar':>7} {'dwell':>6} {'detectados':>11} {'falsos/hora':>12}"]
    out.append("-" * 40)
    for r in rows:
        out.append(f"{r.threshold:>7.2f} {r.dwell_seconds:>6.1f} "
                   f"{r.detected:>4}/{r.total_conceal:<4} {r.false_per_hour:>12.1f}")
    return "\n".join(out)


def main():
    from src.config.settings import AppConfig
    from src.inference.engine import InferenceEngine

    ap = argparse.ArgumentParser(description="Sweep de calibracao")
    ap.add_argument("--conceal", required=True)
    ap.add_argument("--normal", required=True)
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--max-false-per-hour", type=float, default=5.0)
    a = ap.parse_args()

    app = AppConfig.load(a.config)
    engine = InferenceEngine(app.inference)
    grid = [(t, d) for t in (0.5, 0.6, 0.7) for d in (1.0, 1.2, 1.5)]
    rows = sweep(a.conceal, a.normal, engine, grid, app.detection)
    print(format_table(rows))
    best = best_row(rows, a.max_false_per_hour)
    print(f"\nRecomendado (teto {a.max_false_per_hour}/h): limiar {best.threshold}, "
          f"dwell {best.dwell_seconds}s -> pega {best.detected}/{best.total_conceal}, "
          f"{best.false_per_hour} falsos/hora")


if __name__ == "__main__":
    main()
