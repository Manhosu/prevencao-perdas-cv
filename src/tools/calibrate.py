"""Sweep de calibração (spec §8): varre combinações de parâmetros sobre uma
pasta de clipes ROTULADOS e mede detecção × falso-positivo.

    videos/ocultacao/  -> clipes onde DEVE disparar
    videos/normal/     -> movimento comum onde NÃO deve disparar

Produz a tabela que responde 'quantos alertas falsos por dia a equipe aguenta':
escolhe-se a linha que respeita o teto do cliente e maximiza a detecção."""
from __future__ import annotations

import argparse
import copy
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.config.settings import DetectionConfig
from src.tools.replay import replay

log = logging.getLogger(__name__)


@dataclass
class SweepRow:
    threshold: float
    dwell_seconds: float
    detected: int
    total_conceal: int
    false_per_hour: float  # NaN quando normal/ nao produziu segundos medidos


def _clips(d: Path) -> list[Path]:
    return sorted(p for p in Path(d).glob("*.mp4"))


def _fph_or_worst(row: SweepRow) -> float:
    """Chave de ordenação para 'menor falso/hora': NaN (não medido) nunca deve
    parecer melhor que um valor medido, então vira +inf (pior caso)."""
    return float("inf") if math.isnan(row.false_per_hour) else row.false_per_hour


def _format_fph(value: float) -> str:
    return "nao medido" if math.isnan(value) else f"{value:.1f}"


def sweep(conceal_dir, normal_dir, engine, grid, base_cfg: DetectionConfig, every=2) -> list[SweepRow]:
    conceal = _clips(conceal_dir)
    normal = _clips(normal_dir)
    rows: list[SweepRow] = []
    warned_no_normal_data = False

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
                # s.csv_rows[-1]["ts"] é o INÍCIO do último frame processado
                # e subestimaria a duração em ~1 passo de amostragem; em vez
                # disso, reabre o clipe só para ler o fps real e estima a
                # duração como (linhas do CSV) * every / fps, que cobre a
                # janela completa de cada frame amostrado.
                cap = cv2.VideoCapture(str(clip))
                fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
                cap.release()
                normal_seconds += len(s.csv_rows) * every / fps

        if normal_seconds > 0:
            fph = round(false_events / normal_seconds * 3600.0, 2)
        else:
            # pasta normal/ vazia ou sem clipes válidos: não dá pra medir
            # falsos/hora, e 0.0 seria uma promessa falsa ("zero falsos!").
            fph = float("nan")
            if not warned_no_normal_data:
                log.warning(
                    "pasta normal (%s) nao tem clipes validos - falsos/hora "
                    "nao puderam ser medidos (aparecera como 'nao medido' na tabela)",
                    normal_dir,
                )
                warned_no_normal_data = True

        rows.append(SweepRow(threshold, dwell, detected, len(conceal), fph))
    return rows


def best_row(rows: list[SweepRow], max_false_per_hour: float) -> SweepRow:
    # NaN (nao medido) nunca satisfaz "<=", entao linhas sem falsos/hora
    # medidos ja ficam de fora de `ok` automaticamente.
    ok = [r for r in rows if r.false_per_hour <= max_false_per_hour]
    if ok:
        return max(ok, key=lambda r: (r.detected, -r.false_per_hour))
    return min(rows, key=_fph_or_worst)


def format_table(rows: list[SweepRow]) -> str:
    header = f"{'limiar':>7} {'espera(s)':>9} {'detectados':>11} {'falsos/hora':>12}"
    out = [header, "-" * len(header)]
    for r in rows:
        det_str = f"{r.detected}/{r.total_conceal}"
        out.append(f"{r.threshold:>7.2f} {r.dwell_seconds:>9.1f} "
                   f"{det_str:>11} {_format_fph(r.false_per_hour):>12}")
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
          f"{_format_fph(best.false_per_hour)} falsos/hora")


if __name__ == "__main__":
    main()
