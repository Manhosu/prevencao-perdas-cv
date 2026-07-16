"""Roda o pipeline de detecção + ocultação sobre um arquivo de vídeo, como se
fosse uma câmera ao vivo. Produz vídeo anotado (esqueleto + score) e um CSV com
o score por frame. É a ferramenta de validação e calibração sem loja."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.config.settings import AppConfig, DetectionConfig
from src.core.types import KP
from src.detection.concealment import ConcealmentAnalyzer, ConcealmentEvent
from src.detection.tracker import Tracker

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
WRISTS = (KP["left_wrist"], KP["right_wrist"])


@dataclass
class ReplaySummary:
    frames: int
    frames_with_person: int
    events: list[ConcealmentEvent] = field(default_factory=list)
    csv_rows: list[dict] = field(default_factory=list)


def _annotate(frame, poses, tracked, score, fired):
    for pose, person in zip(poses, tracked):
        b = person.bbox
        cor = (0, 0, 255) if fired else (0, 200, 0)
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), cor, 2)
        kp = pose.keypoints
        for a, c in EDGES:
            if kp[a, 2] > 0.35 and kp[c, 2] > 0.35:
                cv2.line(frame, (int(kp[a, 0]), int(kp[a, 1])),
                         (int(kp[c, 0]), int(kp[c, 1])), (255, 180, 0), 2)
        for wi in WRISTS:
            if kp[wi, 2] > 0.35:
                cv2.circle(frame, (int(kp[wi, 0]), int(kp[wi, 1])), 6, (0, 0, 255), -1)
    if fired:
        # Barra vermelha no topo, bem visível — o alerta fica na tela por
        # alert_hold_seconds (não pisca em 1 quadro só), para dar tempo de
        # um humano ver. É o mesmo alerta que iria para o Telegram.
        fw = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (fw, 44), (0, 0, 255), -1)
        cv2.putText(frame, f"OCULTACAO DETECTADA  (score {score:.2f})", (12, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    else:
        cv2.putText(frame, f"score {score:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame


def replay(video_path, cfg: DetectionConfig, engine, out_video=None, out_csv=None,
           every=1, alert_hold_seconds=0.0):
    from src.core.types import PersonPose

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    engine.warmup()
    tracker = Tracker(max_lost_seconds=cfg.guards.track_lost_seconds)
    analyzer = ConcealmentAnalyzer(cfg, fps_hint=fps / every)

    writer = None
    if out_video:
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                 max(1.0, fps / every), (w, h))

    summary = ReplaySummary(0, 0)
    idx = 0
    held_until = -1.0  # o alerta visual fica na tela até este instante
    held_score = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every != 0:
            idx += 1
            continue
        ts = idx / fps
        idx += 1
        summary.frames += 1

        persons, objects = engine.detect(frame)
        tracked = tracker.update(persons, ts)
        max_score, top = 0.0, None
        events = []
        poses = []
        if tracked:
            summary.frames_with_person += 1
            kps = engine.pose(frame, [p.bbox for p in tracked])
            poses = [PersonPose(person=p, keypoints=k) for p, k in zip(tracked, kps)]
            events = analyzer.update(poses, objects, ts)
            for e in events:
                if e.score > max_score:
                    max_score, top = e.score, e
        else:
            analyzer.update([], [], ts)

        summary.events.extend(events)
        row = {
            "frame": summary.frames, "ts": round(ts, 3), "n_persons": len(tracked),
            "max_score": round(max_score, 3),
            "zone": top.zone if top else "",
            "dwell": top.signals["dwell"] if top else "",
            "approach": top.signals["approach"] if top else "",
            "vanish": top.signals["vanish"] if top else "",
            "retract": top.signals["retract"] if top else "",
        }
        summary.csv_rows.append(row)
        if writer is not None:
            # Segura o alerta na tela por alert_hold_seconds após disparar,
            # senão ele pisca em 1 quadro e ninguém vê.
            if events:
                held_until = ts + alert_hold_seconds
                held_score = max_score
            show_alert = ts <= held_until
            draw_score = held_score if show_alert else max_score
            writer.write(_annotate(frame, poses, tracked, draw_score, fired=show_alert))

    cap.release()
    if writer is not None:
        writer.release()
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(summary.csv_rows[0].keys()) if summary.csv_rows
                                else ["frame", "ts", "n_persons", "max_score", "zone",
                                      "dwell", "approach", "vanish", "retract"])
            wr.writeheader()
            wr.writerows(summary.csv_rows)
    return summary


def main():
    from src.inference.engine import InferenceEngine

    ap = argparse.ArgumentParser(description="Replay do detector sobre um video")
    ap.add_argument("video")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--out-video", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--alert-hold", type=float, default=1.5,
                    help="segundos que o alerta OCULTACAO fica na tela apos disparar")
    a = ap.parse_args()

    app = AppConfig.load(a.config)
    engine = InferenceEngine(app.inference)
    summary = replay(a.video, app.detection, engine,
                     out_video=a.out_video, out_csv=a.out_csv, every=a.every,
                     alert_hold_seconds=a.alert_hold)
    print(f"frames: {summary.frames} | com pessoa: {summary.frames_with_person} | "
          f"eventos de ocultacao: {len(summary.events)}")
    for e in summary.events:
        print(f"  t={e.ts:.1f}s id={e.track_id} zona={e.zone} score={e.score} {e.signals}")


if __name__ == "__main__":
    main()
