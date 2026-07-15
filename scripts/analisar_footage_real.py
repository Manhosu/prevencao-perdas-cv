"""Analisa vídeo real de loja com o motor do projeto (detect + pose + track).

Responde à pergunta que decide o projeto: nessas câmeras, o sistema detecta as
pessoas e os keypoints das MÃOS saem confiáveis o suficiente para a heurística
de ocultação funcionar?

Uso:
    python scripts/analisar_footage_real.py <video.mp4> [--device cpu] [--every 5]

Gera:
    <video>_anotado.mp4  — esqueleto + caixa + id por pessoa
    imprime um relatório de qualidade (deteccao, tamanho, confianca de punhos)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.config.settings import InferenceConfig
from src.core.types import BBox, KP
from src.detection.tracker import Tracker
from src.inference.engine import InferenceEngine

# Ligações COCO-17 para desenhar o esqueleto
EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]
WRISTS = (KP["left_wrist"], KP["right_wrist"])
SHOULDERS = (KP["left_shoulder"], KP["right_shoulder"])
HIPS = (KP["left_hip"], KP["right_hip"])
KP_CONF = 0.35


def draw(frame, poses, tracked):
    for pose, person in zip(poses, tracked):
        b = person.bbox
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 200, 0), 2)
        cv2.putText(frame, f"id{person.track_id}", (int(b.x1), int(b.y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        kp = pose.keypoints
        for a, c in EDGES:
            if kp[a, 2] > KP_CONF and kp[c, 2] > KP_CONF:
                cv2.line(frame, (int(kp[a, 0]), int(kp[a, 1])),
                         (int(kp[c, 0]), int(kp[c, 1])), (255, 180, 0), 2)
        for i in range(17):
            if kp[i, 2] > KP_CONF:
                cor = (0, 0, 255) if i in WRISTS else (0, 255, 255)
                r = 6 if i in WRISTS else 3
                cv2.circle(frame, (int(kp[i, 0]), int(kp[i, 1])), r, cor, -1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--every", type=int, default=5, help="processa 1 a cada N frames")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    video = Path(a.video)
    out = Path(a.out) if a.out else video.with_name(video.stem + "_anotado.mp4")

    eng = InferenceEngine(InferenceConfig(device=a.device, detect_bags=True))
    eng.warmup()
    tracker = Tracker(max_lost_seconds=2.0)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             max(1.0, fps / a.every), (w, h))

    n_proc = 0
    n_com_pessoa = 0
    n_pessoas_total = 0
    alturas = []
    conf_punho = []
    conf_ombro = []
    conf_quadril = []
    frames_bolsa = 0
    idx = 0
    ts = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % a.every != 0:
            idx += 1
            continue
        idx += 1
        n_proc += 1
        ts += a.every / fps

        persons, objects = eng.detect(frame)
        tracked = tracker.update(persons, ts)
        if objects:
            frames_bolsa += 1
        if tracked:
            n_com_pessoa += 1
            n_pessoas_total += len(tracked)
            kps = eng.pose(frame, [p.bbox for p in tracked])
            poses = []
            for p, k in zip(tracked, kps):
                alturas.append(p.bbox.height)
                # média da confiança por grupo de keypoint (só quando > 0)
                for grp, alvo in ((WRISTS, conf_punho), (SHOULDERS, conf_ombro), (HIPS, conf_quadril)):
                    vals = [k[i, 2] for i in grp if k[i, 2] > 0]
                    if vals:
                        alvo.append(float(np.mean(vals)))
                # embrulha em objeto simples p/ desenhar
                class _P:  # noqa
                    pass
                pp = _P(); pp.keypoints = k
                poses.append(pp)
            frame = draw(frame, poses, tracked)
            # marca bolsas detectadas
            for o in objects:
                bb = o.bbox
                cv2.rectangle(frame, (int(bb.x1), int(bb.y1)), (int(bb.x2), int(bb.y2)), (255, 0, 255), 2)
                cv2.putText(frame, o.label, (int(bb.x1), int(bb.y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        writer.write(frame)

    cap.release()
    writer.release()

    def pct(v, p):
        return float(np.percentile(v, p)) if v else 0.0

    def m(v):
        return float(np.mean(v)) if v else 0.0

    print(f"\n=== ANÁLISE: {video.name} ===")
    print(f"frames processados: {n_proc}  (1 a cada {a.every})")
    print(f"frames com pessoa: {n_com_pessoa} ({100*n_com_pessoa/max(1,n_proc):.0f}%)")
    print(f"media de pessoas por frame com gente: {n_pessoas_total/max(1,n_com_pessoa):.1f}")
    print(f"frames com bolsa/mochila detectada: {frames_bolsa}")
    print(f"altura da pessoa (px): mediana {pct(alturas,50):.0f}  |  p10 {pct(alturas,10):.0f}  p90 {pct(alturas,90):.0f}")
    print(f"  (min_person_px padrao = 120; abaixo disso a pose e descartada)")
    print(f"confianca media dos keypoints (0-1):")
    print(f"  PUNHOS  : {m(conf_punho):.2f}   <- o mais critico p/ ocultacao")
    print(f"  ombros  : {m(conf_ombro):.2f}")
    print(f"  quadril : {m(conf_quadril):.2f}")
    print(f"video anotado: {out}")


if __name__ == "__main__":
    main()
