"""Grava material de calibração pela webcam, já rotulado.

Uso:
    python dev/record_clips.py --label bolso --seconds 8
    python dev/record_clips.py --label normal --seconds 300

Saída:
    dev/videos/ocultacao/bolso_01.mp4     (labels: bolso, bolsa, roupa, cintura)
    dev/videos/normal/normal_01.mp4       (label: normal)

Encene devagar e depois em ritmo natural. É este material que calibra o
sistema enquanto o vídeo do cliente não chega.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

CONCEAL_LABELS = {"bolso", "bolsa", "roupa", "cintura"}


def _next_path(label: str) -> Path:
    sub = "normal" if label == "normal" else "ocultacao"
    out = Path("dev/videos") / sub
    out.mkdir(parents=True, exist_ok=True)
    n = len(list(out.glob(f"{label}_*.mp4"))) + 1
    return out / f"{label}_{n:02d}.mp4"


def record(label: str, seconds: float, camera: int, fps: int) -> Path:
    if label not in CONCEAL_LABELS | {"normal"}:
        raise SystemExit(f"label inválido: {label} (use {CONCEAL_LABELS} ou 'normal')")
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"não consegui abrir a câmera {camera}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    path = _next_path(label)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"[rec] gravando '{label}' por {seconds}s em {path} — ESC para parar")
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        preview = frame.copy()
        restante = seconds - (time.monotonic() - t0)
        cv2.putText(preview, f"{label}  {restante:4.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("gravando (ESC para parar)", preview)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[rec] salvo: {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="bolso | bolsa | roupa | cintura | normal")
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=int, default=15)
    a = ap.parse_args()
    record(a.label, a.seconds, a.camera, a.fps)
