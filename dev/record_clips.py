"""Grava material de calibração pela webcam, já rotulado.

Uso:
    python dev/record_clips.py --label bolso --seconds 8
    python dev/record_clips.py --label normal --seconds 300

Saída:
    dev/videos/ocultacao/bolso_01.mp4     (labels: bolso, bolsa, roupa, cintura)
    dev/videos/normal/normal_01.mp4       (label: normal)

Encene devagar e depois em ritmo natural. É este material que calibra o
sistema enquanto o vídeo do cliente não chega.

O FPS do arquivo final é o FPS *real* medido durante a gravação (frames
capturados / segundos decorridos), não o valor pedido em --fps: a webcam
raramente entrega exatamente o FPS pedido, e um FPS mentiroso no arquivo
corrompe a duração — que é a base da métrica de permanência em zona de
ocultação. Para isso, os frames são gravados primeiro num arquivo
temporário com um FPS provisório e depois reescritos com o FPS real.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

import cv2

CONCEAL_LABELS = {"bolso", "bolsa", "roupa", "cintura"}

DEFAULT_OUT_DIR = Path("dev/videos")


def _next_path(label: str, base: Path | None = None) -> Path:
    sub = "normal" if label == "normal" else "ocultacao"
    out = (base or DEFAULT_OUT_DIR) / sub
    out.mkdir(parents=True, exist_ok=True)
    indices = []
    for f in out.glob(f"{label}_*.mp4"):
        suffix = f.stem[len(f"{label}_"):]
        if suffix.isdigit():
            indices.append(int(suffix))
    n = max(indices, default=0) + 1
    path = out / f"{label}_{n:02d}.mp4"
    # Defesa extra: nunca sobrescrever, mesmo que o cálculo acima colida
    # por algum motivo (ex.: arquivo criado entre o glob e o uso).
    while path.exists():
        n += 1
        path = out / f"{label}_{n:02d}.mp4"
    return path


def _rewrite_with_fps(
    tmp_path: Path, final_path: Path, fps: float, size: tuple[int, int]
) -> int:
    """Relê os frames gravados com FPS provisório e regrava com o FPS real
    medido, um frame por vez — nunca carrega o vídeo inteiro em memória, o
    que importa para clipes longos (ex.: 300s a 30fps em 1280x720)."""
    reader = cv2.VideoCapture(str(tmp_path))
    writer = cv2.VideoWriter(str(final_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    written = 0
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        reader.release()
        writer.release()
    return written


def record(
    label: str,
    seconds: float,
    camera: int,
    fps: int,
    *,
    capture_factory: Callable[[], object] | None = None,
    out_dir: Path | None = None,
    show_preview: bool = True,
) -> Path:
    """Grava `seconds` segundos rotulados como `label`.

    `capture_factory`, se informado, substitui `cv2.VideoCapture(camera)` —
    permite injetar uma fonte de vídeo falsa em teste (qualquer objeto com
    `isOpened()`, `read()`, `release()` e `get(prop)`), sem precisar de
    webcam física.
    """
    if label not in CONCEAL_LABELS | {"normal"}:
        raise SystemExit(f"label inválido: {label} (use {CONCEAL_LABELS} ou 'normal')")

    capture_factory = capture_factory or (lambda: cv2.VideoCapture(camera))
    cap = capture_factory()
    if not cap.isOpened():
        raise SystemExit(f"não consegui abrir a câmera {camera}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    path = _next_path(label, base=out_dir)

    provisional_fps = fps if fps and fps > 0 else 15
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".mp4", dir=str(path.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    writer = cv2.VideoWriter(
        str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), provisional_fps, (w, h)
    )

    print(f"[rec] gravando '{label}' por {seconds}s em {path} — ESC para parar")
    frame_count = 0
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            frame_count += 1
            if show_preview:
                preview = frame.copy()
                remaining = seconds - (time.monotonic() - t0)
                cv2.putText(preview, f"{label}  {remaining:4.1f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow("gravando (ESC para parar)", preview)
                if cv2.waitKey(1) == 27:
                    break
    finally:
        elapsed = time.monotonic() - t0
        cap.release()
        writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    real_fps = frame_count / elapsed if elapsed > 0 and frame_count > 0 else provisional_fps
    try:
        _rewrite_with_fps(tmp_path, path, real_fps, (w, h))
    finally:
        tmp_path.unlink(missing_ok=True)

    duration = frame_count / real_fps if real_fps > 0 else 0.0
    print(
        f"[rec] salvo: {path} — {real_fps:.2f} fps reais, "
        f"{frame_count} frames, {duration:.1f}s de vídeo"
    )
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="bolso | bolsa | roupa | cintura | normal")
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument(
        "--fps", type=int, default=15,
        help="FPS provisório; o arquivo final usa o FPS real medido na gravação",
    )
    a = ap.parse_args()
    record(a.label, a.seconds, a.camera, a.fps)
