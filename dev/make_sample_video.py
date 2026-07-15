"""Gera vídeo sintético para testar o caminho de captura (não serve para
testar detecção de pessoa — para isso, use dev/record_clips.py)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def synthetic_video(
    path: str | Path,
    seconds: float = 5,
    fps: int = 10,
    size: tuple[int, int] = (640, 360),
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    total = int(seconds * fps)
    for i in range(total):
        img = np.full((h, w, 3), 30, dtype=np.uint8)
        x = int((i / max(1, total - 1)) * (w - 60))
        cv2.rectangle(img, (x, h // 2 - 40), (x + 60, h // 2 + 40), (0, 200, 255), -1)
        cv2.putText(
            img, f"frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        writer.write(img)
    writer.release()
    return path
