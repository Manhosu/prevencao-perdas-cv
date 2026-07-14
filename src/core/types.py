"""Tipos compartilhados por todo o pipeline. Sem dependências de I/O."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# Índices dos keypoints no formato COCO-17 (padrão do YOLO-pose).
KP: dict[str, int] = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


@dataclass(frozen=True)
class BBox:
    """Caixa em pixels do frame completo."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Base da caixa: onde a pessoa toca o chão. É este ponto que decide
        se ela está dentro da zona monitorada."""
        return ((self.x1 + self.x2) / 2, self.y2)

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def expand(self, factor: float) -> "BBox":
        """Cresce a caixa em `factor` (0.2 = +20%) mantendo o centro."""
        dx = self.width * factor / 2
        dy = self.height * factor / 2
        return BBox(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy)

    def clip(self, w: float, h: float) -> "BBox":
        return BBox(
            max(0.0, self.x1), max(0.0, self.y1), min(w, self.x2), min(h, self.y2)
        )


@dataclass
class Frame:
    camera_name: str
    image: np.ndarray  # BGR, HxWx3
    ts: float  # time.monotonic() do momento da captura
    seq: int


@dataclass
class PersonDetection:
    bbox: BBox
    conf: float
    track_id: int | None = None


@dataclass
class ObjectDetection:
    label: str  # 'backpack' | 'handbag'
    bbox: BBox
    conf: float


@dataclass
class PersonPose:
    person: PersonDetection
    keypoints: np.ndarray  # (17, 3) — x, y, conf em pixels do frame completo


class CameraState(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    RECONNECTING = "reconnecting"
