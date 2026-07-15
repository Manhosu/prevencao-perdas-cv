"""Associação de identidade entre frames (IoU + idade).

Manter o track_id estável é requisito da lógica de ocultação: o dwell da mão
na zona é acumulado POR PESSOA. Se o id trocar no meio do gesto, o contador
zera e o furto passa batido. Por isso o track sobrevive a alguns frames sem
detecção (pessoa passa atrás de uma gôndola)."""
from __future__ import annotations

from dataclasses import dataclass

from src.core.types import BBox, PersonDetection

IOU_MATCH_MIN = 0.25


def iou(a: BBox, b: BBox) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    bbox: BBox
    last_seen: float


class Tracker:
    def __init__(self, max_lost_seconds: float = 2.0) -> None:
        self.max_lost_seconds = max_lost_seconds
        self._tracks: list[_Track] = []
        self._next_id = 1

    def active_ids(self) -> set[int]:
        return {t.track_id for t in self._tracks}

    def update(self, persons: list[PersonDetection], ts: float) -> list[PersonDetection]:
        self._tracks = [
            t for t in self._tracks if ts - t.last_seen <= self.max_lost_seconds
        ]

        # Guloso pelo melhor IoU: com poucas pessoas e câmera fixa, resolve.
        pairs = sorted(
            (
                (iou(p.bbox, t.bbox), pi, ti)
                for pi, p in enumerate(persons)
                for ti, t in enumerate(self._tracks)
            ),
            reverse=True,
        )
        taken_p: set[int] = set()
        taken_t: set[int] = set()
        assigned: dict[int, int] = {}  # índice da pessoa -> track_id

        for score, pi, ti in pairs:
            if score < IOU_MATCH_MIN or pi in taken_p or ti in taken_t:
                continue
            track = self._tracks[ti]
            track.bbox = persons[pi].bbox
            track.last_seen = ts
            assigned[pi] = track.track_id
            taken_p.add(pi)
            taken_t.add(ti)

        out: list[PersonDetection] = []
        for pi, p in enumerate(persons):
            if pi in assigned:
                tid = assigned[pi]
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks.append(_Track(tid, p.bbox, ts))
            out.append(PersonDetection(bbox=p.bbox, conf=p.conf, track_id=tid))
        return out
