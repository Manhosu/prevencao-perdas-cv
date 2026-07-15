"""Gate: a pessoa está dentro da área monitorada?

É este filtro que permite rodar 8-10 câmeras num PC de mercado: sem pessoa
na zona, nada de pose. Corredor vazio custa um detect e nada mais."""
from __future__ import annotations

import numpy as np

from src.core.types import PersonDetection

Polygon = list[tuple[float, float]]


class PersonGate:
    def __init__(self, zones: list[Polygon], frame_size: tuple[int, int]) -> None:
        """zones em coordenadas normalizadas (0-1); frame_size = (largura, altura).
        Lista vazia = monitorar o quadro inteiro."""
        w, h = frame_size
        self._polys = [
            np.array([(x * w, y * h) for x, y in poly], dtype=np.float32) for poly in zones
        ]

    def contains(self, person: PersonDetection) -> bool:
        if not self._polys:
            return True
        # O ponto que representa a pessoa é onde ela PISA, não seu centro:
        # uma pessoa alta pode ter o centro fora da zona e os pés dentro.
        import cv2

        x, y = person.bbox.foot_point
        return any(
            cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
            for poly in self._polys
        )

    def filter(self, persons: list[PersonDetection]) -> list[PersonDetection]:
        return [p for p in persons if self.contains(p)]
