"""Lógica das zonas monitoradas — SEM Qt.

Separar isto do widget é o que torna a parte que importa (a geometria) testável
sem depender de tela. O widget PySide6 é uma casca fina em cima disto.

Zonas ficam em coordenadas normalizadas (0–1) sobre o quadro: sobrevivem a troca
de resolução ou substream do DVR. É o mesmo contrato que o PersonGate consome."""
from __future__ import annotations

Point = tuple[float, float]
Polygon = list[Point]

MIN_POINTS = 3  # menos que isso não é área


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


class ZoneModel:
    def __init__(self, zones: list[Polygon] | None = None) -> None:
        self.polygons: list[Polygon] = [list(p) for p in (zones or [])]
        self._current: Polygon = []

    # --- construção ---
    def add_point(self, x_n: float, y_n: float) -> None:
        self._current.append((_clamp01(x_n), _clamp01(y_n)))

    def finish_polygon(self) -> None:
        if len(self._current) >= MIN_POINTS:
            self.polygons.append(self._current)
        self._current = []

    @property
    def current(self) -> Polygon:
        return list(self._current)

    # --- edição ---
    def hit_test(self, x_n: float, y_n: float, tol: float = 0.02) -> tuple[int, int] | None:
        """Qual vértice está sob o cursor (para arrastar)."""
        for pi, poly in enumerate(self.polygons):
            for vi, (px, py) in enumerate(poly):
                if abs(px - x_n) <= tol and abs(py - y_n) <= tol:
                    return pi, vi
        return None

    def move_point(self, poly_i: int, pt_i: int, x_n: float, y_n: float) -> None:
        self.polygons[poly_i][pt_i] = (_clamp01(x_n), _clamp01(y_n))

    def remove_point(self, poly_i: int, pt_i: int) -> None:
        poly = self.polygons[poly_i]
        del poly[pt_i]
        if len(poly) < MIN_POINTS:
            del self.polygons[poly_i]

    def remove_polygon(self, poly_i: int) -> None:
        del self.polygons[poly_i]

    def clear(self) -> None:
        self.polygons = []
        self._current = []

    # --- config ---
    def to_config(self) -> list[Polygon]:
        return [list(p) for p in self.polygons if len(p) >= MIN_POINTS]

    @property
    def covers_whole_frame(self) -> bool:
        """Sem zona = monitorar o quadro inteiro (o padrão que o cliente pediu
        para reduzir o trabalho de configuração por loja)."""
        return not self.to_config()

    # --- tela <-> normalizado ---
    @staticmethod
    def from_pixels(x: float, y: float, w: int, h: int) -> Point:
        return (_clamp01(x / w) if w else 0.0, _clamp01(y / h) if h else 0.0)

    @staticmethod
    def to_pixels(x_n: float, y_n: float, w: int, h: int) -> tuple[float, float]:
        return (x_n * w, y_n * h)
