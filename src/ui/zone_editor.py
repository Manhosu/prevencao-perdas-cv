"""Editor visual de zonas — casca fina de Qt em cima do `ZoneModel`.

O revendedor vê o snapshot da câmera e desenha o polígono da área a monitorar
por cima. Marcar só a gôndola crítica (em vez de vigiar o quadro inteiro) é o
que mais derruba alarme falso — por isso este é o widget mais valioso da
interface.

Toda a matemática (pontos normalizados, arrastar vértice, hit-test, salvar)
vem do `ZoneModel`, que não conhece Qt e é 100% testável sem tela. Este widget
só traduz eventos de mouse em chamadas do model e desenha o resultado — não
reimplementa geometria aqui.

A parte sutil é a conversão de coordenada: o snapshot é escalado para caber no
widget mantendo a proporção (letterbox), então o widget quase sempre tem uma
borda sem imagem. Essa conversão está isolada em `_widget_para_normalizado`
justamente para poder ser testada sem depender de eventos de mouse reais.
"""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import QWidget

from src.ui.zone_model import Point, Polygon, ZoneModel

# Texto mostrado quando não há nenhuma zona marcada (o padrão é vigiar tudo).
_EMPTY_ZONE_MESSAGE = (
    "Sem área marcada — o sistema vigia o quadro inteiro.\n"
    "Clique para marcar a área."
)

_BACKGROUND_COLOR = QColor("#1a1a1a")   # fundo do widget (também a "borda preta" do letterbox)
_OUTLINE_COLOR = QColor(0, 200, 255)
_FILL_COLOR = QColor(0, 200, 255, 60)
_VERTEX_COLOR = QColor(255, 255, 255)
_VERTEX_RADIUS = 5
_HIT_TOL = 0.02  # tolerancia normalizada p/ acertar um vertice existente (mesmo default do model)


def _bgr_to_qimage(img_bgr: np.ndarray) -> QImage:
    """Converte um frame OpenCV (BGR, HxWx3, uint8) para QImage.

    `.copy()` no QImage força uma cópia própria do buffer de pixels: sem ela,
    o QImage aponta pra memória do numpy array, que pode ser realocada/coletada
    assim que a função retornar — um crash silencioso clássico do PySide6.
    """
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return qimg.copy()


class ZoneEditor(QWidget):
    """Widget: mostra o snapshot da câmera e deixa o revendedor desenhar a
    área monitorada por cima, arrastando vértices com o mouse."""

    zonesChanged = Signal()

    def __init__(self, model: ZoneModel | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model if model is not None else ZoneModel()
        self._pixmap: QPixmap | None = None
        self._image_size: tuple[int, int] | None = None  # (w, h) do snapshot original
        self._dragging: tuple[int, int] | None = None  # (poly_i, pt_i) sendo arrastado
        self.setMinimumSize(320, 240)

    # --- API pública ---
    def set_snapshot(self, img_bgr: np.ndarray) -> None:
        """Define a imagem de fundo (BGR, como vem do OpenCV/`Frame.image`)."""
        h, w = img_bgr.shape[:2]
        self._image_size = (w, h)
        self._pixmap = QPixmap.fromImage(_bgr_to_qimage(img_bgr))
        self.update()

    def zones(self) -> list[Polygon]:
        return self.model.to_config()

    def set_zones(self, zones: list[Polygon]) -> None:
        before = self.model.to_config()
        self.model.clear()
        self.model.polygons = [list(p) for p in zones]
        self.update()
        if self.model.to_config() != before:
            self.zonesChanged.emit()

    # --- conversão de coordenada (a parte sutil) ---
    def _image_rect(self) -> QRectF | None:
        """Retângulo (coords do widget) onde o snapshot escalado cai,
        considerando o letterbox. None se não há snapshot ainda."""
        if self._image_size is None:
            return None
        img_w, img_h = self._image_size
        if img_w <= 0 or img_h <= 0:
            return None
        widget_w, widget_h = self.width(), self.height()
        if widget_w <= 0 or widget_h <= 0:
            return None
        scale = min(widget_w / img_w, widget_h / img_h)
        scaled_w, scaled_h = img_w * scale, img_h * scale
        x0 = (widget_w - scaled_w) / 2
        y0 = (widget_h - scaled_h) / 2
        return QRectF(x0, y0, scaled_w, scaled_h)

    def _widget_para_normalizado(self, pos: QPointF) -> Point | None:
        """Converte uma posição do mouse (coords do widget) em (x_n, y_n)
        normalizado sobre a imagem. `None` se o clique caiu na borda preta do
        letterbox ou fora da área da imagem (ou se ainda não há snapshot)."""
        rect = self._image_rect()
        if rect is None or not rect.contains(pos):
            return None
        img_w, img_h = self._image_size
        px = (pos.x() - rect.x()) / rect.width() * img_w
        py = (pos.y() - rect.y()) / rect.height() * img_h
        return ZoneModel.from_pixels(px, py, img_w, img_h)

    def _normalized_to_widget(self, x_n: float, y_n: float, rect: QRectF) -> QPointF:
        return QPointF(rect.x() + x_n * rect.width(), rect.y() + y_n * rect.height())

    # --- eventos de mouse: só traduzem para chamadas do model ---
    def _commit(self, fn, *args) -> None:
        """Executa uma ação que pode alterar os polígonos salvos e emite
        `zonesChanged` só se `zones()` de fato mudou."""
        before = self.model.to_config()
        fn(*args)
        if self.model.to_config() != before:
            self.zonesChanged.emit()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        ponto = self._widget_para_normalizado(event.position())
        if ponto is None:
            super().mousePressEvent(event)
            return
        x_n, y_n = ponto

        if event.button() == Qt.MouseButton.LeftButton:
            alvo = self.model.hit_test(x_n, y_n, tol=_HIT_TOL)
            if alvo is not None:
                self._dragging = alvo  # começa a arrastar o vértice existente
            else:
                self.model.add_point(x_n, y_n)  # ponto novo no polígono corrente
            self.update()
            return

        if event.button() == Qt.MouseButton.RightButton:
            alvo = self.model.hit_test(x_n, y_n, tol=_HIT_TOL)
            if alvo is not None:
                self._commit(self.model.remove_point, *alvo)
            else:
                self._commit(self.model.finish_polygon)  # botao direito em area vazia fecha
            self.update()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging is None:
            super().mouseMoveEvent(event)
            return
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            self._dragging = None
            return
        ponto = self._widget_para_normalizado(event.position())
        if ponto is None:
            return  # cursor saiu da área da imagem: mantém o vértice onde estava
        poly_i, pt_i = self._dragging
        self._commit(self.model.move_point, poly_i, pt_i, *ponto)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseDoubleClickEvent(event)
            return
        # o clique do primeiro toque do duplo-clique já chegou via mousePressEvent
        # (adicionou o ponto); aqui só fecha o polígono.
        self._commit(self.model.finish_polygon)
        self.update()

    # --- desenho ---
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), _BACKGROUND_COLOR)

            rect = self._image_rect()
            if rect is not None and self._pixmap is not None:
                painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
                self._draw_zones(painter, rect)

            if self.model.covers_whole_frame and not self.model.current:
                self._draw_placeholder(painter)
        finally:
            painter.end()

    def _draw_zones(self, painter: QPainter, rect: QRectF) -> None:
        for poly in self.model.polygons:
            points = [self._normalized_to_widget(x, y, rect) for x, y in poly]
            painter.setPen(QPen(_OUTLINE_COLOR, 2))
            painter.setBrush(QBrush(_FILL_COLOR))
            painter.drawPolygon(QPolygonF(points))
            self._draw_vertices(painter, points)

        if self.model.current:
            points = [self._normalized_to_widget(x, y, rect) for x, y in self.model.current]
            painter.setPen(QPen(_OUTLINE_COLOR, 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolyline(QPolygonF(points))
            self._draw_vertices(painter, points)

    def _draw_vertices(self, painter: QPainter, points: list[QPointF]) -> None:
        painter.setPen(QPen(_VERTEX_COLOR, 1))
        painter.setBrush(QBrush(_VERTEX_COLOR))
        for p in points:
            painter.drawEllipse(p, _VERTEX_RADIUS, _VERTEX_RADIUS)

    def _draw_placeholder(self, painter: QPainter) -> None:
        painter.setPen(QPen(_VERTEX_COLOR))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, _EMPTY_ZONE_MESSAGE)
