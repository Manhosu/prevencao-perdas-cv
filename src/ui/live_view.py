"""Grade de câmeras ao vivo — a tela que dá confiança de que o sistema está de
pé. O lojista abre o programa e vê, de cara, cada câmera com o estado
(online/offline/reconectando) e o FPS.

O `LiveViewModel` não conhece Qt: só repassa o status do `Pipeline`, busca o
frame mais recente do slot e desenha a zona monitorada por cima do snapshot
(cv2 é aceitável aqui — não é núcleo de detecção, é só desenho para o
preview). O `LiveViewWidget` é a casca fina de PySide6 que só *lê* esse
estado a cada tique de `QTimer` — nunca roda inferência nem bloqueia a
thread da UI."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from src.ui.zone_model import Polygon, ZoneModel

_ZONE_COLOR_BGR = (0, 200, 255)  # mesma cor do editor de zonas
_ZONE_THICKNESS = 2


class LiveViewModel:
    """Camada entre o `Pipeline` e a tela — sem Qt, 100% testável sem tela."""

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def status(self) -> dict[str, dict]:
        """Repassa `pipeline.status()`: por câmera, `state`/`fps`/`dropped`."""
        return self.pipeline.status()

    def snapshot(self, camera_name: str) -> np.ndarray | None:
        """Frame mais recente da câmera, ou `None` se ainda não chegou nenhum
        (câmera nova) ou a câmera não existe."""
        slot = self.pipeline.slots.get(camera_name)
        if slot is None:
            return None
        frame = slot.peek()
        return frame.image if frame is not None else None

    def overlay_zones(self, img: np.ndarray, zones: list[Polygon]) -> np.ndarray:
        """Desenha os polígonos da zona (coords normalizadas 0–1) sobre uma
        CÓPIA da imagem — nunca altera o frame original, que pode estar sendo
        usado em outro lugar (buffer de evidência, outra aba)."""
        out = img.copy()
        h, w = out.shape[:2]
        for poly in zones:
            if len(poly) < 3:
                continue
            pts = np.array(
                [ZoneModel.to_pixels(x_n, y_n, w, h) for x_n, y_n in poly], dtype=np.int32
            )
            cv2.polylines(out, [pts], isClosed=True, color=_ZONE_COLOR_BGR, thickness=_ZONE_THICKNESS)
        return out

    def resumo(self) -> str:
        """Texto curto tipo '3 de 5 câmeras online', pro lojista bater o olho."""
        status = self.status()
        total = len(status)
        online = sum(1 for info in status.values() if info.get("state") == "online")
        return f"{online} de {total} câmeras online"


# --- widget --------------------------------------------------------------------

_STATE_LABELS = {
    "online": "Online",
    "offline": "Offline",
    "reconnecting": "Reconectando",
}

_STATE_COLORS = {
    "online": "#2ecc71",  # verde
    "offline": "#e74c3c",  # vermelho
    "reconnecting": "#f1c40f",  # amarelo
}

_PREVIEW_WIDTH = 240
_PREVIEW_HEIGHT = 160
_DEFAULT_INTERVAL_MS = 1000
_COLUMNS = 3


def _bgr_to_qimage(img_bgr: np.ndarray) -> QImage:
    """Converte um frame OpenCV (BGR, HxWx3, uint8) para QImage.

    `.copy()` força o QImage a ter buffer próprio: sem isso ele aponta pra
    memória do numpy array, que pode sumir assim que a função retornar."""
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return qimg.copy()


class _CameraCard(QFrame):
    """Um quadro da grade: nome, preview (ou placeholder), badge de estado e FPS."""

    def __init__(self, camera_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.camera_name = camera_name
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self._name_label = QLabel(camera_name)
        self._name_label.setStyleSheet("font-weight: bold;")

        self._preview = QLabel(camera_name)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setFixedSize(_PREVIEW_WIDTH, _PREVIEW_HEIGHT)
        self._preview.setWordWrap(True)
        self._preview.setStyleSheet("background-color: #1a1a1a; color: #cccccc;")

        self._badge = QLabel("●")
        self._state_label = QLabel("—")
        self._fps_label = QLabel("")

        status_row = QHBoxLayout()
        status_row.addWidget(self._badge)
        status_row.addWidget(self._state_label)
        status_row.addStretch()
        status_row.addWidget(self._fps_label)

        layout = QVBoxLayout(self)
        layout.addWidget(self._name_label)
        layout.addWidget(self._preview)
        layout.addLayout(status_row)

    def update_status(self, state: str, fps: float) -> None:
        cor = _STATE_COLORS.get(state, "#888888")
        self._badge.setStyleSheet(f"color: {cor};")
        self._state_label.setText(_STATE_LABELS.get(state, state))
        self._fps_label.setText(f"{fps:.1f} FPS")

    def update_snapshot(self, img: np.ndarray | None) -> None:
        if img is None:
            # sem frame ainda (camera nova ou offline): mostra o nome no lugar
            self._preview.setPixmap(QPixmap())
            self._preview.setText(self.camera_name)
            return
        pixmap = QPixmap.fromImage(_bgr_to_qimage(img)).scaled(
            self._preview.width(),
            self._preview.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(pixmap)


class LiveViewWidget(QWidget):
    """Grade de câmeras ao vivo — casca fina de Qt em cima do `LiveViewModel`.

    Atualiza por `QTimer` (nunca bloqueia): a cada tique só *lê* o status do
    pipeline e o snapshot mais recente de cada slot — a inferência roda nas
    threads do pipeline, nunca aqui."""

    def __init__(
        self,
        model: LiveViewModel,
        parent: QWidget | None = None,
        interval_ms: int = _DEFAULT_INTERVAL_MS,
    ) -> None:
        super().__init__(parent)
        self.model = model
        self._cards: dict[str, _CameraCard] = {}

        self._grid = QGridLayout(self)
        self._build_cards()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(interval_ms)

        self.refresh()

    def _build_cards(self) -> None:
        nomes = sorted(self.model.status().keys())
        for i, nome in enumerate(nomes):
            card = _CameraCard(nome)
            self._cards[nome] = card
            self._grid.addWidget(card, i // _COLUMNS, i % _COLUMNS)

    def refresh(self) -> None:
        """Chamado pelo `QTimer`: só leitura de estado, nunca bloqueia a UI."""
        status = self.model.status()
        for nome, card in self._cards.items():
            info = status.get(nome, {})
            card.update_status(info.get("state", "offline"), info.get("fps", 0.0))
            card.update_snapshot(self.model.snapshot(nome))
