"""Testes do widget do editor de zonas.

O foco é o que é lógica (não pixel-a-pixel): a conversão de coordenada do
widget (com letterbox) para normalizado, e que os eventos de mouse chamam o
`ZoneModel` corretamente. `QT_QPA_PLATFORM=offscreen` vem do conftest.py
(que também garante a `QApplication` única da sessão).
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from src.ui.zone_editor import ZoneEditor


def _frame(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _mouse_event(kind, x, y, button, buttons):
    pos = QPointF(x, y)
    return QMouseEvent(kind, pos, pos, button, buttons, Qt.KeyboardModifier.NoModifier)


def _press(editor, x, y, button=Qt.MouseButton.LeftButton):
    editor.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, x, y, button, button))


def _release(editor, x, y, button=Qt.MouseButton.LeftButton):
    editor.mouseReleaseEvent(
        _mouse_event(QEvent.Type.MouseButtonRelease, x, y, button, Qt.MouseButton.NoButton)
    )


def _click(editor, x, y, button=Qt.MouseButton.LeftButton):
    _press(editor, x, y, button)
    _release(editor, x, y, button)


def _double_click(editor, x, y, button=Qt.MouseButton.LeftButton):
    editor.mouseDoubleClickEvent(
        _mouse_event(QEvent.Type.MouseButtonDblClick, x, y, button, button)
    )


def _drag_move(editor, x, y):
    """Evento de movimento com o botão esquerdo pressionado (arrastando)."""
    editor.mouseMoveEvent(
        _mouse_event(QEvent.Type.MouseMove, x, y, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    )


# --- conversão de coordenada (a parte sutil) -------------------------------


def test_sem_snapshot_conversao_e_none():
    editor = ZoneEditor()
    editor.resize(640, 480)
    assert editor._widget_para_normalizado(QPointF(100, 100)) is None


def test_conversao_sem_letterbox_quando_proporcao_bate():
    """640x480 num widget 640x480: mesma proporção (4:3), escala 1:1, sem
    borda. Confere com o mesmo caso já testado em ZoneModel diretamente."""
    editor = ZoneEditor()
    editor.set_snapshot(_frame(640, 480))
    editor.resize(640, 480)
    x_n, y_n = editor._widget_para_normalizado(QPointF(320, 120))
    assert x_n == pytest.approx(0.5)
    assert y_n == pytest.approx(0.25)


def test_conversao_com_letterbox_vertical():
    """Snapshot 640x480 (4:3) num widget 800x700 (nao e 4:3): a escala fica
    limitada pela largura (1.25x), sobrando faixa preta em cima/embaixo.

    Nota: 640x480 e 800x600 tem a MESMA proporcao (4:3) -- escalar um pro
    outro nao gera letterbox nenhum. Por isso o widget aqui e 800x700 (nao
    800x600): e o jeito de ter uma borda preta de verdade pra testar o `None`.
    """
    editor = ZoneEditor()
    editor.set_snapshot(_frame(640, 480))
    editor.resize(800, 700)

    # imagem escalada cai em (0,50)-(800,650): centro do widget bate com o
    # centro da imagem.
    x_n, y_n = editor._widget_para_normalizado(QPointF(400, 350))
    assert x_n == pytest.approx(0.5)
    assert y_n == pytest.approx(0.5)

    assert editor._widget_para_normalizado(QPointF(400, 10)) is None   # faixa preta de cima
    assert editor._widget_para_normalizado(QPointF(400, 690)) is None  # faixa preta de baixo


def test_conversao_com_letterbox_horizontal():
    """Mesma ideia, mas com o widget mais largo que a proporcao da imagem:
    faixas pretas nas laterais em vez de em cima/embaixo."""
    editor = ZoneEditor()
    editor.set_snapshot(_frame(640, 480))
    editor.resize(1000, 600)

    # imagem escalada cai em (100,0)-(900,600)
    x_n, y_n = editor._widget_para_normalizado(QPointF(500, 300))
    assert x_n == pytest.approx(0.5)
    assert y_n == pytest.approx(0.5)

    assert editor._widget_para_normalizado(QPointF(10, 300)) is None   # faixa preta da esquerda
    assert editor._widget_para_normalizado(QPointF(990, 300)) is None  # faixa preta da direita


# --- eventos de mouse traduzidos em chamadas do model ----------------------


def test_clique_esquerdo_adiciona_ponto_ao_model():
    editor = ZoneEditor()
    editor.set_snapshot(_frame(640, 480))
    editor.resize(640, 480)

    _click(editor, 320, 240)

    assert editor.model.current == [(0.5, 0.5)]
    assert editor.zones() == []  # ainda nao fechou o poligono


def test_arrastar_sobre_vertice_move_o_ponto():
    editor = ZoneEditor()
    editor.set_snapshot(_frame(640, 480))
    editor.resize(640, 480)
    editor.set_zones([[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]])

    _press(editor, 64, 48)     # exatamente sobre o vertice (0.1, 0.1)
    _drag_move(editor, 128, 96)  # arrasta pra (0.2, 0.2)
    _release(editor, 128, 96)

    assert editor.zones()[0][0] == pytest.approx((0.2, 0.2))


def test_duplo_clique_fecha_o_poligono():
    editor = ZoneEditor()
    editor.set_snapshot(_frame(640, 480))
    editor.resize(640, 480)

    _click(editor, 64, 48)     # (0.1, 0.1)
    _click(editor, 576, 48)    # (0.9, 0.1)
    _press(editor, 576, 432)   # (0.9, 0.9) -- o primeiro toque do duplo-clique
    _double_click(editor, 576, 432)

    zonas = editor.zones()
    assert len(zonas) == 1
    assert len(zonas[0]) == 3
    assert zonas[0][0] == pytest.approx((0.1, 0.1))
    assert zonas[0][2] == pytest.approx((0.9, 0.9))


def test_botao_direito_sobre_vertice_remove_o_ponto():
    editor = ZoneEditor()
    editor.set_snapshot(_frame(640, 480))
    editor.resize(640, 480)
    editor.set_zones([[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]])

    _press(editor, 64, 48, button=Qt.MouseButton.RightButton)  # sobre (0.1, 0.1)

    zonas = editor.zones()
    assert len(zonas) == 1
    assert len(zonas[0]) == 3
    assert (0.1, 0.1) not in zonas[0]


def test_botao_direito_em_area_vazia_fecha_o_poligono():
    editor = ZoneEditor()
    editor.set_snapshot(_frame(640, 480))
    editor.resize(640, 480)

    _click(editor, 64, 48)    # (0.1, 0.1)
    _click(editor, 576, 48)   # (0.9, 0.1)
    _click(editor, 576, 432)  # (0.9, 0.9)

    _press(editor, 300, 300, button=Qt.MouseButton.RightButton)  # area vazia

    zonas = editor.zones()
    assert len(zonas) == 1
    assert len(zonas[0]) == 3


# --- zones() / set_zones() --------------------------------------------------


def test_zones_devolve_o_que_o_model_tem():
    editor = ZoneEditor()
    editor.model.polygons = [[(0.2, 0.2), (0.5, 0.2), (0.5, 0.6)]]
    assert editor.zones() == editor.model.to_config()


def test_set_zones_popula_o_model():
    editor = ZoneEditor()
    editor.set_zones([[(0.1, 0.1), (0.6, 0.1), (0.6, 0.6)]])
    assert editor.zones() == [[(0.1, 0.1), (0.6, 0.1), (0.6, 0.6)]]


# --- sinal zonesChanged ------------------------------------------------------


def test_zoneschanged_emitido_ao_arrastar_vertice():
    editor = ZoneEditor()
    editor.set_snapshot(_frame(640, 480))
    editor.resize(640, 480)
    editor.set_zones([[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]])

    chamadas = []
    editor.zonesChanged.connect(lambda: chamadas.append(True))

    _press(editor, 64, 48)
    _drag_move(editor, 128, 96)

    assert len(chamadas) >= 1


def test_zoneschanged_emitido_ao_fechar_poligono():
    editor = ZoneEditor()
    editor.set_snapshot(_frame(640, 480))
    editor.resize(640, 480)

    chamadas = []
    editor.zonesChanged.connect(lambda: chamadas.append(True))

    _click(editor, 64, 48)
    _click(editor, 576, 48)
    _press(editor, 576, 432)
    _double_click(editor, 576, 432)

    assert len(chamadas) >= 1


def test_zoneschanged_nao_emitido_so_ao_adicionar_ponto_do_poligono_corrente():
    """Adicionar ponto ao polígono ainda aberto (`current`) nao muda o que
    `zones()` devolve -- so `finish_polygon`/mover/remover mudam."""
    editor = ZoneEditor()
    editor.set_snapshot(_frame(640, 480))
    editor.resize(640, 480)

    chamadas = []
    editor.zonesChanged.connect(lambda: chamadas.append(True))

    _click(editor, 64, 48)

    assert chamadas == []


def test_zoneschanged_emitido_pelo_set_zones():
    editor = ZoneEditor()
    chamadas = []
    editor.zonesChanged.connect(lambda: chamadas.append(True))

    editor.set_zones([[(0.1, 0.1), (0.6, 0.1), (0.6, 0.6)]])

    assert len(chamadas) >= 1


# --- paintEvent nao explode --------------------------------------------------


def test_paint_sem_snapshot_nao_explode():
    editor = ZoneEditor()
    editor.resize(400, 300)
    pixmap = editor.grab()
    assert not pixmap.isNull()


def test_paint_com_snapshot_e_zona_nao_explode():
    editor = ZoneEditor()
    editor.set_snapshot(_frame(640, 480))
    editor.set_zones([[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]])
    editor.resize(640, 480)
    pixmap = editor.grab()
    assert not pixmap.isNull()
