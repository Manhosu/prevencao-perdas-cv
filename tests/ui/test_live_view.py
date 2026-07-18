"""Testes da grade de câmeras ao vivo.

O foco do model (`LiveViewModel`, sem Qt) é: o status é repassado sem
alteração, o snapshot vem do slot mais recente (e `None` quando vazio), o
overlay desenha a zona sem alterar o frame original, e o resumo conta as
câmeras online corretamente. O widget só é exercitado offscreen (instancia e
`.grab()` sem exceção) — o `QT_QPA_PLATFORM=offscreen` vem do conftest.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.types import Frame
from src.ui.live_view import LiveViewModel, LiveViewWidget


class FakeSlot:
    """Dublê de `LatestFrameSlot`: só o `.peek()` que o model usa."""

    def __init__(self, frame: Frame | None = None) -> None:
        self._frame = frame

    def peek(self) -> Frame | None:
        return self._frame


class FakePipeline:
    """Dublê do `Pipeline` real — sem threads, sem inferência, sem modelo.
    Só os dois atributos que o `LiveViewModel` consome."""

    def __init__(self, status: dict, slots: dict | None = None) -> None:
        self._status = status
        self.slots = slots or {}

    def status(self) -> dict:
        return self._status


def _frame_img(w: int = 8, h: int = 6, value: int = 0) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


# --- status ------------------------------------------------------------------


def test_status_repassa_o_pipeline():
    status = {
        "Caixa 01": {"state": "online", "fps": 8.2, "dropped": 0},
        "Corredor": {"state": "offline", "fps": 0.0, "dropped": 3},
    }
    model = LiveViewModel(FakePipeline(status))
    assert model.status() == status


# --- snapshot ------------------------------------------------------------------


def test_snapshot_vem_do_slot():
    img = _frame_img(value=42)
    slot = FakeSlot(Frame(camera_name="Caixa 01", image=img, ts=1.0, seq=1))
    model = LiveViewModel(FakePipeline({}, {"Caixa 01": slot}))

    snap = model.snapshot("Caixa 01")

    assert snap is not None
    assert np.array_equal(snap, img)


def test_snapshot_none_quando_slot_vazio():
    model = LiveViewModel(FakePipeline({}, {"Caixa 01": FakeSlot(None)}))
    assert model.snapshot("Caixa 01") is None


def test_snapshot_none_para_camera_desconhecida():
    model = LiveViewModel(FakePipeline({}, {}))
    assert model.snapshot("nao existe") is None


# --- overlay_zones ---------------------------------------------------------------


def test_overlay_desenha_e_nao_altera_original():
    img = _frame_img(w=100, h=100, value=0)
    original = img.copy()
    model = LiveViewModel(FakePipeline({}))

    zonas = [[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]]
    resultado = model.overlay_zones(img, zonas)

    assert not np.array_equal(resultado, img)  # a zona foi desenhada na copia
    assert np.array_equal(img, original)  # o frame original nao mudou


def test_overlay_sem_zonas_devolve_copia_intacta():
    img = _frame_img(w=20, h=20, value=7)
    model = LiveViewModel(FakePipeline({}))

    resultado = model.overlay_zones(img, [])

    assert np.array_equal(resultado, img)
    resultado[0, 0] = 255  # mexe só na copia
    assert img[0, 0, 0] == 7  # o original continua intacto (nao e o mesmo buffer)


def test_overlay_ignora_poligono_com_menos_de_tres_pontos():
    img = _frame_img(w=20, h=20, value=3)
    model = LiveViewModel(FakePipeline({}))

    resultado = model.overlay_zones(img, [[(0.1, 0.1), (0.5, 0.5)]])

    assert np.array_equal(resultado, img)  # poligono invalido nao desenha nada


# --- resumo ------------------------------------------------------------------


def test_resumo_conta_as_online():
    status = {
        "Caixa 01": {"state": "online", "fps": 8.0, "dropped": 0},
        "Caixa 02": {"state": "online", "fps": 7.5, "dropped": 0},
        "Corredor": {"state": "offline", "fps": 0.0, "dropped": 5},
        "Estoque": {"state": "reconnecting", "fps": 0.0, "dropped": 1},
    }
    model = LiveViewModel(FakePipeline(status))
    assert model.resumo() == "2 de 4 câmeras online"


def test_resumo_sem_cameras():
    model = LiveViewModel(FakePipeline({}))
    assert model.resumo() == "0 de 0 câmeras online"


def test_resumo_todas_offline():
    status = {"Caixa 01": {"state": "offline", "fps": 0.0, "dropped": 1}}
    model = LiveViewModel(FakePipeline(status))
    assert model.resumo() == "0 de 1 câmeras online"


# --- widget (offscreen) -------------------------------------------------------


def test_widget_instancia_e_grab_sem_excecao():
    status = {
        "Caixa 01": {"state": "online", "fps": 8.0, "dropped": 0},
        "Corredor": {"state": "offline", "fps": 0.0, "dropped": 2},
    }
    img = _frame_img(w=64, h=48, value=10)
    slots = {
        "Caixa 01": FakeSlot(Frame(camera_name="Caixa 01", image=img, ts=1.0, seq=1)),
        "Corredor": FakeSlot(None),
    }
    model = LiveViewModel(FakePipeline(status, slots))
    widget = LiveViewWidget(model)
    widget.resize(640, 480)

    pixmap = widget.grab()

    assert not pixmap.isNull()


def test_widget_refresh_sem_nenhum_snapshot_nao_explode():
    status = {"Sem Sinal": {"state": "reconnecting", "fps": 0.0, "dropped": 0}}
    model = LiveViewModel(FakePipeline(status, {"Sem Sinal": FakeSlot(None)}))
    widget = LiveViewWidget(model)

    widget.refresh()  # so leitura de estado -- nao pode levantar excecao
