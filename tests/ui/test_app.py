"""Testes da janela principal — a peça que amarra tudo que o Plano 4 construiu.

Cobre: instanciar sem exceção com dublês de pipeline/db (nunca o Pipeline real
— isso carregaria modelo YOLO); as 4 abas existem; salvar zonas persiste no
config e `AppConfig.load` relê igual (round-trip); carregar zonas NÃO marca a
tela como "alterada" (ressalva 1 — `ZoneEditor.set_zones()` emite
`zonesChanged` mesmo quando chamado programaticamente, e a janela precisa
bloquear isso ao popular); e o botão "Procurar meu grupo" chama
`descobrir_grupos` (monkeypatchado — nunca rede real).

`QT_QPA_PLATFORM=offscreen` e a `QApplication` da sessão vêm do conftest.py.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.config.settings import AppConfig, CameraConfig, StoreConfig, TelegramConfig
from src.core.types import Frame
from src.storage.db import Database
from src.ui.app import MainWindow, _ZonedLiveViewModel


class FakeSlot:
    """Dublê de `LatestFrameSlot`: só o `.peek()` que o `LiveViewModel` usa."""

    def __init__(self, frame: Frame | None = None) -> None:
        self._frame = frame

    def peek(self) -> Frame | None:
        return self._frame


class FakePipeline:
    """Dublê do `Pipeline` real — sem threads, sem inferência, sem modelo.
    Só os dois atributos que a janela (via `LiveViewModel`) consome."""

    def __init__(self, status: dict | None = None, slots: dict | None = None) -> None:
        self._status = status or {}
        self.slots = slots or {}

    def status(self) -> dict:
        return self._status


def _frame_img(w: int = 64, h: int = 48, value: int = 10) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


@pytest.fixture(autouse=True)
def _sem_dialogos_modais(monkeypatch):
    """`QMessageBox.warning()`/`.information()` abrem um diálogo modal
    (`exec()`); num ambiente automatizado offscreen isso trava o teste para
    sempre, esperando um clique que nunca vem. Substitui por no-op só nesta
    suíte — o comportamento real (mostrar o aviso) continua intacto para o
    usuário de verdade."""
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "app.db")
    d.init_schema()
    yield d
    d.close()


def _cfg(cameras=None) -> AppConfig:
    return AppConfig(
        store=StoreConfig(id="loja1", name="Mercado Teste"),
        telegram=TelegramConfig(bot_token="123:ABC"),
        cameras=cameras or [],
    )


def _janela(tmp_path, db, cfg=None, status=None, slots=None):
    cfg = cfg if cfg is not None else _cfg()
    pipeline = FakePipeline(status=status, slots=slots)
    config_path = tmp_path / "config.json"
    cfg.save(config_path)
    return MainWindow(pipeline, db, cfg, config_path)


# --- instanciação e abas -----------------------------------------------------


def test_janela_instancia_sem_excecao(tmp_path, db):
    win = _janela(tmp_path, db)
    assert win is not None


def test_tem_quatro_abas(tmp_path, db):
    win = _janela(tmp_path, db)
    assert win.tabs.count() == 4
    titulos = [win.tabs.tabText(i) for i in range(4)]
    assert titulos == ["Ao vivo", "Câmeras & Zonas", "Eventos", "Configuração"]


def test_janela_com_cameras_e_status_nao_explode(tmp_path, db):
    camera = CameraConfig(name="Caixa 01", rtsp_url="rtsp://x", zones=[])
    status = {"Caixa 01": {"state": "online", "fps": 5.0, "dropped": 0}}
    slots = {"Caixa 01": FakeSlot(Frame("Caixa 01", _frame_img(), ts=1.0, seq=1))}
    win = _janela(tmp_path, db, cfg=_cfg([camera]), status=status, slots=slots)
    assert win.tabs.count() == 4


# --- ressalva 2: overlay_zones tem caller agora -------------------------------


def test_zoned_live_view_model_desenha_zona_da_camera():
    camera = CameraConfig(
        name="Caixa 01", rtsp_url="rtsp://x",
        zones=[[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]],
    )
    cfg = _cfg([camera])
    img = _frame_img(w=100, h=100, value=0)
    slot = FakeSlot(Frame("Caixa 01", img, ts=1.0, seq=1))
    pipeline = FakePipeline(slots={"Caixa 01": slot})

    model = _ZonedLiveViewModel(pipeline, cfg)
    resultado = model.snapshot("Caixa 01")

    assert resultado is not None
    assert not np.array_equal(resultado, img)  # a zona foi desenhada por cima
    assert np.array_equal(img, _frame_img(w=100, h=100, value=0))  # original intocado


def test_zoned_live_view_model_sem_zona_devolve_snapshot_puro():
    camera = CameraConfig(name="Caixa 01", rtsp_url="rtsp://x", zones=[])
    cfg = _cfg([camera])
    img = _frame_img(w=20, h=20, value=7)
    slot = FakeSlot(Frame("Caixa 01", img, ts=1.0, seq=1))
    pipeline = FakePipeline(slots={"Caixa 01": slot})

    model = _ZonedLiveViewModel(pipeline, cfg)
    resultado = model.snapshot("Caixa 01")

    assert np.array_equal(resultado, img)


def test_zoned_live_view_model_camera_sem_snapshot_devolve_none():
    cfg = _cfg([CameraConfig(name="Caixa 01", rtsp_url="rtsp://x")])
    pipeline = FakePipeline(slots={"Caixa 01": FakeSlot(None)})
    model = _ZonedLiveViewModel(pipeline, cfg)
    assert model.snapshot("Caixa 01") is None


# --- câmeras & zonas: salvar persiste, e round-trip com AppConfig.load -------


def test_salvar_zonas_persiste_e_load_relee_igual(tmp_path, db):
    camera = CameraConfig(name="Caixa 01", rtsp_url="rtsp://x", zones=[])
    config_path = tmp_path / "config.json"
    win = _janela(tmp_path, db, cfg=_cfg([camera]))
    win.config_path = config_path
    win.cfg.save(config_path)

    win._camera_list.setCurrentRow(0)
    novas_zonas = [[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8)]]
    win._zone_editor.set_zones(novas_zonas)

    win._salvar_zonas()

    assert config_path.exists()
    relido = AppConfig.load(config_path)
    assert relido.cameras[0].zones == novas_zonas

    # a própria janela também reflete o valor salvo
    assert win.cfg.cameras[0].zones == novas_zonas


def test_salvar_zonas_sem_camera_selecionada_nao_quebra(tmp_path, db):
    win = _janela(tmp_path, db, cfg=_cfg([]))
    win._salvar_zonas()  # nao ha camera nenhuma: nao pode levantar excecao


# --- ressalva 1: carregar zonas nao marca "alterado" -------------------------


def test_carregar_zonas_do_config_nao_marca_alterado(tmp_path, db):
    camera = CameraConfig(
        name="Caixa 01", rtsp_url="rtsp://x",
        zones=[[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]],
    )
    win = _janela(tmp_path, db, cfg=_cfg([camera]))

    chamadas = []
    win._zone_editor.zonesChanged.connect(lambda: chamadas.append(True))

    win._camera_list.setCurrentRow(0)

    assert chamadas == []
    assert win._zonas_alteradas is False
    assert win._zone_editor.zones() == camera.zones


def test_trocar_de_camera_recarrega_zonas_sem_marcar_alterado(tmp_path, db):
    cam_a = CameraConfig(name="A", rtsp_url="rtsp://a",
                         zones=[[(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)]])
    cam_b = CameraConfig(name="B", rtsp_url="rtsp://b",
                         zones=[[(0.5, 0.5), (0.6, 0.5), (0.6, 0.6)]])
    win = _janela(tmp_path, db, cfg=_cfg([cam_a, cam_b]))

    win._camera_list.setCurrentRow(0)
    assert win._zone_editor.zones() == cam_a.zones
    assert win._zonas_alteradas is False

    chamadas = []
    win._zone_editor.zonesChanged.connect(lambda: chamadas.append(True))
    win._camera_list.setCurrentRow(1)

    assert win._zone_editor.zones() == cam_b.zones
    assert chamadas == []
    assert win._zonas_alteradas is False


def test_editar_zona_manualmente_marca_alterado(tmp_path, db):
    """Contraste com o teste acima: mudança feita pelo usuário (não pelo
    carregamento programático) TEM que marcar a tela como alterada."""
    camera = CameraConfig(name="Caixa 01", rtsp_url="rtsp://x", zones=[])
    win = _janela(tmp_path, db, cfg=_cfg([camera]))
    win._camera_list.setCurrentRow(0)

    win._zone_editor.set_zones([[(0.3, 0.3), (0.4, 0.3), (0.4, 0.4)]])

    assert win._zonas_alteradas is True


# --- adicionar câmera ---------------------------------------------------------


def test_adicionar_camera_generico_usa_url_manual(tmp_path, db):
    win = _janela(tmp_path, db, cfg=_cfg([]))
    win._nome_edit.setText("Nova Câmera")
    win._marca_combo.setCurrentText("Genérico")
    win._url_manual_edit.setText("rtsp://admin:x@10.0.0.5:554/stream1")

    win._adicionar_camera()

    nomes = [c.name for c in win.cfg.cameras]
    assert "Nova Câmera" in nomes
    nova = next(c for c in win.cfg.cameras if c.name == "Nova Câmera")
    assert nova.rtsp_url == "rtsp://admin:x@10.0.0.5:554/stream1"
    # persistiu no disco
    relido = AppConfig.load(win.config_path)
    assert any(c.name == "Nova Câmera" for c in relido.cameras)


def test_adicionar_camera_intelbras_monta_url(tmp_path, db):
    win = _janela(tmp_path, db, cfg=_cfg([]))
    win._nome_edit.setText("Corredor")
    win._marca_combo.setCurrentText("Intelbras/Dahua")
    win._ip_edit.setText("192.168.0.11")
    win._usuario_edit.setText("admin")
    win._senha_edit.setText("s3nh@")
    win._canal_spin.setValue(8)

    win._adicionar_camera()

    nova = next(c for c in win.cfg.cameras if c.name == "Corredor")
    assert nova.rtsp_url == "rtsp://admin:s3nh%40@192.168.0.11:554/cam/realmonitor?channel=8&subtype=1"


def test_adicionar_camera_com_nome_duplicado_nao_adiciona(tmp_path, db):
    camera = CameraConfig(name="Caixa 01", rtsp_url="rtsp://x")
    win = _janela(tmp_path, db, cfg=_cfg([camera]))
    win._nome_edit.setText("Caixa 01")
    win._marca_combo.setCurrentText("Genérico")
    win._url_manual_edit.setText("rtsp://outra")

    win._adicionar_camera()

    assert len(win.cfg.cameras) == 1


# --- testar conexão (thread) --------------------------------------------------


def test_testar_conexao_roda_em_thread_e_atualiza_status(tmp_path, db, monkeypatch):
    chamadas = []

    def fake_test_connection(url, timeout=10):
        chamadas.append(url)
        return True, "Conectou! A câmera está respondendo."

    monkeypatch.setattr("src.ui.app.test_connection", fake_test_connection)

    win = _janela(tmp_path, db, cfg=_cfg([]))
    win._marca_combo.setCurrentText("Genérico")
    win._url_manual_edit.setText("rtsp://minha-camera")

    win._testar_conexao()
    assert win._teste_conexao_worker.wait(2000)
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()

    assert chamadas == ["rtsp://minha-camera"]
    assert "Conectou" in win._conexao_status.text()


# --- eventos -------------------------------------------------------------------


def _inserir_evento(db, camera="Caixa 01", zone="waist", score=0.8):
    from datetime import datetime, timezone
    agora = datetime.now(timezone.utc)
    return db.insert_event(store_id="l", camera_name=camera, ts_utc=agora.isoformat(),
                           ts_local=datetime.now().isoformat(), track_id=1, score=score,
                           zone=zone, signals={}, image_path="/tmp/a.jpg", clip_path=None)


def test_aba_eventos_carrega_do_banco(tmp_path, db):
    _inserir_evento(db)
    win = _janela(tmp_path, db)
    assert win._events_table.rowCount() == 1
    assert win._events_table.item(0, 0).text() == "Caixa 01"


def test_marcar_furto_persiste_no_banco(tmp_path, db):
    eid = _inserir_evento(db)
    win = _janela(tmp_path, db)
    win._events_table.selectRow(0)
    win._marcar_evento("true_positive")
    linhas = win._event_model.load()
    assert linhas[0]["feedback"] == "true_positive"


def test_marcar_falso_alarme_persiste_no_banco(tmp_path, db):
    eid = _inserir_evento(db)
    win = _janela(tmp_path, db)
    win._events_table.selectRow(0)
    win._marcar_evento("false_positive")
    linhas = win._event_model.load()
    assert linhas[0]["feedback"] == "false_positive"


def test_marcar_evento_sem_selecao_nao_quebra(tmp_path, db):
    win = _janela(tmp_path, db)
    win._marcar_evento("true_positive")  # tabela vazia: nao pode levantar excecao


# --- configuração: procurar grupo do telegram (sem rede real) ----------------


def test_botao_procurar_grupo_chama_descobrir_grupos(tmp_path, db, monkeypatch):
    chamadas = []

    def fake_descobrir_grupos(token, session=None):
        chamadas.append(token)
        return [{"chat_id": "-100999", "nome": "Alerta Loja", "tipo": "supergroup"}]

    monkeypatch.setattr("src.ui.app.descobrir_grupos", fake_descobrir_grupos)

    win = _janela(tmp_path, db)
    win._token_edit.setText("999:XYZ")

    win._buscar_grupos()
    assert win._grupos_worker.wait(2000)
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()

    assert chamadas == ["999:XYZ"]
    assert win._grupos_list.count() == 1
    assert "-100999" in win._grupos_list.item(0).text()


def test_botao_procurar_grupo_sem_token_nao_chama_rede(tmp_path, db, monkeypatch):
    chamadas = []
    monkeypatch.setattr("src.ui.app.descobrir_grupos",
                        lambda token, session=None: chamadas.append(token) or [])

    win = _janela(tmp_path, db)
    win._token_edit.setText("")
    win._buscar_grupos()

    assert chamadas == []


def test_selecionar_grupo_preenche_chat_id(tmp_path, db, monkeypatch):
    def fake_descobrir_grupos(token, session=None):
        return [{"chat_id": "-100999", "nome": "Alerta Loja", "tipo": "supergroup"}]

    monkeypatch.setattr("src.ui.app.descobrir_grupos", fake_descobrir_grupos)

    win = _janela(tmp_path, db)
    win._token_edit.setText("999:XYZ")
    win._buscar_grupos()
    win._grupos_worker.wait(2000)
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()

    win._grupos_list.setCurrentRow(0)
    win._on_grupo_selecionado(win._grupos_list.item(0))

    assert win.cfg.telegram.chat_id == "-100999"
