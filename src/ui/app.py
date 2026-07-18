"""Janela principal — amarra tudo que o Plano 4 construiu: preview ao vivo,
editor de zonas, cadastro de câmera por assistente, histórico de eventos e a
configuração do Telegram/benchmark. É a última peça que dá "cara de produto"
ao sistema: o revendedor abre um programa só, não um script.

Duas ressalvas de implementadores anteriores, tratadas aqui:

1. `ZoneEditor.set_zones()` emite `zonesChanged` mesmo quando chamado
   programaticamente (é o comportamento testado e intencional do widget).
   Ao CARREGAR as zonas do config nesta janela (troca de câmera selecionada),
   isso dispararia um "marcado como alterado" espúrio — por isso todo
   preenchimento programático do editor acontece dentro de um
   `QSignalBlocker`, e só a EDIÇÃO feita pelo usuário passa e marca
   `_zonas_alteradas`.

2. `LiveViewModel.overlay_zones` existia e era testado, mas nenhum caller o
   usava. `_ZonedLiveViewModel` (abaixo) estende `LiveViewModel` — sem
   modificar o arquivo original — para desenhar a zona monitorada de cada
   `CameraConfig` sobre o preview da aba "Ao vivo".

Regra que atravessa a janela inteira: A UI NUNCA roda inferência nem
bloqueia. Testar conexão RTSP, procurar o grupo do Telegram e o teste de
capacidade são todos I/O ou CPU pesada — cada um roda em uma `_ThreadWorker`
(QThread) própria, e a UI só recebe o resultado por sinal.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QSignalBlocker, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.alerts.telegram_alert import descobrir_grupos
from src.config.settings import AppConfig, CameraConfig
from src.ui.camera_form import MARCAS, build_rtsp_url, test_connection
from src.ui.event_log import EventLogModel
from src.ui.live_view import LiveViewModel, LiveViewWidget
from src.ui.zone_editor import ZoneEditor

log = logging.getLogger(__name__)

_MARCA_GENERICA = "Genérico"


class _ThreadWorker(QThread):
    """Roda uma função qualquer fora da thread da UI e devolve o resultado
    (ou o erro) por sinal. É o mecanismo único usado por "Testar conexão",
    "Procurar meu grupo" e "Teste de capacidade" — nenhuma dessas operações
    pode travar a tela, e nenhuma pode derrubar a janela se falhar."""

    resultado = Signal(object)
    erro = Signal(str)

    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            r = self._fn(*self._args, **self._kwargs)
        except Exception as e:  # falha de rede/hardware nunca derruba a UI
            log.exception("falha na thread de trabalho da UI")
            self.erro.emit(str(e))
        else:
            self.resultado.emit(r)


class _ZonedLiveViewModel(LiveViewModel):
    """Estende `LiveViewModel` (sem modificá-lo) para desenhar, no preview da
    aba "Ao vivo", a zona monitorada de cada câmera — a mesma zona que o
    `PersonGate` consome. `overlay_zones` já existia e era testado; era só
    questão de ligar as duas pontas."""

    def __init__(self, pipeline, cfg: AppConfig) -> None:
        super().__init__(pipeline)
        self._cfg = cfg

    def snapshot(self, camera_name: str):
        img = super().snapshot(camera_name)
        if img is None:
            return None
        cam = next((c for c in self._cfg.cameras if c.name == camera_name), None)
        zones = cam.zones if cam is not None else []
        if not zones:
            return img
        return self.overlay_zones(img, zones)


class MainWindow(QWidget):
    """A janela principal: `pipeline`/`db`/`cfg` são os mesmos objetos que o
    `main.py` headless já constrói — a UI só lê e desenha, nunca inicia nem
    para o pipeline por conta própria (isso continua sendo responsabilidade
    do `main.py`)."""

    def __init__(self, pipeline, db, cfg: AppConfig, config_path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.pipeline = pipeline
        self.db = db
        self.cfg = cfg
        self.config_path = config_path

        self._plain_live_model = LiveViewModel(pipeline)
        self._zonas_alteradas = False
        self._eventos: list[dict] = []
        self._grupos_encontrados: list[dict] = []
        self._teste_conexao_worker: _ThreadWorker | None = None
        self._grupos_worker: _ThreadWorker | None = None
        self._benchmark_worker: _ThreadWorker | None = None

        self.setWindowTitle("Prevenção de Perdas")

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_live_tab(), "Ao vivo")
        self.tabs.addTab(self._build_cameras_tab(), "Câmeras & Zonas")
        self.tabs.addTab(self._build_events_tab(), "Eventos")
        self.tabs.addTab(self._build_config_tab(), "Configuração")

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

        if self.cfg.cameras:
            self._camera_list.setCurrentRow(0)

    # --- utilitário compartilhado ------------------------------------------

    def _camera_by_name(self, nome: str) -> CameraConfig | None:
        return next((c for c in self.cfg.cameras if c.name == nome), None)

    # --- aba 1: Ao vivo -------------------------------------------------------

    def _build_live_tab(self) -> QWidget:
        self._live_model = _ZonedLiveViewModel(self.pipeline, self.cfg)
        self._live_widget = LiveViewWidget(self._live_model)
        return self._live_widget

    # --- aba 2: Câmeras & Zonas -------------------------------------------------

    def _build_cameras_tab(self) -> QWidget:
        container = QWidget()
        outer = QHBoxLayout(container)

        left = QVBoxLayout()
        left.addWidget(QLabel("Câmeras cadastradas"))
        self._camera_list = QListWidget()
        self._camera_list.addItems([c.name for c in self.cfg.cameras])
        self._camera_list.currentTextChanged.connect(self._on_camera_selected)
        left.addWidget(self._camera_list)

        form = QFormLayout()
        self._nome_edit = QLineEdit()
        self._marca_combo = QComboBox()
        self._marca_combo.addItems(list(MARCAS.keys()))
        self._marca_combo.currentTextChanged.connect(self._on_marca_changed)
        self._ip_edit = QLineEdit()
        self._usuario_edit = QLineEdit()
        self._senha_edit = QLineEdit()
        self._senha_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._canal_spin = QSpinBox()
        self._canal_spin.setRange(1, 999)
        self._canal_spin.setValue(1)
        self._porta_spin = QSpinBox()
        self._porta_spin.setRange(1, 65535)
        self._porta_spin.setValue(554)
        self._substream_check = QCheckBox("Usar substream (recomendado)")
        self._substream_check.setChecked(True)
        self._url_manual_edit = QLineEdit()
        self._url_manual_edit.setPlaceholderText("rtsp://usuario:senha@ip:porta/...")

        form.addRow("Nome da câmera", self._nome_edit)
        form.addRow("Marca", self._marca_combo)
        form.addRow("IP", self._ip_edit)
        form.addRow("Usuário", self._usuario_edit)
        form.addRow("Senha", self._senha_edit)
        form.addRow("Canal", self._canal_spin)
        form.addRow("Porta", self._porta_spin)
        form.addRow("", self._substream_check)
        form.addRow("URL manual (marca Genérico)", self._url_manual_edit)
        left.addLayout(form)

        botoes = QHBoxLayout()
        self._testar_btn = QPushButton("Testar conexão")
        self._testar_btn.clicked.connect(self._testar_conexao)
        self._adicionar_btn = QPushButton("Adicionar câmera")
        self._adicionar_btn.clicked.connect(self._adicionar_camera)
        botoes.addWidget(self._testar_btn)
        botoes.addWidget(self._adicionar_btn)
        left.addLayout(botoes)

        self._conexao_status = QLabel("")
        self._conexao_status.setWordWrap(True)
        left.addWidget(self._conexao_status)
        left.addStretch()

        right = QVBoxLayout()
        right.addWidget(QLabel("Área monitorada (clique para marcar, duplo clique fecha)"))
        self._zone_editor = ZoneEditor()
        self._zone_editor.zonesChanged.connect(self._on_zonas_changed)
        right.addWidget(self._zone_editor, 1)

        self._zonas_status = QLabel("")
        right.addWidget(self._zonas_status)
        self._salvar_zonas_btn = QPushButton("Salvar zonas")
        self._salvar_zonas_btn.clicked.connect(self._salvar_zonas)
        right.addWidget(self._salvar_zonas_btn)

        outer.addLayout(left, 1)
        outer.addLayout(right, 2)

        self._on_marca_changed(self._marca_combo.currentText())
        return container

    def _on_marca_changed(self, marca: str) -> None:
        generico = marca == _MARCA_GENERICA
        self._url_manual_edit.setEnabled(generico)
        for w in (self._ip_edit, self._usuario_edit, self._senha_edit,
                  self._canal_spin, self._porta_spin, self._substream_check):
            w.setEnabled(not generico)

    def _on_camera_selected(self, nome: str) -> None:
        if not nome:
            return
        cam = self._camera_by_name(nome)
        if cam is None:
            return
        img = self._plain_live_model.snapshot(nome)
        if img is not None:
            self._zone_editor.set_snapshot(img)
        # RESSALVA 1: carregar as zonas do config é preenchimento
        # programático, não edição do usuário — bloqueia o sinal para não
        # marcar a tela como "alterada" nem disparar um re-render/save
        # indevido.
        with QSignalBlocker(self._zone_editor):
            self._zone_editor.set_zones(cam.zones)
        self._zonas_alteradas = False
        self._zonas_status.setText("")

    def _on_zonas_changed(self) -> None:
        self._zonas_alteradas = True
        self._zonas_status.setText("Zonas alteradas — clique em \"Salvar zonas\".")

    def _salvar_zonas(self) -> None:
        item = self._camera_list.currentItem()
        if item is None:
            QMessageBox.warning(self, "Salvar zonas", "Selecione uma câmera primeiro.")
            return
        cam = self._camera_by_name(item.text())
        if cam is None:
            return
        cam.zones = self._zone_editor.zones()
        self.cfg.save(self.config_path)
        self._zonas_alteradas = False
        self._zonas_status.setText("Zonas salvas.")

    def _url_do_formulario(self) -> str | None:
        marca = self._marca_combo.currentText()
        if marca == _MARCA_GENERICA:
            url = self._url_manual_edit.text().strip()
            return url or None
        try:
            return build_rtsp_url(
                marca,
                self._ip_edit.text().strip(),
                self._usuario_edit.text(),
                self._senha_edit.text(),
                self._canal_spin.value(),
                substream=self._substream_check.isChecked(),
                porta=self._porta_spin.value(),
            )
        except ValueError as e:
            QMessageBox.warning(self, "URL inválida", str(e))
            return None

    def _testar_conexao(self) -> None:
        url = self._url_do_formulario()
        if not url:
            QMessageBox.warning(self, "Testar conexão", "Preencha os dados da câmera primeiro.")
            return
        self._conexao_status.setText("Testando conexão...")
        self._testar_btn.setEnabled(False)
        self._teste_conexao_worker = _ThreadWorker(test_connection, url)
        self._teste_conexao_worker.resultado.connect(self._on_teste_conexao_resultado)
        self._teste_conexao_worker.erro.connect(self._on_teste_conexao_erro)
        self._teste_conexao_worker.finished.connect(lambda: self._testar_btn.setEnabled(True))
        self._teste_conexao_worker.start()

    def _on_teste_conexao_resultado(self, resultado) -> None:
        _ok, msg = resultado
        self._conexao_status.setText(msg)

    def _on_teste_conexao_erro(self, msg: str) -> None:
        self._conexao_status.setText(f"Erro inesperado ao testar a conexão: {msg}")

    def _adicionar_camera(self) -> None:
        nome = self._nome_edit.text().strip()
        if not nome:
            QMessageBox.warning(self, "Adicionar câmera", "Informe o nome da câmera.")
            return
        if self._camera_by_name(nome) is not None:
            QMessageBox.warning(self, "Adicionar câmera",
                                f"Já existe uma câmera chamada \"{nome}\".")
            return
        url = self._url_do_formulario()
        if not url:
            QMessageBox.warning(self, "Adicionar câmera",
                                "Informe a URL RTSP (ou os dados da câmera).")
            return
        try:
            nova = CameraConfig(name=nome, rtsp_url=url)
        except Exception as e:
            QMessageBox.warning(self, "Adicionar câmera", str(e))
            return

        self.cfg.cameras.append(nova)
        self.cfg.save(self.config_path)
        self._camera_list.addItem(nome)
        self._nome_edit.clear()

    # --- aba 3: Eventos --------------------------------------------------------

    def _build_events_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        self._event_model = EventLogModel(self.db)
        self._events_table = QTableWidget(0, 6)
        self._events_table.setHorizontalHeaderLabels(
            ["Câmera", "Hora", "Score", "Zona", "Enviado", "Feedback"]
        )
        self._events_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._events_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._events_table, 1)

        botoes = QHBoxLayout()
        self._furto_btn = QPushButton("Foi furto")
        self._furto_btn.clicked.connect(lambda: self._marcar_evento("true_positive"))
        self._falso_btn = QPushButton("Foi falso alarme")
        self._falso_btn.clicked.connect(lambda: self._marcar_evento("false_positive"))
        self._atualizar_eventos_btn = QPushButton("Atualizar")
        self._atualizar_eventos_btn.clicked.connect(self._carregar_eventos)
        botoes.addWidget(self._furto_btn)
        botoes.addWidget(self._falso_btn)
        botoes.addStretch()
        botoes.addWidget(self._atualizar_eventos_btn)
        layout.addLayout(botoes)

        self._carregar_eventos()
        return container

    def _carregar_eventos(self) -> None:
        self._eventos = self._event_model.load()
        self._events_table.setRowCount(len(self._eventos))
        for row, ev in enumerate(self._eventos):
            valores = (
                ev["camera"], ev["hora"], str(ev["score"]), ev["zona"],
                "sim" if ev["enviado"] else "não", ev["feedback"] or "",
            )
            for col, valor in enumerate(valores):
                self._events_table.setItem(row, col, QTableWidgetItem(valor))

    def _marcar_evento(self, feedback: str) -> None:
        row = self._events_table.currentRow()
        if row < 0 or row >= len(self._eventos):
            QMessageBox.information(self, "Marcar evento", "Selecione um evento na tabela.")
            return
        event_id = self._eventos[row]["id"]
        if feedback == "true_positive":
            self._event_model.mark_true_positive(event_id)
        else:
            self._event_model.mark_false_positive(event_id)
        self._carregar_eventos()

    # --- aba 4: Configuração -----------------------------------------------------

    def _build_config_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        form = QFormLayout()
        self._token_edit = QLineEdit(self.cfg.telegram.bot_token)
        self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Token do bot Telegram", self._token_edit)
        layout.addLayout(form)

        self._buscar_grupo_btn = QPushButton("Procurar meu grupo")
        self._buscar_grupo_btn.clicked.connect(self._buscar_grupos)
        layout.addWidget(self._buscar_grupo_btn)

        self._grupos_list = QListWidget()
        self._grupos_list.itemClicked.connect(self._on_grupo_selecionado)
        layout.addWidget(self._grupos_list)

        self._grupo_status = QLabel("")
        self._grupo_status.setWordWrap(True)
        layout.addWidget(self._grupo_status)

        self._salvar_telegram_btn = QPushButton("Salvar configuração do Telegram")
        self._salvar_telegram_btn.clicked.connect(self._salvar_telegram)
        layout.addWidget(self._salvar_telegram_btn)

        self._benchmark_btn = QPushButton("Teste de capacidade")
        self._benchmark_btn.clicked.connect(self._rodar_benchmark)
        layout.addWidget(self._benchmark_btn)

        self._benchmark_output = QPlainTextEdit()
        self._benchmark_output.setReadOnly(True)
        layout.addWidget(self._benchmark_output, 1)

        return container

    def _buscar_grupos(self) -> None:
        token = self._token_edit.text().strip()
        if not token:
            QMessageBox.warning(self, "Procurar meu grupo",
                                "Informe o token do bot primeiro.")
            return
        self._grupo_status.setText("Procurando grupos...")
        self._buscar_grupo_btn.setEnabled(False)
        self._grupos_worker = _ThreadWorker(descobrir_grupos, token)
        self._grupos_worker.resultado.connect(self._on_grupos_encontrados)
        self._grupos_worker.erro.connect(self._on_grupos_erro)
        self._grupos_worker.finished.connect(lambda: self._buscar_grupo_btn.setEnabled(True))
        self._grupos_worker.start()

    def _on_grupos_encontrados(self, grupos: list) -> None:
        self._grupos_encontrados = grupos
        self._grupos_list.clear()
        if not grupos:
            self._grupo_status.setText(
                "Nenhum grupo encontrado. Crie um grupo, adicione o bot, mande uma "
                "mensagem qualquer e tente de novo."
            )
            return
        for g in grupos:
            self._grupos_list.addItem(QListWidgetItem(f"{g['nome']} ({g['chat_id']})"))
        self._grupo_status.setText(f"{len(grupos)} grupo(s) encontrado(s). Selecione um.")

    def _on_grupos_erro(self, msg: str) -> None:
        self._grupo_status.setText(f"Erro ao procurar grupos: {msg}")

    def _on_grupo_selecionado(self, item: QListWidgetItem) -> None:
        idx = self._grupos_list.row(item)
        if idx < 0 or idx >= len(self._grupos_encontrados):
            return
        grupo = self._grupos_encontrados[idx]
        self.cfg.telegram.chat_id = grupo["chat_id"]
        self._grupo_status.setText(f"Grupo selecionado: {grupo['nome']}")

    def _salvar_telegram(self) -> None:
        self.cfg.telegram.bot_token = self._token_edit.text().strip()
        self.cfg.save(self.config_path)
        self._grupo_status.setText("Configuração do Telegram salva.")

    def _rodar_benchmark(self) -> None:
        self._benchmark_output.setPlainText(
            "Rodando teste de capacidade... isso pode levar alguns segundos."
        )
        self._benchmark_btn.setEnabled(False)

        inference_cfg = self.cfg.inference

        def _tarefa() -> str:
            # import tardio: a UI nao pode exigir o motor de inferencia (nem
            # carregar modelo YOLO) so para abrir a janela -- so quando o
            # usuario de fato pede o teste de capacidade.
            from src.inference.engine import InferenceEngine
            from src.tools.benchmark import benchmark

            engine = InferenceEngine(inference_cfg)
            report = benchmark(engine, workers=inference_cfg.workers)
            return report.as_text()

        self._benchmark_worker = _ThreadWorker(_tarefa)
        self._benchmark_worker.resultado.connect(self._benchmark_output.setPlainText)
        self._benchmark_worker.erro.connect(
            lambda msg: self._benchmark_output.setPlainText(f"Falha no teste de capacidade: {msg}")
        )
        self._benchmark_worker.finished.connect(lambda: self._benchmark_btn.setEnabled(True))
        self._benchmark_worker.start()
