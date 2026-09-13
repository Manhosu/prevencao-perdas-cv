"""Tela de liberação do computador.

É a primeira coisa que o instalador vê numa máquina nova, e a única tela do
sistema que aparece antes de qualquer coisa funcionar. Por isso ela precisa
resolver sozinha o que a pessoa tem que fazer, sem manual:

  1. mostra o código DESTA máquina, pronto para copiar;
  2. explica que esse código vai para o fornecedor;
  3. recebe a licença de volta e libera.

Nenhuma operação aqui é pesada (é só conferência de assinatura), então tudo
roda na thread da interface sem travar nada."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from src.licensing import fingerprint
from src.licensing.activation import Estado, ativar


class ActivationDialog(QDialog):
    """Diálogo de ativação. `aceito` fica True quando a máquina foi liberada."""

    def __init__(self, pasta_dados: Path, mensagem: str = "", parent=None) -> None:
        super().__init__(parent)
        self.pasta_dados = Path(pasta_dados)
        self.aceito = False
        self.setWindowTitle("Liberar este computador")

        layout = QVBoxLayout(self)

        explicacao = QLabel(
            mensagem
            or "Este computador ainda não foi liberado para usar o sistema."
        )
        explicacao.setWordWrap(True)
        layout.addWidget(explicacao)

        layout.addWidget(QLabel(
            "\n1) Envie o código abaixo para o seu fornecedor:"))

        linha_codigo = QHBoxLayout()
        self._codigo = QLineEdit(fingerprint.codigo_da_maquina())
        self._codigo.setReadOnly(True)
        # Fonte maior: esse código é lido em voz alta ou copiado à mão quando
        # a loja não tem o WhatsApp aberto na mesma máquina.
        fonte = self._codigo.font()
        fonte.setPointSize(fonte.pointSize() + 4)
        fonte.setBold(True)
        self._codigo.setFont(fonte)
        self._copiar_btn = QPushButton("Copiar")
        self._copiar_btn.clicked.connect(self._copiar_codigo)
        linha_codigo.addWidget(self._codigo, 1)
        linha_codigo.addWidget(self._copiar_btn)
        layout.addLayout(linha_codigo)

        layout.addWidget(QLabel(
            "\n2) Cole abaixo a licença que ele enviar:"))
        self._licenca_edit = QPlainTextEdit()
        self._licenca_edit.setPlaceholderText("PP1....")
        self._licenca_edit.setFixedHeight(90)
        layout.addWidget(self._licenca_edit)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        botoes = QHBoxLayout()
        self._ativar_btn = QPushButton("Liberar computador")
        self._ativar_btn.setDefault(True)
        self._ativar_btn.clicked.connect(self._ativar)
        self._fechar_btn = QPushButton("Fechar")
        self._fechar_btn.clicked.connect(self.reject)
        botoes.addStretch()
        botoes.addWidget(self._fechar_btn)
        botoes.addWidget(self._ativar_btn)
        layout.addLayout(botoes)

    def _copiar_codigo(self) -> None:
        area = QApplication.clipboard()
        if area is not None:
            area.setText(self._codigo.text())
        self._status.setText("Código copiado. Envie para o seu fornecedor.")

    def _ativar(self) -> None:
        token = self._licenca_edit.toPlainText().strip()
        if not token:
            self._status.setText("Cole a licença que o fornecedor enviou.")
            return
        resultado = ativar(self.pasta_dados, token)
        if resultado.estado is Estado.OK:
            self.aceito = True
            self._status.setText("Computador liberado.")
            self.accept()
            return
        self._status.setText(resultado.mensagem)


def pedir_ativacao(pasta_dados: Path, mensagem: str = "") -> bool:
    """Abre o diálogo e devolve True se a máquina foi liberada.

    Usado no boot: sem isto, o programa instalado numa máquina não liberada
    abriria e fecharia sozinho, que foi exatamente o bug de campo de julho."""
    dialogo = ActivationDialog(pasta_dados, mensagem)
    dialogo.setWindowModality(Qt.WindowModality.ApplicationModal)
    # É a primeira e única janela da primeira abertura. Se ela abrir atrás de
    # outra coisa, o lojista não vê o que fazer e fecha o programa achando que
    # travou. Tem que vir na frente.
    dialogo.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    dialogo.show()
    dialogo.raise_()
    dialogo.activateWindow()
    dialogo.exec()
    return dialogo.aceito
