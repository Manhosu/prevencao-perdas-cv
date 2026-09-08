"""A tela que o instalador vê numa máquina nova.

É a única tela que aparece antes de o sistema funcionar, então ela tem que
carregar sozinha a instrução inteira: qual código enviar e onde colar a
resposta."""
import pytest

from src.licensing import activation, fingerprint
from src.licensing.activation import Estado, avaliar
from src.licensing.license import Licenca, emitir, gerar_par_de_chaves
from src.ui.activation_dialog import ActivationDialog


@pytest.fixture
def chave(monkeypatch):
    privada, publica = gerar_par_de_chaves()
    monkeypatch.setattr(activation, "_chave_publica", lambda: publica)
    return privada


def _licenca_desta_maquina(chave, cliente="Mercado Julie"):
    return emitir(
        Licenca(maquina=fingerprint.codigo_da_maquina(), cliente=cliente,
                emitida_em="2026-09-07", expira_em=None),
        chave,
    )


def test_mostra_o_codigo_desta_maquina(tmp_path):
    dlg = ActivationDialog(tmp_path)
    assert dlg._codigo.text() == fingerprint.codigo_da_maquina()
    assert dlg._codigo.isReadOnly()  # não é campo de digitação


def test_colar_licenca_valida_libera_a_maquina(tmp_path, chave):
    dlg = ActivationDialog(tmp_path)
    dlg._licenca_edit.setPlainText(_licenca_desta_maquina(chave))

    dlg._ativar()

    assert dlg.aceito is True
    assert avaliar(tmp_path).estado is Estado.OK


def test_licenca_de_outra_maquina_explica_e_nao_libera(tmp_path, chave):
    outra = fingerprint.codigo_da_maquina({"guid": "outro-pc", "volume": "FF00FF00"})
    token = emitir(Licenca(maquina=outra, cliente="Outro", emitida_em="2026-09-07"),
                   chave)
    dlg = ActivationDialog(tmp_path)
    dlg._licenca_edit.setPlainText(token)

    dlg._ativar()

    assert dlg.aceito is False
    assert "outro computador" in dlg._status.text()


def test_texto_qualquer_nao_derruba_a_tela(tmp_path, chave):
    dlg = ActivationDialog(tmp_path)
    dlg._licenca_edit.setPlainText("isso nao e uma licenca")

    dlg._ativar()

    assert dlg.aceito is False
    assert dlg._status.text()  # explicou em vez de quebrar


def test_campo_vazio_pede_a_licenca(tmp_path, chave):
    dlg = ActivationDialog(tmp_path)
    dlg._ativar()
    assert dlg.aceito is False
    assert "Cole a licença" in dlg._status.text()


def test_botao_copiar_leva_o_codigo_para_a_area_de_transferencia(tmp_path):
    from PySide6.QtWidgets import QApplication

    dlg = ActivationDialog(tmp_path)
    dlg._copiar_codigo()

    assert QApplication.clipboard().text() == fingerprint.codigo_da_maquina()
