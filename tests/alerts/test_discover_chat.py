"""Descoberta automática do grupo do Telegram.

É o que dá autonomia ao revendedor: em cada loja nova ele cria o grupo, adiciona
o MESMO bot, manda um "oi", e o sistema acha o chat_id sozinho — sem precisar
pedir nada para o desenvolvedor."""
import pytest

from src.alerts.telegram_alert import descobrir_grupos


class FakeResp:
    def __init__(self, payload, status=200):
        self.status_code = status
        self._p = payload

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, payload=None, boom=False, status=200):
        self.payload = payload or {"ok": True, "result": []}
        self.boom = boom
        self.status = status
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if self.boom:
            raise ConnectionError("rede fora")
        return FakeResp(self.payload, self.status)


def _update(chat_id, tipo="supergroup", titulo="Alerta Loja"):
    return {"message": {"chat": {"id": chat_id, "type": tipo, "title": titulo}}}


def test_encontra_o_grupo():
    sess = FakeSession({"ok": True, "result": [_update(-1001234, titulo="Alerta Mercado 2")]})
    grupos = descobrir_grupos("123:ABC", session=sess)
    assert len(grupos) == 1
    assert grupos[0]["chat_id"] == "-1001234"
    assert grupos[0]["nome"] == "Alerta Mercado 2"
    assert grupos[0]["tipo"] == "supergroup"


def test_usa_o_token_na_url():
    sess = FakeSession()
    descobrir_grupos("123:ABC", session=sess)
    assert "bot123:ABC/getUpdates" in sess.calls[0]


def test_ignora_conversa_privada_e_prioriza_grupo():
    """Conversa 1-a-1 com o bot nao serve — o alerta vai pro grupo da equipe."""
    sess = FakeSession({"ok": True, "result": [
        _update(555, tipo="private", titulo=None),
        _update(-100999, tipo="group", titulo="Equipe Loja"),
    ]})
    grupos = descobrir_grupos("123:ABC", session=sess)
    assert [g["chat_id"] for g in grupos] == ["-100999"]


def test_sem_grupo_devolve_lista_vazia():
    sess = FakeSession({"ok": True, "result": []})
    assert descobrir_grupos("123:ABC", session=sess) == []


def test_nao_repete_o_mesmo_grupo():
    sess = FakeSession({"ok": True, "result": [
        _update(-100777, titulo="Loja"), _update(-100777, titulo="Loja"),
    ]})
    assert len(descobrir_grupos("123:ABC", session=sess)) == 1


def test_erro_de_rede_nao_levanta():
    assert descobrir_grupos("123:ABC", session=FakeSession(boom=True)) == []


def test_token_invalido_nao_levanta():
    sess = FakeSession({"ok": False, "description": "Unauthorized"}, status=401)
    assert descobrir_grupos("123:ABC", session=sess) == []


def test_token_vazio_nem_tenta():
    sess = FakeSession()
    assert descobrir_grupos("", session=sess) == []
    assert sess.calls == []
