from datetime import datetime

import pytest

from src.alerts.telegram_alert import TelegramSender
from src.config.settings import TelegramConfig


class FakeResp:
    def __init__(self, ok=True, status=200):
        self.status_code = status
        self._ok = ok

    def json(self):
        return {"ok": self._ok}


class FakeSession:
    def __init__(self, resp=None, boom=False):
        self.resp = resp or FakeResp()
        self.boom = boom
        self.calls = []

    def post(self, url, data=None, files=None, timeout=None):
        self.calls.append({"url": url, "data": data, "files": files})
        if self.boom:
            raise ConnectionError("rede fora")
        return self.resp


def _cfg(**kw):
    base = dict(bot_token="123:ABC", chat_id="-100999")
    base.update(kw)
    return TelegramConfig(**base)


def test_not_configured_when_token_missing():
    s = TelegramSender(TelegramConfig(bot_token="", chat_id=""), session=FakeSession())
    assert s.configured is False


def test_configured_with_token_and_chat():
    assert TelegramSender(_cfg(), session=FakeSession()).configured is True


def test_desligado_nao_envia_foto(tmp_path):
    """Modo silencioso (bug de campo 23/jul: alertas errados assustaram a dona).
    Com enabled=False o sistema NÃO manda no Telegram — nem faz a chamada HTTP —
    mesmo com credenciais válidas. A evidência continua sendo gravada no fluxo do
    _on_result (antes do envio); aqui garantimos que o envio é o que fica mudo."""
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    sess = FakeSession()
    s = TelegramSender(_cfg(enabled=False), session=sess)

    assert s.send_photo(img, "legenda") is False
    assert sess.calls == []  # nenhuma chamada de rede


def test_desligado_nao_envia_mensagem_de_sistema():
    """Vale também para as mensagens de câmera offline/online — no modo teste
    o grupo fica 100% mudo."""
    sess = FakeSession()
    s = TelegramSender(_cfg(enabled=False), session=sess)
    assert s.send_message("Câmera Cam10 voltou ao normal.") is False
    assert sess.calls == []


def test_ligado_e_o_padrao_e_envia():
    """enabled é True por padrão: instalações atuais não mudam de comportamento."""
    assert _cfg().enabled is True
    sess = FakeSession()
    s = TelegramSender(_cfg(), session=sess)
    assert s.send_message("oi") is True
    assert len(sess.calls) == 1


def test_configured_independe_de_enabled():
    """`configured` = tem credenciais; `enabled` = master switch de envio. São
    coisas distintas: desligar não é o mesmo que estar sem token."""
    s = TelegramSender(_cfg(enabled=False), session=FakeSession())
    assert s.configured is True
    assert s.can_send is False


def test_send_photo_posts_to_sendphoto(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    sess = FakeSession()
    s = TelegramSender(_cfg(), session=sess)

    assert s.send_photo(img, "legenda") is True
    assert sess.calls[0]["url"].endswith("/sendPhoto")
    assert "123:ABC" in sess.calls[0]["url"]
    assert sess.calls[0]["data"]["chat_id"] == "-100999"
    assert sess.calls[0]["data"]["caption"] == "legenda"


def test_send_video_posts_to_sendvideo(tmp_path):
    v = tmp_path / "a.mp4"
    v.write_bytes(b"x")
    sess = FakeSession()
    s = TelegramSender(_cfg(), session=sess)
    assert s.send_video(v, "leg") is True
    assert sess.calls[0]["url"].endswith("/sendVideo")


def test_returns_false_on_network_error(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    s = TelegramSender(_cfg(), session=FakeSession(boom=True))
    assert s.send_photo(img, "x") is False  # nao levanta, so reporta


def test_returns_false_when_telegram_says_not_ok(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    s = TelegramSender(_cfg(), session=FakeSession(resp=FakeResp(ok=False, status=400)))
    assert s.send_photo(img, "x") is False


def test_does_not_send_when_not_configured(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    sess = FakeSession()
    s = TelegramSender(TelegramConfig(bot_token="", chat_id=""), session=sess)
    assert s.send_photo(img, "x") is False
    assert sess.calls == []  # nem tentou


def test_caption_is_portuguese_and_has_the_facts():
    s = TelegramSender(_cfg(), session=FakeSession())
    cap = s.caption_for("Mercado Central", "Corredor 3",
                        datetime(2026, 7, 16, 14, 35, 9), "waist")
    assert "Mercado Central" in cap
    assert "Corredor 3" in cap
    assert "14:35" in cap
    assert "16/07/2026" in cap
    # a zona aparece em portugues, nao o codigo interno
    assert "waist" not in cap
    assert any(p in cap.lower() for p in ("cintura", "bolso"))
