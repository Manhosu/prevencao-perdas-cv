"""Envio ao Telegram (só o HTTP). A fila/retry fica no alert_queue.

Nunca levanta exceção para o chamador: devolve True/False. Uma falha de rede não
pode derrubar quem chamou."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import requests

from src.config.settings import TelegramConfig

log = logging.getLogger(__name__)

API = "https://api.telegram.org"
TIMEOUT = 20

# O lojista lê isso — nada de código interno na legenda.
ZONA_PT = {
    "waist": "mão na cintura/bolso",
    "torso": "mão sob a roupa",
    "back_waist": "mão na cintura (de costas)",
    "bag": "mão na bolsa/mochila",
}


def descobrir_grupos(bot_token: str, session=None) -> list[dict]:
    """Descobre sozinho os grupos onde o bot foi adicionado.

    É o que dá AUTONOMIA ao revendedor: em cada loja nova ele cria o grupo,
    adiciona o MESMO bot, manda qualquer mensagem, e o sistema acha o chat_id
    sozinho — sem precisar pedir nada para o desenvolvedor. Um bot serve todas
    as lojas; o que muda por loja é só o grupo.

    Devolve [{chat_id, nome, tipo}]. Nunca levanta: erro de rede ou token
    inválido devolvem lista vazia (quem chama mostra a mensagem ao usuário).
    """
    if not bot_token:
        return []
    sess = session or requests.Session()
    try:
        r = sess.get(f"{API}/bot{bot_token}/getUpdates", timeout=TIMEOUT)
        if r.status_code != 200:
            log.warning("Telegram recusou getUpdates: %s", r.status_code)
            return []
        payload = r.json()
        if not payload.get("ok"):
            return []
    except Exception as e:
        log.warning("nao consegui consultar o Telegram: %s", e)
        return []

    encontrados: dict[str, dict] = {}
    for u in payload.get("result", []):
        for chave in ("message", "my_chat_member", "channel_post", "edited_message"):
            chat = (u.get(chave) or {}).get("chat") or {}
            tipo = chat.get("type")
            # conversa 1-a-1 com o bot nao serve: o alerta vai para o grupo da equipe
            if tipo not in ("group", "supergroup", "channel"):
                continue
            cid = str(chat.get("id"))
            encontrados.setdefault(cid, {
                "chat_id": cid,
                "nome": chat.get("title") or "(sem nome)",
                "tipo": tipo,
            })
    return list(encontrados.values())


class TelegramSender:
    def __init__(self, cfg: TelegramConfig, session=None) -> None:
        self.cfg = cfg
        self._session = session or requests.Session()

    @property
    def configured(self) -> bool:
        """Tem credenciais (token + chat_id). NÃO implica que vai enviar —
        para isso ver `can_send` (respeita o modo silencioso)."""
        return bool(self.cfg.bot_token and self.cfg.chat_id)

    @property
    def can_send(self) -> bool:
        """Só envia se tem credenciais E o envio está ligado. É o gate único
        de todo envio ao Telegram (foto, vídeo, mensagem de sistema)."""
        return self.configured and self.cfg.enabled

    def caption_for(self, store_name: str, camera_name: str,
                    ts_local: datetime, zone: str) -> str:
        gesto = ZONA_PT.get(zone, "ocultação de produto")
        return (f"⚠️ Possível ocultação de produto\n"
                f"🏪 {store_name}\n"
                f"📷 {camera_name}\n"
                f"🕒 {ts_local.strftime('%d/%m/%Y %H:%M:%S')}\n"
                f"👀 {gesto}")

    def caption_panico(self, store_name: str, camera_name: str,
                       ts_local: datetime, arma: bool = False) -> str:
        """Legenda do alerta de PÂNICO (modo caixa): mãos acima da cabeça =
        possível assalto. Tom de urgência, distinto do alerta de ocultação.

        `arma=True` acrescenta a linha de arma — só vem True quando o detector
        de arma (gated no pânico) viu uma arma óbvia no quadro; o gesto é o
        gatilho, a arma é contexto extra."""
        linha_arma = "\n🔫 Possível ARMA no local" if arma else ""
        return (f"🚨 ALERTA DE PÂNICO — mãos levantadas no caixa\n"
                f"🏪 {store_name}\n"
                f"📷 {camera_name}\n"
                f"🕒 {ts_local.strftime('%d/%m/%Y %H:%M:%S')}"
                f"{linha_arma}\n"
                f"🆘 Possível assalto — verifique agora")

    def _post(self, metodo: str, campo: str, caminho: Path, caption: str) -> bool:
        if not self.can_send:
            return False
        url = f"{API}/bot{self.cfg.bot_token}/{metodo}"
        try:
            with open(caminho, "rb") as f:
                r = self._session.post(
                    url,
                    data={"chat_id": self.cfg.chat_id, "caption": caption},
                    files={campo: f},
                    timeout=TIMEOUT,
                )
            ok = r.status_code == 200 and r.json().get("ok", False)
            if not ok:
                log.warning("Telegram recusou o envio (%s): %s", metodo, r.status_code)
            return bool(ok)
        except Exception as e:
            log.warning("falha ao enviar para o Telegram (%s): %s", metodo, e)
            return False

    def send_photo(self, image_path, caption: str) -> bool:
        return self._post("sendPhoto", "photo", Path(image_path), caption)

    def send_video(self, video_path, caption: str) -> bool:
        return self._post("sendVideo", "video", Path(video_path), caption)

    def send_message(self, text: str) -> bool:
        if not self.can_send:
            return False
        url = f"{API}/bot{self.cfg.bot_token}/sendMessage"
        try:
            r = self._session.post(
                url, data={"chat_id": self.cfg.chat_id, "text": text}, timeout=TIMEOUT
            )
            return r.status_code == 200 and r.json().get("ok", False)
        except Exception as e:
            log.warning("falha ao enviar mensagem ao Telegram: %s", e)
            return False
