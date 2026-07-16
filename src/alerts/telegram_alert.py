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


class TelegramSender:
    def __init__(self, cfg: TelegramConfig, session=None) -> None:
        self.cfg = cfg
        self._session = session or requests.Session()

    @property
    def configured(self) -> bool:
        return bool(self.cfg.bot_token and self.cfg.chat_id)

    def caption_for(self, store_name: str, camera_name: str,
                    ts_local: datetime, zone: str) -> str:
        gesto = ZONA_PT.get(zone, "ocultação de produto")
        return (f"⚠️ Possível ocultação de produto\n"
                f"🏪 {store_name}\n"
                f"📷 {camera_name}\n"
                f"🕒 {ts_local.strftime('%d/%m/%Y %H:%M:%S')}\n"
                f"👀 {gesto}")

    def _post(self, metodo: str, campo: str, caminho: Path, caption: str) -> bool:
        if not self.configured:
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
        if not self.configured:
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
