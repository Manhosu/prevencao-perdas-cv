"""Montador de URL RTSP por marca — SEM Qt (só funções, testáveis sem tela).

O revendedor instala o sistema em várias lojas, cada uma com um DVR/câmera
diferente. Ele não sabe (nem precisa saber) o que é RTSP: escolhe a marca no
formulário, digita IP/usuário/senha/canal, e a URL sai pronta. É isso que
torna a configuração de cada câmera nova uma questão de minutos, não de
procurar o manual do fabricante.

A senha é escapada com `urllib.parse.quote` porque senhas com `@` ou `:` são
comuns (ex.: "s3nh@") e, sem escapar, o `@` da senha seria confundido com o
separador `usuário@host` da URL — a câmera errada (ou nenhuma) seria acessada.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import quote, unquote, urlparse

log = logging.getLogger(__name__)

# Marca -> descrição do padrão de URL usado (mostrada ao revendedor no combo).
MARCAS: dict[str, str] = {
    "Intelbras/Dahua": "rtsp://usuario:senha@IP:porta/cam/realmonitor?channel=N&subtype=0|1",
    "Hikvision": "rtsp://usuario:senha@IP:porta/Streaming/Channels/N01|N02",
    "Genérico": "URL RTSP informada manualmente (sem padrão fixo de fabricante)",
}

_MSG_OK = "Conectou! A câmera está respondendo."
_MSG_FALHA = "Não consegui conectar — confira IP, usuário, senha e canal."

_HIKVISION_CHANNEL_RE = re.compile(r"^/Streaming/Channels/(\d+)(0[12])$")


def build_rtsp_url(
    marca: str,
    ip: str,
    usuario: str,
    senha: str,
    canal: int,
    substream: bool = True,
    porta: int = 554,
) -> str:
    """Monta a URL RTSP no formato do fabricante escolhido.

    A senha é sempre escapada (`quote(senha, safe='')`) para que caracteres
    especiais (`@`, `:`, etc.) não quebrem a autoridade da URL.
    """
    senha_escapada = quote(senha, safe="")

    if marca == "Intelbras/Dahua":
        subtype = 1 if substream else 0
        return (
            f"rtsp://{usuario}:{senha_escapada}@{ip}:{porta}"
            f"/cam/realmonitor?channel={canal}&subtype={subtype}"
        )

    if marca == "Hikvision":
        sufixo = "02" if substream else "01"
        return f"rtsp://{usuario}:{senha_escapada}@{ip}:{porta}/Streaming/Channels/{canal}{sufixo}"

    if marca == "Genérico":
        raise ValueError(
            "Marca 'Genérico' não tem padrão de URL fixo — informe a URL RTSP "
            "completa manualmente."
        )

    raise ValueError(f"Marca desconhecida: {marca!r}. Marcas suportadas: {', '.join(MARCAS)}.")


def parse_rtsp_url(url: str) -> dict | None:
    """Caminho inverso de `build_rtsp_url`: extrai os campos do formulário a
    partir de uma URL RTSP já salva, para permitir editar a câmera sem ter
    que redigitar tudo do zero. Devolve `None` se a URL não for RTSP."""
    try:
        partes = urlparse(url)
    except Exception:
        return None

    if partes.scheme != "rtsp":
        return None

    campos: dict = {
        "marca": "Genérico",
        "ip": partes.hostname,
        "porta": partes.port or 554,
        "usuario": unquote(partes.username) if partes.username else None,
        "senha": unquote(partes.password) if partes.password else None,
        "canal": None,
        "substream": None,
    }

    if partes.path == "/cam/realmonitor":
        query = dict(par.split("=", 1) for par in partes.query.split("&") if "=" in par)
        canal = query.get("channel")
        subtype = query.get("subtype")
        campos["marca"] = "Intelbras/Dahua"
        campos["canal"] = int(canal) if canal is not None else None
        campos["substream"] = (subtype == "1") if subtype is not None else None
        return campos

    m = _HIKVISION_CHANNEL_RE.match(partes.path)
    if m:
        canal_str, sufixo = m.groups()
        campos["marca"] = "Hikvision"
        campos["canal"] = int(canal_str)
        campos["substream"] = sufixo == "02"
        return campos

    return campos


def test_connection(url: str, timeout: float = 10) -> tuple[bool, str]:
    """Abre a câmera com OpenCV para o revendedor confirmar, na hora, que a
    URL montada funciona — sem nunca levantar exceção para a tela."""
    import cv2  # import tardio: este módulo não deve exigir cv2 só para montar URLs

    cap = None
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(timeout) * 1000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(timeout) * 1000)
        except Exception:
            pass  # nem toda build do OpenCV tem essas props; segue sem elas

        if not cap.isOpened():
            return False, _MSG_FALHA

        ok, frame = cap.read()
        if not ok or frame is None:
            return False, _MSG_FALHA

        return True, _MSG_OK
    except Exception:
        log.exception("erro inesperado testando conexão RTSP")
        return False, _MSG_FALHA
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
