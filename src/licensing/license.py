"""Emissão e verificação da licença. Lógica pura, sem disco e sem Qt.

Assinatura **assimétrica** (Ed25519), e essa escolha é o coração da proteção:
o programa carrega só a chave PÚBLICA, que pode estar à vista de todo mundo.
Quem emite licença é quem tem a chave PRIVADA, que nunca sai da máquina do
revendedor e nunca entra no repositório.

Isso importa porque o código-fonte deste projeto pertence ao cliente e está
publicado. Se a validação dependesse de um segredo dentro do código (uma
senha fixa, um hash conhecido), qualquer pessoa com acesso ao repositório
emitiria licenças. Com assinatura, ler o código inteiro não ajuda a forjar
nada.

Formato do token entregue ao instalador:

    PP1.<payload em base64url>.<assinatura em base64url>

O payload traz o código da máquina, então a licença só vale naquele
computador. Repassar o token para outra máquina não funciona — é o que
sustenta a promessa de "um código, uma instalação"."""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import date

PREFIXO = "PP1"


class LicencaInvalida(Exception):
    """Licença ausente, corrompida, forjada ou de outra máquina.

    A mensagem é lida pelo instalador na loja, então é sempre em português e
    sem jargão."""


@dataclass(frozen=True)
class Licenca:
    maquina: str
    cliente: str
    emitida_em: str
    expira_em: str | None = None

    def expirada(self, hoje: date | None = None) -> bool:
        if not self.expira_em:
            return False
        return (hoje or date.today()).isoformat() > self.expira_em


def _b64e(dados: bytes) -> str:
    return base64.urlsafe_b64encode(dados).decode("ascii").rstrip("=")


def _b64d(texto: str) -> bytes:
    faltando = (-len(texto)) % 4  # o padding foi removido na emissão
    return base64.urlsafe_b64decode(texto + "=" * faltando)


def _payload_bytes(licenca: Licenca) -> bytes:
    """Serialização canônica do que é assinado.

    `sort_keys` e separadores fixos são obrigatórios: a verificação assina de
    novo o mesmo dicionário, e qualquer diferença de espaço ou de ordem faria
    uma licença legítima ser recusada."""
    corpo = {
        "v": 1,
        "m": licenca.maquina,
        "c": licenca.cliente,
        "i": licenca.emitida_em,
        "e": licenca.expira_em,
    }
    return json.dumps(corpo, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def emitir(licenca: Licenca, chave_privada_pem: bytes) -> str:
    """Gera o token assinado. Roda SÓ na máquina do revendedor."""
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    chave = load_pem_private_key(chave_privada_pem, password=None)
    payload = _payload_bytes(licenca)
    assinatura = chave.sign(payload)
    return f"{PREFIXO}.{_b64e(payload)}.{_b64e(assinatura)}"


def verificar(token: str, chave_publica_pem: bytes) -> Licenca:
    """Confere a assinatura e devolve a licença. Levanta `LicencaInvalida`.

    Não confere a máquina nem a validade: quem faz isso é `activation.py`,
    que sabe em qual computador está rodando e em que dia."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    partes = (token or "").strip().split(".")
    if len(partes) != 3 or partes[0] != PREFIXO:
        raise LicencaInvalida(
            "Este código de licença não tem o formato esperado. "
            "Confira se ele foi copiado inteiro."
        )
    try:
        payload = _b64d(partes[1])
        assinatura = _b64d(partes[2])
    except Exception as e:
        raise LicencaInvalida(
            "Este código de licença está corrompido. Peça um novo ao seu fornecedor."
        ) from e

    try:
        chave = load_pem_public_key(chave_publica_pem)
        chave.verify(assinatura, payload)
    except InvalidSignature as e:
        raise LicencaInvalida(
            "Este código de licença não é válido (assinatura não confere). "
            "Peça um código novo ao seu fornecedor."
        ) from e
    except Exception as e:
        raise LicencaInvalida(
            "Não foi possível conferir este código de licença."
        ) from e

    try:
        corpo = json.loads(payload.decode("utf-8"))
        return Licenca(
            maquina=corpo["m"],
            cliente=corpo["c"],
            emitida_em=corpo["i"],
            expira_em=corpo["e"],
        )
    except Exception as e:
        raise LicencaInvalida("Este código de licença está incompleto.") from e


def gerar_par_de_chaves() -> tuple[bytes, bytes]:
    """Cria um par novo (privada, pública) em PEM. Roda uma única vez, na
    máquina do revendedor. A privada nunca entra no repositório."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    privada = Ed25519PrivateKey.generate()
    pem_privada = privada.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pem_publica = privada.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return pem_privada, pem_publica
