"""Chave privada protegida por senha.

O problema que isto resolve é de entrega, não de criptografia da licença: o
revendedor recebe tudo por um canal só (o chat da Workana), e a chave privada
não pode viajar solta num link — quem baixasse o link emitiria licença.

Aqui a chave privada é guardada CIFRADA (AES-256-GCM, com a senha derivada por
scrypt). O arquivo `.enc` pode ir em qualquer link: sem a senha, é lixo. A
senha, curta, vai separada no texto do chat. Só as duas coisas juntas emitem
licença.

Formato do arquivo: MAGIC + salt(16) + nonce(12) + texto cifrado."""
from __future__ import annotations

_MAGIC = b"PPKEY1\n"
_SALT = 16
_NONCE = 12
# scrypt custoso o suficiente para uma senha curta valer a pena; barato para o
# revendedor, que decifra uma vez por emissão.
_N, _R, _P = 2 ** 15, 8, 1


class SenhaIncorreta(Exception):
    """Senha errada ou arquivo da chave corrompido."""


def _derivar(senha: str, salt: bytes) -> bytes:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    return Scrypt(salt=salt, length=32, n=_N, r=_R, p=_P).derive(senha.encode("utf-8"))


def cifrar(pem: bytes, senha: str) -> bytes:
    """Cifra a chave privada (bytes do PEM) com a senha."""
    import os

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    salt = os.urandom(_SALT)
    nonce = os.urandom(_NONCE)
    ct = AESGCM(_derivar(senha, salt)).encrypt(nonce, pem, None)
    return _MAGIC + salt + nonce + ct


def decifrar(dados: bytes, senha: str) -> bytes:
    """Devolve os bytes do PEM. Levanta SenhaIncorreta se a senha não bate."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not dados.startswith(_MAGIC):
        raise SenhaIncorreta("Este arquivo de chave não é válido.")
    corpo = dados[len(_MAGIC):]
    salt, nonce, ct = corpo[:_SALT], corpo[_SALT:_SALT + _NONCE], corpo[_SALT + _NONCE:]
    try:
        return AESGCM(_derivar(senha, salt)).decrypt(nonce, ct, None)
    except InvalidTag as e:
        raise SenhaIncorreta("Senha incorreta.") from e


def parece_cifrada(dados: bytes) -> bool:
    return dados.startswith(_MAGIC)
