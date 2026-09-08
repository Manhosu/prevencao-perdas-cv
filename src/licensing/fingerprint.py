"""Identificação da máquina — o que amarra uma licença a UM computador.

O código aparece na tela da instalação, o instalador manda pro revendedor, e
o revendedor emite a licença daquela máquina. Copiar a pasta instalada pra
outro PC não adianta: os identificadores são outros e a licença não vale.

Duas fontes independentes, de propósito:

  * `guid`   — MachineGuid do Windows (registro). Sobrevive a troca de disco;
               muda quando o Windows é reinstalado.
  * `volume` — número de série do volume do sistema. Sobrevive a reinstalação
               do Windows; muda quando o disco é formatado ou trocado.

Nenhuma das duas sobrevive à troca de máquina, que é o caso que se quer
barrar. Já uma manutenção legítima (formatou, trocou o HD) derruba só UMA
delas — e é isso que o modo de tolerância em `activation.py` usa para avisar
em vez de deixar a loja sem vigilância de um dia pro outro.

As duas coletas são defensivas: qualquer falha vira `None`, nunca exceção. Um
PC que não entregue nenhum identificador é tratado como não identificável por
quem chama, e não derruba o programa aqui.
"""
from __future__ import annotations

import hashlib
import sys

# Comprimento de cada metade do código visível. 8 caracteres base32 = 40 bits,
# folga enorme para o número de instalações deste produto e curto o bastante
# para o instalador ditar por telefone se precisar.
_METADE = 8
# Sem I, O, 0 e 1: o instalador às vezes dita esse código por telefone, e
# esses quatro são os que geram erro de transcrição. Sobram 32 caracteres.
_ALFABETO = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _resumo(valor: str | None, sal: str) -> str:
    """Condensa um identificador em `_METADE` caracteres legíveis.

    O identificador cru nunca aparece na tela nem viaja em mensagem: o que
    circula é este resumo. `sal` separa os espaços das duas fontes, para que
    um mesmo valor bruto não produza o mesmo pedaço nas duas metades."""
    if not valor:
        return "-" * _METADE
    digest = hashlib.sha256(f"{sal}:{valor}".encode("utf-8")).digest()
    numero = int.from_bytes(digest[:8], "big")
    saida = []
    for _ in range(_METADE):
        numero, resto = divmod(numero, len(_ALFABETO))
        saida.append(_ALFABETO[resto])
    return "".join(saida)


def machine_guid() -> str | None:
    """MachineGuid do registro do Windows. `None` fora do Windows ou se o
    registro não puder ser lido (política de TI, permissão)."""
    if not sys.platform.startswith("win"):
        return None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Cryptography",
            0,
            winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
        ) as chave:
            valor, _ = winreg.QueryValueEx(chave, "MachineGuid")
        return str(valor) or None
    except Exception:
        return None


def volume_serial() -> str | None:
    """Número de série do volume do sistema. `None` fora do Windows ou se a
    chamada da API falhar."""
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes

        serial = ctypes.c_uint32(0)
        ok = ctypes.windll.kernel32.GetVolumeInformationW(
            ctypes.c_wchar_p("C:\\"),
            None, 0,
            ctypes.byref(serial),
            None, None, None, 0,
        )
        if not ok or serial.value == 0:
            return None
        return f"{serial.value:08X}"
    except Exception:
        return None


def coletar() -> dict[str, str | None]:
    """As duas fontes cruas desta máquina."""
    return {"guid": machine_guid(), "volume": volume_serial()}


def partes(identificadores: dict[str, str | None]) -> tuple[str, str]:
    """As duas metades resumidas, na ordem (guid, volume)."""
    return (
        _resumo(identificadores.get("guid"), "guid"),
        _resumo(identificadores.get("volume"), "volume"),
    )


def codigo_da_maquina(identificadores: dict[str, str | None] | None = None) -> str:
    """O código que aparece na tela, no formato `XXXX-XXXX-YYYY-YYYY`.

    É o que o instalador copia e manda pro revendedor emitir a licença."""
    ids = coletar() if identificadores is None else identificadores
    guid, volume = partes(ids)
    blocos = [guid[:4], guid[4:], volume[:4], volume[4:]]
    return "-".join(blocos)


def identificavel(identificadores: dict[str, str | None] | None = None) -> bool:
    """True se pelo menos uma das fontes respondeu.

    Uma máquina sem nenhuma delas não pode ser licenciada com segurança: duas
    máquinas diferentes produziriam o mesmo código."""
    ids = coletar() if identificadores is None else identificadores
    return any(ids.get(chave) for chave in ("guid", "volume"))
