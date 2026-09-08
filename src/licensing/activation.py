"""Estado da licença nesta máquina: ler, conferir e decidir se o programa roda.

Regra de comparação (ver `fingerprint.py` para as duas fontes):

  * as duas metades batem  → liberado;
  * só uma bate            → TOLERÂNCIA. É a manutenção legítima (formatou o
    PC, trocou o HD). O sistema continua vigiando e avisa, por
    `TOLERANCIA_DIAS`, que precisa de uma licença nova;
  * nenhuma bate           → é outra máquina. Bloqueia.

A tolerância existe por uma razão de campo, não por caridade: quem comprou
está usando isso como sistema de segurança de uma loja. Desligar a vigilância
sem aviso porque o cliente trocou um disco causaria um estrago maior do que a
cópia que a trava evita."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path

from src.licensing import fingerprint
from src.licensing.license import Licenca, LicencaInvalida, verificar

# Dias que o sistema continua funcionando quando só um dos identificadores
# bate. Tempo folgado para o lojista falar com quem instalou e receber a
# licença nova sem ficar sem vigilância.
TOLERANCIA_DIAS = 15

ARQUIVO_LICENCA = "licenca.txt"
ARQUIVO_ESTADO = "licenca_estado.json"


class Estado(str, Enum):
    OK = "ok"
    SEM_LICENCA = "sem_licenca"
    TOLERANCIA = "tolerancia"
    BLOQUEADA = "bloqueada"


@dataclass
class Ativacao:
    estado: Estado
    mensagem: str = ""
    licenca: Licenca | None = None
    dias_restantes: int | None = None

    @property
    def pode_rodar(self) -> bool:
        return self.estado in (Estado.OK, Estado.TOLERANCIA)


def _chave_publica() -> bytes:
    from src.licensing.chave_publica import CHAVE_PUBLICA_PEM

    return CHAVE_PUBLICA_PEM.encode("ascii")


def caminho_licenca(pasta: Path) -> Path:
    return Path(pasta) / ARQUIVO_LICENCA


def ler_token(pasta: Path) -> str:
    try:
        return caminho_licenca(pasta).read_text(encoding="utf-8-sig").strip()
    except OSError:
        return ""


def gravar_token(pasta: Path, token: str) -> None:
    destino = caminho_licenca(pasta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(token.strip(), encoding="utf-8")


def _estado_path(pasta: Path) -> Path:
    return Path(pasta) / ARQUIVO_ESTADO


def _ler_inicio_tolerancia(pasta: Path) -> str | None:
    try:
        dados = json.loads(_estado_path(pasta).read_text(encoding="utf-8"))
        valor = dados.get("tolerancia_desde")
        return str(valor) if valor else None
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def _gravar_inicio_tolerancia(pasta: Path, quando: str | None) -> None:
    """Grava (ou limpa) a data em que a divergência apareceu pela primeira vez.

    Falha de escrita não pode derrubar o programa: no pior caso a contagem
    recomeça na próxima abertura, o que é o lado seguro do erro (o cliente
    ganha tempo, não perde)."""
    try:
        alvo = _estado_path(pasta)
        if quando is None:
            alvo.unlink(missing_ok=True)
            return
        alvo.parent.mkdir(parents=True, exist_ok=True)
        alvo.write_text(json.dumps({"tolerancia_desde": quando}), encoding="utf-8")
    except OSError:
        pass


def _dias_entre(inicio: str, hoje: date) -> int:
    try:
        return (hoje - date.fromisoformat(inicio)).days
    except ValueError:
        return 0


def avaliar(
    pasta: Path,
    identificadores: dict[str, str | None] | None = None,
    hoje: date | None = None,
) -> Ativacao:
    """Decide se este computador pode rodar o sistema."""
    hoje = hoje or date.today()
    ids = fingerprint.coletar() if identificadores is None else identificadores

    token = ler_token(pasta)
    if not token:
        return Ativacao(
            Estado.SEM_LICENCA,
            "Este computador ainda não foi liberado. Informe o código da máquina "
            "ao seu fornecedor para receber a licença.",
        )

    try:
        licenca = verificar(token, _chave_publica())
    except LicencaInvalida as e:
        return Ativacao(Estado.BLOQUEADA, str(e))

    if licenca.expirada(hoje):
        return Ativacao(
            Estado.BLOQUEADA,
            f"A licença deste computador venceu em {licenca.expira_em}. "
            "Fale com seu fornecedor para renovar.",
            licenca=licenca,
        )

    if not fingerprint.identificavel(ids):
        # Sem nenhuma fonte, duas máquinas diferentes gerariam o mesmo código —
        # liberar aqui transformaria uma licença numa chave-mestra.
        return Ativacao(
            Estado.BLOQUEADA,
            "Não foi possível identificar este computador com segurança. "
            "Fale com seu fornecedor.",
            licenca=licenca,
        )

    guid_atual, volume_atual = fingerprint.partes(ids)
    blocos = licenca.maquina.split("-")
    if len(blocos) != 4:
        return Ativacao(Estado.BLOQUEADA,
                        "Este código de licença está incompleto.", licenca=licenca)
    guid_licenca = blocos[0] + blocos[1]
    volume_licenca = blocos[2] + blocos[3]

    bate_guid = guid_atual == guid_licenca
    bate_volume = volume_atual == volume_licenca

    if bate_guid and bate_volume:
        _gravar_inicio_tolerancia(pasta, None)  # voltou ao normal
        return Ativacao(Estado.OK, licenca=licenca)

    if not bate_guid and not bate_volume:
        return Ativacao(
            Estado.BLOQUEADA,
            "Esta licença foi emitida para outro computador. Cada licença vale "
            "para uma máquina. Peça uma licença para este computador ao seu "
            "fornecedor.",
            licenca=licenca,
        )

    # Só uma das fontes bate: manutenção legítima (formatou, trocou o disco).
    inicio = _ler_inicio_tolerancia(pasta)
    if inicio is None:
        inicio = hoje.isoformat()
        _gravar_inicio_tolerancia(pasta, inicio)
    restantes = TOLERANCIA_DIAS - _dias_entre(inicio, hoje)
    if restantes <= 0:
        return Ativacao(
            Estado.BLOQUEADA,
            "Este computador mudou (formatação ou troca de disco) e o prazo para "
            "regularizar a licença terminou. Peça uma licença nova ao seu "
            "fornecedor.",
            licenca=licenca,
        )
    return Ativacao(
        Estado.TOLERANCIA,
        f"Este computador mudou desde a liberação. O sistema continua "
        f"funcionando por mais {restantes} dia(s). Peça uma licença nova ao seu "
        f"fornecedor informando o código atual da máquina.",
        licenca=licenca,
        dias_restantes=restantes,
    )


def ativar(pasta: Path, token: str,
           identificadores: dict[str, str | None] | None = None,
           hoje: date | None = None) -> Ativacao:
    """Confere um token digitado e, se servir para esta máquina, grava.

    Só grava quando o resultado libera o uso: token de outra máquina não pode
    substituir uma licença boa que já esteja no lugar."""
    ids = fingerprint.coletar() if identificadores is None else identificadores
    hoje = hoje or date.today()

    try:
        licenca = verificar(token, _chave_publica())
    except LicencaInvalida as e:
        return Ativacao(Estado.BLOQUEADA, str(e))

    if licenca.expirada(hoje):
        return Ativacao(
            Estado.BLOQUEADA,
            f"Esta licença venceu em {licenca.expira_em}.", licenca=licenca)

    if not fingerprint.identificavel(ids):
        return Ativacao(
            Estado.BLOQUEADA,
            "Não foi possível identificar este computador com segurança.",
            licenca=licenca)

    esperado = fingerprint.codigo_da_maquina(ids)
    if licenca.maquina != esperado:
        return Ativacao(
            Estado.BLOQUEADA,
            "Esta licença foi emitida para outro computador. Confira se o "
            f"código enviado foi o desta máquina: {esperado}",
            licenca=licenca)

    gravar_token(pasta, token)
    _gravar_inicio_tolerancia(pasta, None)
    return Ativacao(Estado.OK, "Computador liberado.", licenca=licenca)
