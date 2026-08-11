"""Separa o que o app LÊ do que o app GRAVA.

Instalado em `C:\\Program Files\\PrevencaoPerdas`, a pasta do programa é
SÓ-LEITURA para usuário comum (o Windows nega escrita sem elevação). O app
nasceu rodando de um checkout de desenvolvimento, onde o diretório atual era
gravável e já tinha `config/config.json` — e por isso, instalado, ele abria e
fechava sozinho: `AppConfig.load("config/config.json")` não achava o arquivo e
o processo saía com código 2 antes de qualquer janela aparecer.

A regra aqui é simples:

  * `resource_dir()` — o que veio EMBARCADO no instalador (modelos, cache
    OpenVINO, config de exemplo). Só-leitura. Fica no bundle do PyInstaller.
  * `data_dir()` — o que é DA LOJA (config.json com o token, banco de
    eventos, fotos e clipes). Gravável. Fica em `%LOCALAPPDATA%`.

Em desenvolvimento nada muda: os dois caem no diretório atual, que é como o
projeto sempre rodou.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

APP_DIR_NAME = "PrevencaoPerdas"


def is_frozen() -> bool:
    """True quando rodando empacotado pelo PyInstaller (o .exe da loja)."""
    return bool(getattr(sys, "frozen", False))


def resource_dir() -> Path:
    """Onde estão os arquivos embarcados no instalador (só-leitura).

    No PyInstaller é `sys._MEIPASS` (a pasta `_internal` no modo onedir).
    Em desenvolvimento é a raiz do repositório."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Onde o app pode GRAVAR. Instalado, `%LOCALAPPDATA%\\PrevencaoPerdas`.

    Nunca a pasta de instalação: em Program Files a escrita é negada e o
    programa morre sem explicar por quê."""
    if not is_frozen():
        return Path.cwd()
    base = os.environ.get("LOCALAPPDATA")
    raiz = Path(base) if base else Path.home()
    destino = raiz / APP_DIR_NAME
    destino.mkdir(parents=True, exist_ok=True)
    return destino


def default_config_path() -> Path:
    """O config.json da loja. Instalado vai pra área gravável; em dev
    continua sendo `config/config.json`, como sempre foi."""
    if is_frozen():
        return data_dir() / "config.json"
    return data_dir() / "config" / "config.json"


def _exemplo_preparado() -> dict:
    """O `config.example.json` do bundle, já ajustado para esta máquina.

    Levanta FileNotFoundError se o instalador foi gerado sem o exemplo."""
    exemplo = resource_dir() / "config" / "config.example.json"
    if not exemplo.exists():
        raise FileNotFoundError(
            f"config de exemplo não encontrado no pacote: {exemplo}. "
            "O instalador foi gerado sem o config.example.json."
        )

    dados = json.loads(exemplo.read_text(encoding="utf-8-sig"))

    # Modelos: caminho ABSOLUTO pro bundle. Relativo faria o engine procurar
    # (e tentar exportar) o cache OpenVINO dentro de Program Files.
    modelos = resource_dir() / "models"
    inferencia = dados.setdefault("inference", {})
    for chave in ("person_model", "pose_model"):
        if chave in inferencia:
            inferencia[chave] = str(modelos / Path(inferencia[chave]).name)

    # Evidência (fotos/clipes) na área gravável, não na pasta do programa.
    dados.setdefault("evidence", {})["dir"] = str(data_dir() / "evidence")

    # Loja nova começa sem câmera: a tela inicial orienta a cadastrar a
    # primeira. Herdar a câmera de exemplo só geraria erro de conexão.
    dados["cameras"] = []
    return dados


def _merge_missing(destino: dict, exemplo: dict) -> bool:
    """Copia para `destino` apenas as chaves AUSENTES, recursivamente.

    Nunca sobrescreve valor já presente — é o que protege o token do
    Telegram, as câmeras cadastradas e a calibração daquela loja. Devolve
    True se algo foi acrescentado."""
    mudou = False
    for chave, valor in exemplo.items():
        if chave not in destino:
            destino[chave] = valor
            mudou = True
        elif isinstance(valor, dict) and isinstance(destino[chave], dict):
            mudou = _merge_missing(destino[chave], valor) or mudou
    return mudou


def ensure_config(path: str | Path) -> bool:
    """Prepara o `config.json` da loja. Devolve True se CRIOU o arquivo
    (primeira execução), False se ele já existia.

    Duas responsabilidades:

    1. **Primeira execução numa loja nova:** cria o arquivo a partir do
       exemplo embarcado, senão o app morre com exit 2 antes de abrir a
       janela que cadastra câmera e token.
    2. **Atualização de versão:** acrescenta ao config existente as chaves
       que a versão nova passou a ter. Sem isto, quem ATUALIZA fica com um
       config antigo, e toda feature nova nasce desligada no default do
       código sem ninguém perceber — foi o que aconteceu com a detecção de
       arma (05/ago/2026): o bloco `detection.weapon` só entrou no exemplo
       depois, então quem já tinha instalado continuou com `weapon.enabled`
       no default False e testou um recurso que nunca chegou a carregar.

    Só-acrescenta-o-que-falta é a parte crítica da migração: sobrescrever
    apagaria o token do Telegram, as câmeras e a calibração daquela loja."""
    destino = Path(path)

    if not destino.exists():
        dados = _exemplo_preparado()
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(dados, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        return True

    # --- config já existe: migra as chaves novas, preservando as antigas ---
    try:
        atual = json.loads(destino.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        # Config corrompido não pode virar crash mudo aqui: deixa seguir para
        # o AppConfig.load, que devolve a mensagem em português explicando o
        # que está errado no arquivo.
        return False
    if not isinstance(atual, dict):
        return False

    try:
        exemplo = _exemplo_preparado()
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        # Bundle sem exemplo é erro de build, mas uma loja com config válido
        # não pode deixar de abrir por causa disso.
        return False

    # `cameras` fica de fora da migração: o exemplo preparado traz lista
    # vazia, e a lista da loja é dado do usuário, nunca default a completar.
    exemplo.pop("cameras", None)

    if _merge_missing(atual, exemplo):
        try:
            destino.write_text(json.dumps(atual, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        except OSError:
            # Sem permissão de escrita: o app segue rodando com o config em
            # memória; só não persiste a migração.
            pass
    return False
