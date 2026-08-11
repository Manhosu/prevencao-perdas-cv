"""Onde o app LÊ e onde o app GRAVA quando está instalado.

Bug de campo (20/jul/2026): o cliente instalou em duas máquinas e o programa
abria e fechava sozinho. Causa: o app tratava a pasta de instalação como
diretório de trabalho gravável e não tinha bootstrap de primeira execução.
Instalado em `C:\\Program Files\\PrevencaoPerdas`:

  1. não existe `config/config.json`  -> ConfigError -> exit 2  (o sintoma)
  2. `data/app.db` e `evidence/` cairiam em Program Files -> acesso negado
  3. o cache OpenVINO seria exportado em Program Files -> acesso negado

Estes testes fixam a separação que resolve os três: recurso embarcado é
SÓ-LEITURA (vem do bundle), dado do usuário é GRAVÁVEL (vem do LOCALAPPDATA).
"""
import json

import pytest

from src.config.paths import (
    data_dir,
    default_config_path,
    ensure_config,
    is_frozen,
    resource_dir,
)
from src.config.settings import AppConfig


@pytest.fixture
def congelado(tmp_path, monkeypatch):
    """Simula o app rodando empacotado pelo PyInstaller.

    `sys.frozen` e `sys._MEIPASS` são o que o próprio PyInstaller injeta em
    runtime; LOCALAPPDATA é redirecionado pro tmp pra não sujar a máquina."""
    bundle = tmp_path / "bundle"
    (bundle / "models").mkdir(parents=True)
    (bundle / "config").mkdir(parents=True)
    local = tmp_path / "LocalAppData"
    local.mkdir()

    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys._MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    return bundle, local


def _exemplo(bundle, **extra):
    """Grava um config.example.json mínimo e válido dentro do bundle."""
    base = {
        "store": {"id": "loja", "name": "Loja"},
        "telegram": {"bot_token": "", "chat_id": ""},
        "inference": {"person_model": "models/yolo11n.pt",
                      "pose_model": "models/yolo11n-pose.pt"},
        "detection": {"threshold": 0.6, "dwell_seconds": 1.2,
                      "window_seconds": 3.0, "cooldown_seconds": 30.0},
        "evidence": {"dir": "evidence", "retention_days": 30},
        "cameras": [{"name": "Exemplo", "rtsp_url": "rtsp://x/1"}],
    }
    base.update(extra)
    (bundle / "config" / "config.example.json").write_text(
        json.dumps(base), encoding="utf-8")


# --- modo desenvolvimento: nada pode mudar -------------------------------

def test_dev_nao_e_congelado():
    assert is_frozen() is False


def test_dev_grava_no_diretorio_atual(tmp_path, monkeypatch):
    """Em dev o comportamento de hoje é preservado: tudo relativo ao cwd."""
    monkeypatch.chdir(tmp_path)
    assert data_dir() == tmp_path


def test_dev_mantem_o_caminho_de_config_de_sempre(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert default_config_path() == tmp_path / "config" / "config.json"


# --- modo instalado ------------------------------------------------------

def test_instalado_le_recurso_do_bundle(congelado):
    bundle, _ = congelado
    assert resource_dir() == bundle


def test_instalado_grava_no_localappdata(congelado):
    """O ponto central: NUNCA gravar na pasta de instalação (Program Files)."""
    bundle, local = congelado
    destino = data_dir()
    assert local in destino.parents or destino == local / "PrevencaoPerdas"
    assert bundle not in destino.parents


def test_instalado_cria_o_diretorio_de_dados(congelado):
    assert data_dir().is_dir()


def test_instalado_config_fica_no_diretorio_gravavel(congelado):
    assert default_config_path().parent == data_dir()


# --- bootstrap de primeira execução --------------------------------------

def test_primeira_execucao_cria_o_config(congelado):
    """Sem isto o app morre com exit 2 numa instalação limpa."""
    bundle, _ = congelado
    _exemplo(bundle)
    destino = default_config_path()
    assert not destino.exists()

    criou = ensure_config(destino)

    assert criou is True
    assert destino.exists()


def test_config_criado_abre_sem_erro(congelado):
    """O teste que prova que o app não morre mais no boot: o config semeado
    tem que passar pelo AppConfig.load de verdade."""
    bundle, _ = congelado
    _exemplo(bundle)
    destino = default_config_path()

    ensure_config(destino)

    AppConfig.load(destino)  # levanta ConfigError se inválido


def test_nao_sobrescreve_config_existente(congelado):
    """Proteção crítica: sobrescrever apagaria o token do Telegram e as
    câmeras já cadastradas da loja a cada abertura do programa."""
    bundle, _ = congelado
    _exemplo(bundle)
    destino = default_config_path()
    ensure_config(destino)
    original = destino.read_text(encoding="utf-8")
    destino.write_text(original.replace('"name": "Loja"', '"name": "Mercado do Ze"'),
                       encoding="utf-8")

    criou = ensure_config(destino)

    assert criou is False
    assert "Mercado do Ze" in destino.read_text(encoding="utf-8")


def test_config_criado_aponta_para_os_modelos_embarcados(congelado):
    """Bug 3: o cache OpenVINO mora junto dos modelos embarcados. Se o caminho
    for relativo, o engine tenta exportar dentro de Program Files."""
    bundle, _ = congelado
    _exemplo(bundle)
    destino = default_config_path()

    ensure_config(destino)

    cfg = AppConfig.load(destino)
    assert cfg.inference.person_model.startswith(str(bundle))
    assert cfg.inference.pose_model.startswith(str(bundle))


def test_config_criado_grava_evidencia_em_area_gravavel(congelado):
    """Bug 2: evidência não pode cair na pasta de instalação."""
    bundle, _ = congelado
    _exemplo(bundle)
    destino = default_config_path()

    ensure_config(destino)

    cfg = AppConfig.load(destino)
    assert cfg.evidence.dir.startswith(str(data_dir()))


def test_config_criado_comeca_sem_camera(congelado):
    """Loja nova não herda a câmera de exemplo: a UI orienta a cadastrar."""
    bundle, _ = congelado
    _exemplo(bundle)
    destino = default_config_path()

    ensure_config(destino)

    assert AppConfig.load(destino).cameras == []


def test_sem_exemplo_no_bundle_nao_explode(congelado):
    """Bundle sem o exemplo é erro de build, mas não pode virar crash mudo
    na cara do lojista."""
    destino = default_config_path()
    with pytest.raises(FileNotFoundError):
        ensure_config(destino)


# --- migração ao ATUALIZAR de versão -------------------------------------
#
# Bug de campo (05/ago/2026): a detecção de arma foi entregue ligada no
# `config.example.json`, mas quem ATUALIZOU por cima de uma instalação
# anterior continuou com o config antigo — sem o bloco `detection.weapon`.
# Como o default do código é `enabled: False`, o recurso nunca chegou a
# carregar, e o cliente testou arma contra um detector que não existia.
# A migração abaixo é o que impede que toda feature nova nasça desligada
# em quem já tem o sistema instalado.

def test_atualizacao_acrescenta_bloco_novo_da_versao(congelado):
    bundle, _ = congelado
    _exemplo(bundle)
    destino = default_config_path()
    ensure_config(destino)  # loja instalou a versão ANTIGA

    # a versão nova do app passa a trazer um bloco que aquele config não tem
    _exemplo(bundle, detection={"threshold": 0.6, "dwell_seconds": 1.2,
                                "window_seconds": 3.0, "cooldown_seconds": 30.0,
                                "weapon": {"enabled": True}})

    criou = ensure_config(destino)

    assert criou is False  # não é primeira execução: só migrou
    dados = json.loads(destino.read_text(encoding="utf-8"))
    assert dados["detection"]["weapon"]["enabled"] is True
    AppConfig.load(destino)  # e o resultado continua sendo um config válido


def test_atualizacao_preserva_o_que_a_loja_configurou(congelado):
    """A trava crítica da migração: acrescentar o que falta NUNCA pode
    encostar no token, nas câmeras ou na calibração daquela loja."""
    bundle, _ = congelado
    _exemplo(bundle)
    destino = default_config_path()
    ensure_config(destino)

    dados = json.loads(destino.read_text(encoding="utf-8"))
    dados["telegram"]["bot_token"] = "TOKEN-DA-LOJA"
    dados["detection"]["threshold"] = 0.85  # calibração feita em campo
    dados["cameras"] = [{"name": "Caixa", "rtsp_url": "rtsp://x/9"}]
    destino.write_text(json.dumps(dados), encoding="utf-8")

    _exemplo(bundle, detection={"threshold": 0.6, "weapon": {"enabled": True}})
    ensure_config(destino)

    novo = json.loads(destino.read_text(encoding="utf-8"))
    assert novo["telegram"]["bot_token"] == "TOKEN-DA-LOJA"
    assert novo["detection"]["threshold"] == 0.85
    assert [c["name"] for c in novo["cameras"]] == ["Caixa"]
    assert novo["detection"]["weapon"]["enabled"] is True  # e o novo entrou


def test_migracao_nao_explode_com_config_corrompido(congelado):
    """Config quebrado à mão segue para o AppConfig.load, que explica o erro
    em português — não pode morrer aqui, sem mensagem."""
    bundle, _ = congelado
    _exemplo(bundle)
    destino = default_config_path()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("{ isso nao e json", encoding="utf-8")

    assert ensure_config(destino) is False
    assert destino.read_text(encoding="utf-8") == "{ isso nao e json"


def test_bundle_sem_exemplo_nao_derruba_loja_ja_configurada(congelado):
    """Erro de build não pode impedir de abrir uma loja que já tem config."""
    bundle, _ = congelado
    _exemplo(bundle)
    destino = default_config_path()
    ensure_config(destino)
    (bundle / "config" / "config.example.json").unlink()

    assert ensure_config(destino) is False  # não levanta
