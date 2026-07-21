"""O boot do app numa instalação limpa.

Regressão do bug de campo (20/jul/2026): instalado, o programa abria e fechava
sozinho. `main()` carregava `config/config.json` — que não existe numa loja
nova — e saía com código 2 antes de abrir qualquer janela.

O boot precisa se virar sozinho: se não há config, semeia um a partir do
exemplo embarcado e segue. Sem isso não existe caminho de "acabei de instalar"
até "programa aberto", porque a tela que cadastra câmera e token está DENTRO
do programa que não abre.
"""
import json

import pytest

from src.config.settings import ConfigError
from src.main import carregar_config


@pytest.fixture
def instalado(tmp_path, monkeypatch):
    """App empacotado, com o exemplo embarcado e LOCALAPPDATA no tmp."""
    bundle = tmp_path / "bundle"
    (bundle / "models").mkdir(parents=True)
    (bundle / "config").mkdir(parents=True)
    (bundle / "config" / "config.example.json").write_text(json.dumps({
        "store": {"id": "loja", "name": "Loja"},
        "telegram": {"bot_token": "", "chat_id": ""},
        "inference": {"person_model": "models/yolo11n.pt",
                      "pose_model": "models/yolo11n-pose.pt"},
        "detection": {"threshold": 0.6, "dwell_seconds": 1.2,
                      "window_seconds": 3.0, "cooldown_seconds": 30.0},
        "evidence": {"dir": "evidence", "retention_days": 30},
        "cameras": [{"name": "Exemplo", "rtsp_url": "rtsp://x/1"}],
    }), encoding="utf-8")

    local = tmp_path / "LocalAppData"
    local.mkdir()
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys._MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    return bundle, local


def test_boot_sem_config_nao_falha(instalado):
    """O coração do bug: instalação limpa não pode morrer no load."""
    from src.config.paths import default_config_path

    cfg = carregar_config(default_config_path())

    assert cfg is not None
    assert cfg.cameras == []


def test_boot_cria_o_config_no_disco(instalado):
    from src.config.paths import default_config_path

    destino = default_config_path()
    assert not destino.exists()

    carregar_config(destino)

    assert destino.exists()


def test_segunda_abertura_preserva_o_que_a_loja_configurou(instalado):
    """Abrir o programa de novo não pode apagar token nem câmeras."""
    from src.config.paths import default_config_path

    destino = default_config_path()
    carregar_config(destino)
    dados = json.loads(destino.read_text(encoding="utf-8"))
    dados["telegram"]["bot_token"] = "123:TOKEN-DA-LOJA"
    destino.write_text(json.dumps(dados), encoding="utf-8")

    cfg = carregar_config(destino)

    assert cfg.telegram.bot_token == "123:TOKEN-DA-LOJA"


def test_config_corrompido_ainda_reclama(instalado):
    """Semear não pode virar desculpa pra engolir erro de config inválido:
    se o arquivo existe e está quebrado, o erro tem que aparecer."""
    from src.config.paths import default_config_path

    destino = default_config_path()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("{ isso nao e json", encoding="utf-8")

    with pytest.raises(ConfigError):
        carregar_config(destino)
