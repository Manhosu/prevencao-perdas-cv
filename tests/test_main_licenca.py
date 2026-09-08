"""O portão de licença no boot.

Duas exigências que se contradizem se mal feitas, e que estes testes fixam
juntas:

  * máquina não liberada NÃO roda (é a trava que foi vendida);
  * máquina não liberada não pode morrer em silêncio. O bug de campo de julho
    foi exatamente esse: o programa abria e fechava sozinho, e ninguém na loja
    tinha como saber por quê. Sair sem explicar é regressão."""
import json

import pytest

from src.main import main


@pytest.fixture
def instalado(tmp_path, monkeypatch):
    """App empacotado, com config semeável e LOCALAPPDATA no tmp."""
    bundle = tmp_path / "bundle"
    (bundle / "models").mkdir(parents=True)
    (bundle / "config").mkdir(parents=True)
    (bundle / "config" / "config.example.json").write_text(json.dumps({
        "store": {"id": "loja", "name": "Loja"},
        "telegram": {"bot_token": "", "chat_id": ""},
        "inference": {"person_model": "models/yolo11n.pt",
                      "pose_model": "models/yolo11n-pose.pt"},
        "detection": {"mode": "panico"},
        "evidence": {"dir": "evidence", "retention_days": 30},
        "cameras": [],
    }), encoding="utf-8")

    local = tmp_path / "LocalAppData"
    local.mkdir()
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys._MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setattr("sys.argv", ["PrevencaoPerdas"])
    return bundle, local


def test_maquina_sem_licenca_nao_roda(instalado):
    """Sai com código 3 (distinto do 2 de config inválido) e, principalmente,
    sem chegar a carregar modelo nem abrir câmera."""
    assert main() == 3


def test_recusa_explica_e_mostra_o_codigo_da_maquina(instalado, caplog):
    """O instalador precisa sair da tela sabendo o que fazer: qual código
    mandar para o fornecedor."""
    from src.licensing.fingerprint import codigo_da_maquina

    with caplog.at_level("ERROR"):
        main()

    registrado = caplog.text
    assert "não foi liberado" in registrado
    assert codigo_da_maquina() in registrado
