"""O portão de licença no boot, e o que o lojista VÊ quando o programa não abre.

Duas exigências que se contradizem se mal feitas, e que estes testes fixam
juntas:

  * máquina não liberada NÃO roda (é a trava que foi vendida);
  * o programa nunca pode sumir calado. Bug de julho: abria e fechava sem
    explicar. Relato de set/2026: "dá erro e fecha sozinho" na primeira
    abertura sem licença. Com janela, todo motivo de encerramento vira caixa
    de mensagem; sem janela, vai para o log."""
import json
import logging
import sys

import pytest

import src.main as app_main
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
    # O log em arquivo fica fora destes testes: ele penduraria no logger raiz
    # um handler apontando para um tmp que some.
    monkeypatch.setattr(app_main, "configurar_log", lambda: None)
    return bundle, local


@pytest.fixture
def avisos(monkeypatch):
    """Captura as caixas de mensagem em vez de abrir janela."""
    registro: list[tuple[str, str]] = []
    monkeypatch.setattr(app_main, "_avisar",
                        lambda titulo, texto: registro.append((titulo, texto)))
    return registro


@pytest.fixture
def com_janela(monkeypatch):
    monkeypatch.setattr("sys.argv", ["PrevencaoPerdas", "--ui"])
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")


# --- sem janela (headless) --------------------------------------------------

def test_maquina_sem_licenca_nao_roda(instalado):
    """Sai com código 3 (distinto do 2 de config inválido) e, principalmente,
    sem chegar a carregar modelo nem abrir câmera."""
    assert main() == 3


def test_recusa_explica_e_mostra_o_codigo_da_maquina(instalado, caplog):
    """Sem janela, o instalador precisa achar no log o que fazer: qual código
    mandar para o fornecedor."""
    from src.licensing.fingerprint import codigo_da_maquina

    with caplog.at_level("ERROR"):
        main()

    registrado = caplog.text
    assert "não foi liberado" in registrado
    assert codigo_da_maquina() in registrado


# --- com janela (o app instalado) -------------------------------------------

def test_com_janela_sem_licenca_nao_registra_erro(
        instalado, com_janela, avisos, monkeypatch, caplog):
    """Máquina ainda não liberada é o estado normal da primeira abertura.
    Registrar isso como ERROR era o que o lojista lia como defeito."""
    monkeypatch.setattr("src.ui.activation_dialog.pedir_ativacao",
                        lambda *a, **k: False)

    with caplog.at_level(logging.INFO):
        main()

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_fechar_a_tela_sem_licenca_explica_em_vez_de_sumir(
        instalado, com_janela, avisos, monkeypatch):
    from src.licensing.fingerprint import codigo_da_maquina

    monkeypatch.setattr("src.ui.activation_dialog.pedir_ativacao",
                        lambda *a, **k: False)

    assert main() == 3
    assert len(avisos) == 1
    titulo, texto = avisos[0]
    assert "não liberado" in titulo
    assert codigo_da_maquina() in texto


def test_config_quebrado_com_janela_avisa(instalado, com_janela, avisos):
    """Antes, config inválido com janela saía com código 2 sem mostrar nada:
    o mesmo "abre e fecha" por outro caminho."""
    from src.config.paths import default_config_path

    destino = default_config_path()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("{ isso nao e json", encoding="utf-8")

    assert main() == 2
    assert len(avisos) == 1
    assert "configuração" in avisos[0][0].lower()


def test_falha_inesperada_vira_mensagem_com_o_caminho_do_log(
        instalado, com_janela, avisos, monkeypatch):
    """Antivírus bloqueando uma DLL, disco cheio, o que for: o lojista vê o
    motivo e onde está o registro para mandar ao suporte."""
    def explode():
        raise RuntimeError("DLL bloqueada pelo antivírus")

    monkeypatch.setattr(app_main, "main", explode)

    assert app_main.executar() == 1
    titulo, texto = avisos[0]
    assert "DLL bloqueada" in texto
    assert "app.log" in texto


def test_falha_inesperada_sem_janela_nao_abre_caixa(instalado, avisos, monkeypatch):
    def explode():
        raise RuntimeError("x")

    monkeypatch.setattr(app_main, "main", explode)

    assert app_main.executar() == 1
    assert avisos == []


# --- log do app instalado ---------------------------------------------------

def test_log_instalado_nao_quebra_sem_console(tmp_path, monkeypatch):
    """Empacotado sem console, sys.stdout e sys.stderr são None, e o Ultralytics
    imprime direto neles. Sem redirecionar, a primeira linha impressa derruba o
    processo antes de qualquer janela aparecer."""
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(app_main, "_LOG_PRONTO", False)
    raiz = logging.getLogger()
    handlers_antes = list(raiz.handlers)
    nivel_antes = raiz.level
    try:
        app_main.configurar_log()

        print("barra de progresso do tqdm")        # não pode explodir
        sys.stderr.write("aviso do ultralytics\n")
        logging.getLogger("main").info("registro de teste")

        log = app_main.arquivo_de_log()
        assert log.exists()
        assert (log.parent / "saida.log").exists()
    finally:
        for h in list(raiz.handlers):
            if h not in handlers_antes:
                h.close()
                raiz.removeHandler(h)
        raiz.setLevel(nivel_antes)
        if sys.stdout is not None:
            sys.stdout.close()
