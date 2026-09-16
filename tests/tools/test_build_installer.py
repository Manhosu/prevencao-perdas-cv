"""Testes da lógica de montagem do instalador Windows (Task 7, Plano 4).

O build real (chamar o PyInstaller/Inno Setup de verdade) é lento e pesado —
fica marcado @pytest.mark.slow e é pulado se o PyInstaller não estiver
instalado (`pytest.importorskip`). Aqui testamos só a lógica testável e
rápida: os argumentos montados, os pares de --add-data, e o conteúdo do
`installer/setup.iss`.

O teste mais importante desta suíte é `test_config_json_nunca_entra`: o
`config.json` de uma loja tem o token do Telegram do revendedor, e um
vazamento dele dentro do instalador (que é distribuído para várias lojas)
seria grave. Esse teste garante que nenhum argumento do comando, em nenhum
dos pares de --add-data, referencia `config.json` — só `config.example.json`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_installer import (
    APP_NAME,
    HIDDEN_IMPORTS,
    arquivos_de_dados,
    montar_comando_pyinstaller,
)

RAIZ = Path(__file__).resolve().parent.parent.parent


def test_comando_usa_modo_onedir_e_nome_do_app():
    comando = montar_comando_pyinstaller(RAIZ)
    assert "--onedir" in comando
    i = comando.index("--name")
    assert comando[i + 1] == APP_NAME
    assert APP_NAME == "PrevencaoPerdas"


def test_app_da_loja_abre_sem_janela_de_console():
    """Relato de campo (set/2026): "dá erro e fecha sozinho" na primeira
    abertura. Com console, o lojista via log com ERROR numa janela preta e
    fechava; falha real sumia junto. Sem console o log vai para arquivo e o
    motivo aparece em caixa de mensagem."""
    comando = montar_comando_pyinstaller(RAIZ)
    assert "--windowed" in comando


def test_entrypoint_e_src_main_com_ui():
    comando = montar_comando_pyinstaller(RAIZ)
    entrypoint = str(Path("src") / "main.py")
    assert any(entrypoint in c for c in comando)


def test_modelos_pt_entram_no_add_data():
    pares = arquivos_de_dados(RAIZ)
    origens = [str(o) for o, _ in pares]
    assert any(o.endswith(".pt") for o in origens), (
        "os modelos .pt tem que ser embarcados — sem internet liberada na "
        "loja, baixar modelo na 1a execucao seria uma falha em campo"
    )


def test_pastas_openvino_entram_no_add_data():
    pares = arquivos_de_dados(RAIZ)
    origens = [str(o) for o, _ in pares]
    assert any("_openvino_model" in o for o in origens)


def test_config_example_entra_no_add_data():
    pares = arquivos_de_dados(RAIZ)
    origens = [str(o) for o, _ in pares]
    assert any("config.example.json" in o for o in origens)


def test_config_json_nunca_entra():
    """O vazamento que não pode acontecer: config.json tem o token do
    revendedor. Nem nos dados, nem em nenhum argumento do comando."""
    pares = arquivos_de_dados(RAIZ)
    for origem, destino in pares:
        assert "config.json" not in str(origem)
        assert "config.json" not in destino

    comando = montar_comando_pyinstaller(RAIZ)
    for arg in comando:
        assert "config.json" not in arg


def test_hidden_imports_esperados_estao_presentes():
    comando = montar_comando_pyinstaller(RAIZ)
    for modulo in ("ultralytics", "openvino", "cv2", "PySide6"):
        assert modulo in HIDDEN_IMPORTS
        assert modulo in comando
    assert comando.count("--hidden-import") == len(HIDDEN_IMPORTS)


def test_add_data_usa_ponto_e_virgula_no_windows(monkeypatch):
    monkeypatch.setattr("scripts.build_installer.platform.system", lambda: "Windows")
    comando = montar_comando_pyinstaller(RAIZ)
    pares_add_data = [comando[i + 1] for i, a in enumerate(comando) if a == "--add-data"]
    assert pares_add_data, "esperava pelo menos um --add-data (modelos + config.example.json)"
    for par in pares_add_data:
        # separador do Windows: ";" (não ":", que colide com a letra de unidade)
        assert len(par.split(";")) == 2


def test_add_data_usa_dois_pontos_fora_do_windows(monkeypatch):
    monkeypatch.setattr("scripts.build_installer.platform.system", lambda: "Linux")
    comando = montar_comando_pyinstaller(RAIZ)
    pares_add_data = [comando[i + 1] for i, a in enumerate(comando) if a == "--add-data"]
    assert pares_add_data
    for par in pares_add_data:
        assert par.split(":")[-1] in ("models", "config") or par.rsplit(":", 1)[-1].startswith("models")


def test_arquivos_de_dados_devolve_pares_origem_destino():
    pares = arquivos_de_dados(RAIZ)
    assert pares, "deveria achar modelos + config.example.json neste repo"
    for origem, destino in pares:
        assert isinstance(origem, Path)
        assert isinstance(destino, str)
        assert origem.exists(), f"origem inexistente: {origem}"


# --- installer/setup.iss ---

ISS_PATH = RAIZ / "installer" / "setup.iss"


def _conteudo_iss() -> str:
    return ISS_PATH.read_text(encoding="utf-8")


def test_setup_iss_existe():
    assert ISS_PATH.exists()


def test_setup_iss_nome_em_portugues():
    conteudo = _conteudo_iss()
    assert "Prevenção de Perdas" in conteudo


def test_setup_iss_instala_em_autopf_prevencaoperdas():
    conteudo = _conteudo_iss()
    assert "{autopf}\\PrevencaoPerdas" in conteudo


def test_setup_iss_tem_atalho_desktop_e_menu_iniciar():
    conteudo = _conteudo_iss()
    assert "{autodesktop}" in conteudo
    assert "{group}" in conteudo


def test_setup_iss_tem_opcao_iniciar_com_windows():
    conteudo = _conteudo_iss().lower()
    assert "iniciar com o windows" in conteudo
    assert "currentversion\\run" in conteudo


def test_setup_iss_nao_desativa_desinstalador():
    conteudo = _conteudo_iss()
    assert "Uninstallable=no" not in conteudo


def test_setup_iss_nao_menciona_config_json():
    conteudo = _conteudo_iss()
    assert "config.json" not in conteudo


# --- build real (lento, pesado — pulável) ---


@pytest.mark.slow
def test_build_real_gera_o_executavel():
    """Roda o PyInstaller de verdade. Lento (minutos) e pesado (GBs) —
    pulado por padrão. Requer PyInstaller instalado."""
    pytest.importorskip("PyInstaller")
    import subprocess

    comando = montar_comando_pyinstaller(RAIZ)
    resultado = subprocess.run(comando, cwd=RAIZ)
    assert resultado.returncode == 0
    assert (RAIZ / "dist" / APP_NAME / f"{APP_NAME}.exe").exists()


# --- runtime do OpenVINO (bug de campo 20/jul/2026) ----------------------
#
# O cliente instalou e o programa fechava sozinho. Depois de corrigir o
# bootstrap do config, apareceu o erro real por baixo:
#
#   RuntimeError: Unable to read the model: "yolo11n.xml"
#   Available frontends: jax pytorch
#
# O OpenVINO carrega frontends e plugins de dispositivo por DLL dinâmica em
# runtime. A análise estática do PyInstaller não enxerga esse carregamento,
# então o bundle saiu com openvino.dll mas SEM `openvino_ir_frontend.dll`
# (que lê o .xml) e SEM `openvino_intel_cpu_plugin.dll` (que roda a
# inferência). O pacote parecia completo e quebrava só na primeira detecção.

def test_coleta_os_binarios_do_openvino():
    """Sem isto o .exe empacota um OpenVINO que não lê o próprio modelo."""
    comando = montar_comando_pyinstaller(Path("/proj"))
    assert "--collect-binaries" in comando
    assert "openvino" in comando[comando.index("--collect-binaries") + 1]


# --- remoção do plugin de GPU (bug de campo 16/set) ----------------------
#
# Numa maquina com placa Intel de driver antigo, abrir o programa dava
# "Ponto de entrada nao encontrado — clCreateBufferWithProperties em
# openvino_intel_gpu_plugin.dll" e fechava. O ultralytics chama
# core.available_devices, que carrega o plugin de GPU para enumerar; esse
# plugin depende de uma funcao de OpenCL que o driver antigo nao tem, e o
# carregamento trava no proprio Windows. O sistema roda na CPU e nao usa GPU,
# entao o plugin sai do bundle.

def test_remove_o_plugin_de_gpu_do_bundle(tmp_path):
    from scripts.build_installer import (
        APP_NAME,
        remover_plugins_nao_usados,
    )

    libs = tmp_path / APP_NAME / "_internal" / "openvino" / "libs"
    libs.mkdir(parents=True)
    (libs / "openvino_intel_gpu_plugin.dll").write_bytes(b"x")
    (libs / "openvino_intel_npu_plugin.dll").write_bytes(b"x")
    (libs / "openvino_intel_cpu_plugin.dll").write_bytes(b"x")  # este FICA

    removidos = remover_plugins_nao_usados(tmp_path)

    nomes = {p.name for p in removidos}
    assert "openvino_intel_gpu_plugin.dll" in nomes
    assert not (libs / "openvino_intel_gpu_plugin.dll").exists()
    # a CPU (que o sistema de fato usa) nunca pode ser removida
    assert (libs / "openvino_intel_cpu_plugin.dll").exists()


def test_remocao_de_plugin_e_idempotente(tmp_path):
    """Rodar o build de novo (bundle já limpo) não pode quebrar."""
    from scripts.build_installer import APP_NAME, remover_plugins_nao_usados

    (tmp_path / APP_NAME / "_internal" / "openvino" / "libs").mkdir(parents=True)
    assert remover_plugins_nao_usados(tmp_path) == []
