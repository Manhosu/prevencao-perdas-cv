"""Monta e roda o build do instalador Windows (PyInstaller + Inno Setup).

Task 7 (Plano 4): o que transforma o projeto em produto — o revendedor
instala na loja com um clique, sem Python, sem terminal.

Modo `onedir` (não `onefile`): abre mais rápido e atualiza melhor — trocar
um arquivo dentro da pasta é mais simples/robusto do que reconstruir um
único executável monolítico.

EMBARCA os modelos (`models/*.pt` e `models/*_openvino_model/`) e o
`config/config.example.json`. É crítico: a loja pode não ter internet
liberada, e baixar modelo na primeira execução seria uma falha em campo.

NUNCA embarca `config/config.json` — tem o token do Telegram do revendedor
daquela loja específica. Só o `config.example.json` (o modelo em branco)
entra no instalador; `config.json` é gerado/editado na própria loja, pela
aba "Configuração" da UI (Task 6).

Uso:
    .venv\\Scripts\\python.exe scripts/build_installer.py
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

APP_NAME = "PrevencaoPerdas"
ENTRYPOINT = Path("src") / "main.py"

# Módulos que o PyInstaller costuma não enxergar sozinho com este stack
# (import dinâmico/plugin do ultralytics, backend nativo do openvino,
# bindings C++ do cv2, plugins do PySide6) — sem isto o executável abre e
# quebra na primeira chamada real, não no build.
HIDDEN_IMPORTS = [
    "ultralytics",
    "openvino",
    "cv2",
    "PySide6",
]


def _separador_add_data() -> str:
    """No Windows o --add-data usa ';' (o ':' colide com a letra de
    unidade, ex.: 'C:'). Fora do Windows (build cruzado/CI) usa ':'."""
    return ";" if platform.system() == "Windows" else ":"


def arquivos_de_dados(raiz: Path) -> list[tuple[Path, str]]:
    """Pares (origem, destino) para os `--add-data` do PyInstaller.

    - `models/*.pt`: os pesos do YOLO (detecção de pessoa e pose).
    - `models/*_openvino_model/`: o cache exportado do OpenVINO (evita
      reexportar/otimizar o modelo na loja, o que pode falhar sem internet).
    - `config/config.example.json`: o modelo de configuração em branco.

    `config/config.json` NUNCA entra aqui — tem o token do Telegram do
    revendedor daquela loja.
    """
    pares: list[tuple[Path, str]] = []

    models_dir = raiz / "models"
    if models_dir.is_dir():
        for pt in sorted(models_dir.glob("*.pt")):
            pares.append((pt, "models"))
        for openvino_dir in sorted(models_dir.glob("*_openvino_model")):
            if openvino_dir.is_dir():
                pares.append((openvino_dir, f"models/{openvino_dir.name}"))

    config_example = raiz / "config" / "config.example.json"
    if config_example.exists():
        pares.append((config_example, "config"))

    return pares


def montar_comando_pyinstaller(raiz: Path) -> list[str]:
    """Monta a lista de argumentos do comando do PyInstaller.

    Só monta a lista — não executa nada. É o que torna isto testável sem
    rodar o build de verdade (lento, minutos, GBs de saída)."""
    separador = _separador_add_data()

    comando = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(raiz / ENTRYPOINT),
        "--name",
        APP_NAME,
        "--onedir",
        "--noconfirm",
    ]

    for origem, destino in arquivos_de_dados(raiz):
        comando += ["--add-data", f"{origem}{separador}{destino}"]

    for modulo in HIDDEN_IMPORTS:
        comando += ["--hidden-import", modulo]

    return comando


def main() -> int:
    """Roda o build de verdade: PyInstaller e, se o Inno Setup (`ISCC`)
    estiver disponível no PATH, o instalador final também."""
    raiz = Path(__file__).resolve().parent.parent
    comando = montar_comando_pyinstaller(raiz)

    print("Gerando executável com PyInstaller (modo onedir)...")
    print(" ".join(comando))
    resultado = subprocess.run(comando, cwd=raiz)
    if resultado.returncode != 0:
        print("Falha ao gerar o executável com PyInstaller.")
        return resultado.returncode
    print(f"Executável gerado em dist/{APP_NAME}/")

    iscc = shutil.which("ISCC") or shutil.which("iscc")
    setup_iss = raiz / "installer" / "setup.iss"
    if iscc and setup_iss.exists():
        print("Gerando instalador com Inno Setup...")
        resultado_iss = subprocess.run([iscc, str(setup_iss)], cwd=raiz)
        if resultado_iss.returncode != 0:
            print("Falha ao gerar o instalador com Inno Setup.")
            return resultado_iss.returncode
        print("Instalador gerado com sucesso.")
    else:
        print(
            "Inno Setup (ISCC) não encontrado no PATH — pulei a geração do "
            "instalador final. Instale o Inno Setup "
            "(https://jrsoftware.org/isinfo.php) e rode de novo, ou compile "
            "installer/setup.iss manualmente pelo IDE do Inno Setup."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
