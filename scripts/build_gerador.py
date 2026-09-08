"""Compila o GERADOR DE LICENÇA — a ferramenta do revendedor.

Deliberadamente separado de `build_installer.py`: este executável **nunca**
pode entrar no instalador que vai para a loja. Quem tiver o gerador junto com
a chave privada emite licenças à vontade, e a trava inteira perde o sentido.

    .venv\\Scripts\\python.exe scripts/build_gerador.py

Sai em `dist_gerador/GerarLicenca.exe`. O revendedor deixa o `.exe` e o
`chave_privada.pem` na mesma pasta e emite licença com dois cliques — sem
Python instalado, sem terminal, sem digitar comando.

A chave privada NÃO é embutida no executável de propósito: embutir colocaria
o segredo dentro de um arquivo que pode ser copiado junto por engano."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
NOME = "GerarLicenca"


def montar_comando(raiz: Path) -> list[str]:
    return [
        sys.executable, "-m", "PyInstaller",
        str(raiz / "scripts" / "gerar_licenca.py"),
        "--name", NOME,
        "--onefile",       # arquivo único: fácil de mandar e de guardar
        "--console",       # é um programa de perguntas e respostas
        "--noconfirm",
        "--distpath", str(raiz / "dist_gerador"),
        # `gerar_licenca.py` importa `src.licensing` depois de mexer no
        # sys.path em runtime; a análise estática do PyInstaller não vê isso.
        "--paths", str(raiz),
        "--hidden-import", "src.licensing.license",
        "--hidden-import", "cryptography",
    ]


def main() -> int:
    resultado = subprocess.run(montar_comando(RAIZ), cwd=RAIZ)
    if resultado.returncode != 0:
        print("Falha ao gerar o programa de licença.")
        return resultado.returncode
    print(f"\nPronto: dist_gerador/{NOME}.exe")
    print("Coloque o 'chave_privada.pem' na mesma pasta do executável.")
    print("NÃO mande este programa para nenhum cliente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
