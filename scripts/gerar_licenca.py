"""Ferramenta do REVENDEDOR: cria o par de chaves e emite licenças.

Este arquivo é a única peça do projeto que usa a chave privada. Ele roda na
máquina de quem vende, nunca na loja.

    # uma única vez, na primeira configuração:
    python scripts/gerar_licenca.py chaves --privada C:/MinhasChaves/chave_privada.pem

    # a cada instalação nova (o código vem da tela do instalador):
    python scripts/gerar_licenca.py emitir \\
        --maquina ABCD-EFGH-IJKL-MNPQ \\
        --cliente "Mercado Julie" \\
        --privada C:/MinhasChaves/chave_privada.pem

`chaves` grava a chave PÚBLICA dentro do projeto (`src/licensing/chave_publica.py`,
que vai no instalador) e a PRIVADA no caminho que você escolher — fora do
repositório, sempre.

Perder a chave privada significa não conseguir emitir mais nenhuma licença
para as instalações que já existem. Guarde uma cópia em lugar seguro."""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
# Quem usa isto é o revendedor, rodando `python scripts/gerar_licenca.py` na
# raiz do projeto. Sem esta linha o import de `src` só funcionaria com
# `python -m scripts.gerar_licenca`, o que é detalhe de programador.
sys.path.insert(0, str(RAIZ))

from src.licensing.license import Licenca, emitir, gerar_par_de_chaves  # noqa: E402

MODULO_PUBLICO = RAIZ / "src" / "licensing" / "chave_publica.py"

_CABECALHO = '''"""Chave pública de licenciamento — gerada por scripts/gerar_licenca.py.

Pode ser lida por qualquer pessoa: ela só CONFERE assinatura, não cria. Quem
emite licença é a chave privada, que fica com o revendedor e nunca entra no
repositório. Trocar esta chave invalida todas as licenças já emitidas."""

CHAVE_PUBLICA_PEM = """\\
'''


def _escrever_modulo_publico(pem: bytes) -> None:
    conteudo = _CABECALHO + pem.decode("ascii").strip() + '\n"""\n'
    MODULO_PUBLICO.write_text(conteudo, encoding="utf-8")


def cmd_chaves(args) -> int:
    destino = Path(args.privada)
    if destino.exists() and not args.forcar:
        print(f"ERRO: {destino} já existe. Use --forcar apenas se souber que "
              "quer invalidar todas as licenças já emitidas.")
        return 1
    privada, publica = gerar_par_de_chaves()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(privada)
    _escrever_modulo_publico(publica)
    print(f"Chave privada gravada em: {destino}")
    print(f"Chave pública gravada em: {MODULO_PUBLICO}")
    print()
    print("GUARDE UMA CÓPIA DA CHAVE PRIVADA em lugar seguro (pendrive, cofre de")
    print("senhas). Sem ela você não emite mais licenças para as instalações que")
    print("já existem. E nunca coloque esse arquivo no repositório.")
    return 0


def cmd_emitir(args) -> int:
    privada = Path(args.privada) if args.privada else caminho_padrao_chave()
    if not privada.exists():
        print(f"ERRO: chave privada não encontrada em {privada}")
        return 1
    maquina = args.maquina.strip().upper()
    if len(maquina.split("-")) != 4:
        print("ERRO: o código da máquina tem o formato XXXX-XXXX-YYYY-YYYY "
              "(copie exatamente o que aparece na tela do cliente).")
        return 1

    licenca = Licenca(
        maquina=maquina,
        cliente=args.cliente.strip(),
        emitida_em=date.today().isoformat(),
        expira_em=args.validade,
    )
    token = emitir(licenca, privada.read_bytes())
    print()
    print(f"Licença de: {licenca.cliente}")
    print(f"Máquina:    {licenca.maquina}")
    print(f"Validade:   {licenca.expira_em or 'sem prazo'}")
    print()
    print("Mande o código abaixo para quem está instalando:")
    print()
    print(token)
    print()
    if args.saida:
        Path(args.saida).write_text(token, encoding="utf-8")
        print(f"(também gravado em {args.saida})")
    return 0


def caminho_padrao_chave() -> Path:
    """Onde procurar a chave privada quando ninguém passou `--privada`.

    Empacotado, é a pasta do próprio `.exe`: o revendedor deixa a chave ao
    lado do programa e não digita caminho nenhum. Em desenvolvimento, a raiz
    do projeto."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "chave_privada.pem"
    return RAIZ / "chave_privada.pem"


def _perguntar(rotulo: str, obrigatorio: bool = True) -> str:
    while True:
        valor = input(rotulo).strip()
        if valor or not obrigatorio:
            return valor
        print("  (precisa preencher)")


def modo_interativo() -> int:
    """O que roda quando o revendedor dá dois cliques no programa.

    Existe porque a alternativa era ele abrir terminal e digitar comando com
    parâmetros — o que, na prática, significaria me ligar a cada instalação."""
    print("=" * 58)
    print("  GERADOR DE LICENÇA — Prevenção de Perdas")
    print("=" * 58)
    print()

    chave = caminho_padrao_chave()
    if not chave.exists():
        print(f"ERRO: não encontrei a chave privada em:\n  {chave}\n")
        print("Coloque o arquivo 'chave_privada.pem' na mesma pasta deste")
        print("programa e abra de novo.")
        input("\nPressione Enter para sair.")
        return 1

    print("Peça ao cliente o código que aparece na tela dele.\n")
    maquina = _perguntar("Código da máquina (XXXX-XXXX-XXXX-XXXX): ").upper()
    if len(maquina.split("-")) != 4:
        print("\nERRO: o código tem 4 blocos separados por hífen.")
        input("\nPressione Enter para sair.")
        return 1
    cliente = _perguntar("Nome da loja/cliente: ")
    validade = _perguntar(
        "Validade AAAA-MM-DD (Enter = sem prazo): ", obrigatorio=False) or None

    licenca = Licenca(maquina=maquina, cliente=cliente,
                      emitida_em=date.today().isoformat(), expira_em=validade)
    try:
        token = emitir(licenca, chave.read_bytes())
    except Exception as e:
        print(f"\nERRO ao gerar a licença: {e}")
        input("\nPressione Enter para sair.")
        return 1

    seguro = "".join(c if c.isalnum() else "-" for c in cliente)[:40]
    destino = chave.parent / f"licenca-{seguro}.txt"
    try:
        destino.write_text(token, encoding="utf-8")
        gravado = f"\nTambém salvo em: {destino}"
    except OSError:
        gravado = ""

    print("\n" + "=" * 58)
    print(f"  Licença de: {cliente}")
    print(f"  Máquina:    {maquina}")
    print(f"  Validade:   {validade or 'sem prazo'}")
    print("=" * 58)
    print("\nMANDE O CÓDIGO ABAIXO PARA QUEM ESTÁ INSTALANDO:\n")
    print(token)
    print(gravado)
    input("\nPressione Enter para sair.")
    return 0


def main(argv=None) -> int:
    # Sem argumentos = dois cliques no programa. Vai para o modo guiado.
    if not (argv if argv is not None else sys.argv[1:]):
        return modo_interativo()

    ap = argparse.ArgumentParser(description="Licenciamento — Prevenção de Perdas")
    sub = ap.add_subparsers(dest="comando", required=True)

    p_chaves = sub.add_parser("chaves", help="cria o par de chaves (uma vez só)")
    p_chaves.add_argument("--privada", required=True,
                          help="onde gravar a chave privada (FORA do repositório)")
    p_chaves.add_argument("--forcar", action="store_true",
                          help="sobrescreve a chave existente (invalida licenças)")
    p_chaves.set_defaults(func=cmd_chaves)

    p_emitir = sub.add_parser("emitir", help="emite a licença de uma máquina")
    p_emitir.add_argument("--maquina", required=True,
                          help="código que aparece na tela do cliente")
    p_emitir.add_argument("--cliente", required=True, help="nome da loja/cliente")
    p_emitir.add_argument("--validade", default=None,
                          help="vencimento AAAA-MM-DD (padrão: sem prazo)")
    p_emitir.add_argument("--privada", default=None,
                          help="caminho da chave privada (padrão: ao lado do programa)")
    p_emitir.add_argument("--saida", default=None, help="grava o código num arquivo")
    p_emitir.set_defaults(func=cmd_emitir)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
