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

from src.licensing import keyfile  # noqa: E402
from src.licensing.license import Licenca, emitir, gerar_par_de_chaves  # noqa: E402

MODULO_PUBLICO = RAIZ / "src" / "licensing" / "chave_publica.py"


def caminho_padrao_chave() -> Path:
    """Onde procurar a chave quando ninguém passou `--privada`.

    Empacotado, é a pasta do próprio `.exe`. Em desenvolvimento, a raiz do
    projeto."""
    base = Path(sys.executable).parent if getattr(sys, "frozen", False) else RAIZ
    return base / "chave_privada.pem"


def _chave_privada_bytes(caminho: Path | None, pedir_senha=None) -> bytes:
    """Devolve os bytes do PEM, decifrando com senha se o arquivo for `.enc`.

    Aceita as duas formas:
      * `chave_privada.pem` — chave em claro (uso no desenvolvimento);
      * `chave_privada.enc` — chave cifrada por senha (entrega ao revendedor,
        pode viajar em link porque sem a senha é inútil).

    Se `caminho` foi dado, usa exatamente esse arquivo. Senão procura o `.pem`
    e, na falta dele, o `.enc` ao lado do programa.

    `pedir_senha` é injetável para teste; por padrão lê do console. Usamos
    input() (senha visível), não getpass: no console do executável empacotado
    o getpass fica travado à espera de um terminal que ele não controla, e o
    programa congelaria na cara do revendedor. A senha aparecer na tela é
    aceitável — é a máquina dele, emitindo licença sozinho."""
    if pedir_senha is None:
        pedir_senha = lambda: input("Senha da chave: ").strip()  # noqa: E731

    if caminho is not None:
        alvo = Path(caminho)
    else:
        alvo = caminho_padrao_chave()
        if not alvo.exists():
            enc = alvo.with_suffix(".enc")
            if enc.exists():
                alvo = enc
    if not alvo.exists():
        raise FileNotFoundError(alvo)

    dados = alvo.read_bytes()
    if not keyfile.parece_cifrada(dados):
        return dados
    for tentativa in range(3):
        senha = pedir_senha()
        try:
            return keyfile.decifrar(dados, senha)
        except keyfile.SenhaIncorreta as e:
            faltam = 2 - tentativa
            print(f"  {e}" + (f" Tente de novo ({faltam} restante(s))." if faltam else ""))
    raise keyfile.SenhaIncorreta("Senha incorreta nas três tentativas.")

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
    maquina = args.maquina.strip().upper()
    if len(maquina.split("-")) != 4:
        print("ERRO: o código da máquina tem o formato XXXX-XXXX-YYYY-YYYY "
              "(copie exatamente o que aparece na tela do cliente).")
        return 1
    try:
        pem = _chave_privada_bytes(args.privada)
    except FileNotFoundError as e:
        print(f"ERRO: chave privada não encontrada em {e}")
        return 1
    except keyfile.SenhaIncorreta as e:
        print(f"ERRO: {e}")
        return 1

    licenca = Licenca(
        maquina=maquina,
        cliente=args.cliente.strip(),
        emitida_em=date.today().isoformat(),
        expira_em=args.validade,
    )
    token = emitir(licenca, pem)
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

    pasta = caminho_padrao_chave().parent
    if not (caminho_padrao_chave().exists()
            or caminho_padrao_chave().with_suffix(".enc").exists()):
        print("ERRO: não encontrei a chave nesta pasta:\n"
              f"  {pasta}\n")
        print("Coloque o arquivo da chave (chave_privada.pem ou chave_privada.enc)")
        print("na mesma pasta deste programa e abra de novo.")
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

    # Chave cifrada pede a senha aqui dentro (uma vez por emissão): a chave
    # decifrada nunca é gravada em disco, só usada em memória.
    try:
        pem = _chave_privada_bytes(None)
    except keyfile.SenhaIncorreta as e:
        print(f"\n{e}")
        input("\nPressione Enter para sair.")
        return 1
    except FileNotFoundError:
        print("\nERRO: não encontrei o arquivo da chave nesta pasta.")
        input("\nPressione Enter para sair.")
        return 1

    licenca = Licenca(maquina=maquina, cliente=cliente,
                      emitida_em=date.today().isoformat(), expira_em=validade)
    try:
        token = emitir(licenca, pem)
    except Exception as e:
        print(f"\nERRO ao gerar a licença: {e}")
        input("\nPressione Enter para sair.")
        return 1

    seguro = "".join(c if c.isalnum() else "-" for c in cliente)[:40]
    destino = pasta / f"licenca-{seguro}.txt"
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


def cmd_proteger(args) -> int:
    """Cifra a chave_privada.pem com uma senha, gerando o .enc que é entregue
    ao revendedor. O .enc pode ir em link: sem a senha, não abre."""
    origem = Path(args.privada)
    if not origem.exists():
        print(f"ERRO: não encontrei {origem}")
        return 1
    destino = Path(args.saida) if args.saida else origem.with_suffix(".enc")
    destino.write_bytes(keyfile.cifrar(origem.read_bytes(), args.senha))
    # confere o ciclo antes de confiar no arquivo entregue
    assert keyfile.decifrar(destino.read_bytes(), args.senha) == origem.read_bytes()
    print(f"Chave cifrada gravada em: {destino}")
    print("Entregue este .enc junto com o gerador; mande a SENHA por um canal "
          "separado (texto do chat).")
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

    p_prot = sub.add_parser(
        "proteger", help="cifra a chave privada com uma senha (gera o .enc de entrega)")
    p_prot.add_argument("--privada", required=True, help="chave_privada.pem a cifrar")
    p_prot.add_argument("--senha", required=True, help="senha que abrirá a chave")
    p_prot.add_argument("--saida", default=None,
                        help="arquivo .enc de saída (padrão: ao lado do .pem)")
    p_prot.set_defaults(func=cmd_proteger)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
