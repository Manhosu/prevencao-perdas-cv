"""O gerador de licenças — a ferramenta que fica só com o revendedor.

Quem usa isto não é programador. Se emitir uma licença exigisse abrir
terminal e montar comando com parâmetros, na prática significaria o
revendedor ligar para o desenvolvedor a cada instalação. Por isso o programa
sem argumentos entra em modo guiado, e a chave é procurada ao lado dele."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import gerar_licenca  # noqa: E402
from src.licensing import fingerprint, keyfile  # noqa: E402
from src.licensing.license import LicencaInvalida, verificar  # noqa: E402


@pytest.fixture
def chaves(tmp_path, monkeypatch):
    """Par real gravado num tmp, com o gerador apontando pra lá."""
    privada, publica = gerar_licenca.gerar_par_de_chaves()
    destino = tmp_path / "chave_privada.pem"
    destino.write_bytes(privada)
    monkeypatch.setattr(gerar_licenca, "caminho_padrao_chave", lambda: destino)
    return destino, publica


def _respostas(monkeypatch, *valores):
    """Simula o revendedor digitando; o último Enter fecha o programa."""
    fila = list(valores) + [""]
    monkeypatch.setattr("builtins.input", lambda *_: fila.pop(0))


@pytest.fixture
def chave_cifrada(tmp_path, monkeypatch):
    """A forma REAL de entrega: só o .enc ao lado do programa, aberto por senha.
    Não há .pem em claro no disco do revendedor."""
    privada, publica = gerar_licenca.gerar_par_de_chaves()
    enc = tmp_path / "chave_privada.enc"
    enc.write_bytes(keyfile.cifrar(privada, "PP-ZRJE-TWW8-8T92"))
    # caminho_padrao_chave aponta para o .pem (que não existe); o gerador cai
    # no .enc ao lado, como faz empacotado.
    monkeypatch.setattr(gerar_licenca, "caminho_padrao_chave",
                        lambda: tmp_path / "chave_privada.pem")
    return enc, publica


def _com_senha(monkeypatch, senha, *respostas):
    """A senha é lida por input() junto com as demais respostas. Entrega a
    senha na posição certa: código, cliente, validade, SENHA, Enter final."""
    fila = list(respostas) + [senha, ""]
    monkeypatch.setattr("builtins.input", lambda *_: fila.pop(0))


def test_modo_guiado_emite_licenca_valida(chaves, monkeypatch, capsys):
    _, publica = chaves
    maquina = fingerprint.codigo_da_maquina({"guid": "g", "volume": "V1"})
    _respostas(monkeypatch, maquina, "Mercado Julie", "")

    assert gerar_licenca.modo_interativo() == 0

    saida = capsys.readouterr().out
    token = next(l.strip() for l in saida.splitlines() if l.startswith("PP1."))
    licenca = verificar(token, publica)
    assert licenca.maquina == maquina
    assert licenca.cliente == "Mercado Julie"
    assert licenca.expira_em is None


def test_modo_guiado_aceita_validade(chaves, monkeypatch, capsys):
    _, publica = chaves
    maquina = fingerprint.codigo_da_maquina({"guid": "g", "volume": "V1"})
    _respostas(monkeypatch, maquina, "Loja", "2027-03-31")

    gerar_licenca.modo_interativo()

    saida = capsys.readouterr().out
    token = next(l.strip() for l in saida.splitlines() if l.startswith("PP1."))
    assert verificar(token, publica).expira_em == "2027-03-31"


def test_modo_guiado_grava_arquivo_ao_lado_da_chave(chaves, monkeypatch, capsys):
    destino_chave, _ = chaves
    maquina = fingerprint.codigo_da_maquina({"guid": "g", "volume": "V1"})
    _respostas(monkeypatch, maquina, "Mercado Julie", "")

    gerar_licenca.modo_interativo()

    gravados = list(destino_chave.parent.glob("licenca-*.txt"))
    assert len(gravados) == 1
    assert gravados[0].read_text(encoding="utf-8").startswith("PP1.")


def test_codigo_de_maquina_torto_e_recusado_com_explicacao(chaves, monkeypatch, capsys):
    """Erro de digitação não pode virar licença que só falha na loja."""
    _respostas(monkeypatch, "ABC-DEF", "Loja", "")

    assert gerar_licenca.modo_interativo() == 1
    assert "4 blocos" in capsys.readouterr().out


def test_sem_a_chave_privada_explica_o_que_fazer(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gerar_licenca, "caminho_padrao_chave",
                        lambda: tmp_path / "nao-existe.pem")
    _respostas(monkeypatch)

    assert gerar_licenca.modo_interativo() == 1
    assert "chave_privada.pem" in capsys.readouterr().out


def test_licenca_emitida_nao_vale_para_outra_maquina(chaves, monkeypatch, capsys):
    """A promessa comercial, conferida na saída real do gerador."""
    from src.licensing.activation import Estado, avaliar, gravar_token

    _, publica = chaves
    ids_cliente = {"guid": "pc-do-cliente", "volume": "V1"}
    _respostas(monkeypatch,
               fingerprint.codigo_da_maquina(ids_cliente), "Mercado Julie", "")
    gerar_licenca.modo_interativo()
    token = next(l.strip() for l in capsys.readouterr().out.splitlines()
                 if l.startswith("PP1."))

    import src.licensing.activation as act
    monkeypatch.setattr(act, "_chave_publica", lambda: publica)
    pasta = Path(chaves[0]).parent
    gravar_token(pasta, token)

    assert avaliar(pasta, ids_cliente).estado is Estado.OK
    outro = {"guid": "outro-pc", "volume": "V9"}
    assert avaliar(pasta, outro).estado is Estado.BLOQUEADA


def test_verificar_recusa_lixo(chaves):
    _, publica = chaves
    with pytest.raises(LicencaInvalida):
        verificar("nao e licenca", publica)


# --- chave cifrada por senha (a forma de entrega) ---------------------------

def test_gerador_emite_com_chave_cifrada_e_senha_certa(
        chave_cifrada, monkeypatch, capsys):
    _, publica = chave_cifrada
    maquina = fingerprint.codigo_da_maquina({"guid": "g", "volume": "V1"})
    # ordem das perguntas: código, cliente, validade(Enter), SENHA, Enter final
    _com_senha(monkeypatch, "PP-ZRJE-TWW8-8T92", maquina, "Mercado Julie", "")

    assert gerar_licenca.modo_interativo() == 0

    token = next(l.strip() for l in capsys.readouterr().out.splitlines()
                 if l.startswith("PP1."))
    assert verificar(token, publica).maquina == maquina


def test_gerador_com_senha_errada_nao_emite(chave_cifrada, monkeypatch, capsys):
    """O ponto que torna seguro pôr o .enc num link: senha errada não gera
    licença nenhuma. Três tentativas erradas, depois o Enter de sair."""
    maquina = fingerprint.codigo_da_maquina({"guid": "g", "volume": "V1"})
    fila = [maquina, "Mercado Julie", "", "errada1", "errada2", "errada3", ""]
    monkeypatch.setattr("builtins.input", lambda *_: fila.pop(0))

    assert gerar_licenca.modo_interativo() == 1

    saida = capsys.readouterr().out
    assert not any(l.strip().startswith("PP1.") for l in saida.splitlines())
    assert "Senha incorreta" in saida


def test_proteger_gera_enc_que_so_a_senha_abre(tmp_path):
    privada, _ = gerar_licenca.gerar_par_de_chaves()
    pem = tmp_path / "chave_privada.pem"
    pem.write_bytes(privada)

    class Args:
        privada = str(pem)
        senha = "PP-ABCD-EFGH-JKLM"
        saida = str(tmp_path / "chave_privada.enc")

    assert gerar_licenca.cmd_proteger(Args()) == 0
    dados = Path(Args.saida).read_bytes()
    assert keyfile.parece_cifrada(dados)
    assert keyfile.decifrar(dados, "PP-ABCD-EFGH-JKLM") == privada
    assert b"PRIVATE KEY" not in dados  # a chave em claro não vaza no arquivo
