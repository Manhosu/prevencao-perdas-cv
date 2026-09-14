"""A chave privada protegida por senha.

O que estes testes garantem: o arquivo cifrado pode ir num link aberto sem
comprometer nada, porque só a senha certa devolve a chave. Errou a senha,
não vaza nem meia chave — vem erro, não um PEM parcial."""
import pytest

from src.licensing import keyfile
from src.licensing.license import gerar_par_de_chaves


@pytest.fixture(scope="module")
def pem():
    privada, _ = gerar_par_de_chaves()
    return privada


def test_ciclo_completo_devolve_a_chave_original(pem):
    cifrada = keyfile.cifrar(pem, "PP-ABCD-EFGH-JKLM")
    assert keyfile.decifrar(cifrada, "PP-ABCD-EFGH-JKLM") == pem


def test_senha_errada_nao_devolve_a_chave(pem):
    cifrada = keyfile.cifrar(pem, "senha-certa")
    with pytest.raises(keyfile.SenhaIncorreta):
        keyfile.decifrar(cifrada, "senha-errada")


def test_arquivo_cifrado_nao_contem_a_chave_em_claro(pem):
    """O ponto de pôr num link: o PEM não aparece dentro do arquivo cifrado."""
    cifrada = keyfile.cifrar(pem, "x")
    assert b"PRIVATE KEY" not in cifrada
    assert pem[30:60] not in cifrada


def test_um_byte_trocado_e_detectado(pem):
    cifrada = bytearray(keyfile.cifrar(pem, "x"))
    cifrada[-1] ^= 0x01
    with pytest.raises(keyfile.SenhaIncorreta):
        keyfile.decifrar(bytes(cifrada), "x")


def test_arquivo_que_nao_e_chave_da_erro_claro(pem):
    with pytest.raises(keyfile.SenhaIncorreta):
        keyfile.decifrar(b"qualquer coisa", "x")


def test_cada_cifragem_usa_salt_e_nonce_novos(pem):
    """Duas cifragens da mesma chave com a mesma senha não podem sair iguais,
    senão vazaria que é a mesma chave."""
    assert keyfile.cifrar(pem, "x") != keyfile.cifrar(pem, "x")


def test_parece_cifrada_distingue_pem_de_enc(pem):
    assert keyfile.parece_cifrada(keyfile.cifrar(pem, "x")) is True
    assert keyfile.parece_cifrada(pem) is False
