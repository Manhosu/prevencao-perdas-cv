"""Assinatura da licença.

O ponto que estes testes protegem: o código-fonte do projeto pertence ao
cliente e está publicado. A validação NÃO pode depender de um segredo dentro
do código. Só quem tem a chave privada emite licença, e ler o repositório
inteiro não ajuda a forjar nenhuma."""
import pytest

from src.licensing.license import (
    Licenca,
    LicencaInvalida,
    emitir,
    gerar_par_de_chaves,
    verificar,
)


@pytest.fixture(scope="module")
def par():
    return gerar_par_de_chaves()


def _licenca(**kw):
    base = dict(maquina="ABCD-EFGH-IJKL-MNPQ", cliente="Mercado Julie",
                emitida_em="2026-09-07", expira_em=None)
    base.update(kw)
    return Licenca(**base)


def test_emitir_e_verificar_devolve_a_mesma_licenca(par):
    privada, publica = par
    token = emitir(_licenca(), privada)
    lida = verificar(token, publica)
    assert lida.maquina == "ABCD-EFGH-IJKL-MNPQ"
    assert lida.cliente == "Mercado Julie"
    assert lida.emitida_em == "2026-09-07"


def test_licenca_de_outra_chave_e_recusada(par):
    """Alguém que gere o próprio par (o código é público) não consegue emitir
    licença válida para as instalações que já existem."""
    _, publica = par
    outra_privada, _ = gerar_par_de_chaves()
    token = emitir(_licenca(), outra_privada)
    with pytest.raises(LicencaInvalida):
        verificar(token, publica)


def test_alterar_o_conteudo_invalida_a_assinatura(par):
    """Trocar a máquina no token (para reaproveitar numa segunda instalação)
    quebra a assinatura."""
    privada, publica = par
    token = emitir(_licenca(), privada)
    prefixo, payload, assinatura = token.split(".")
    adulterado = f"{prefixo}.{payload[:-4]}XXXX.{assinatura}"
    with pytest.raises(LicencaInvalida):
        verificar(adulterado, publica)


@pytest.mark.parametrize("ruim", ["", "   ", "lixo", "PP1.so-duas-partes",
                                  "XX9.aaa.bbb", "PP1.###.###"])
def test_token_malformado_da_erro_em_portugues(par, ruim):
    _, publica = par
    with pytest.raises(LicencaInvalida) as e:
        verificar(ruim, publica)
    assert str(e.value)  # mensagem para o instalador, não stacktrace
    assert "Traceback" not in str(e.value)


def test_licenca_sem_prazo_nunca_expira(par):
    from datetime import date
    privada, publica = par
    lida = verificar(emitir(_licenca(), privada), publica)
    assert lida.expirada(date(2099, 1, 1)) is False


def test_licenca_com_prazo_expira_no_dia_seguinte(par):
    """O campo de validade entra pronto mesmo sem ser usado agora: é o que
    permite vender por mensalidade depois sem trocar o sistema."""
    from datetime import date
    privada, publica = par
    lida = verificar(emitir(_licenca(expira_em="2026-12-31"), privada), publica)
    assert lida.expirada(date(2026, 12, 31)) is False
    assert lida.expirada(date(2027, 1, 1)) is True


def test_acentos_no_nome_do_cliente_sobrevivem(par):
    privada, publica = par
    token = emitir(_licenca(cliente="Armazém São João"), privada)
    assert verificar(token, publica).cliente == "Armazém São João"
