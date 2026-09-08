"""Decisão de rodar ou não neste computador.

Os dois extremos que estes testes fixam:

  * copiar a pasta instalada para outra máquina NÃO funciona (é a trava que
    foi vendida);
  * cliente que formatou o PC ou trocou o disco NÃO fica sem o sistema de
    segurança de um dia pro outro (é a promessa de não travar quem pagou)."""
from datetime import date

import pytest

from src.licensing import activation, fingerprint
from src.licensing.activation import Estado, ativar, avaliar, gravar_token
from src.licensing.license import Licenca, emitir, gerar_par_de_chaves

MAQUINA = {"guid": "guid-da-loja", "volume": "AB12CD34"}
OUTRA = {"guid": "guid-de-outro-pc", "volume": "FF99EE88"}
HOJE = date(2026, 9, 7)


@pytest.fixture(autouse=True)
def chave(monkeypatch):
    """Par próprio para os testes: nada aqui depende da chave real do
    revendedor, que nem está no repositório."""
    privada, publica = gerar_par_de_chaves()
    monkeypatch.setattr(activation, "_chave_publica", lambda: publica)
    return privada


def _token(chave, ids=MAQUINA, cliente="Mercado Julie", expira_em=None):
    return emitir(
        Licenca(maquina=fingerprint.codigo_da_maquina(ids), cliente=cliente,
                emitida_em="2026-09-01", expira_em=expira_em),
        chave,
    )


def test_sem_licenca_nao_roda_e_explica_o_que_fazer(tmp_path):
    r = avaliar(tmp_path, MAQUINA, HOJE)
    assert r.estado is Estado.SEM_LICENCA
    assert r.pode_rodar is False
    assert "código da máquina" in r.mensagem


def test_licenca_da_maquina_libera(tmp_path, chave):
    gravar_token(tmp_path, _token(chave))
    r = avaliar(tmp_path, MAQUINA, HOJE)
    assert r.estado is Estado.OK
    assert r.pode_rodar is True
    assert r.licenca.cliente == "Mercado Julie"


def test_pasta_copiada_para_outro_pc_nao_roda(tmp_path, chave):
    """O cenário que a trava existe para impedir: o dono da loja copia a
    instalação inteira, licença junto, para outra máquina."""
    gravar_token(tmp_path, _token(chave, ids=MAQUINA))
    r = avaliar(tmp_path, OUTRA, HOJE)
    assert r.estado is Estado.BLOQUEADA
    assert r.pode_rodar is False
    assert "outro computador" in r.mensagem


def test_licenca_vencida_bloqueia(tmp_path, chave):
    gravar_token(tmp_path, _token(chave, expira_em="2026-08-31"))
    r = avaliar(tmp_path, MAQUINA, HOJE)
    assert r.estado is Estado.BLOQUEADA
    assert "venceu" in r.mensagem


def test_troca_de_disco_entra_em_tolerancia_e_continua_vigiando(tmp_path, chave):
    """Metade do identificador mudou: é manutenção, não pirataria. O sistema
    segue rodando e avisa."""
    gravar_token(tmp_path, _token(chave, ids=MAQUINA))
    disco_novo = {"guid": MAQUINA["guid"], "volume": "NOVO0001"}

    r = avaliar(tmp_path, disco_novo, HOJE)

    assert r.estado is Estado.TOLERANCIA
    assert r.pode_rodar is True
    assert r.dias_restantes == activation.TOLERANCIA_DIAS


def test_tolerancia_conta_a_partir_do_primeiro_dia_e_acaba(tmp_path, chave):
    gravar_token(tmp_path, _token(chave, ids=MAQUINA))
    disco_novo = {"guid": MAQUINA["guid"], "volume": "NOVO0001"}

    avaliar(tmp_path, disco_novo, HOJE)                       # marca o início
    meio = avaliar(tmp_path, disco_novo, date(2026, 9, 12))   # 5 dias depois
    assert meio.estado is Estado.TOLERANCIA
    assert meio.dias_restantes == activation.TOLERANCIA_DIAS - 5

    fim = avaliar(tmp_path, disco_novo, date(2026, 10, 30))
    assert fim.estado is Estado.BLOQUEADA
    assert fim.pode_rodar is False


def test_licenca_nova_limpa_a_tolerancia(tmp_path, chave):
    """Depois de reemitir para a máquina nova, o relógio da tolerância zera —
    senão o cliente regularizado seria bloqueado pelo prazo antigo."""
    gravar_token(tmp_path, _token(chave, ids=MAQUINA))
    disco_novo = {"guid": MAQUINA["guid"], "volume": "NOVO0001"}
    avaliar(tmp_path, disco_novo, HOJE)  # entra em tolerância

    ativar(tmp_path, _token(chave, ids=disco_novo), disco_novo, HOJE)

    depois = avaliar(tmp_path, disco_novo, date(2026, 10, 30))
    assert depois.estado is Estado.OK


def test_ativar_grava_e_libera(tmp_path, chave):
    r = ativar(tmp_path, _token(chave), MAQUINA, HOJE)
    assert r.estado is Estado.OK
    assert avaliar(tmp_path, MAQUINA, HOJE).estado is Estado.OK


def test_ativar_com_licenca_de_outra_maquina_nao_apaga_a_boa(tmp_path, chave):
    """Colar um código errado não pode destruir a licença que já funcionava."""
    gravar_token(tmp_path, _token(chave, ids=MAQUINA))

    r = ativar(tmp_path, _token(chave, ids=OUTRA), MAQUINA, HOJE)

    assert r.estado is Estado.BLOQUEADA
    assert fingerprint.codigo_da_maquina(MAQUINA) in r.mensagem  # diz o código certo
    assert avaliar(tmp_path, MAQUINA, HOJE).estado is Estado.OK


def test_ativar_recusa_token_forjado(tmp_path):
    outra_privada, _ = gerar_par_de_chaves()
    r = ativar(tmp_path, _token(outra_privada), MAQUINA, HOJE)
    assert r.estado is Estado.BLOQUEADA
    assert activation.caminho_licenca(tmp_path).exists() is False


def test_maquina_sem_identificador_nao_roda(tmp_path, chave):
    """Sem nada que identifique o PC, uma licença serviria em qualquer um."""
    gravar_token(tmp_path, _token(chave))
    r = avaliar(tmp_path, {"guid": None, "volume": None}, HOJE)
    assert r.estado is Estado.BLOQUEADA


def test_licenca_com_bom_do_bloco_de_notas_ainda_vale(tmp_path, chave):
    """O instalador cola o código num arquivo pelo Bloco de Notas; o BOM não
    pode invalidar uma licença legítima."""
    token = _token(chave)
    activation.caminho_licenca(tmp_path).write_text(token, encoding="utf-8-sig")
    assert avaliar(tmp_path, MAQUINA, HOJE).estado is Estado.OK
