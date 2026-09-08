"""O código da máquina é o que amarra a licença a UM computador.

Duas propriedades sustentam a promessa comercial "um código, uma instalação":
o mesmo PC sempre gera o mesmo código, e PCs diferentes geram códigos
diferentes. Sem a primeira, cliente legítimo é bloqueado ao reabrir o
programa; sem a segunda, uma licença vira chave-mestra."""
from src.licensing import fingerprint


def test_codigo_tem_o_formato_de_quatro_blocos():
    codigo = fingerprint.codigo_da_maquina({"guid": "g-1", "volume": "AB12CD34"})
    blocos = codigo.split("-")
    assert len(blocos) == 4
    assert all(len(b) == 4 for b in blocos)


def test_mesma_maquina_gera_sempre_o_mesmo_codigo():
    ids = {"guid": "g-1", "volume": "AB12CD34"}
    assert fingerprint.codigo_da_maquina(ids) == fingerprint.codigo_da_maquina(ids)


def test_maquinas_diferentes_geram_codigos_diferentes():
    a = fingerprint.codigo_da_maquina({"guid": "g-1", "volume": "AB12CD34"})
    b = fingerprint.codigo_da_maquina({"guid": "g-2", "volume": "FF99EE88"})
    assert a != b


def test_trocar_o_disco_muda_so_a_segunda_metade():
    """Formatou/trocou o HD: o MachineGuid continua, o serial do volume muda.
    A metade preservada é o que permite o modo de tolerância em vez de um
    bloqueio seco na cara de um cliente pagante."""
    antes = fingerprint.codigo_da_maquina({"guid": "g-1", "volume": "AB12CD34"})
    depois = fingerprint.codigo_da_maquina({"guid": "g-1", "volume": "ZZ00YY11"})
    assert antes.split("-")[:2] == depois.split("-")[:2]
    assert antes.split("-")[2:] != depois.split("-")[2:]


def test_codigo_nao_expoe_o_identificador_cru():
    """O que circula em mensagem é o resumo, nunca o MachineGuid do cliente."""
    codigo = fingerprint.codigo_da_maquina(
        {"guid": "11112222-3333-4444-5555-666677778888", "volume": "AB12CD34"})
    assert "1111" not in codigo
    assert "AB12CD34" not in codigo


def test_alfabeto_sem_caracteres_ambiguos():
    """O instalador pode ter que ditar isso por telefone: 0/O e 1/I fora."""
    codigo = fingerprint.codigo_da_maquina({"guid": "g-1", "volume": "AB12CD34"})
    assert not (set(codigo) & set("O01I"))


def test_maquina_sem_nenhuma_fonte_nao_e_identificavel():
    """Sem identificador, dois PCs gerariam o mesmo código e uma licença
    serviria em qualquer um deles."""
    assert fingerprint.identificavel({"guid": None, "volume": None}) is False


def test_uma_fonte_basta_para_identificar():
    assert fingerprint.identificavel({"guid": None, "volume": "AB12CD34"}) is True
