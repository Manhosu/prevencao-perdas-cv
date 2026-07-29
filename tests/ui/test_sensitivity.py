"""Mapeamento do controle de sensibilidade (slider) <-> limiar de detecção.

Pedido de campo: o revendedor pediu "como faz a sensibilidade" e a UI não tinha
controle nenhum — só editando o config.json na mão (com risco de BOM do
Notepad). Este é o miolo PURO do controle, testável sem Qt: converte a posição
do slider (0 = conservador/menos alertas, 100 = sensível/mais alertas) no
`detection.threshold` e vice-versa.
"""
from src.ui.sensitivity import rotulo, slider_para_threshold, threshold_para_slider


def test_slider_no_minimo_e_o_mais_conservador():
    """0 = limiar alto = dispara só no gesto forte = menos alertas."""
    assert slider_para_threshold(0) == 0.90


def test_slider_no_maximo_e_o_mais_sensivel():
    """100 = limiar baixo = pega gesto sutil = mais alertas."""
    assert slider_para_threshold(100) == 0.55


def test_meio_do_slider_fica_no_meio_do_limiar():
    assert 0.72 <= slider_para_threshold(50) <= 0.73  # ~0.725, meio da faixa


def test_ida_e_volta_aproxima():
    """threshold_para_slider é o inverso — a posição inicial do slider vem do
    threshold salvo. Como o threshold é arredondado a 2 casas, aceitamos ±1 de
    slider (imperceptível: 1/100 do curso)."""
    for s in (0, 25, 50, 75, 100):
        t = slider_para_threshold(s)
        assert abs(threshold_para_slider(t) - s) <= 1


def test_threshold_fora_da_faixa_nao_quebra():
    """Config antigo pode ter threshold 0.60 ou até fora de [0.55,0.90] — o
    slider tem que clampar sem estourar."""
    assert threshold_para_slider(0.60) == threshold_para_slider(0.60)  # não levanta
    assert 0 <= threshold_para_slider(0.30) <= 100   # abaixo da faixa -> clamp
    assert 0 <= threshold_para_slider(0.99) <= 100   # acima da faixa -> clamp


def test_rotulo_descreve_o_ponto():
    assert "onservador" in rotulo(0.88)
    assert "ensível" in rotulo(0.58)
    assert rotulo(0.72)  # meio tem algum rótulo, não vazio
