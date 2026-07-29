"""Controle de sensibilidade — mapeamento puro slider <-> limiar de detecção.

Sem Qt de propósito (testável sem tela). O revendedor não deve pensar em
"threshold 0.85"; ele mexe num slider de "conservador → sensível". Este módulo
traduz.

  slider 0   = mais CONSERVADOR = limiar alto (0.90) = só o gesto forte = menos alertas
  slider 100 = mais SENSÍVEL    = limiar baixo (0.55) = pega o sutil     = mais alertas
"""
from __future__ import annotations

THRESH_CONSERVADOR = 0.90   # slider 0
THRESH_SENSIVEL = 0.55      # slider 100
_FAIXA = THRESH_CONSERVADOR - THRESH_SENSIVEL


def slider_para_threshold(slider: int) -> float:
    """Posição do slider (0–100) -> `detection.threshold` (2 casas)."""
    s = max(0, min(100, int(slider)))
    return round(THRESH_CONSERVADOR - (s / 100.0) * _FAIXA, 2)


def threshold_para_slider(threshold: float) -> int:
    """Limiar salvo -> posição inicial do slider. Clampa limiares fora da faixa
    (config antigo pode ter 0.60, ou valores fora de [0.55, 0.90])."""
    bruto = (THRESH_CONSERVADOR - float(threshold)) / _FAIXA * 100.0
    return int(round(max(0.0, min(100.0, bruto))))


def rotulo(threshold: float) -> str:
    """Rótulo amigável do ponto atual, para o revendedor entender o efeito."""
    if threshold >= 0.82:
        return "Conservador — menos alertas, só o gesto óbvio"
    if threshold >= 0.68:
        return "Equilibrado"
    return "Sensível — mais alertas, pega gestos sutis"
