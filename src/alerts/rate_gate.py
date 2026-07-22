"""Trava anti-enxurrada POR CÂMERA — a segunda de duas.

O `ConcealmentAnalyzer` já tem cooldown POR PESSOA (não redispara a mesma
pessoa por N segundos). Mas numa loja movimentada o problema não é a mesma
pessoa: é que CADA pessoa nova abre um episódio e pode disparar. Com 6 câmeras
de teto onde qualquer um parado na frente da prateleira dispara (ver o achado
de campo de 21/jul), isso vira centenas de mensagens por hora.

Este gate limita POR CÂMERA: depois que uma câmera manda um alerta, ela fica
muda por `min_interval_seconds`, não importa quantas pessoas disparem nela
nesse meio-tempo. Não tenta distinguir furto de não-furto (inviável nessas
câmeras) — só impede que a equipe seja soterrada. É a ponte que mantém a loja
usável enquanto a detecção não fica afiada.
"""
from __future__ import annotations


class CameraAlertGate:
    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval = max(0.0, float(min_interval_seconds))
        self._last: dict[str, float] = {}
        self._suprimidos: dict[str, int] = {}

    def allow(self, camera_name: str, ts: float) -> bool:
        """True se um alerta desta câmera deve sair agora; False se ela ainda
        está no intervalo mudo do último alerta que passou.

        `min_interval_seconds == 0` desliga o gate (todo alerta passa)."""
        if self.min_interval <= 0.0:
            return True
        anterior = self._last.get(camera_name)
        if anterior is not None and ts - anterior < self.min_interval:
            self._suprimidos[camera_name] = self._suprimidos.get(camera_name, 0) + 1
            return False
        self._last[camera_name] = ts
        return True

    def suprimidos(self, camera_name: str) -> int:
        """Quantos alertas o gate engoliu nesta câmera (para log/relatório)."""
        return self._suprimidos.get(camera_name, 0)
