"""Detecção de arma — SÓ é usada DENTRO do alerta de pânico (mãos acima da
cabeça). Nunca roda no dia a dia, então NÃO gera alarme falso com celular: o
gatilho é sempre o gesto de rendição; a arma é um extra que enriquece o alerta
quando o modelo vê uma arma óbvia no quadro.

O modelo é fraco de propósito no recall (pega ~2/3 de arma escancarada de
perto). Isso é aceitável aqui porque o gesto já garante o alerta — a leitura
de arma é marketing/contexto, não o detector principal. Por isso fica OFF por
padrão e o modelo carrega preguiçosamente (só quando ligado)."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Rótulos que os modelos de arma da comunidade usam (variam por dataset).
PALAVRAS_ARMA = {
    "weapon", "pistol", "gun", "handgun", "firearm", "rifle",
    "revolver", "senapan", "knife",
}


class WeaponDetector:
    """Carrega o modelo de arma sob demanda e diz se há arma no frame.

    Nunca levanta exceção para o chamador: qualquer falha (modelo ausente,
    erro de inferência) vira `False` — a ausência da leitura de arma não pode
    derrubar o alerta de pânico, que é o que de fato importa."""

    def __init__(self, model_path: str, threshold: float = 0.40) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self._model = None
        self._falhou = False

    def _ensure(self):
        if self._model is None and not self._falhou:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
            except Exception as e:
                self._falhou = True
                log.warning("não consegui carregar o modelo de arma (%s): %s",
                            self.model_path, e)
        return self._model

    def has_weapon(self, image) -> bool:
        """True se há uma arma acima do limiar no frame. Chamada só no pânico."""
        model = self._ensure()
        if model is None:
            return False
        try:
            r = model(image, verbose=False)[0]
            for b in r.boxes:
                nome = model.names[int(b.cls)].lower()
                if nome in PALAVRAS_ARMA and float(b.conf) >= self.threshold:
                    return True
            return False
        except Exception as e:
            log.warning("falha na detecção de arma: %s", e)
            return False
