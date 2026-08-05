"""WeaponDetector: gated no pânico. Testa a lógica (limiar, rótulos, robustez a
erro) com um modelo dublê — sem carregar o modelo real."""
import numpy as np

from src.detection.weapon import WeaponDetector


class _Box:
    # 0-dim (como o tensor por-caixa do ultralytics): int()/float() convertem.
    def __init__(self, cls, conf):
        self.cls = np.array(cls); self.conf = np.array(conf)


class _Result:
    def __init__(self, boxes): self.boxes = boxes


class _FakeModel:
    """Dublê do YOLO: nomes fixos e devolve caixas roteirizadas."""
    def __init__(self, names, boxes):
        self.names = names; self._boxes = boxes
    def __call__(self, image, verbose=False):
        return [_Result(self._boxes)]


def _det(names, boxes, threshold=0.40):
    d = WeaponDetector("nao-usado.pt", threshold)
    d._model = _FakeModel(names, boxes)   # injeta, pula o carregamento
    return d


IMG = np.zeros((64, 64, 3), np.uint8)


def test_arma_acima_do_limiar_dispara():
    d = _det({0: "Gun"}, [_Box(0, 0.55)])
    assert d.has_weapon(IMG) is True


def test_arma_abaixo_do_limiar_nao_dispara():
    d = _det({0: "Gun"}, [_Box(0, 0.20)], threshold=0.40)
    assert d.has_weapon(IMG) is False


def test_classe_que_nao_e_arma_nao_dispara():
    d = _det({0: "person", 1: "cell phone"}, [_Box(0, 0.99), _Box(1, 0.99)])
    assert d.has_weapon(IMG) is False


def test_sem_caixas_nao_dispara():
    assert _det({0: "pistol"}, []).has_weapon(IMG) is False


def test_modelo_ausente_devolve_false_sem_quebrar():
    """Modelo que não carrega (arquivo inexistente) → False, nunca exceção:
    a ausência da leitura de arma não pode derrubar o alerta de pânico."""
    d = WeaponDetector("caminho/inexistente/weapon.pt", 0.4)
    assert d.has_weapon(IMG) is False
