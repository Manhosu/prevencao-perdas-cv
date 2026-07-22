"""Limite anti-enxurrada POR CÂMERA.

Bug de campo (21/jul): a loja gerou 300+ mensagens numa tarde. O cooldown do
analyzer é POR PESSOA — numa loja movimentada, cada pessoa nova abre episódio e
pode disparar, então o volume explode. Este gate é a segunda trava, POR CÂMERA:
depois que uma câmera alerta, ela fica muda por um intervalo mínimo, não importa
quantas pessoas disparem nela. Reduz o volume de mensagens sem depender de
distinguir furto de não-furto (que, nas câmeras de teto da loja, não é viável).
"""
from src.alerts.rate_gate import CameraAlertGate


def test_primeiro_alerta_da_camera_passa():
    g = CameraAlertGate(min_interval_seconds=60.0)
    assert g.allow("Cam 13", ts=100.0) is True


def test_segundo_alerta_dentro_do_intervalo_e_barrado():
    g = CameraAlertGate(min_interval_seconds=60.0)
    g.allow("Cam 13", ts=100.0)
    assert g.allow("Cam 13", ts=130.0) is False


def test_alerta_depois_do_intervalo_passa():
    g = CameraAlertGate(min_interval_seconds=60.0)
    g.allow("Cam 13", ts=100.0)
    assert g.allow("Cam 13", ts=161.0) is True


def test_exatamente_no_intervalo_passa():
    """No limite (ts + intervalo) já libera — não fica preso por arredondamento."""
    g = CameraAlertGate(min_interval_seconds=60.0)
    g.allow("Cam 13", ts=100.0)
    assert g.allow("Cam 13", ts=160.0) is True


def test_cameras_sao_independentes():
    """Uma câmera muda não pode calar as outras."""
    g = CameraAlertGate(min_interval_seconds=60.0)
    g.allow("Cam 13", ts=100.0)
    assert g.allow("Cam 12", ts=101.0) is True


def test_intervalo_zero_desliga_o_gate():
    """min_interval=0 = recurso desligado: todo alerta passa."""
    g = CameraAlertGate(min_interval_seconds=0.0)
    assert g.allow("Cam 13", ts=100.0) is True
    assert g.allow("Cam 13", ts=100.0) is True


def test_liberar_reinicia_a_janela():
    """Depois que passa, a janela reconta a partir do novo alerta."""
    g = CameraAlertGate(min_interval_seconds=60.0)
    g.allow("Cam 13", ts=100.0)      # passa
    assert g.allow("Cam 13", ts=161.0) is True   # passa, nova âncora
    assert g.allow("Cam 13", ts=200.0) is False  # 39s depois do novo âncora: barra


def test_conta_suprimidos_por_camera():
    """Para o log/relatório: quantos alertas o gate engoliu em cada câmera."""
    g = CameraAlertGate(min_interval_seconds=60.0)
    g.allow("Cam 13", ts=100.0)
    g.allow("Cam 13", ts=110.0)  # suprimido
    g.allow("Cam 13", ts=120.0)  # suprimido
    assert g.suprimidos("Cam 13") == 2
    assert g.suprimidos("Cam 12") == 0
