import pytest

from src.config.settings import DetectionConfig
from src.detection.signals import WristHistory, compute_signals

CFG = DetectionConfig()  # dwell 1.2s, window 3.0s, vanish_max 3.0, gap_frames 2
FPS = 5.0
DT = 1.0 / FPS


def _feed(hist, seq, start=0.0):
    """seq: lista de (x_n, y_n, conf, zone, reach). Um item por frame a 5fps."""
    t = start
    for (x_n, y_n, conf, zone, reach) in seq:
        hist.observe(x_n, y_n, conf, zone, reach, t)
        t += DT
    return t


def test_dwell_rises_with_time_in_zone():
    hist = WristHistory()
    # 6 frames (1.2s a 5fps) com o punho na zona 'waist'
    now = _feed(hist, [(0.4, -0.1, 0.8, "waist", False)] * 6)
    s = compute_signals(hist, CFG, now - DT)
    assert s.dwell == pytest.approx(1.0, abs=0.05)  # atingiu dwell_seconds
    assert s.zone == "waist"


def test_dwell_partial():
    hist = WristHistory()
    now = _feed(hist, [(0.4, -0.1, 0.8, "waist", False)] * 3)  # 0.6s de 1.2s
    s = compute_signals(hist, CFG, now - DT)
    assert 0.4 < s.dwell < 0.6


def test_dwell_tolerates_short_gap():
    hist = WristHistory()
    seq = [(0.4, -0.1, 0.8, "waist", False)] * 3
    seq += [(0.4, -0.1, 0.8, None, False)]        # 1 frame fora (gap<=2)
    seq += [(0.4, -0.1, 0.8, "waist", False)] * 3
    now = _feed(hist, seq)
    s = compute_signals(hist, CFG, now - DT)
    assert s.dwell > 0.9  # o gap curto não zerou


def test_dwell_resets_after_long_gap():
    hist = WristHistory()
    seq = [(0.4, -0.1, 0.8, "waist", False)] * 3
    seq += [(0.9, 0.5, 0.8, None, False)] * 4     # 4 frames fora (gap>2)
    seq += [(0.4, -0.1, 0.8, "waist", False)] * 2
    now = _feed(hist, seq)
    s = compute_signals(hist, CFG, now - DT)
    assert s.dwell < 0.5  # recomeçou a contagem


def test_approach_when_wrist_came_from_reach():
    hist = WristHistory()
    seq = [(1.1, 0.5, 0.8, None, True)] * 2        # veio da prateleira (reach)
    seq += [(0.4, -0.1, 0.8, "waist", False)] * 2  # entrou na zona
    now = _feed(hist, seq)
    s = compute_signals(hist, CFG, now - DT)
    assert s.approach > 0.5


def test_no_approach_when_hand_was_already_at_waist():
    hist = WristHistory()
    seq = [(0.4, -0.1, 0.8, "waist", False)] * 4   # sempre na cintura, nunca reach
    now = _feed(hist, seq)
    s = compute_signals(hist, CFG, now - DT)
    assert s.approach < 0.2


def test_vanish_when_wrist_disappears_inside_zone():
    hist = WristHistory()
    seq = [(0.4, -0.1, 0.8, "waist", False)] * 3   # visível na zona
    seq += [(0.4, -0.1, 0.10, "waist", False)]     # conf caiu < kp_conf_min DENTRO da zona
    now = _feed(hist, seq)
    s = compute_signals(hist, CFG, now - DT)
    assert s.vanish > 0.8


def test_no_vanish_when_wrist_disappears_far_from_body():
    hist = WristHistory()
    seq = [(1.2, 0.5, 0.8, None, True)] * 2        # longe do corpo
    seq += [(1.2, 0.5, 0.05, None, True)]          # sumiu longe — não é ocultação
    now = _feed(hist, seq)
    s = compute_signals(hist, CFG, now - DT)
    assert s.vanish < 0.2


def test_vanish_expires_after_max_seconds():
    hist = WristHistory()
    seq = [(0.4, -0.1, 0.8, "waist", False)] * 2
    # 20 frames (4s > vanish_max 3.0) com punho sumido
    seq += [(0.4, -0.1, 0.05, "waist", False)] * 20
    now = _feed(hist, seq)
    s = compute_signals(hist, CFG, now - DT)
    assert s.vanish < 0.2  # expirou


def test_retract_when_hand_returns_and_rises():
    hist = WristHistory()
    seq = [(0.4, -0.1, 0.8, "waist", False)] * 4   # ficou na zona (>0.5*dwell)
    seq += [(0.3, 0.6, 0.8, "torso", False)]       # reapareceu subindo (Δy_n>0.3)
    now = _feed(hist, seq)
    s = compute_signals(hist, CFG, now - DT)
    assert s.retract > 0.5


def test_prune_drops_old_observations():
    hist = WristHistory()
    _feed(hist, [(0.4, -0.1, 0.8, "waist", False)] * 30)  # 6s de dados
    hist.prune(now=6.0, window_seconds=3.0)
    # só as observações dos últimos 3s permanecem
    assert all(o.ts >= 3.0 for o in hist.observations)
