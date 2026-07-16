import numpy as np
import pytest

from src.config.settings import DetectionConfig
from src.core.types import BBox, KP, ObjectDetection, PersonDetection, PersonPose
from src.detection.concealment import ConcealmentAnalyzer, ConcealmentEvent

FPS = 5.0
DT = 1.0 / FPS


def _pose(track_id, wrist_xy, *, upright=True, wrist_conf=0.9, bbox=None):
    """Pessoa em pé com o punho DIREITO na posição dada (px do frame)."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[KP["left_shoulder"]] = [90, 100, 0.9]
    kp[KP["right_shoulder"]] = [110, 100, 0.9]
    kp[KP["left_hip"]] = [92, 200, 0.9]
    kp[KP["right_hip"]] = [108, 200, 0.9]
    kp[KP["nose"]] = [100, 80, 0.9]
    kp[KP["left_eye"]] = [96, 78, 0.9]
    kp[KP["right_eye"]] = [104, 78, 0.9]
    kp[KP["right_wrist"]] = [wrist_xy[0], wrist_xy[1], wrist_conf]
    b = bbox or BBox(80, 60, 120, 300)
    return PersonPose(person=PersonDetection(bbox=b, conf=0.9, track_id=track_id), keypoints=kp)


def _run(analyzer, frames):
    """frames: lista de (wrist_xy, wrist_conf). Um por frame a 5fps.
    Devolve todos os eventos emitidos."""
    events = []
    t = 0.0
    for (wrist_xy, conf) in frames:
        ev = analyzer.update([_pose(1, wrist_xy, wrist_conf=conf)], [], t)
        events.extend(ev)
        t += DT
    return events


def test_conceal_gesture_fires_event():
    """Mão vem da prateleira, desce ao bolso e fica lá > dwell → dispara.

    Posições calculadas no referencial do corpo desta pose (ombros em
    y=100, quadris em y=200, escala=100px): (200, 90) cai fora das zonas
    de cintura/tórax e satisfaz in_reach (braço esticado para cima, pegando
    algo na prateleira). (130, 205) cai dentro da zona 'waist' (bolso).
    O punho fica visível por 2 frames entrando no bolso e depois SOME
    (conf cai) — fisicamente correto: a mão dentro do bolso deixa de ser
    vista pelo modelo de pose, e é o sinal `vanish` que sustenta o score
    acima do limiar (ver INSIGHT da revisão da Task 2: o score tem pico
    transitório sustentado pelo vanish, não por dwell+approach sozinhos)."""
    a = ConcealmentAnalyzer(DetectionConfig(), fps_hint=FPS)
    frames = [((200, 90), 0.9)] * 3            # reach (braço estendido para a prateleira)
    frames += [((130, 205), 0.9)] * 2          # punho visível entrando no bolso (cintura)
    frames += [((130, 205), 0.05)] * 8         # punho some dentro do bolso, permanece
    events = _run(a, frames)
    assert len(events) >= 1
    e = events[0]
    assert e.track_id == 1
    assert e.zone in ("waist", "torso")
    assert e.score >= DetectionConfig().threshold
    assert set(e.signals) >= {"dwell", "approach", "vanish", "retract"}


def test_scratching_belly_does_not_fire():
    """Mão encosta rápido no tórax e sai — dwell insuficiente, não dispara."""
    a = ConcealmentAnalyzer(DetectionConfig(), fps_hint=FPS)
    frames = [((108, 150), 0.9)]              # 1 frame no tórax (0.2s << dwell 1.2s)
    frames += [((160, 120), 0.9)] * 10        # volta pra longe
    events = _run(a, frames)
    assert events == []


def test_hand_into_clothes_fires_via_vanish():
    """Mão vai ao tórax e o punho SOME (mão sob a blusa) → vanish sustenta o score.

    (200, 90) é reach (fora de zona, braço esticado). (108, 150) cai dentro
    da zona 'torso' neste referencial de corpo."""
    a = ConcealmentAnalyzer(DetectionConfig(), fps_hint=FPS)
    frames = [((200, 90), 0.9)] * 2            # reach
    frames += [((108, 150), 0.9)] * 3          # mão no tórax, visível
    frames += [((108, 150), 0.05)] * 6         # punho some DENTRO da zona (conf ~0)
    events = _run(a, frames)
    assert len(events) >= 1
    assert events[0].signals["vanish"] > 0.5


def test_cooldown_prevents_duplicate_alerts():
    a = ConcealmentAnalyzer(DetectionConfig(cooldown_seconds=30.0), fps_hint=FPS)
    frames = [((200, 90), 0.9)] * 3 + [((130, 205), 0.9)] * 2 + [((130, 205), 0.05)] * 8
    frames += [((130, 205), 0.05)] * 10        # continua na zona logo depois
    events = _run(a, frames)
    assert len(events) == 1  # o cooldown segura o segundo


def test_small_person_is_ignored():
    """Pessoa menor que min_person_px → pose não confiável, não avalia."""
    cfg = DetectionConfig()
    cfg.guards.min_person_px = 500  # força o descarte
    a = ConcealmentAnalyzer(cfg, fps_hint=FPS)
    frames = [((200, 90), 0.9)] * 3 + [((130, 205), 0.9)] * 2 + [((130, 205), 0.05)] * 8
    assert _run(a, frames) == []


def test_low_pose_quality_is_ignored():
    cfg = DetectionConfig()
    cfg.guards.pose_quality_min = 0.99  # quase nada passa
    a = ConcealmentAnalyzer(cfg, fps_hint=FPS)
    frames = [((200, 90), 0.9)] * 3 + [((130, 205), 0.9)] * 2 + [((130, 205), 0.05)] * 8
    assert _run(a, frames) == []


def test_bag_zone_detects_hand_in_backpack():
    """Punho dentro da bbox de uma mochila associada à pessoa → zona 'bag'.

    A mochila fica posicionada de forma que seu interior, no referencial do
    corpo, caia FORA das zonas 'waist'/'torso' (senão classify_zone já
    resolveria a zona corporal antes do fallback de mochila ser avaliado) —
    ex.: mochila usada ao lado do corpo, além do envelope de cintura/tórax."""
    a = ConcealmentAnalyzer(DetectionConfig(), fps_hint=FPS)
    bag = ObjectDetection(label="backpack", bbox=BBox(170, 120, 220, 190), conf=0.8)
    events = []
    t = 0.0
    # reach primeiro
    for _ in range(2):
        events += a.update([_pose(1, (200, 90))], [bag], t); t += DT
    # mão entra na bbox da mochila, fica visível 2 frames e depois some
    for i in range(10):
        conf = 0.9 if i < 2 else 0.05
        events += a.update([_pose(1, (195, 140), wrist_conf=conf)], [bag], t); t += DT
    assert any(e.zone == "bag" for e in events)


def test_long_visible_dwell_fires():
    """Mão vem da prateleira e entra na cintura, mas o punho fica VISÍVEL o
    tempo todo (câmera lateral, sem oclusão) — nunca há `vanish`. Mesmo assim,
    permanência longa o bastante para saturar o dwell (>= dwell_seconds) tem
    que disparar sozinha, sem depender do sinal `vanish`.

    (140, 190) cai dentro da zona 'waist' neste referencial de corpo (ombros
    y=100, quadris y=200, escala~100): dx=40, dy=-10 -> x_n=0.4, y_n=0.1,
    dentro de waist_x=[0.10,0.85] e waist_y=[-0.45,0.25]."""
    a = ConcealmentAnalyzer(DetectionConfig(), fps_hint=FPS)
    frames = [((200, 90), 0.9)] * 3             # reach (braço estendido para a prateleira)
    frames += [((140, 190), 0.9)] * 14          # punho na cintura, sempre visível e confiante
    events = _run(a, frames)
    assert len(events) >= 1
    e = events[0]
    assert e.track_id == 1
    assert e.zone in ("waist", "torso")
    assert e.signals["dwell"] >= 1.0
    assert e.signals["vanish"] == 0.0


def test_incidental_hand_near_body_does_not_fire():
    """Falso-positivo medido em footage real: mão fica perto do corpo (zona
    de cintura) por movimento normal — NUNCA veio da prateleira (sem reach
    antes) e NUNCA some (conf sempre alta). Dwell puro satura, mas sem
    approach nem vanish isso não é uma assinatura de ocultação real e não
    deve disparar.

    Mesma posição (140, 190) do teste test_long_visible_dwell_fires (zona
    'waist'), mas SEM os 3 frames de reach que o precedem lá."""
    a = ConcealmentAnalyzer(DetectionConfig(), fps_hint=FPS)
    frames = [((140, 190), 0.9)] * 14           # já nasce dentro da zona, sempre visível
    events = _run(a, frames)
    assert events == []


def test_no_refire_from_stale_approach_after_cooldown():
    """Pessoa faz reach+conceal e dispara. Depois que o cooldown expira, a
    mão volta INCIDENTALMENTE à zona (visível, SEM novo reach, SEM sumir) —
    não deve disparar um segundo evento.

    Usa cooldown_seconds == window_seconds (3.0), o menor cooldown válido
    (a config recusaria cooldown < window). Entre o disparo e a reentrada, o
    punho fica parado numa posição neutra: (100, 220) não é reach nem zona
    neste referencial (origem no quadril-médio (100,200), escala 100) —
    x_n=0.0, y_n=-0.2, fora de waist/torso/reach."""
    a = ConcealmentAnalyzer(DetectionConfig(cooldown_seconds=3.0, window_seconds=3.0), fps_hint=FPS)
    fire_frames = [((200, 90), 0.9)] * 3 + [((130, 205), 0.9)] * 2 + [((130, 205), 0.05)] * 8
    neutral_frames = [((100, 220), 0.9)] * 7   # cobre o resto do cooldown, sem reach/zona
    reentry_frames = [((130, 205), 0.9)] * 14  # reentrada visível, sem novo reach, sem sumir
    events = _run(a, fire_frames + neutral_frames + reentry_frames)
    assert len(events) == 1


def test_wrist_history_cleared_on_fire():
    """A correção principal do vazamento: ao disparar um evento, o histórico
    de observações dos punhos do track é limpo. Sem essa limpeza, as
    observações do episódio que acabou de disparar (reach, entrada na zona,
    vanish) continuam ocupando a janela deslizante depois do cooldown —
    memória que um novo episódio não deveria herdar. Este teste falha sem a
    limpeza (histórico não fica vazio) e passa com ela.

    Verifica o estado IMEDIATAMENTE após o frame que dispara (não depois de
    rodar frames adicionais, que naturalmente reencheriam o histórico com
    observações novas e mascarariam a ausência da limpeza)."""
    a = ConcealmentAnalyzer(DetectionConfig(cooldown_seconds=3.0, window_seconds=3.0), fps_hint=FPS)
    frames = [((200, 90), 0.9)] * 3 + [((130, 205), 0.9)] * 2 + [((130, 205), 0.05)] * 8
    t = 0.0
    fired = False
    for wrist_xy, conf in frames:
        ev = a.update([_pose(1, wrist_xy, wrist_conf=conf)], [], t)
        t += DT
        if ev:
            fired = True
            break
    assert fired
    st = a._tracks[1]
    for hist in st.wrists.values():
        assert hist.observations == []


def test_per_track_state_isolation():
    """Duas pessoas: só a que faz o gesto dispara."""
    a = ConcealmentAnalyzer(DetectionConfig(), fps_hint=FPS)
    events = []
    t = 0.0
    for i in range(13):
        if i < 3:
            wrist_ladrao, conf_ladrao = (200, 90), 0.9   # reach
        elif i < 5:
            wrist_ladrao, conf_ladrao = (130, 205), 0.9  # entra no bolso, visível
        else:
            wrist_ladrao, conf_ladrao = (130, 205), 0.05  # some dentro do bolso
        p_ladrao = _pose(1, wrist_ladrao, wrist_conf=conf_ladrao)
        p_inocente = _pose(2, (160, 120), bbox=BBox(300, 60, 340, 300))
        events += a.update([p_ladrao, p_inocente], [], t)
        t += DT
    assert {e.track_id for e in events} == {1}


def test_sensitivity_dial_catches_subtle_gesture_when_relaxed():
    """O 'botao de sensibilidade' (require_approach_or_vanish).

    Gesto sutil: a mao JA esta junto ao corpo e permanece na zona de ocultacao
    (baixar um produto na bolsa sem esticar o braco antes e sem o punho sumir).
    - Conservador (True): NAO dispara (evita alarme falso de mao incidental).
    - Sensivel (False): dispara (pega o gesto sutil).
    Este e o trade-off central da calibracao — os dois lados travados por teste.
    """
    frames = [((140, 190), 0.9)] * 14  # punho parado na zona 'waist', sempre visivel

    conservador = ConcealmentAnalyzer(
        DetectionConfig(require_approach_or_vanish=True), fps_hint=FPS)
    assert _run(conservador, frames) == []

    sensivel = ConcealmentAnalyzer(
        DetectionConfig(require_approach_or_vanish=False), fps_hint=FPS)
    eventos = _run(sensivel, frames)
    assert len(eventos) >= 1
    assert eventos[0].zone == "waist"
