"""Os quatro sinais da heurística de ocultação (spec §6.3), sobre uma janela
deslizante de observações de UM punho, de UMA pessoa rastreada.

O sinal mais importante é `vanish`: quando alguém enfia a mão no bolso, sob a
blusa ou na bolsa, o punho DESAPARECE dos keypoints. Tratamos punho que some
DENTRO da zona do corpo como evidência positiva, não como dado faltante — sem
isso, 'sob a roupa' e 'dentro da mochila' seriam quase invisíveis."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config.settings import DetectionConfig


@dataclass
class Observation:
    x_n: float
    y_n: float
    conf: float
    zone: str | None
    reach: bool
    ts: float


@dataclass
class Signals:
    dwell: float
    approach: float
    vanish: float
    retract: float
    zone: str | None


@dataclass
class WristHistory:
    fps_hint: float = 5.0
    observations: list[Observation] = field(default_factory=list)

    def observe(self, x_n, y_n, conf, zone, reach, ts) -> None:
        self.observations.append(Observation(x_n, y_n, conf, zone, reach, ts))

    def prune(self, now: float, window_seconds: float) -> None:
        cutoff = now - window_seconds
        self.observations = [o for o in self.observations if o.ts >= cutoff]


def compute_signals(hist: WristHistory, cfg: DetectionConfig, now: float) -> Signals:
    obs = hist.observations
    if not obs:
        return Signals(0.0, 0.0, 0.0, 0.0, None)

    g = cfg.guards
    fps = hist.fps_hint
    dwell_frames_target = max(1.0, cfg.dwell_seconds * fps)
    gap_allow = g.gap_frames

    # zona corrente = zona da observação mais recente com punho confiável
    cur_zone = None
    for o in reversed(obs):
        if o.conf >= g.kp_conf_min:
            cur_zone = o.zone
            break

    # --- dwell: frames consecutivos (tolerando gap) numa zona de ocultação ---
    # Importante: usamos o streak CORRENTE (o valor ao final do loop), não o
    # maior streak já visto na janela. Um gap longo precisa realmente "zerar
    # a contagem" — se guardássemos o máximo histórico, uma permanência
    # anterior ao gap sobreviveria como "recorde" e o dwell nunca cairia,
    # mesmo depois da pessoa ter saído e reentrado na zona.
    streak = 0.0
    gap = 0
    active_zone = None
    for o in obs:
        in_zone = o.zone is not None and o.conf >= g.kp_conf_min
        # punho que sumiu DENTRO da zona conta como permanência (não quebra o streak)
        vanished_in_zone = o.conf < g.kp_conf_min and o.zone is not None
        if in_zone or vanished_in_zone:
            if active_zone is None:
                active_zone = o.zone
            streak += 1
            gap = 0
        else:
            gap += 1
            if gap > gap_allow:
                streak = 0.0
                active_zone = None
    current_streak = streak
    dwell = min(1.0, current_streak / dwell_frames_target)

    # --- approach: o punho esteve em 'reach' na janela ANTES de entrar na zona ---
    approach = 0.0
    first_zone_ts = next((o.ts for o in obs if o.zone is not None), None)
    if first_zone_ts is not None:
        reach_before = [o for o in obs if o.reach and o.ts < first_zone_ts]
        if reach_before:
            age = now - max(o.ts for o in reach_before)
            approach = max(0.0, 1.0 - age / max(1e-3, cfg.window_seconds))

    # --- vanish: punho sumido cuja última posição conhecida era na zona ---
    vanish = 0.0
    last_known = next((o for o in reversed(obs) if o.conf >= g.kp_conf_min), None)
    latest = obs[-1]
    if latest.conf < g.kp_conf_min and last_known is not None and last_known.zone is not None:
        gap_since = now - last_known.ts
        if gap_since <= g.vanish_max_seconds:
            vanish = 1.0 if gap_since >= 0 else 0.0
            # dentro do período de graça é sempre forte; depois, decai até expirar
            if gap_since > g.vanish_grace_seconds:
                vanish = max(0.0, 1.0 - (gap_since - g.vanish_grace_seconds) /
                             max(1e-3, g.vanish_max_seconds - g.vanish_grace_seconds))

    # --- retract: após permanência, o punho reaparece e SOBE (Δy_n > 0.3 em ~1s) ---
    retract = 0.0
    half_dwell = 0.5 * dwell_frames_target
    if current_streak >= half_dwell:
        recent = [o for o in obs if o.conf >= g.kp_conf_min and o.ts >= now - 1.0]
        if len(recent) >= 2:
            dy = recent[-1].y_n - recent[0].y_n
            if dy > 0.3:
                retract = min(1.0, dy / 0.6)

    return Signals(dwell, approach, vanish, retract, cur_zone or active_zone)
