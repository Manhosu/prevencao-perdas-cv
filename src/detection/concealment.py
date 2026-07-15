"""Analisador de ocultação: junta coordenadas do corpo + sinais + máquina de
estados por pessoa rastreada, e emite eventos (spec §6.4, §6.5, §6.6).

Um evento sai quando: score >= limiar E o dwell mínimo foi atingido E o
cooldown do track está livre. Todos os pesos e limiares vêm do DetectionConfig
— calibrar é editar número, nunca código."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.config.settings import DetectionConfig
from src.core.types import BBox, KP, ObjectDetection, PersonPose
from src.detection.body_frame import BodyFrame, classify_zone, in_reach
from src.detection.signals import Signals, WristHistory, compute_signals

WRISTS = (KP["left_wrist"], KP["right_wrist"])


class State(str, Enum):
    IDLE = "idle"
    APPROACHING = "approaching"
    CONCEALING = "concealing"
    ALERT = "alert"
    COOLDOWN = "cooldown"


@dataclass
class ConcealmentEvent:
    track_id: int
    score: float
    zone: str
    signals: dict
    ts: float


@dataclass
class _TrackState:
    state: State = State.IDLE
    wrists: dict[int, WristHistory] = field(default_factory=dict)
    last_seen: float = 0.0
    cooldown_until: float = 0.0


class ConcealmentAnalyzer:
    def __init__(self, cfg: DetectionConfig, fps_hint: float = 5.0) -> None:
        self.cfg = cfg
        self.fps_hint = fps_hint
        self._tracks: dict[int, _TrackState] = {}

    def update(
        self, poses: list[PersonPose], objects: list[ObjectDetection], ts: float
    ) -> list[ConcealmentEvent]:
        g = self.cfg.guards
        events: list[ConcealmentEvent] = []
        alive: set[int] = set()

        for pose in poses:
            tid = pose.person.track_id
            if tid is None:
                continue
            alive.add(tid)
            st = self._tracks.setdefault(tid, _TrackState())
            st.last_seen = ts

            # Guardas: pessoa pequena demais ou pose ruim → ignora
            if pose.person.bbox.height < g.min_person_px:
                continue
            bf = BodyFrame.from_keypoints(pose.keypoints, pose.person.bbox, g)
            if bf is None or bf.quality < g.pose_quality_min:
                continue

            bag = self._nearest_bag(bf, objects)

            # Atualiza o histórico de cada punho
            best: tuple[float, Signals] | None = None
            for wi in WRISTS:
                w = pose.keypoints[wi]
                x_n, y_n = bf.to_body_coords((float(w[0]), float(w[1])))
                zone = classify_zone(x_n, y_n, self.cfg.geometry, bf.facing_back)
                if zone is None and bag is not None and self._in_bag(w, bag):
                    zone = "bag"
                reach = in_reach(x_n, y_n, self.cfg.geometry)
                hist = st.wrists.setdefault(wi, WristHistory(self.fps_hint))
                hist.observe(x_n, y_n, float(w[2]), zone, reach, ts)
                hist.prune(ts, self.cfg.window_seconds)
                sig = compute_signals(hist, self.cfg, ts)
                sc = self._score(sig, bf.quality)
                if best is None or sc > best[0]:
                    best = (sc, sig)

            if best is None:
                continue
            score, sig = best

            ev = self._advance(st, tid, score, sig, ts)
            if ev is not None:
                events.append(ev)

        # descarta tracks perdidos há muito tempo
        for tid in list(self._tracks):
            if tid not in alive and ts - self._tracks[tid].last_seen > g.track_lost_seconds:
                del self._tracks[tid]

        return events

    # --- score (spec §6.4) ---
    def _score(self, s: Signals, quality: float) -> float:
        w = self.cfg.weights
        zw = self.cfg.zone_weights
        bruto = (w.dwell * s.dwell + w.approach * s.approach +
                 w.vanish * s.vanish + w.retract * s.retract)
        zone_weight = getattr(zw, s.zone, 1.0) if s.zone else 0.0
        return float(np.clip(bruto * zone_weight * quality, 0.0, 1.0))

    # --- máquina de estados (spec §6.5) ---
    def _advance(self, st, tid, score, sig, ts) -> ConcealmentEvent | None:
        if st.state == State.COOLDOWN:
            if ts >= st.cooldown_until:
                st.state = State.IDLE
            else:
                return None

        dwell_ok = sig.dwell >= 1.0 or sig.vanish > 0.5
        if score >= self.cfg.threshold and dwell_ok and sig.zone:
            st.state = State.ALERT
            ev = ConcealmentEvent(
                track_id=tid, score=round(score, 3), zone=sig.zone,
                signals={"dwell": round(sig.dwell, 3), "approach": round(sig.approach, 3),
                         "vanish": round(sig.vanish, 3), "retract": round(sig.retract, 3)},
                ts=ts,
            )
            st.state = State.COOLDOWN
            st.cooldown_until = ts + self.cfg.cooldown_seconds
            return ev

        st.state = State.CONCEALING if sig.zone else State.IDLE
        return None

    # --- associação de bolsa/mochila (spec §6.2) ---
    def _nearest_bag(self, bf: BodyFrame, objects) -> ObjectDetection | None:
        best = None
        best_d = 1.2 * bf.scale
        for o in objects:
            if o.label not in ("backpack", "handbag"):
                continue
            cx, cy = o.bbox.center
            d = float(np.hypot(cx - bf.shoulder_mid[0], cy - bf.shoulder_mid[1]))
            if d <= best_d:
                best, best_d = o, d
        return best

    def _in_bag(self, wrist, bag: ObjectDetection) -> bool:
        b = bag.bbox.expand(0.1)
        return b.contains(float(wrist[0]), float(wrist[1]))
