"""Detector de pânico — as duas mãos acima da cabeça por N segundos → alerta.

O "botão de emergência virtual" do caixa (pivô 29/jul, depois de o detector de
furto se mostrar inviável na câmera de teto). Ao contrário do furto, este é um
gesto DELIBERADO e sem ambiguidade: as duas mãos acima da cabeça, mantidas por
alguns segundos. A pose acerta isso com confiança de qualquer ângulo — e num
assalto é o sinal certo (o caixa levanta as mãos, forçado ou pra avisar).

Lógica pura, sem I/O nem Qt — testável sem câmera. Mesma interface do
ConcealmentAnalyzer (`update(poses, ts) -> list[eventos]`) para encaixar no
pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import KP, PersonPose

NOSE = KP["nose"]
LWRIST = KP["left_wrist"]
RWRIST = KP["right_wrist"]


@dataclass
class PanicConfig:
    hold_seconds: float = 2.0        # tempo com as mãos acima para disparar
    cooldown_seconds: float = 30.0   # silêncio após um disparo (não spamma)
    wrist_conf_min: float = 0.30     # confiança mínima do punho/nariz
    track_lost_seconds: float = 2.0  # descarta track sumido há mais que isto


@dataclass
class PanicEvent:
    track_id: int
    ts: float
    kind: str = "panico"
    # campos abaixo existem só para o evento fluir pela MESMA infra de evidência
    # (recorder + banco) que o evento de ocultação — não têm significado próprio.
    score: float = 1.0
    zone: str = "panico"
    signals: dict = field(default_factory=dict)


@dataclass
class _State:
    up_since: float | None = None    # quando as mãos subiram (None = estão baixas)
    cooldown_until: float = 0.0
    last_seen: float = 0.0


class PanicDetector:
    def __init__(self, cfg: PanicConfig) -> None:
        self.cfg = cfg
        self._tracks: dict[int, _State] = {}

    def _maos_acima(self, kp) -> bool:
        """True se AS DUAS mãos estão acima do nariz (y menor = mais alto na
        imagem), com confiança suficiente. Uma mão só não conta — acenar ou
        pegar produto na prateleira alta não é pânico."""
        c = self.cfg.wrist_conf_min
        nose, lw, rw = kp[NOSE], kp[LWRIST], kp[RWRIST]
        if nose[2] < c or lw[2] < c or rw[2] < c:
            return False
        return lw[1] < nose[1] and rw[1] < nose[1]

    def update(self, poses: list[PersonPose], ts: float) -> list[PanicEvent]:
        events: list[PanicEvent] = []
        alive: set[int] = set()

        for pose in poses:
            tid = pose.person.track_id
            if tid is None:
                continue
            alive.add(tid)
            st = self._tracks.setdefault(tid, _State())
            st.last_seen = ts

            acima = self._maos_acima(pose.keypoints)

            # Em cooldown: só acompanha o estado das mãos, não dispara.
            if ts < st.cooldown_until:
                st.up_since = ts if acima else None
                continue

            if not acima:
                st.up_since = None
                continue

            if st.up_since is None:
                st.up_since = ts
            elif ts - st.up_since >= self.cfg.hold_seconds:
                events.append(PanicEvent(track_id=tid, ts=ts))
                st.cooldown_until = ts + self.cfg.cooldown_seconds
                st.up_since = None  # o próximo pânico é um novo episódio

        # limpa tracks perdidos há muito tempo
        for tid in list(self._tracks):
            if tid not in alive and ts - self._tracks[tid].last_seen > self.cfg.track_lost_seconds:
                del self._tracks[tid]

        return events
