"""Detector de pânico — as duas mãos acima da cabeça por N segundos → alerta.

O "botão de emergência virtual" do caixa (pivô 29/jul, depois de o detector de
furto se mostrar inviável na câmera de teto). Ao contrário do furto, este é um
gesto DELIBERADO e sem ambiguidade: as duas mãos acima da cabeça, mantidas por
alguns segundos. A pose acerta isso com confiança — de frente (referência no
nariz) e de costas (referência nos ombros, ver `_linha_da_cabeca`), que é o
caso do caixa com a câmera atrás. Num assalto é o sinal certo (o caixa levanta
as mãos, forçado ou pra avisar).

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
LSHOULDER = KP["left_shoulder"]
RSHOULDER = KP["right_shoulder"]


@dataclass
class PanicConfig:
    hold_seconds: float = 2.0        # tempo com as mãos acima para disparar
    cooldown_seconds: float = 30.0   # silêncio após um disparo (não spamma)
    wrist_conf_min: float = 0.30     # confiança mínima do punho/nariz/ombro
    track_lost_seconds: float = 2.0  # descarta track sumido há mais que isto
    # De COSTAS o nariz não aparece, então a referência de "altura da cabeça"
    # vem dos ombros: a cabeça fica acima da linha dos ombros por esta fração
    # da largura deles. 0.5 ≈ topo da cabeça — exige a mão claramente acima,
    # não só na altura do ombro (senão pegar algo numa prateleira à altura do
    # ombro viraria falso pânico). Só é usada quando o rosto está oculto.
    head_above_shoulder: float = 0.5


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

    def _linha_da_cabeca(self, kp) -> float | None:
        """Altura (y na imagem) da cabeça, para comparar com os punhos.

        De FRENTE, é o nariz (comportamento validado em campo, mantido igual).
        De COSTAS o nariz some — e era isso que impedia o pânico de disparar
        com o caixa de costas para a câmera. Nesse caso a referência vem dos
        ombros: a cabeça está acima da linha dos ombros por uma fração da
        largura deles. Sem nariz E sem os dois ombros confiáveis, não há como
        estabelecer a referência → None (não dispara, lado seguro)."""
        c = self.cfg.wrist_conf_min
        nose = kp[NOSE]
        if nose[2] >= c:
            return float(nose[1])
        ls, rs = kp[LSHOULDER], kp[RSHOULDER]
        if ls[2] >= c and rs[2] >= c:
            largura = abs(float(ls[0]) - float(rs[0]))
            if largura < 1.0:  # ombros sobrepostos (perfil): referência não confiável
                return None
            ombro_y = (float(ls[1]) + float(rs[1])) / 2
            return ombro_y - self.cfg.head_above_shoulder * largura
        return None

    def _maos_acima(self, kp) -> bool:
        """True se AS DUAS mãos estão acima da cabeça, com confiança
        suficiente. Uma mão só não conta — acenar ou pegar produto na
        prateleira alta não é pânico. Funciona de frente e de costas (ver
        `_linha_da_cabeca`)."""
        c = self.cfg.wrist_conf_min
        lw, rw = kp[LWRIST], kp[RWRIST]
        if lw[2] < c or rw[2] < c:
            return False
        cabeca = self._linha_da_cabeca(kp)
        if cabeca is None:
            return False
        return lw[1] < cabeca and rw[1] < cabeca

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
