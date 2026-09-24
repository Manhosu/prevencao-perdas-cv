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
LHIP = KP["left_hip"]
RHIP = KP["right_hip"]


@dataclass
class PanicConfig:
    hold_seconds: float = 2.0        # tempo com as mãos acima para disparar
    cooldown_seconds: float = 30.0   # silêncio após um disparo (não spamma)
    wrist_conf_min: float = 0.30     # confiança mínima do punho/nariz/ombro
    track_lost_seconds: float = 2.0  # descarta track sumido há mais que isto
    # De COSTAS o nariz não aparece, então a referência de "altura da cabeça"
    # vem dos ombros: a cabeça fica acima da linha dos ombros por esta fração
    # da largura deles. Só é usada quando `detectar_de_costas` está ligado.
    head_above_shoulder: float = 0.5
    # Detecção com a pessoa DE COSTAS (rosto oculto → referência nos ombros).
    # OFF por padrão, e isto é uma decisão de segurança: em câmera de TETO (o
    # caso comum das lojas) "mãos acima da cabeça" não vira geometria
    # confiável — uma pessoa curvada sobre o balcão projeta os punhos acima da
    # linha dos ombros e dispara pânico FALSO (relato de campo 24/set, câmera
    # de teto). Ligar só faz sentido numa câmera na ALTURA da pessoa, atrás do
    # caixa, onde o corpo aparece em pé. Mesmo ligado, a guarda de "em pé"
    # (ombros claramente acima dos quadris) evita o falso de teto.
    detectar_de_costas: bool = False


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

        De FRENTE, é o nariz (comportamento validado em campo: 0 falso em 39
        fotos). Este caminho é o padrão e não mudou.

        De COSTAS o nariz some. Só então, E se `detectar_de_costas` estiver
        ligado, a referência vem dos ombros. É opcional/OFF porque em câmera de
        teto esse caminho gera falso positivo (pessoa curvada sobre o balcão).
        Mesmo ligado, exige o corpo EM PÉ (ombros acima dos quadris) — o que
        rejeita a câmera de teto, onde ombros e quadris caem quase no mesmo y."""
        c = self.cfg.wrist_conf_min
        nose = kp[NOSE]
        if nose[2] >= c:
            return float(nose[1])
        if not self.cfg.detectar_de_costas:
            return None
        ls, rs = kp[LSHOULDER], kp[RSHOULDER]
        if ls[2] < c or rs[2] < c:
            return None
        largura = abs(float(ls[0]) - float(rs[0]))
        if largura < 1.0:  # ombros sobrepostos (perfil): referência não confiável
            return None
        # Guarda "em pé": a referência pelos ombros só é confiável se o tronco
        # estiver vertical na imagem. Em câmera de teto o corpo é visto de cima
        # e ombros/quadris caem quase no mesmo y — esta guarda barra esse caso,
        # que é a fonte do falso positivo. Sem quadris confiáveis, não arrisca.
        lh, rh = kp[LHIP], kp[RHIP]
        if lh[2] < c or rh[2] < c:
            return None
        ombro_y = (float(ls[1]) + float(rs[1])) / 2
        quadril_y = (float(lh[1]) + float(rh[1])) / 2
        if quadril_y - ombro_y < 0.5 * largura:
            return None  # tronco não está vertical (câmera de teto): não dispara
        return ombro_y - self.cfg.head_above_shoulder * largura

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
