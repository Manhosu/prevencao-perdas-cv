# Plano 2 — Lógica de Ocultação (marco de 50%)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transformar o fluxo de `PersonPose` (que o Plano 1 já entrega por câmera) em **eventos de ocultação** — a inteligência que decide "esta pessoa levou a mão ao bolso/bolsa/roupa e ficou lá tempo suficiente". É o coração do produto e o marco que libera 50% do pagamento.

**Architecture:** Núcleo puro em `src/detection/`. A partir dos keypoints COCO-17, derivamos um **sistema de coordenadas ancorado no corpo** (origem no quadril, escala pelo tronco) que normaliza pessoa perto/longe e inclinada sem calibrar por câmera. Sobre esse espaço definimos **zonas de ocultação**; uma **janela deslizante por punho, por track** alimenta **quatro sinais** (permanência, trajetória, sumiço do punho, retração); uma **máquina de estados por track** converte os sinais em `score` e dispara evento quando passa do limiar. Um **modo replay** roda tudo sobre arquivo de vídeo, e um **sweep** mede acerto × falso-positivo sobre uma pasta de clipes rotulados — convertendo calibração em medição.

**Tech Stack:** Python 3.13, numpy, OpenCV (só no replay, para anotar/ler vídeo), pytest. `src/detection/` continua sem I/O.

## Global Constraints

- **`src/detection/` é núcleo puro:** sem `cv2.VideoCapture`, `sqlite3`, `requests`, `PySide6`, RTSP. Recebe dataclasses, devolve dataclasses. (O `replay.py` e o `calibrate.py` vivem em `src/tools/` e PODEM usar cv2.)
- **Toda constante da heurística vem do `DetectionConfig`** (já existe, do Plano 1) — pesos, limiares, geometria, guardas. Calibrar é editar JSON, nunca código. É a razão de existir do projeto.
- **A heurística é testável com keypoints sintéticos** — sequências de coordenadas construídas à mão. Nenhum teste do núcleo depende de vídeo real. Vídeo só aparece nos testes `@pytest.mark.slow` do replay.
- **Coordenadas do corpo (spec §6.1):** `hip_mid`, `shoulder_mid`, escala `S = ||shoulder_mid − hip_mid||`, eixo vertical `û` (do quadril para o ombro), horizontal `v̂`. Punho `w` → `x_n = ((w−hip_mid)·v̂)/S`, `y_n = ((w−hip_mid)·û)/S`. `y_n`: 0 = linha do quadril, 1 = linha do ombro. `x_n`: 0 = eixo do corpo.
- **Fallback de escala:** quando os quadris têm confiança `< kp_conf_min`, `S = 0.55 × altura_bbox`.
- **Índices COCO-17** vêm de `src.core.types.KP`. Punhos 9/10, ombros 5/6, quadris 11/12, nariz 0, olhos 1/2.
- Nomes de código em inglês; comentários/logs em português; commits em português com prefixo convencional.
- Roda os testes com `.venv\Scripts\python.exe -m pytest`. `pythonpath = .` já está no `pytest.ini`.

## Interfaces já existentes (do Plano 1 — NÃO modificar)

- `src.core.types`: `BBox`, `PersonDetection(bbox, conf, track_id)`, `PersonPose(person, keypoints)` (keypoints `(17,3)` em px do frame completo, coluna 2 = confiança), `ObjectDetection(label, bbox, conf)`, `KP`.
- `src.config.settings`: `DetectionConfig` com `.threshold`, `.dwell_seconds`, `.window_seconds`, `.cooldown_seconds`, `.weights.{dwell,approach,vanish,retract}`, `.zone_weights.{waist,torso,back_waist,bag}`, `.geometry.{waist_y,waist_x,torso_y,torso_x_max,reach_y_min,reach_x_min}`, `.guards.{kp_conf_min,pose_quality_min,min_person_px,vanish_grace_seconds,vanish_max_seconds,gap_frames,track_lost_seconds}`.
- `src.pipeline.Pipeline` / `FrameResult(camera_name, persons: list[PersonPose], objects, had_person)`.

---

## Estrutura de arquivos deste plano

| Arquivo | Responsabilidade |
|---|---|
| `src/detection/body_frame.py` | coordenadas do corpo + classificação de zona de um ponto |
| `src/detection/signals.py` | histórico por punho + os 4 sinais sobre a janela |
| `src/detection/concealment.py` | máquina de estados por track + score + emissão de evento |
| `src/tools/replay.py` | roda o pipeline+concealment sobre arquivo de vídeo, anota, exporta CSV |
| `src/tools/calibrate.py` | sweep de parâmetros sobre pasta de clipes rotulados |

---

## Task 1: Sistema de coordenadas do corpo

**Files:**
- Create: `src/detection/body_frame.py`
- Test: `tests/detection/test_body_frame.py`

**Interfaces:**
- Consumes: `DetectionConfig`, `KP`, `BBox`.
- Produces:
  - `BodyFrame.from_keypoints(kp: np.ndarray, bbox: BBox, guards) -> BodyFrame | None` (None se não há tronco confiável e nem bbox utilizável).
  - `BodyFrame` com `.hip_mid`, `.shoulder_mid`, `.scale`, `.u`, `.v`, `.quality` (média conf ombros+quadris), `.facing_back: bool`.
  - `.to_body_coords(point_xy) -> tuple[x_n, y_n]`.
  - `classify_zone(x_n, y_n, geometry, facing_back) -> str | None` devolve `"waist" | "torso" | "back_waist" | None`.
  - `in_reach(x_n, y_n, geometry) -> bool`.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/detection/test_body_frame.py`:

```python
import numpy as np
import pytest

from src.config.settings import Guards, Geometry
from src.core.types import BBox, KP
from src.detection.body_frame import BodyFrame, classify_zone, in_reach

G = Guards()
GEO = Geometry()


def _kp(**pts):
    """Constrói (17,3) com confiança 0.9 nos keypoints dados, 0 no resto.
    pts: nome_coco -> (x, y)."""
    a = np.zeros((17, 3), dtype=np.float32)
    for name, (x, y) in pts.items():
        a[KP[name]] = [x, y, 0.9]
    return a


def _upright():
    """Pessoa em pé: ombros em y=100, quadris em y=200 (y cresce p/ baixo na imagem).
    Corpo vertical, tronco de 100px."""
    return _kp(
        left_shoulder=(90, 100), right_shoulder=(110, 100),
        left_hip=(92, 200), right_hip=(108, 200),
        nose=(100, 80), left_eye=(96, 78), right_eye=(104, 78),
    )


def test_builds_frame_from_upright_person():
    bf = BodyFrame.from_keypoints(_upright(), BBox(80, 60, 120, 300), G)
    assert bf is not None
    assert bf.hip_mid == pytest.approx((100, 200), abs=1)
    assert bf.shoulder_mid == pytest.approx((100, 100), abs=1)
    assert bf.scale == pytest.approx(100, abs=2)  # ||ombro-quadril||
    assert bf.quality > 0.8


def test_wrist_at_hip_line_is_yn_zero():
    bf = BodyFrame.from_keypoints(_upright(), BBox(80, 60, 120, 300), G)
    x_n, y_n = bf.to_body_coords((100, 200))  # exatamente no hip_mid
    assert x_n == pytest.approx(0, abs=0.05)
    assert y_n == pytest.approx(0, abs=0.05)


def test_wrist_at_shoulder_line_is_yn_one():
    bf = BodyFrame.from_keypoints(_upright(), BBox(80, 60, 120, 300), G)
    _, y_n = bf.to_body_coords((100, 100))  # na linha do ombro
    assert y_n == pytest.approx(1.0, abs=0.05)


def test_wrist_below_hip_is_negative_yn():
    bf = BodyFrame.from_keypoints(_upright(), BBox(80, 60, 120, 300), G)
    _, y_n = bf.to_body_coords((100, 250))  # abaixo do quadril (bolso/coxa)
    assert y_n < 0


def test_lateral_offset_is_xn():
    bf = BodyFrame.from_keypoints(_upright(), BBox(80, 60, 120, 300), G)
    x_n, _ = bf.to_body_coords((150, 200))  # 50px à direita do eixo, S=100
    assert abs(x_n) == pytest.approx(0.5, abs=0.05)


def test_scale_normalizes_distance():
    """Pessoa 2x mais longe (metade do tamanho) → mesmas coords de corpo."""
    near = BodyFrame.from_keypoints(_upright(), BBox(80, 60, 120, 300), G)
    far = _kp(
        left_shoulder=(45, 50), right_shoulder=(55, 50),
        left_hip=(46, 100), right_hip=(54, 100),
        nose=(50, 40), left_eye=(48, 39), right_eye=(52, 39),
    )
    bf_far = BodyFrame.from_keypoints(far, BBox(40, 30, 60, 150), G)
    # punho "no bolso": near (100,250) dy=+50 sobre S=100 → yn=-0.5
    # far equivalente (50,125) dy=+25 sobre S=50 → yn=-0.5
    _, yn_near = near.to_body_coords((100, 250))
    _, yn_far = bf_far.to_body_coords((50, 125))
    assert yn_near == pytest.approx(yn_far, abs=0.05)


def test_fallback_scale_when_hips_missing():
    kp = _kp(left_shoulder=(90, 100), right_shoulder=(110, 100), nose=(100, 80))
    bf = BodyFrame.from_keypoints(kp, BBox(80, 60, 120, 300), G)  # bbox altura 240
    assert bf is not None
    assert bf.scale == pytest.approx(0.55 * 240, abs=1)


def test_returns_none_when_no_torso_and_no_bbox():
    kp = np.zeros((17, 3), dtype=np.float32)
    assert BodyFrame.from_keypoints(kp, BBox(0, 0, 0, 0), G) is None


def test_facing_back_when_face_not_visible():
    kp = _upright()
    kp[KP["nose"]] = [0, 0, 0.0]
    kp[KP["left_eye"]] = [0, 0, 0.0]
    kp[KP["right_eye"]] = [0, 0, 0.0]
    bf = BodyFrame.from_keypoints(kp, BBox(80, 60, 120, 300), G)
    assert bf.facing_back is True


def test_facing_front_when_face_visible():
    bf = BodyFrame.from_keypoints(_upright(), BBox(80, 60, 120, 300), G)
    assert bf.facing_back is False


def test_classify_zone_waist():
    # waist_y [-0.45,0.25], waist_x [0.10,0.85]
    assert classify_zone(0.4, -0.1, GEO, facing_back=False) == "waist"


def test_classify_zone_torso():
    # torso_y [0.15,0.85], torso_x_max 0.55 — precisa NÃO cair em waist antes
    assert classify_zone(0.2, 0.5, GEO, facing_back=False) == "torso"


def test_classify_zone_back_waist_only_when_facing_back():
    assert classify_zone(0.4, -0.1, GEO, facing_back=True) == "back_waist"


def test_classify_zone_none_when_far_from_body():
    assert classify_zone(1.2, 0.5, GEO, facing_back=False) is None


def test_in_reach_arm_extended():
    # reach: y_n > 0.9 OU |x_n| > 0.95
    assert in_reach(1.1, 0.5, GEO) is True
    assert in_reach(0.2, 0.95, GEO) is True
    assert in_reach(0.3, 0.2, GEO) is False
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `.venv\Scripts\python.exe -m pytest tests/detection/test_body_frame.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.detection.body_frame'`

- [ ] **Step 3: Implementar `src/detection/body_frame.py`**

```python
"""Sistema de coordenadas ancorado no corpo (spec §6.1).

Converte a posição de um punho para um espaço normalizado pelo tronco da
pessoa. Isso resolve, SEM calibrar por câmera, os três problemas que quebram
heurística ingênua: pessoa perto vs. longe (a escala S normaliza), pessoa
inclinada (os eixos acompanham o corpo) e câmeras em alturas diferentes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config.settings import Geometry, Guards
from src.core.types import BBox, KP

Point = tuple[float, float]


def _mid(kp: np.ndarray, a: int, b: int, conf_min: float) -> tuple[Point | None, float]:
    """Ponto médio entre dois keypoints e a confiança média (0 se ausente)."""
    ca, cb = float(kp[a, 2]), float(kp[b, 2])
    if ca < conf_min and cb < conf_min:
        return None, 0.0
    # usa só os confiáveis; se um só, devolve ele
    pts = [(kp[i, 0], kp[i, 1]) for i in (a, b) if kp[i, 2] >= conf_min]
    x = float(np.mean([p[0] for p in pts]))
    y = float(np.mean([p[1] for p in pts]))
    return (x, y), (ca + cb) / 2


@dataclass
class BodyFrame:
    hip_mid: Point
    shoulder_mid: Point
    scale: float
    u: Point  # eixo vertical do corpo (quadril -> ombro), unitário
    v: Point  # eixo horizontal do corpo, unitário
    quality: float
    facing_back: bool

    @classmethod
    def from_keypoints(cls, kp: np.ndarray, bbox: BBox, guards: Guards) -> "BodyFrame | None":
        cmin = guards.kp_conf_min
        shoulder_mid, conf_sh = _mid(kp, KP["left_shoulder"], KP["right_shoulder"], cmin)
        hip_mid, conf_hip = _mid(kp, KP["left_hip"], KP["right_hip"], cmin)

        if shoulder_mid is None and hip_mid is None:
            return None  # sem tronco

        # Escala e eixo vertical
        if shoulder_mid is not None and hip_mid is not None:
            dx = shoulder_mid[0] - hip_mid[0]
            dy = shoulder_mid[1] - hip_mid[1]
            scale = float(np.hypot(dx, dy))
            if scale < 1e-3:
                scale = 0.55 * bbox.height
                u = (0.0, -1.0)
            else:
                u = (dx / scale, dy / scale)
        else:
            # Fallback: só um dos dois presente → usa altura da bbox e vertical da imagem
            scale = 0.55 * bbox.height
            u = (0.0, -1.0)  # "para cima" na imagem
            if hip_mid is None:
                # estima quadril abaixo do ombro
                hip_mid = (shoulder_mid[0] - u[0] * scale, shoulder_mid[1] - u[1] * scale)
            if shoulder_mid is None:
                shoulder_mid = (hip_mid[0] + u[0] * scale, hip_mid[1] + u[1] * scale)

        if scale < 1e-3:
            return None

        # Eixo horizontal = perpendicular ao vertical
        v = (-u[1], u[0])
        quality = (conf_sh + conf_hip) / 2 if (conf_sh and conf_hip) else max(conf_sh, conf_hip)

        # De costas: rosto (nariz + olhos) sem confiança
        face = [kp[KP[n], 2] for n in ("nose", "left_eye", "right_eye")]
        facing_back = all(c < cmin for c in face)

        return cls(hip_mid, shoulder_mid, scale, u, v, float(quality), facing_back)

    def to_body_coords(self, point: Point) -> tuple[float, float]:
        dx = point[0] - self.hip_mid[0]
        dy = point[1] - self.hip_mid[1]
        y_n = (dx * self.u[0] + dy * self.u[1]) / self.scale
        x_n = (dx * self.v[0] + dy * self.v[1]) / self.scale
        return x_n, y_n


def classify_zone(x_n: float, y_n: float, geo: Geometry, facing_back: bool) -> str | None:
    """Zona de ocultação de um ponto em coordenadas do corpo, ou None.
    Ordem de prioridade: cintura (frente/costas) antes de tórax."""
    ax = abs(x_n)
    wy0, wy1 = geo.waist_y
    wx0, wx1 = geo.waist_x
    if wy0 <= y_n <= wy1 and wx0 <= ax <= wx1:
        return "back_waist" if facing_back else "waist"
    ty0, ty1 = geo.torso_y
    if ty0 <= y_n <= ty1 and ax <= geo.torso_x_max:
        return "torso"
    return None


def in_reach(x_n: float, y_n: float, geo: Geometry) -> bool:
    """Braço estendido para longe do corpo (pegando item na prateleira)."""
    return y_n > geo.reach_y_min or abs(x_n) > geo.reach_x_min
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `.venv\Scripts\python.exe -m pytest tests/detection/test_body_frame.py -v`
Expected: PASS — 15 passed

- [ ] **Step 5: Commit**

```bash
git add src/detection/body_frame.py tests/detection/test_body_frame.py
git commit -m "feat: sistema de coordenadas do corpo e classificacao de zonas de ocultacao"
```

---

## Task 2: Sinais sobre a janela deslizante

**Files:**
- Create: `src/detection/signals.py`
- Test: `tests/detection/test_signals.py`

**Interfaces:**
- Consumes: `DetectionConfig`, `BodyFrame`, `classify_zone`, `in_reach`.
- Produces:
  - `WristHistory(fps_hint=5.0)` acumula observações de UM punho.
  - `.observe(x_n, y_n, conf, zone, reach, ts)` registra um frame.
  - `.prune(now, window_seconds)` descarta observações fora da janela.
  - `compute_signals(hist, cfg, now) -> Signals` com `.dwell, .approach, .vanish, .retract` (cada 0..1) e `.zone` (a zona corrente, ou None).

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/detection/test_signals.py`:

```python
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
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `.venv\Scripts\python.exe -m pytest tests/detection/test_signals.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.detection.signals'`

- [ ] **Step 3: Implementar `src/detection/signals.py`**

```python
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
    streak = 0.0
    best_streak = 0.0
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
            best_streak = max(best_streak, streak)
        else:
            gap += 1
            if gap > gap_allow:
                streak = 0.0
                active_zone = None
    dwell = min(1.0, best_streak / dwell_frames_target)

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
    if best_streak >= half_dwell:
        recent = [o for o in obs if o.conf >= g.kp_conf_min and o.ts >= now - 1.0]
        if len(recent) >= 2:
            dy = recent[-1].y_n - recent[0].y_n
            if dy > 0.3:
                retract = min(1.0, dy / 0.6)

    return Signals(dwell, approach, vanish, retract, cur_zone or active_zone)
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `.venv\Scripts\python.exe -m pytest tests/detection/test_signals.py -v`
Expected: PASS — 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/detection/signals.py tests/detection/test_signals.py
git commit -m "feat: os quatro sinais de ocultacao (dwell, approach, vanish, retract)"
```

---

## Task 3: Analisador de ocultação (score + máquina de estados + evento)

**Files:**
- Create: `src/detection/concealment.py`
- Test: `tests/detection/test_concealment.py`

**Interfaces:**
- Consumes: `DetectionConfig`, `PersonPose`, `ObjectDetection`, `BodyFrame`, `WristHistory`, `compute_signals`, `classify_zone`, `in_reach`, `KP`.
- Produces:
  - `ConcealmentEvent(track_id, score, zone, signals: dict, ts)`.
  - `ConcealmentAnalyzer(cfg: DetectionConfig, fps_hint=5.0)`.
  - `.update(poses: list[PersonPose], objects: list[ObjectDetection], ts: float) -> list[ConcealmentEvent]` — processa um frame de UMA câmera; devolve eventos disparados neste frame.
  - Estados internos por track: `IDLE, APPROACHING, CONCEALING, ALERT, COOLDOWN`.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/detection/test_concealment.py`:

```python
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
    """Mão vem da prateleira, desce ao bolso e fica lá > dwell → dispara."""
    a = ConcealmentAnalyzer(DetectionConfig(), fps_hint=FPS)
    frames = [((160, 130), 0.9)] * 3          # reach (braço estendido, alto/lateral)
    frames += [((108, 210), 0.9)] * 10        # punho na cintura/bolso, permanece
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
    """Mão vai ao tórax e o punho SOME (mão sob a blusa) → vanish sustenta o score."""
    a = ConcealmentAnalyzer(DetectionConfig(), fps_hint=FPS)
    frames = [((160, 130), 0.9)] * 2          # reach
    frames += [((108, 150), 0.9)] * 3         # mão no tórax, visível
    frames += [((108, 150), 0.05)] * 6        # punho some DENTRO da zona (conf ~0)
    events = _run(a, frames)
    assert len(events) >= 1
    assert events[0].signals["vanish"] > 0.5


def test_cooldown_prevents_duplicate_alerts():
    a = ConcealmentAnalyzer(DetectionConfig(cooldown_seconds=30.0), fps_hint=FPS)
    frames = [((160, 130), 0.9)] * 3 + [((108, 210), 0.9)] * 10
    frames += [((108, 210), 0.9)] * 10        # continua na zona logo depois
    events = _run(a, frames)
    assert len(events) == 1  # o cooldown segura o segundo


def test_small_person_is_ignored():
    """Pessoa menor que min_person_px → pose não confiável, não avalia."""
    cfg = DetectionConfig()
    cfg.guards.min_person_px = 500  # força o descarte
    a = ConcealmentAnalyzer(cfg, fps_hint=FPS)
    frames = [((160, 130), 0.9)] * 3 + [((108, 210), 0.9)] * 10
    assert _run(a, frames) == []


def test_low_pose_quality_is_ignored():
    cfg = DetectionConfig()
    cfg.guards.pose_quality_min = 0.99  # quase nada passa
    a = ConcealmentAnalyzer(cfg, fps_hint=FPS)
    frames = [((160, 130), 0.9)] * 3 + [((108, 210), 0.9)] * 10
    assert _run(a, frames) == []


def test_bag_zone_detects_hand_in_backpack():
    """Punho dentro da bbox de uma mochila associada à pessoa → zona 'bag'."""
    a = ConcealmentAnalyzer(DetectionConfig(), fps_hint=FPS)
    bag = ObjectDetection(label="backpack", bbox=BBox(120, 150, 170, 220), conf=0.8)
    events = []
    t = 0.0
    # reach primeiro
    for _ in range(2):
        events += a.update([_pose(1, (160, 120))], [bag], t); t += DT
    # mão entra na bbox da mochila e permanece
    for _ in range(10):
        events += a.update([_pose(1, (145, 185))], [bag], t); t += DT
    assert any(e.zone == "bag" for e in events)


def test_per_track_state_isolation():
    """Duas pessoas: só a que faz o gesto dispara."""
    a = ConcealmentAnalyzer(DetectionConfig(), fps_hint=FPS)
    events = []
    t = 0.0
    for i in range(13):
        wrist_ladrao = (108, 210) if i >= 3 else (160, 130)
        p_ladrao = _pose(1, wrist_ladrao)
        p_inocente = _pose(2, (160, 120), bbox=BBox(300, 60, 340, 300))
        events += a.update([p_ladrao, p_inocente], [], t)
        t += DT
    assert {e.track_id for e in events} == {1}
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `.venv\Scripts\python.exe -m pytest tests/detection/test_concealment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.detection.concealment'`

- [ ] **Step 3: Implementar `src/detection/concealment.py`**

```python
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
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `.venv\Scripts\python.exe -m pytest tests/detection/test_concealment.py -v`
Expected: PASS — 8 passed. Se algum teste de gesto não disparar, ajustar as POSIÇÕES sintéticas dos punhos (não os limiares do config) até o gesto óbvio disparar e o gesto inocente não — os defaults do config são o alvo da calibração real, mas os testes de lógica devem passar com eles.

- [ ] **Step 5: Rodar a suíte inteira**

Run: `.venv\Scripts\python.exe -m pytest -q -m "not slow"`
Expected: PASS — 129 (Plano 1) + 34 novos, zero regressões.

- [ ] **Step 6: Commit**

```bash
git add src/detection/concealment.py tests/detection/test_concealment.py
git commit -m "feat: analisador de ocultacao (score + maquina de estados + evento)"
```

---

## Task 4: Plugar a ocultação no Pipeline

**Files:**
- Modify: `src/pipeline.py` (adicionar um `ConcealmentAnalyzer` por câmera e emitir eventos)
- Test: `tests/test_pipeline_concealment.py`

**Interfaces:**
- `FrameResult` ganha campo `events: list[ConcealmentEvent] = field(default_factory=list)`.
- `Pipeline` cria um `ConcealmentAnalyzer` por câmera (como faz com `Tracker`), chama `.update(poses, objects, ts)` em `process_frame`, e anexa os eventos ao `FrameResult`. O callback `on_result` passa a receber os eventos.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_pipeline_concealment.py`:

```python
import numpy as np

from src.config.settings import AppConfig, CameraConfig, StoreConfig
from src.core.types import BBox, Frame, KP, PersonDetection
from src.pipeline import Pipeline


class ScriptedEngine:
    """Engine dublê: devolve uma pessoa cujo punho segue um roteiro."""

    def __init__(self, wrist_script):
        self.wrist_script = wrist_script
        self.i = 0

    def detect(self, image):
        return [PersonDetection(bbox=BBox(80, 60, 120, 300), conf=0.9)], []

    def pose(self, image, boxes):
        wx, wy, wc = self.wrist_script[min(self.i, len(self.wrist_script) - 1)]
        self.i += 1
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[KP["left_shoulder"]] = [90, 100, 0.9]
        kp[KP["right_shoulder"]] = [110, 100, 0.9]
        kp[KP["left_hip"]] = [92, 200, 0.9]
        kp[KP["right_hip"]] = [108, 200, 0.9]
        kp[KP["nose"]] = [100, 80, 0.9]
        kp[KP["left_eye"]] = [96, 78, 0.9]
        kp[KP["right_eye"]] = [104, 78, 0.9]
        kp[KP["right_wrist"]] = [wx, wy, wc]
        return [kp]

    def warmup(self):
        pass


def _cfg():
    return AppConfig(store=StoreConfig(id="l", name="L"),
                     cameras=[CameraConfig(name="cam1", rtsp_url="rtsp://x", target_fps=5, zones=[])])


def test_pipeline_emits_concealment_event():
    script = [(160, 130, 0.9)] * 3 + [(108, 210, 0.9)] * 12
    p = Pipeline(_cfg(), ScriptedEngine(script))
    all_events = []
    t = 0.0
    for _ in range(len(script)):
        r = p.process_frame(Frame("cam1", np.zeros((360, 200, 3), np.uint8), t, 1))
        all_events.extend(r.events)
        t += 0.2
    assert len(all_events) >= 1
    assert all_events[0].zone in ("waist", "torso")


def test_pipeline_no_event_for_normal_movement():
    script = [(160, 120, 0.9)] * 15  # mão sempre longe do corpo
    p = Pipeline(_cfg(), ScriptedEngine(script))
    t = 0.0
    events = []
    for _ in range(len(script)):
        r = p.process_frame(Frame("cam1", np.zeros((360, 200, 3), np.uint8), t, 1))
        events.extend(r.events)
        t += 0.2
    assert events == []
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pipeline_concealment.py -v`
Expected: FAIL — `AttributeError: 'FrameResult' object has no attribute 'events'`

- [ ] **Step 3: Modificar `src/pipeline.py`**

No topo, adicionar import:
```python
from src.detection.concealment import ConcealmentAnalyzer, ConcealmentEvent
```

Em `FrameResult`, adicionar o campo:
```python
    events: list[ConcealmentEvent] = field(default_factory=list)
```

No `Pipeline.__init__`, criar um analisador por câmera (ao lado dos `_trackers`):
```python
        self._analyzers: dict[str, ConcealmentAnalyzer] = {
            c.name: ConcealmentAnalyzer(
                c.effective_detection(cfg.detection),
                fps_hint=c.target_fps,
            )
            for c in self.cameras
        }
```

Em `process_frame`, no caminho COM pessoa, depois de montar `poses`, chamar o analisador e anexar os eventos:
```python
        poses = [PersonPose(person=p, keypoints=k) for p, k in zip(tracked, keypoints)]
        events = self._analyzers[frame.camera_name].update(poses, objects, frame.ts)
        return FrameResult(frame.camera_name, poses, objects, had_person=True, events=events)
```

E no caminho SEM pessoa, ainda chamar o analisador com lista vazia (para o cooldown/track_lost avançarem no tempo):
```python
            self._trackers[frame.camera_name].update([], frame.ts)
            self._analyzers[frame.camera_name].update([], [], frame.ts)
            return FrameResult(frame.camera_name, had_person=False)
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pipeline_concealment.py tests/test_pipeline.py -v`
Expected: PASS — os 2 novos + os 6 do Plano 1 (zero regressão no pipeline).

- [ ] **Step 5: Rodar a suíte inteira**

Run: `.venv\Scripts\python.exe -m pytest -q -m "not slow"`
Expected: PASS — zero regressões.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline.py tests/test_pipeline_concealment.py
git commit -m "feat: pipeline emite eventos de ocultacao por camera"
```

---

## Task 5: Modo replay (rodar sobre arquivo de vídeo)

**Files:**
- Create: `src/tools/replay.py`
- Test: `tests/tools/test_replay.py`

**Interfaces:**
- Consumes: `AppConfig`/`DetectionConfig`, `InferenceEngine`, `Tracker`, `ConcealmentAnalyzer`, cv2.
- Produces:
  - `replay(video_path, cfg, engine, out_video=None, out_csv=None, every=1) -> ReplaySummary`.
  - `ReplaySummary(frames, frames_with_person, events: list[ConcealmentEvent], csv_rows)`.
  - CSV com uma linha por frame processado: `frame, ts, n_persons, max_score, zone, dwell, approach, vanish, retract`.
  - Vídeo anotado: esqueleto + score em tempo real + marca vermelha quando dispara.
  - CLI: `python -m src.tools.replay <video> --config config/config.json --out-video x.mp4 --out-csv x.csv`.

**Por que existe:** valida a detecção sem loja e é como se calibra em cima do vídeo real do cliente. Quando o material do João Lucas chega, a calibração vira medição.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/tools/test_replay.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from src.config.settings import DetectionConfig
from src.core.types import BBox, KP, PersonDetection
from src.tools.replay import replay


class ScriptedEngine:
    def __init__(self, script):
        self.script = script
        self.i = 0

    def detect(self, image):
        return [PersonDetection(bbox=BBox(80, 60, 120, 300), conf=0.9)], []

    def pose(self, image, boxes):
        wx, wy, wc = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        kp = np.zeros((17, 3), dtype=np.float32)
        for name, xy in (("left_shoulder", (90, 100)), ("right_shoulder", (110, 100)),
                         ("left_hip", (92, 200)), ("right_hip", (108, 200)),
                         ("nose", (100, 80)), ("left_eye", (96, 78)), ("right_eye", (104, 78))):
            kp[KP[name]] = [xy[0], xy[1], 0.9]
        kp[KP["right_wrist"]] = [wx, wy, wc]
        return [kp]

    def warmup(self):
        pass


def _make_video(path, n=15, size=(200, 360)):
    import cv2
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
    for _ in range(n):
        vw.write(np.zeros((h, w, 3), dtype=np.uint8))
    vw.release()


def test_replay_detects_event_and_writes_outputs(tmp_path):
    video = tmp_path / "in.mp4"
    _make_video(video, n=15)
    script = [(160, 130, 0.9)] * 3 + [(108, 210, 0.9)] * 12
    out_csv = tmp_path / "out.csv"
    out_video = tmp_path / "out.mp4"

    summary = replay(video, DetectionConfig(), ScriptedEngine(script),
                     out_video=out_video, out_csv=out_csv)

    assert summary.frames == 15
    assert summary.frames_with_person == 15
    assert len(summary.events) >= 1
    assert out_csv.exists()
    assert out_video.exists()
    # o CSV tem cabeçalho + uma linha por frame
    lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("frame,ts,")
    assert len(lines) == 16


def test_replay_no_event_for_normal(tmp_path):
    video = tmp_path / "in.mp4"
    _make_video(video, n=12)
    script = [(160, 120, 0.9)] * 12
    summary = replay(video, DetectionConfig(), ScriptedEngine(script))
    assert summary.events == []
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `.venv\Scripts\python.exe -m pytest tests/tools/test_replay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tools.replay'`

- [ ] **Step 3: Implementar `src/tools/replay.py`**

```python
"""Roda o pipeline de detecção + ocultação sobre um arquivo de vídeo, como se
fosse uma câmera ao vivo. Produz vídeo anotado (esqueleto + score) e um CSV com
o score por frame. É a ferramenta de validação e calibração sem loja."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.config.settings import AppConfig, DetectionConfig
from src.core.types import KP
from src.detection.concealment import ConcealmentAnalyzer, ConcealmentEvent
from src.detection.tracker import Tracker

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
WRISTS = (KP["left_wrist"], KP["right_wrist"])


@dataclass
class ReplaySummary:
    frames: int
    frames_with_person: int
    events: list[ConcealmentEvent] = field(default_factory=list)
    csv_rows: list[dict] = field(default_factory=list)


def _annotate(frame, poses, tracked, score, fired):
    for pose, person in zip(poses, tracked):
        b = person.bbox
        cor = (0, 0, 255) if fired else (0, 200, 0)
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), cor, 2)
        kp = pose.keypoints
        for a, c in EDGES:
            if kp[a, 2] > 0.35 and kp[c, 2] > 0.35:
                cv2.line(frame, (int(kp[a, 0]), int(kp[a, 1])),
                         (int(kp[c, 0]), int(kp[c, 1])), (255, 180, 0), 2)
        for wi in WRISTS:
            if kp[wi, 2] > 0.35:
                cv2.circle(frame, (int(kp[wi, 0]), int(kp[wi, 1])), 6, (0, 0, 255), -1)
    cv2.putText(frame, f"score {score:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255) if fired else (255, 255, 255), 2)
    if fired:
        cv2.putText(frame, "OCULTACAO", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return frame


def replay(video_path, cfg: DetectionConfig, engine, out_video=None, out_csv=None, every=1):
    from src.core.types import PersonPose

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    engine.warmup()
    tracker = Tracker(max_lost_seconds=cfg.guards.track_lost_seconds)
    analyzer = ConcealmentAnalyzer(cfg, fps_hint=fps / every)

    writer = None
    if out_video:
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                 max(1.0, fps / every), (w, h))

    summary = ReplaySummary(0, 0)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every != 0:
            idx += 1
            continue
        ts = idx / fps
        idx += 1
        summary.frames += 1

        persons, objects = engine.detect(frame)
        tracked = tracker.update(persons, ts)
        max_score, top = 0.0, None
        events = []
        poses = []
        if tracked:
            summary.frames_with_person += 1
            kps = engine.pose(frame, [p.bbox for p in tracked])
            poses = [PersonPose(person=p, keypoints=k) for p, k in zip(tracked, kps)]
            events = analyzer.update(poses, objects, ts)
            for e in events:
                if e.score > max_score:
                    max_score, top = e.score, e
        else:
            analyzer.update([], [], ts)

        summary.events.extend(events)
        row = {
            "frame": summary.frames, "ts": round(ts, 3), "n_persons": len(tracked),
            "max_score": round(max_score, 3),
            "zone": top.zone if top else "",
            "dwell": top.signals["dwell"] if top else "",
            "approach": top.signals["approach"] if top else "",
            "vanish": top.signals["vanish"] if top else "",
            "retract": top.signals["retract"] if top else "",
        }
        summary.csv_rows.append(row)
        if writer is not None:
            writer.write(_annotate(frame, poses, tracked, max_score, fired=bool(events)))

    cap.release()
    if writer is not None:
        writer.release()
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(summary.csv_rows[0].keys()) if summary.csv_rows
                                else ["frame", "ts", "n_persons", "max_score", "zone",
                                      "dwell", "approach", "vanish", "retract"])
            wr.writeheader()
            wr.writerows(summary.csv_rows)
    return summary


def main():
    from src.inference.engine import InferenceEngine

    ap = argparse.ArgumentParser(description="Replay do detector sobre um video")
    ap.add_argument("video")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--out-video", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()

    app = AppConfig.load(a.config)
    engine = InferenceEngine(app.inference)
    summary = replay(a.video, app.detection, engine,
                     out_video=a.out_video, out_csv=a.out_csv, every=a.every)
    print(f"frames: {summary.frames} | com pessoa: {summary.frames_with_person} | "
          f"eventos de ocultacao: {len(summary.events)}")
    for e in summary.events:
        print(f"  t={e.ts:.1f}s id={e.track_id} zona={e.zone} score={e.score} {e.signals}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `.venv\Scripts\python.exe -m pytest tests/tools/test_replay.py -v`
Expected: PASS — 2 passed

- [ ] **Step 5: Rodar o replay REAL sobre o footage do cliente**

```bash
python -m src.tools.replay "c:/Workana/material-cliente/videos/VID-WA0099.mp4" \
  --config config/config.piloto.json --out-video evidencia_ch11.mp4 --out-csv ch11.csv --every 3
```
Expected: roda sem erro, imprime nº de frames/pessoas/eventos. Como o vídeo é movimento normal, o esperado é **poucos ou zero** eventos (é o material de falso-positivo). Guardar o vídeo anotado — é o que se mostra ao cliente.

- [ ] **Step 6: Commit**

```bash
git add src/tools/replay.py tests/tools/test_replay.py
git commit -m "feat: modo replay (detector sobre video, anota score e exporta CSV)"
```

---

## Task 6: Sweep de calibração

**Files:**
- Create: `src/tools/calibrate.py`
- Test: `tests/tools/test_calibrate.py`

**Interfaces:**
- Consumes: `replay`, `DetectionConfig`, `InferenceEngine`.
- Produces:
  - `sweep(conceal_dir, normal_dir, engine, grid, base_cfg) -> list[SweepRow]`.
  - `SweepRow(threshold, dwell_seconds, detected, total_conceal, false_per_hour)`.
  - Para cada combinação da grade, roda o replay em cada clipe de `conceal_dir` (conta quantos dispararam ≥1 evento = detecção) e de `normal_dir` (conta eventos = falsos), e normaliza os falsos por hora de vídeo normal.
  - `best_row(rows, max_false_per_hour) -> SweepRow` escolhe a linha que respeita o teto de falsos e maximiza a detecção.
  - CLI: `python -m src.tools.calibrate --conceal dir --normal dir --config ...`.

**Por que existe:** converte a pergunta do João Lucas ("quantos alertas falsos por dia a equipe aguenta?") na escolha objetiva dos parâmetros. Calibração vira medição.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/tools/test_calibrate.py`:

```python
from src.config.settings import DetectionConfig
from src.tools.calibrate import SweepRow, best_row


def test_best_row_respects_false_ceiling():
    rows = [
        SweepRow(threshold=0.5, dwell_seconds=1.0, detected=9, total_conceal=10, false_per_hour=12.0),
        SweepRow(threshold=0.6, dwell_seconds=1.2, detected=8, total_conceal=10, false_per_hour=4.0),
        SweepRow(threshold=0.7, dwell_seconds=1.5, detected=5, total_conceal=10, false_per_hour=1.0),
    ]
    # teto de 5 falsos/hora → a de threshold 0.5 (12/h) é descartada
    best = best_row(rows, max_false_per_hour=5.0)
    assert best.threshold == 0.6
    assert best.detected == 8


def test_best_row_returns_least_false_when_none_meet_ceiling():
    rows = [
        SweepRow(threshold=0.5, dwell_seconds=1.0, detected=10, total_conceal=10, false_per_hour=30.0),
        SweepRow(threshold=0.9, dwell_seconds=2.0, detected=3, total_conceal=10, false_per_hour=20.0),
    ]
    best = best_row(rows, max_false_per_hour=5.0)
    assert best.false_per_hour == 20.0  # a menos ruim
```

Criar também um teste `@pytest.mark.slow` que exercita `sweep` sobre 1 clipe sintético em cada pasta (usa a mesma `ScriptedEngine` do replay via um pequeno helper) — opcional, mas recomendado para garantir que a montagem de pastas funciona.

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `.venv\Scripts\python.exe -m pytest tests/tools/test_calibrate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tools.calibrate'`

- [ ] **Step 3: Implementar `src/tools/calibrate.py`**

```python
"""Sweep de calibração (spec §8): varre combinações de parâmetros sobre uma
pasta de clipes ROTULADOS e mede detecção × falso-positivo.

    videos/ocultacao/  -> clipes onde DEVE disparar
    videos/normal/     -> movimento comum onde NÃO deve disparar

Produz a tabela que responde 'quantos alertas falsos por dia a equipe aguenta':
escolhe-se a linha que respeita o teto do cliente e maximiza a detecção."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import DetectionConfig
from src.tools.replay import replay


@dataclass
class SweepRow:
    threshold: float
    dwell_seconds: float
    detected: int
    total_conceal: int
    false_per_hour: float


def _clips(d: Path) -> list[Path]:
    return sorted(p for p in Path(d).glob("*.mp4"))


def sweep(conceal_dir, normal_dir, engine, grid, base_cfg: DetectionConfig, every=2) -> list[SweepRow]:
    conceal = _clips(conceal_dir)
    normal = _clips(normal_dir)
    rows: list[SweepRow] = []

    for threshold, dwell in grid:
        cfg = copy.deepcopy(base_cfg)
        cfg.threshold = threshold
        cfg.dwell_seconds = dwell

        detected = 0
        for clip in conceal:
            s = replay(clip, cfg, engine, every=every)
            if s.events:
                detected += 1

        false_events = 0
        normal_seconds = 0.0
        for clip in normal:
            s = replay(clip, cfg, engine, every=every)
            false_events += len(s.events)
            if s.csv_rows:
                normal_seconds += s.csv_rows[-1]["ts"]
        fph = (false_events / normal_seconds * 3600.0) if normal_seconds > 0 else 0.0

        rows.append(SweepRow(threshold, dwell, detected, len(conceal), round(fph, 2)))
    return rows


def best_row(rows: list[SweepRow], max_false_per_hour: float) -> SweepRow:
    ok = [r for r in rows if r.false_per_hour <= max_false_per_hour]
    if ok:
        return max(ok, key=lambda r: (r.detected, -r.false_per_hour))
    return min(rows, key=lambda r: r.false_per_hour)


def format_table(rows: list[SweepRow]) -> str:
    out = [f"{'limiar':>7} {'dwell':>6} {'detectados':>11} {'falsos/hora':>12}"]
    out.append("-" * 40)
    for r in rows:
        out.append(f"{r.threshold:>7.2f} {r.dwell_seconds:>6.1f} "
                   f"{r.detected:>4}/{r.total_conceal:<4} {r.false_per_hour:>12.1f}")
    return "\n".join(out)


def main():
    from src.config.settings import AppConfig
    from src.inference.engine import InferenceEngine

    ap = argparse.ArgumentParser(description="Sweep de calibracao")
    ap.add_argument("--conceal", required=True)
    ap.add_argument("--normal", required=True)
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--max-false-per-hour", type=float, default=5.0)
    a = ap.parse_args()

    app = AppConfig.load(a.config)
    engine = InferenceEngine(app.inference)
    grid = [(t, d) for t in (0.5, 0.6, 0.7) for d in (1.0, 1.2, 1.5)]
    rows = sweep(a.conceal, a.normal, engine, grid, app.detection)
    print(format_table(rows))
    best = best_row(rows, a.max_false_per_hour)
    print(f"\nRecomendado (teto {a.max_false_per_hour}/h): limiar {best.threshold}, "
          f"dwell {best.dwell_seconds}s -> pega {best.detected}/{best.total_conceal}, "
          f"{best.false_per_hour} falsos/hora")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `.venv\Scripts\python.exe -m pytest tests/tools/test_calibrate.py -v`
Expected: PASS — 2 passed (+ o slow, se escrito).

- [ ] **Step 5: Commit**

```bash
git add src/tools/calibrate.py tests/tools/test_calibrate.py
git commit -m "feat: sweep de calibracao (deteccao x falso-positivo sobre clipes rotulados)"
```

---

## Fechamento do Plano 2 (marco de 50%)

- [ ] **Suíte completa**

Run: `.venv\Scripts\python.exe -m pytest -q -m "not slow"`
Expected: PASS — Plano 1 (129) + Plano 2 (~57) sem regressões.

- [ ] **Demonstração real (o que se mostra no marco de 50%)**

1. Gravar 4–6 encenações pela webcam: `python dev/record_clips.py --label bolso --seconds 8` (e `bolsa`, `roupa`, `cintura`).
2. Rodar o replay sobre uma delas e confirmar que **dispara**, com vídeo anotado:
   `python -m src.tools.replay dev/videos/ocultacao/bolso_01.mp4 --config config/config.json --out-video demo.mp4`
3. Rodar o replay sobre o footage real do cliente (movimento normal) e confirmar que **quase não dispara** (falso-positivo baixo).
4. Rodar o sweep sobre as duas pastas e gerar a tabela detecção × falso/hora.

- [ ] **Revisão final da branch** (subagent-driven-development exige)

Gerar o pacote da branch inteira e despachar o revisor final no modelo mais capaz, com foco em: a heurística dispara nos gestos óbvios e NÃO nos inocentes? o núcleo continua puro? o replay/sweep casam com o `ConcealmentAnalyzer` real?

- [ ] **Merge para `master`** e seguir para o Plano 3 (evidência + Telegram + watchdog + UI + instalador).

**Estado ao fim deste plano:** o sistema detecta comportamento de ocultação em vídeo real, com score calibrável e ferramenta de medição de falso-positivo. Isto é o **marco de 50%**: "câmeras capturando + detecção de comportamento funcionando".
