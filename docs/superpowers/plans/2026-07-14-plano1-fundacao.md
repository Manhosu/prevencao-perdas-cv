# Plano 1 — Fundação: Captura e Inferência

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entregar o esqueleto do sistema rodando ponta a ponta: várias câmeras RTSP capturando com reconexão automática, YOLO detectando pessoas dentro das zonas configuradas, pose e tracking funcionando, e um teste de capacidade que diz quantas câmeras o PC aguenta.

**Architecture:** Processo único multi-thread. Uma `CameraThread` por câmera lê RTSP e publica o frame mais recente num slot de tamanho 1 (frame velho é descartado). Um pool de workers consome esses slots via um escalonador que prioriza câmeras com pessoa recente, roda o gate de pessoa (YOLO detect) e, só quando há alguém dentro da zona, roda pose no recorte da pessoa e mantém o `track_id` com ByteTrack. O núcleo (`detection/`) não conhece RTSP nem banco — recebe frame e config, devolve dados.

**Tech Stack:** Python 3.11, OpenCV, Ultralytics (YOLO11n + YOLO11n-pose), OpenVINO, pydantic v2, SQLite, pytest, MediaMTX (DVR simulado), imageio-ffmpeg.

## Global Constraints

- **Python 3.11+**, Windows como plataforma alvo. Todo caminho de arquivo via `pathlib`.
- **Nenhum parâmetro de detecção hardcoded.** Tudo vem de `config.json` validado por pydantic, com override por câmera. Esta é a restrição central do projeto: quando o vídeo real do cliente chegar, calibrar deve ser mexer em número, nunca em código.
- **`src/detection/` é núcleo puro:** não importa `cv2.VideoCapture`, `requests`, `sqlite3` nem PySide6. Recebe frame + config, devolve dataclasses.
- **Coordenadas de zona são normalizadas (0–1)** sobre o quadro, nunca em pixels — sobrevivem a troca de resolução/substream.
- **Modelos:** `yolo11n.pt` (detect) e `yolo11n-pose.pt` (pose). Keypoints no formato COCO-17.
- **Testes que exigem download de modelo ou binário externo** são marcados `@pytest.mark.slow` e ficam fora do ciclo rápido (`pytest -m "not slow"`).
- Mensagens de log e de UI em **português**; nomes de código em **inglês**.
- Commits em português, prefixo convencional (`feat:`, `test:`, `fix:`, `chore:`).

---

## Estrutura de arquivos deste plano

| Arquivo | Responsabilidade |
|---|---|
| `requirements.txt` | dependências |
| `pytest.ini` | config de testes e marcadores |
| `src/core/types.py` | dataclasses compartilhadas (Frame, BBox, PersonDetection, PersonPose…) |
| `src/config/settings.py` | modelos pydantic, carga/validação, merge de overrides por câmera |
| `config/config.example.json` | configuração de exemplo |
| `src/storage/db.py` | SQLite: schema, eventos, status de câmera, purga |
| `dev/dvr_sim.py` | DVR simulado (MediaMTX + ffmpeg servindo vídeos em loop) |
| `dev/record_clips.py` | grava material de teste próprio pela webcam |
| `src/capture/frame_slot.py` | `LatestFrameSlot` — slot thread-safe de 1 frame |
| `src/capture/rtsp_capture.py` | `CameraThread` — RTSP, amostragem, reconexão, heartbeat |
| `src/inference/engine.py` | carga de modelos, export OpenVINO, `detect()` e `pose()` |
| `src/detection/person_gate.py` | pessoa está dentro do polígono da zona? |
| `src/detection/pose_estimator.py` | keypoints no recorte da pessoa |
| `src/detection/tracker.py` | ByteTrack → `track_id` estável |
| `src/inference/scheduler.py` | round-robin ponderado por atividade recente |
| `src/inference/worker_pool.py` | pool de workers de inferência |
| `src/main.py` | orquestração headless |
| `src/tools/benchmark.py` | teste de capacidade do PC |

---

## Task 1: Scaffold e tipos do núcleo

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `.gitignore`
- Create: `src/core/types.py`
- Test: `tests/core/test_types.py`

**Interfaces:**
- Produces: `BBox(x1,y1,x2,y2)` com `.width`, `.height`, `.center`, `.foot_point`, `.expand(f)`, `.contains(x,y)`; `Frame(camera_name, image, ts, seq)`; `PersonDetection(bbox, conf, track_id)`; `ObjectDetection(label, bbox, conf)`; `PersonPose(person, keypoints)`; `CameraState` enum; dicionário `KP` com os índices COCO-17.

- [ ] **Step 1: Criar `requirements.txt`**

```
ultralytics==8.3.*
opencv-python==4.10.*
numpy<2.2
openvino==2024.*
onnxruntime==1.20.*
pydantic==2.*
requests==2.32.*
PySide6==6.8.*
imageio-ffmpeg==0.5.*
psutil==6.*
pytest==8.*
pytest-timeout==2.*
```

- [ ] **Step 2: Criar `pytest.ini`**

```ini
[pytest]
testpaths = tests
pythonpath = .
timeout = 120
markers =
    slow: exige download de modelo ou binário externo (rodar com -m slow)
    rtsp: exige o DVR simulado (MediaMTX) em execução
```

- [ ] **Step 3: Criar `.gitignore`**

```
.venv/
__pycache__/
*.pyc
models/*.pt
models/*_openvino_model/
models/*.onnx
config/config.json
data/
evidence/
logs/
dev/bin/
dev/videos/
.pytest_cache/
```

- [ ] **Step 4: Escrever o teste que falha**

Criar `tests/core/test_types.py`:

```python
import numpy as np
import pytest

from src.core.types import BBox, CameraState, Frame, PersonDetection, PersonPose, KP


def test_bbox_geometry():
    b = BBox(10, 20, 110, 220)
    assert b.width == 100
    assert b.height == 200
    assert b.center == (60, 120)
    assert b.foot_point == (60, 220)


def test_bbox_contains():
    b = BBox(0, 0, 10, 10)
    assert b.contains(5, 5)
    assert not b.contains(11, 5)


def test_bbox_expand_grows_around_center():
    b = BBox(10, 10, 20, 20).expand(0.2)
    assert b.x1 == pytest.approx(9.0)
    assert b.x2 == pytest.approx(21.0)
    assert b.y1 == pytest.approx(9.0)
    assert b.y2 == pytest.approx(21.0)


def test_frame_holds_image_and_sequence():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    f = Frame(camera_name="cam1", image=img, ts=1.0, seq=7)
    assert f.camera_name == "cam1"
    assert f.seq == 7
    assert f.image.shape == (4, 4, 3)


def test_person_pose_keypoints_shape():
    kps = np.zeros((17, 3), dtype=np.float32)
    p = PersonPose(person=PersonDetection(bbox=BBox(0, 0, 1, 1), conf=0.9), keypoints=kps)
    assert p.keypoints.shape == (17, 3)
    assert p.person.track_id is None


def test_keypoint_index_map_is_coco17():
    assert KP["left_wrist"] == 9
    assert KP["right_wrist"] == 10
    assert KP["left_hip"] == 11
    assert KP["right_hip"] == 12
    assert KP["left_shoulder"] == 5
    assert len(KP) == 17


def test_camera_state_values():
    assert CameraState.ONLINE.value == "online"
    assert CameraState.OFFLINE.value == "offline"
    assert CameraState.RECONNECTING.value == "reconnecting"
```

- [ ] **Step 5: Rodar o teste e confirmar que falha**

Run: `pytest tests/core/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.types'`

- [ ] **Step 6: Implementar `src/core/types.py`**

Criar também os `__init__.py` vazios em `src/`, `src/core/`, `tests/`, `tests/core/`.

```python
"""Tipos compartilhados por todo o pipeline. Sem dependências de I/O."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# Índices dos keypoints no formato COCO-17 (padrão do YOLO-pose).
KP: dict[str, int] = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


@dataclass(frozen=True)
class BBox:
    """Caixa em pixels do frame completo."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Base da caixa: onde a pessoa toca o chão. É este ponto que decide
        se ela está dentro da zona monitorada."""
        return ((self.x1 + self.x2) / 2, self.y2)

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def expand(self, factor: float) -> "BBox":
        """Cresce a caixa em `factor` (0.2 = +20%) mantendo o centro."""
        dx = self.width * factor / 2
        dy = self.height * factor / 2
        return BBox(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy)

    def clip(self, w: float, h: float) -> "BBox":
        return BBox(
            max(0.0, self.x1), max(0.0, self.y1), min(w, self.x2), min(h, self.y2)
        )


@dataclass
class Frame:
    camera_name: str
    image: np.ndarray  # BGR, HxWx3
    ts: float  # time.monotonic() do momento da captura
    seq: int


@dataclass
class PersonDetection:
    bbox: BBox
    conf: float
    track_id: int | None = None


@dataclass
class ObjectDetection:
    label: str  # 'backpack' | 'handbag'
    bbox: BBox
    conf: float


@dataclass
class PersonPose:
    person: PersonDetection
    keypoints: np.ndarray  # (17, 3) — x, y, conf em pixels do frame completo


class CameraState(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    RECONNECTING = "reconnecting"
```

- [ ] **Step 7: Rodar os testes e confirmar que passam**

Run: `pytest tests/core/test_types.py -v`
Expected: PASS — 6 passed

- [ ] **Step 8: Commit**

```bash
git add requirements.txt pytest.ini .gitignore src/ tests/
git commit -m "feat: scaffold do projeto e tipos do núcleo (BBox, Frame, PersonPose)"
```

---

## Task 2: Configuração (pydantic) com override por câmera

**Files:**
- Create: `src/config/settings.py`, `config/config.example.json`
- Test: `tests/config/test_settings.py`

**Interfaces:**
- Consumes: nada.
- Produces: `AppConfig.load(path) -> AppConfig`; campos `store`, `telegram`, `inference`, `detection`, `evidence`, `watchdog`, `cameras`. `CameraConfig.effective_detection(base: DetectionConfig) -> DetectionConfig` aplica os overrides. Classes: `StoreConfig`, `TelegramConfig`, `InferenceConfig`, `DetectionConfig`, `Weights`, `ZoneWeights`, `Geometry`, `Guards`, `EvidenceConfig`, `WatchdogConfig`, `CameraConfig`, `AppConfig`. Exceção `ConfigError`.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/config/test_settings.py`:

```python
import json

import pytest

from src.config.settings import AppConfig, ConfigError, DetectionConfig


def _minimal(tmp_path, **overrides):
    data = {
        "store": {"id": "loja1", "name": "Loja 1"},
        "telegram": {"bot_token": "t", "chat_id": "c"},
        "cameras": [
            {
                "name": "Caixa 01",
                "rtsp_url": "rtsp://user:pw@10.0.0.1:554/ch1",
                "zones": [[[0.2, 0.3], [0.8, 0.3], [0.8, 0.9], [0.2, 0.9]]],
            }
        ],
    }
    data.update(overrides)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_loads_with_defaults(tmp_path):
    cfg = AppConfig.load(_minimal(tmp_path))
    assert cfg.store.id == "loja1"
    assert cfg.detection.threshold == 0.60
    assert cfg.detection.dwell_seconds == 1.2
    assert cfg.detection.weights.vanish == 0.30
    assert cfg.cameras[0].target_fps == 5.0
    assert cfg.cameras[0].enabled is True


def test_camera_override_replaces_only_given_keys(tmp_path):
    cfg = AppConfig.load(_minimal(tmp_path))
    cfg.cameras[0].overrides = {"threshold": 0.75, "guards": {"min_person_px": 60}}
    eff = cfg.cameras[0].effective_detection(cfg.detection)
    assert eff.threshold == 0.75
    assert eff.guards.min_person_px == 60
    # o que não foi sobrescrito segue o padrão global
    assert eff.dwell_seconds == cfg.detection.dwell_seconds
    assert eff.guards.kp_conf_min == cfg.detection.guards.kp_conf_min
    # e o global não é mutado
    assert cfg.detection.threshold == 0.60


def test_rejects_unknown_override_key(tmp_path):
    cfg = AppConfig.load(_minimal(tmp_path))
    cfg.cameras[0].overrides = {"nao_existe": 1}
    with pytest.raises(ConfigError, match="nao_existe"):
        cfg.cameras[0].effective_detection(cfg.detection)


def test_rejects_zone_outside_unit_square(tmp_path):
    p = _minimal(tmp_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["cameras"][0]["zones"] = [[[0.2, 0.3], [1.4, 0.3], [0.8, 0.9]]]
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="normalizad"):
        AppConfig.load(p)


def test_rejects_zone_with_less_than_three_points(tmp_path):
    p = _minimal(tmp_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["cameras"][0]["zones"] = [[[0.2, 0.3], [0.8, 0.3]]]
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="3 pontos"):
        AppConfig.load(p)


def test_rejects_duplicate_camera_names(tmp_path):
    p = _minimal(tmp_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["cameras"].append(dict(data["cameras"][0]))
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicado"):
        AppConfig.load(p)


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="não encontrado"):
        AppConfig.load(tmp_path / "nao_existe.json")


def test_empty_zones_means_whole_frame(tmp_path):
    p = _minimal(tmp_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["cameras"][0]["zones"] = []
    p.write_text(json.dumps(data), encoding="utf-8")
    cfg = AppConfig.load(p)
    assert cfg.cameras[0].zones == []


def test_detection_defaults_match_spec():
    d = DetectionConfig()
    assert d.window_seconds == 3.0
    assert d.cooldown_seconds == 30.0
    assert d.geometry.waist_y == (-0.45, 0.25)
    assert d.geometry.torso_x_max == 0.55
    assert d.guards.vanish_max_seconds == 3.0
    assert d.zone_weights.back_waist == 1.05
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `pytest tests/config/test_settings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.config.settings'`

- [ ] **Step 3: Implementar `src/config/settings.py`**

```python
"""Configuração da aplicação. Toda constante de detecção mora aqui —
calibrar o sistema é editar JSON, nunca código."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ConfigError(Exception):
    """Erro de configuração legível para o usuário final (vai para a UI)."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StoreConfig(_Strict):
    id: str
    name: str


class TelegramConfig(_Strict):
    bot_token: str = ""
    chat_id: str = ""
    send_photo: bool = True
    send_clip: bool = True
    rate_limit_per_min: int = 15


class InferenceConfig(_Strict):
    device: str = "openvino"  # openvino | onnx | cpu
    person_model: str = "models/yolo11n.pt"
    pose_model: str = "models/yolo11n-pose.pt"
    detect_size: int = 640
    pose_on_crop: bool = True
    workers: int = 2
    detect_bags: bool = True


class Weights(_Strict):
    dwell: float = 0.40
    approach: float = 0.20
    vanish: float = 0.30
    retract: float = 0.10


class ZoneWeights(_Strict):
    waist: float = 1.00
    torso: float = 0.95
    back_waist: float = 1.05
    bag: float = 1.00


class Geometry(_Strict):
    waist_y: tuple[float, float] = (-0.45, 0.25)
    waist_x: tuple[float, float] = (0.10, 0.85)
    torso_y: tuple[float, float] = (0.15, 0.85)
    torso_x_max: float = 0.55
    reach_y_min: float = 0.9
    reach_x_min: float = 0.95


class Guards(_Strict):
    kp_conf_min: float = 0.35
    pose_quality_min: float = 0.40
    min_person_px: int = 120
    vanish_grace_seconds: float = 0.4
    vanish_max_seconds: float = 3.0
    gap_frames: int = 2
    track_lost_seconds: float = 2.0


class DetectionConfig(_Strict):
    threshold: float = 0.60
    dwell_seconds: float = 1.2
    window_seconds: float = 3.0
    cooldown_seconds: float = 30.0
    weights: Weights = Field(default_factory=Weights)
    zone_weights: ZoneWeights = Field(default_factory=ZoneWeights)
    geometry: Geometry = Field(default_factory=Geometry)
    guards: Guards = Field(default_factory=Guards)


class EvidenceConfig(_Strict):
    dir: str = "evidence"
    retention_days: int = 30
    clip_pre_seconds: float = 2.0
    clip_post_seconds: float = 4.0


class WatchdogConfig(_Strict):
    offline_after_seconds: float = 30.0
    notify: bool = True


Polygon = list[tuple[float, float]]


class CameraConfig(_Strict):
    name: str
    rtsp_url: str
    enabled: bool = True
    target_fps: float = 5.0
    zones: list[Polygon] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("zones")
    @classmethod
    def _check_zones(cls, zones: list[Polygon]) -> list[Polygon]:
        for poly in zones:
            if len(poly) < 3:
                raise ValueError("cada zona precisa de pelo menos 3 pontos")
            for x, y in poly:
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    raise ValueError(
                        "pontos da zona devem estar em coordenadas normalizadas (0 a 1)"
                    )
        return zones

    def effective_detection(self, base: DetectionConfig) -> DetectionConfig:
        """Aplica os overrides desta câmera sobre a configuração global.
        Câmera de teto e câmera lateral não compartilham o mesmo limiar —
        é isso que torna a calibração por câmera possível."""
        if not self.overrides:
            return base
        merged = _deep_merge(base.model_dump(), self.overrides)
        try:
            return DetectionConfig(**merged)
        except ValidationError as e:
            raise ConfigError(
                f"overrides inválidos na câmera '{self.name}': {e}"
            ) from e


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class AppConfig(_Strict):
    store: StoreConfig
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    watchdog: WatchdogConfig = Field(default_factory=WatchdogConfig)
    cameras: list[CameraConfig] = Field(default_factory=list)

    @field_validator("cameras")
    @classmethod
    def _unique_names(cls, cams: list[CameraConfig]) -> list[CameraConfig]:
        seen: set[str] = set()
        for c in cams:
            if c.name in seen:
                raise ValueError(f"nome de câmera duplicado: '{c.name}'")
            seen.add(c.name)
        return cams

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"arquivo de configuração não encontrado: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ConfigError(f"JSON inválido em {p}: {e}") from e
        try:
            return cls(**data)
        except ValidationError as e:
            raise ConfigError(f"configuração inválida em {p}:\n{e}") from e

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `pytest tests/config/test_settings.py -v`
Expected: PASS — 9 passed

- [ ] **Step 5: Criar `config/config.example.json`**

```json
{
  "store": { "id": "mercado-piloto", "name": "Mercado Piloto" },
  "telegram": {
    "bot_token": "COLOCAR_TOKEN_DO_BOTFATHER",
    "chat_id": "COLOCAR_ID_DO_GRUPO",
    "send_photo": true,
    "send_clip": true,
    "rate_limit_per_min": 15
  },
  "inference": {
    "device": "openvino",
    "person_model": "models/yolo11n.pt",
    "pose_model": "models/yolo11n-pose.pt",
    "detect_size": 640,
    "pose_on_crop": true,
    "workers": 2,
    "detect_bags": true
  },
  "detection": {
    "threshold": 0.6,
    "dwell_seconds": 1.2,
    "window_seconds": 3.0,
    "cooldown_seconds": 30.0
  },
  "evidence": { "dir": "evidence", "retention_days": 30 },
  "watchdog": { "offline_after_seconds": 30.0, "notify": true },
  "cameras": [
    {
      "name": "Corredor Bebidas",
      "rtsp_url": "rtsp://user:senha@192.168.0.10:554/cam/realmonitor?channel=7&subtype=1",
      "target_fps": 5,
      "zones": [[[0.2, 0.3], [0.8, 0.3], [0.8, 0.9], [0.2, 0.9]]]
    }
  ]
}
```

- [ ] **Step 6: Commit**

```bash
git add src/config/ config/config.example.json tests/config/
git commit -m "feat: configuração pydantic com override de detecção por câmera"
```

---

## Task 3: Banco de dados (SQLite)

**Files:**
- Create: `src/storage/db.py`
- Test: `tests/storage/test_db.py`

**Interfaces:**
- Consumes: `CameraState` de `src.core.types`.
- Produces: `Database(path)` com `init_schema()`, `insert_event(...) -> int`, `mark_sent(event_id)`, `set_feedback(event_id, value)`, `list_events(limit, since) -> list[sqlite3.Row]`, `upsert_camera_status(name, state, last_frame_ts)`, `get_camera_status(name)`, `purge_older_than(days) -> list[Path]`, `close()`.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/storage/test_db.py`:

```python
import json
from datetime import datetime, timedelta, timezone

import pytest

from src.core.types import CameraState
from src.storage.db import Database


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "app.db")
    d.init_schema()
    yield d
    d.close()


def _insert(db, **kw):
    defaults = dict(
        store_id="loja1",
        camera_name="Caixa 01",
        ts_utc=datetime.now(timezone.utc).isoformat(),
        ts_local=datetime.now().isoformat(),
        track_id=7,
        score=0.81,
        zone="waist",
        signals={"dwell": 1.0, "approach": 1.0, "vanish": 0.0, "retract": 0.0},
        image_path="evidence/a.jpg",
        clip_path=None,
    )
    defaults.update(kw)
    return db.insert_event(**defaults)


def test_insert_and_list_event(db):
    eid = _insert(db)
    assert eid > 0
    rows = db.list_events(limit=10)
    assert len(rows) == 1
    assert rows[0]["camera_name"] == "Caixa 01"
    assert rows[0]["score"] == pytest.approx(0.81)
    assert rows[0]["sent_telegram"] == 0
    assert rows[0]["feedback"] is None
    # a decomposição do score é preservada: sem isso, falso positivo em campo
    # é indepurável
    assert json.loads(rows[0]["signals_json"])["dwell"] == 1.0


def test_mark_sent(db):
    eid = _insert(db)
    db.mark_sent(eid)
    assert db.list_events(limit=1)[0]["sent_telegram"] == 1


def test_set_feedback(db):
    eid = _insert(db)
    db.set_feedback(eid, "false_positive")
    assert db.list_events(limit=1)[0]["feedback"] == "false_positive"


def test_set_feedback_rejects_invalid_value(db):
    eid = _insert(db)
    with pytest.raises(ValueError):
        db.set_feedback(eid, "talvez")


def test_list_events_is_newest_first(db):
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    _insert(db, ts_utc=old.isoformat(), camera_name="Antiga")
    _insert(db, camera_name="Nova")
    rows = db.list_events(limit=10)
    assert rows[0]["camera_name"] == "Nova"


def test_camera_status_upsert(db):
    db.upsert_camera_status("Caixa 01", CameraState.ONLINE, "2026-07-14T10:00:00")
    db.upsert_camera_status("Caixa 01", CameraState.OFFLINE, "2026-07-14T10:05:00")
    row = db.get_camera_status("Caixa 01")
    assert row["state"] == "offline"
    assert row["last_frame_ts"] == "2026-07-14T10:05:00"


def test_purge_removes_old_events_and_returns_files(db, tmp_path):
    img = tmp_path / "old.jpg"
    img.write_bytes(b"x")
    old = datetime.now(timezone.utc) - timedelta(days=40)
    _insert(db, ts_utc=old.isoformat(), image_path=str(img))
    _insert(db)  # recente

    removed = db.purge_older_than(days=30)

    assert [str(p) for p in removed] == [str(img)]
    assert len(db.list_events(limit=10)) == 1
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `pytest tests/storage/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.storage.db'`

- [ ] **Step 3: Implementar `src/storage/db.py`**

```python
"""Persistência local. `signals_json` guarda a decomposição do score —
é o que torna um falso positivo em campo depurável.
`store_id` custa zero hoje e evita reescrever o schema no dia em que o
cliente quiser um painel com várias lojas."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.core.types import CameraState

VALID_FEEDBACK = {"true_positive", "false_positive"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY,
  store_id TEXT NOT NULL,
  camera_name TEXT NOT NULL,
  ts_utc TEXT NOT NULL,
  ts_local TEXT NOT NULL,
  track_id INTEGER,
  score REAL NOT NULL,
  zone TEXT NOT NULL,
  signals_json TEXT NOT NULL,
  image_path TEXT,
  clip_path TEXT,
  sent_telegram INTEGER NOT NULL DEFAULT 0,
  feedback TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts_utc);

CREATE TABLE IF NOT EXISTS camera_status (
  camera_name TEXT PRIMARY KEY,
  state TEXT NOT NULL,
  last_frame_ts TEXT,
  since TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_meta (key TEXT PRIMARY KEY, value TEXT);
"""


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")

    def init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(SCHEMA)

    def insert_event(
        self,
        *,
        store_id: str,
        camera_name: str,
        ts_utc: str,
        ts_local: str,
        track_id: int | None,
        score: float,
        zone: str,
        signals: dict,
        image_path: str | None,
        clip_path: str | None,
    ) -> int:
        with self._conn:
            cur = self._conn.execute(
                """INSERT INTO events (store_id, camera_name, ts_utc, ts_local,
                       track_id, score, zone, signals_json, image_path, clip_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    store_id,
                    camera_name,
                    ts_utc,
                    ts_local,
                    track_id,
                    score,
                    zone,
                    json.dumps(signals),
                    image_path,
                    clip_path,
                ),
            )
        return int(cur.lastrowid)

    def mark_sent(self, event_id: int) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE events SET sent_telegram=1 WHERE id=?", (event_id,)
            )

    def set_feedback(self, event_id: int, value: str) -> None:
        if value not in VALID_FEEDBACK:
            raise ValueError(f"feedback inválido: {value}")
        with self._conn:
            self._conn.execute(
                "UPDATE events SET feedback=? WHERE id=?", (value, event_id)
            )

    def list_events(self, limit: int = 100, since: str | None = None) -> list[sqlite3.Row]:
        if since:
            return list(
                self._conn.execute(
                    "SELECT * FROM events WHERE ts_utc >= ? ORDER BY ts_utc DESC LIMIT ?",
                    (since, limit),
                )
            )
        return list(
            self._conn.execute(
                "SELECT * FROM events ORDER BY ts_utc DESC LIMIT ?", (limit,)
            )
        )

    def upsert_camera_status(
        self, camera_name: str, state: CameraState, last_frame_ts: str | None
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                """INSERT INTO camera_status (camera_name, state, last_frame_ts, since)
                   VALUES (?,?,?,?)
                   ON CONFLICT(camera_name) DO UPDATE SET
                     state=excluded.state,
                     last_frame_ts=excluded.last_frame_ts,
                     since=CASE WHEN camera_status.state != excluded.state
                                THEN excluded.since ELSE camera_status.since END""",
                (camera_name, state.value, last_frame_ts, now),
            )

    def get_camera_status(self, camera_name: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM camera_status WHERE camera_name=?", (camera_name,)
        ).fetchone()

    def purge_older_than(self, days: int) -> list[Path]:
        """Apaga eventos mais velhos que `days` e devolve os arquivos de
        evidência que devem ser removidos do disco."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = list(
            self._conn.execute(
                "SELECT image_path, clip_path FROM events WHERE ts_utc < ?", (cutoff,)
            )
        )
        files = [
            Path(p)
            for r in rows
            for p in (r["image_path"], r["clip_path"])
            if p and Path(p).exists()
        ]
        with self._conn:
            self._conn.execute("DELETE FROM events WHERE ts_utc < ?", (cutoff,))
        for f in files:
            f.unlink(missing_ok=True)
        return files

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `pytest tests/storage/test_db.py -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/storage/ tests/storage/
git commit -m "feat: persistência SQLite (eventos, status de câmera, purga)"
```

---

## Task 4: DVR simulado e material de teste próprio

**Files:**
- Create: `dev/dvr_sim.py`, `dev/record_clips.py`, `dev/make_sample_video.py`
- Test: `tests/dev/test_dvr_sim.py`

**Interfaces:**
- Produces: `DvrSim(videos: dict[str, Path], port=8554)` com `start()`, `stop()`, `kill_stream(channel)`, `url(channel) -> str`, e uso como context manager. Módulo `make_sample_video.synthetic_video(path, seconds, fps, size)`.

**Por que este task existe:** sem ele, a reconexão RTSP — a falha nº 1 em campo — só seria testada na loja do cliente. Com ele, derrubamos o servidor no meio do teste e verificamos que o sistema volta sozinho.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/dev/test_dvr_sim.py`:

```python
import cv2
import pytest

from dev.dvr_sim import DvrSim
from dev.make_sample_video import synthetic_video


@pytest.mark.slow
@pytest.mark.rtsp
def test_dvr_sim_serves_rtsp_frames(tmp_path):
    video = tmp_path / "cam1.mp4"
    synthetic_video(video, seconds=3, fps=10, size=(640, 360))

    with DvrSim({"ch1": video}) as sim:
        cap = cv2.VideoCapture(sim.url("ch1"), cv2.CAP_FFMPEG)
        assert cap.isOpened(), "não abriu o stream RTSP do DVR simulado"
        ok, frame = cap.read()
        cap.release()

    assert ok
    assert frame.shape[:2] == (360, 640)


def test_url_format():
    sim = DvrSim({}, port=9554)
    assert sim.url("ch3") == "rtsp://127.0.0.1:9554/ch3"


def test_synthetic_video_has_expected_length(tmp_path):
    p = tmp_path / "v.mp4"
    synthetic_video(p, seconds=2, fps=10, size=(320, 240))
    cap = cv2.VideoCapture(str(p))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert 15 <= n <= 25  # ~20 frames, tolerando o encoder
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `pytest tests/dev/test_dvr_sim.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'dev.dvr_sim'`

- [ ] **Step 3: Implementar `dev/make_sample_video.py`**

```python
"""Gera vídeo sintético para testar o caminho de captura (não serve para
testar detecção de pessoa — para isso, use dev/record_clips.py)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def synthetic_video(
    path: str | Path,
    seconds: float = 5,
    fps: int = 10,
    size: tuple[int, int] = (640, 360),
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    total = int(seconds * fps)
    for i in range(total):
        img = np.full((h, w, 3), 30, dtype=np.uint8)
        x = int((i / max(1, total - 1)) * (w - 60))
        cv2.rectangle(img, (x, h // 2 - 40), (x + 60, h // 2 + 40), (0, 200, 255), -1)
        cv2.putText(
            img, f"frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        writer.write(img)
    writer.release()
    return path
```

- [ ] **Step 4: Implementar `dev/dvr_sim.py`**

```python
"""DVR simulado: sobe um MediaMTX local e publica vídeos em loop como canais
RTSP. Permite testar reconexão derrubando um canal no meio do teste —
exatamente o que acontece quando o DVR do cliente reinicia."""
from __future__ import annotations

import platform
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import imageio_ffmpeg

MEDIAMTX_VERSION = "v1.9.3"
BIN_DIR = Path("dev/bin")


def _mediamtx_binary() -> Path:
    exe = BIN_DIR / ("mediamtx.exe" if platform.system() == "Windows" else "mediamtx")
    if exe.exists():
        return exe
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    asset = (
        f"mediamtx_{MEDIAMTX_VERSION}_windows_amd64.zip"
        if platform.system() == "Windows"
        else f"mediamtx_{MEDIAMTX_VERSION}_linux_amd64.tar.gz"
    )
    url = (
        f"https://github.com/bluenviron/mediamtx/releases/download/"
        f"{MEDIAMTX_VERSION}/{asset}"
    )
    dest = BIN_DIR / asset
    print(f"[dvr_sim] baixando MediaMTX de {url}")
    urllib.request.urlretrieve(url, dest)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(dest) as z:
            z.extractall(BIN_DIR)
    else:
        shutil.unpack_archive(str(dest), str(BIN_DIR))
    dest.unlink(missing_ok=True)
    return exe


class DvrSim:
    """Uso:
    with DvrSim({"ch1": Path("a.mp4"), "ch2": Path("b.mp4")}) as sim:
        cv2.VideoCapture(sim.url("ch1"))
    """

    def __init__(self, videos: dict[str, Path], port: int = 8554) -> None:
        self.videos = {k: Path(v) for k, v in videos.items()}
        self.port = port
        self._server: subprocess.Popen | None = None
        self._publishers: dict[str, subprocess.Popen] = {}

    def url(self, channel: str) -> str:
        return f"rtsp://127.0.0.1:{self.port}/{channel}"

    def start(self) -> "DvrSim":
        exe = _mediamtx_binary()
        cfg = BIN_DIR / f"mediamtx-{self.port}.yml"
        cfg.write_text(
            f"rtspAddress: :{self.port}\nhls: no\nwebrtc: no\nrtmp: no\napi: no\n",
            encoding="utf-8",
        )
        self._server = subprocess.Popen(
            [str(exe), str(cfg)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)  # o servidor sobe em <1s
        for ch in self.videos:
            self._publish(ch)
        time.sleep(1.5)  # dá tempo do ffmpeg começar a publicar
        return self

    def _publish(self, channel: str) -> None:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        self._publishers[channel] = subprocess.Popen(
            [
                ffmpeg, "-re", "-stream_loop", "-1",
                "-i", str(self.videos[channel]),
                "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                "-an", "-f", "rtsp", self.url(channel),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def kill_stream(self, channel: str) -> None:
        """Simula queda de um canal (cabo solto, DVR reiniciando)."""
        p = self._publishers.pop(channel, None)
        if p:
            p.terminate()
            p.wait(timeout=5)

    def restore_stream(self, channel: str) -> None:
        self._publish(channel)
        time.sleep(1.5)

    def stop(self) -> None:
        for ch in list(self._publishers):
            self.kill_stream(ch)
        if self._server:
            self._server.terminate()
            self._server.wait(timeout=5)
            self._server = None

    def __enter__(self) -> "DvrSim":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
```

- [ ] **Step 5: Implementar `dev/record_clips.py`**

Este script resolve o maior risco do projeto: **material real sem depender do cliente.** Grava clipes rotulados pela webcam, no formato que o sweep de calibração (Plano 2) espera.

```python
"""Grava material de calibração pela webcam, já rotulado.

Uso:
    python dev/record_clips.py --label bolso --seconds 8
    python dev/record_clips.py --label normal --seconds 300

Saída:
    dev/videos/ocultacao/bolso_01.mp4     (labels: bolso, bolsa, roupa, cintura)
    dev/videos/normal/normal_01.mp4       (label: normal)

Encene devagar e depois em ritmo natural. É este material que calibra o
sistema enquanto o vídeo do cliente não chega.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

CONCEAL_LABELS = {"bolso", "bolsa", "roupa", "cintura"}


def _next_path(label: str) -> Path:
    sub = "normal" if label == "normal" else "ocultacao"
    out = Path("dev/videos") / sub
    out.mkdir(parents=True, exist_ok=True)
    n = len(list(out.glob(f"{label}_*.mp4"))) + 1
    return out / f"{label}_{n:02d}.mp4"


def record(label: str, seconds: float, camera: int, fps: int) -> Path:
    if label not in CONCEAL_LABELS | {"normal"}:
        raise SystemExit(f"label inválido: {label} (use {CONCEAL_LABELS} ou 'normal')")
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"não consegui abrir a câmera {camera}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    path = _next_path(label)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"[rec] gravando '{label}' por {seconds}s em {path} — ESC para parar")
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        preview = frame.copy()
        restante = seconds - (time.monotonic() - t0)
        cv2.putText(preview, f"{label}  {restante:4.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("gravando (ESC para parar)", preview)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[rec] salvo: {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="bolso | bolsa | roupa | cintura | normal")
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=int, default=15)
    a = ap.parse_args()
    record(a.label, a.seconds, a.camera, a.fps)
```

- [ ] **Step 6: Rodar os testes rápidos**

Criar `dev/__init__.py` e `tests/dev/__init__.py` vazios.

Run: `pytest tests/dev/test_dvr_sim.py -v -m "not slow"`
Expected: PASS — 2 passed, 1 deselected

- [ ] **Step 7: Rodar o teste de integração (baixa o MediaMTX na primeira vez)**

Run: `pytest tests/dev/test_dvr_sim.py -v -m slow`
Expected: PASS — 1 passed. Se falhar por firewall do Windows, autorizar `mediamtx.exe` na rede privada.

- [ ] **Step 8: Commit**

```bash
git add dev/ tests/dev/
git commit -m "feat: DVR simulado (MediaMTX) e gravador de clipes de calibração"
```

---

## Task 5: LatestFrameSlot

**Files:**
- Create: `src/capture/frame_slot.py`
- Test: `tests/capture/test_frame_slot.py`

**Interfaces:**
- Consumes: `Frame` de `src.core.types`.
- Produces: `LatestFrameSlot()` com `put(frame)`, `get() -> Frame | None` (consome e esvazia), `peek() -> Frame | None` (não consome), `dropped` (contador).

**Por quê:** em vigilância ao vivo, frame velho é lixo. Uma fila com buffer acumularia atraso crescente quando a inferência não acompanha o stream; descartar o antigo mantém a latência limitada e faz o sistema degradar suavemente sob carga em vez de travar.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/capture/test_frame_slot.py`:

```python
import threading

import numpy as np

from src.capture.frame_slot import LatestFrameSlot
from src.core.types import Frame


def _frame(seq: int) -> Frame:
    return Frame("cam1", np.zeros((2, 2, 3), dtype=np.uint8), ts=float(seq), seq=seq)


def test_get_returns_none_when_empty():
    assert LatestFrameSlot().get() is None


def test_put_then_get():
    slot = LatestFrameSlot()
    slot.put(_frame(1))
    assert slot.get().seq == 1


def test_get_consumes_the_frame():
    slot = LatestFrameSlot()
    slot.put(_frame(1))
    slot.get()
    assert slot.get() is None


def test_second_put_overwrites_and_counts_drop():
    slot = LatestFrameSlot()
    slot.put(_frame(1))
    slot.put(_frame(2))
    assert slot.get().seq == 2  # o frame velho foi descartado, não enfileirado
    assert slot.dropped == 1


def test_peek_does_not_consume():
    slot = LatestFrameSlot()
    slot.put(_frame(5))
    assert slot.peek().seq == 5
    assert slot.get().seq == 5


def test_thread_safety_under_concurrent_writes():
    slot = LatestFrameSlot()

    def writer(start: int):
        for i in range(start, start + 200):
            slot.put(_frame(i))

    threads = [threading.Thread(target=writer, args=(i * 1000,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert slot.get() is not None
    assert slot.dropped == 800 - 1
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `pytest tests/capture/test_frame_slot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.capture.frame_slot'`

- [ ] **Step 3: Implementar `src/capture/frame_slot.py`**

```python
"""Slot de um frame só. Frame velho é descartado, nunca enfileirado:
em vídeo ao vivo, atraso acumulado é pior que frame perdido."""
from __future__ import annotations

import threading

from src.core.types import Frame


class LatestFrameSlot:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._dropped = 0

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def put(self, frame: Frame) -> None:
        with self._lock:
            if self._frame is not None:
                self._dropped += 1
            self._frame = frame

    def get(self) -> Frame | None:
        with self._lock:
            f, self._frame = self._frame, None
            return f

    def peek(self) -> Frame | None:
        with self._lock:
            return self._frame
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `pytest tests/capture/test_frame_slot.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/capture/ tests/capture/
git commit -m "feat: LatestFrameSlot (descarta frame velho, latência limitada)"
```

---

## Task 6: Captura RTSP com reconexão

**Files:**
- Create: `src/capture/rtsp_capture.py`
- Test: `tests/capture/test_rtsp_capture.py`

**Interfaces:**
- Consumes: `LatestFrameSlot`, `Frame`, `CameraState`, `CameraConfig`.
- Produces: `CameraThread(camera: CameraConfig, slot: LatestFrameSlot, backoff_max=30.0, open_capture=None)` com `start()`, `stop()`, propriedades `state`, `last_frame_ts` (monotonic ou `None`), `effective_fps`. O parâmetro `open_capture` é injetável para teste (fábrica que devolve algo com `isOpened()`, `read()`, `release()`).

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/capture/test_rtsp_capture.py`:

```python
import time

import numpy as np
import pytest

from src.capture.frame_slot import LatestFrameSlot
from src.capture.rtsp_capture import CameraThread
from src.config.settings import CameraConfig
from src.core.types import CameraState


class FakeCapture:
    """Dublê de cv2.VideoCapture com falha programável."""

    def __init__(self, fail_after: int | None = None, open_ok: bool = True):
        self.fail_after = fail_after
        self._open = open_ok
        self.reads = 0
        self.released = False

    def isOpened(self):  # noqa: N802 (assinatura do OpenCV)
        return self._open

    def read(self):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        return True, np.zeros((360, 640, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def _cam(**kw) -> CameraConfig:
    return CameraConfig(
        name="cam1", rtsp_url="rtsp://fake/ch1", target_fps=kw.pop("target_fps", 20), **kw
    )


def test_publishes_frames_and_goes_online():
    slot = LatestFrameSlot()
    cap = FakeCapture()
    t = CameraThread(_cam(), slot, open_capture=lambda url: cap)
    t.start()
    try:
        deadline = time.monotonic() + 3
        while slot.peek() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert slot.peek() is not None
        assert t.state == CameraState.ONLINE
        assert t.last_frame_ts is not None
    finally:
        t.stop()
    assert cap.released


def test_samples_at_target_fps():
    slot = LatestFrameSlot()
    t = CameraThread(_cam(target_fps=5), slot, open_capture=lambda url: FakeCapture())
    t.start()
    seqs = []
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            f = slot.get()
            if f:
                seqs.append(f.seq)
            time.sleep(0.01)
    finally:
        t.stop()
    # 5 fps por ~2s: aceita folga de agendamento do Windows
    assert 6 <= len(seqs) <= 14


def test_reconnects_after_stream_dies():
    slot = LatestFrameSlot()
    caps: list[FakeCapture] = []

    def factory(url):
        # o primeiro morre depois de 3 leituras; o segundo é saudável
        cap = FakeCapture(fail_after=3 if not caps else None)
        caps.append(cap)
        return cap

    t = CameraThread(_cam(), slot, backoff_max=0.2, open_capture=factory)
    t.start()
    try:
        deadline = time.monotonic() + 5
        while len(caps) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(caps) >= 2, "não reconectou depois da queda do stream"
        assert caps[0].released, "não liberou a captura morta"

        deadline = time.monotonic() + 3
        while t.state != CameraState.ONLINE and time.monotonic() < deadline:
            time.sleep(0.05)
        assert t.state == CameraState.ONLINE
    finally:
        t.stop()


def test_state_is_reconnecting_while_open_fails():
    slot = LatestFrameSlot()
    t = CameraThread(
        _cam(), slot, backoff_max=0.2, open_capture=lambda url: FakeCapture(open_ok=False)
    )
    t.start()
    try:
        deadline = time.monotonic() + 3
        while t.state != CameraState.RECONNECTING and time.monotonic() < deadline:
            time.sleep(0.05)
        assert t.state == CameraState.RECONNECTING
        assert slot.peek() is None
    finally:
        t.stop()


def test_backoff_is_capped():
    t = CameraThread(_cam(), LatestFrameSlot(), backoff_max=5.0, open_capture=lambda u: None)
    assert t._next_backoff(0.0) == 1.0
    assert t._next_backoff(1.0) == 2.0
    assert t._next_backoff(4.0) == 5.0
    assert t._next_backoff(5.0) == 5.0


@pytest.mark.slow
@pytest.mark.rtsp
def test_captures_from_simulated_dvr(tmp_path):
    from dev.dvr_sim import DvrSim
    from dev.make_sample_video import synthetic_video

    video = tmp_path / "ch1.mp4"
    synthetic_video(video, seconds=5, fps=10)

    with DvrSim({"ch1": video}) as sim:
        slot = LatestFrameSlot()
        cam = CameraConfig(name="cam1", rtsp_url=sim.url("ch1"), target_fps=5)
        t = CameraThread(cam, slot)
        t.start()
        try:
            deadline = time.monotonic() + 15
            while slot.peek() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            assert slot.peek() is not None, "não recebeu frame do DVR simulado"

            # derruba o canal: o sistema tem que perceber e voltar sozinho
            sim.kill_stream("ch1")
            deadline = time.monotonic() + 20
            while t.state == CameraState.ONLINE and time.monotonic() < deadline:
                time.sleep(0.2)
            assert t.state != CameraState.ONLINE

            sim.restore_stream("ch1")
            deadline = time.monotonic() + 30
            while t.state != CameraState.ONLINE and time.monotonic() < deadline:
                time.sleep(0.2)
            assert t.state == CameraState.ONLINE, "não reconectou após o canal voltar"
        finally:
            t.stop()
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `pytest tests/capture/test_rtsp_capture.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.capture.rtsp_capture'`

- [ ] **Step 3: Implementar `src/capture/rtsp_capture.py`**

```python
"""Captura RTSP: uma thread por câmera. Reconexão automática com backoff —
é a falha nº 1 em campo (DVR reinicia, cabo solta, rede oscila) e o sistema
tem que voltar sozinho, sem ninguém perceber que caiu."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import cv2

from src.capture.frame_slot import LatestFrameSlot
from src.config.settings import CameraConfig
from src.core.types import CameraState, Frame

log = logging.getLogger(__name__)

MAX_READ_FAILURES = 5  # leituras falhas seguidas antes de considerar o stream morto


def _open_rtsp(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    # buffer mínimo: queremos o frame mais novo, não a fila do driver
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class CameraThread:
    def __init__(
        self,
        camera: CameraConfig,
        slot: LatestFrameSlot,
        backoff_max: float = 30.0,
        open_capture: Callable[[str], object] | None = None,
    ) -> None:
        self.camera = camera
        self.slot = slot
        self.backoff_max = backoff_max
        self._open = open_capture or _open_rtsp
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state = CameraState.OFFLINE
        self._last_frame_ts: float | None = None
        self._seq = 0
        self._fps_window: list[float] = []

    # --- estado observável ---

    @property
    def state(self) -> CameraState:
        with self._lock:
            return self._state

    @property
    def last_frame_ts(self) -> float | None:
        with self._lock:
            return self._last_frame_ts

    @property
    def effective_fps(self) -> float:
        with self._lock:
            if len(self._fps_window) < 2:
                return 0.0
            span = self._fps_window[-1] - self._fps_window[0]
            return (len(self._fps_window) - 1) / span if span > 0 else 0.0

    def _set_state(self, state: CameraState) -> None:
        with self._lock:
            if self._state != state:
                log.info("câmera '%s': %s", self.camera.name, state.value)
                self._state = state

    # --- ciclo de vida ---

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"capture-{self.camera.name}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        self._set_state(CameraState.OFFLINE)

    def _next_backoff(self, current: float) -> float:
        return min(self.backoff_max, max(1.0, current * 2))

    # --- laço principal ---

    def _run(self) -> None:
        backoff = 0.0
        interval = 1.0 / max(0.1, self.camera.target_fps)

        while not self._stop.is_set():
            cap = self._open(self.camera.rtsp_url)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                self._set_state(CameraState.RECONNECTING)
                backoff = self._next_backoff(backoff)
                log.warning(
                    "câmera '%s': falha ao conectar, nova tentativa em %.0fs",
                    self.camera.name, backoff,
                )
                self._stop.wait(backoff)
                continue

            backoff = 0.0
            failures = 0
            next_sample = time.monotonic()

            while not self._stop.is_set():
                ok, image = cap.read()
                if not ok or image is None:
                    failures += 1
                    if failures >= MAX_READ_FAILURES:
                        log.warning("câmera '%s': stream morreu", self.camera.name)
                        break
                    time.sleep(0.05)
                    continue

                failures = 0
                now = time.monotonic()
                if now < next_sample:
                    continue  # amostragem: descarta o frame sem processar

                next_sample = now + interval
                self._seq += 1
                self.slot.put(Frame(self.camera.name, image, now, self._seq))
                with self._lock:
                    self._last_frame_ts = now
                    self._fps_window.append(now)
                    if len(self._fps_window) > 20:
                        self._fps_window.pop(0)
                self._set_state(CameraState.ONLINE)

            cap.release()
            if not self._stop.is_set():
                self._set_state(CameraState.RECONNECTING)
                backoff = self._next_backoff(backoff)
                self._stop.wait(backoff)
```

- [ ] **Step 4: Rodar os testes rápidos**

Run: `pytest tests/capture/test_rtsp_capture.py -v -m "not slow"`
Expected: PASS — 5 passed, 1 deselected

- [ ] **Step 5: Rodar o teste de integração contra o DVR simulado**

Run: `pytest tests/capture/test_rtsp_capture.py -v -m slow`
Expected: PASS — 1 passed. Este é o teste que prova que o sistema sobrevive à queda do DVR.

- [ ] **Step 6: Commit**

```bash
git add src/capture/rtsp_capture.py tests/capture/test_rtsp_capture.py
git commit -m "feat: captura RTSP com amostragem, heartbeat e reconexão com backoff"
```

---

## Task 7: Engine de inferência (YOLO + OpenVINO)

**Files:**
- Create: `src/inference/engine.py`
- Test: `tests/inference/test_engine.py`

**Interfaces:**
- Consumes: `InferenceConfig`, `BBox`, `PersonDetection`, `ObjectDetection`.
- Produces: `InferenceEngine(cfg: InferenceConfig)` com `detect(image) -> tuple[list[PersonDetection], list[ObjectDetection]]`, `pose(image, boxes) -> list[np.ndarray]` (cada um `(17,3)` em pixels do frame completo), `warmup()`. Função `export_openvino(pt_path) -> Path` (cacheia; não re-exporta se já existe).

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/inference/test_engine.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from src.config.settings import InferenceConfig
from src.core.types import BBox
from src.inference.engine import InferenceEngine, export_openvino


class FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = np.array(xyxy, dtype=np.float32)
        self.cls = np.array(cls, dtype=np.float32)
        self.conf = np.array(conf, dtype=np.float32)

    def __len__(self):
        return len(self.cls)


class FakeResult:
    def __init__(self, boxes=None, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


class FakeKeypoints:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)  # (n, 17, 3)


def test_detect_splits_persons_and_bags(monkeypatch):
    cfg = InferenceConfig(device="cpu", detect_bags=True)
    eng = InferenceEngine(cfg)

    # COCO: 0=person, 24=backpack, 26=handbag, 2=car (deve ser ignorado)
    boxes = FakeBoxes(
        xyxy=[[10, 10, 60, 200], [70, 20, 100, 60], [0, 0, 5, 5]],
        cls=[0, 24, 2],
        conf=[0.9, 0.7, 0.95],
    )
    monkeypatch.setattr(eng, "_person_model", lambda *a, **k: [FakeResult(boxes=boxes)])

    persons, objects = eng.detect(np.zeros((240, 320, 3), dtype=np.uint8))

    assert len(persons) == 1
    assert persons[0].bbox == BBox(10, 10, 60, 200)
    assert persons[0].conf == pytest.approx(0.9)
    assert len(objects) == 1
    assert objects[0].label == "backpack"


def test_detect_ignores_bags_when_disabled(monkeypatch):
    eng = InferenceEngine(InferenceConfig(device="cpu", detect_bags=False))
    boxes = FakeBoxes(xyxy=[[70, 20, 100, 60]], cls=[24], conf=[0.7])
    monkeypatch.setattr(eng, "_person_model", lambda *a, **k: [FakeResult(boxes=boxes)])

    persons, objects = eng.detect(np.zeros((240, 320, 3), dtype=np.uint8))

    assert persons == []
    assert objects == []


def test_pose_on_crop_returns_keypoints_in_full_frame_coords(monkeypatch):
    """A pose roda no recorte da pessoa (resolução efetiva muito maior em
    pessoa pequena), mas devolve coordenadas do frame completo."""
    eng = InferenceEngine(InferenceConfig(device="cpu", pose_on_crop=True))

    kp_local = np.zeros((1, 17, 3), dtype=np.float32)
    kp_local[0, 9] = [10.0, 20.0, 0.9]  # punho esquerdo a (10,20) DENTRO do recorte
    monkeypatch.setattr(
        eng, "_pose_model", lambda *a, **k: [FakeResult(keypoints=FakeKeypoints(kp_local))]
    )

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    box = BBox(100, 50, 200, 250)  # recorte expandido começa antes de (100,50)
    kps = eng.pose(image, [box])

    assert len(kps) == 1
    exp = box.expand(0.1).clip(640, 480)
    assert kps[0][9][0] == pytest.approx(10.0 + exp.x1)
    assert kps[0][9][1] == pytest.approx(20.0 + exp.y1)
    assert kps[0][9][2] == pytest.approx(0.9)


def test_pose_returns_zeros_when_model_finds_nothing(monkeypatch):
    eng = InferenceEngine(InferenceConfig(device="cpu"))
    monkeypatch.setattr(
        eng, "_pose_model", lambda *a, **k: [FakeResult(keypoints=None)]
    )
    kps = eng.pose(np.zeros((480, 640, 3), dtype=np.uint8), [BBox(0, 0, 100, 200)])
    assert len(kps) == 1
    assert kps[0].shape == (17, 3)
    assert kps[0][:, 2].sum() == 0.0  # confiança zero = "não sei"


def test_pose_skips_degenerate_box(monkeypatch):
    eng = InferenceEngine(InferenceConfig(device="cpu"))
    called = []
    monkeypatch.setattr(
        eng, "_pose_model", lambda *a, **k: called.append(1) or [FakeResult()]
    )
    kps = eng.pose(np.zeros((480, 640, 3), dtype=np.uint8), [BBox(10, 10, 10, 10)])
    assert kps[0][:, 2].sum() == 0.0
    assert not called, "não deve chamar o modelo para caixa degenerada"


@pytest.mark.slow
def test_real_model_detects_person_in_recorded_clip():
    """Roda o modelo de verdade sobre material próprio.
    Grave antes: python dev/record_clips.py --label normal --seconds 5"""
    import cv2

    clips = sorted(Path("dev/videos/normal").glob("*.mp4"))
    if not clips:
        pytest.skip("sem material: rode dev/record_clips.py --label normal --seconds 5")

    cap = cv2.VideoCapture(str(clips[0]))
    ok, frame = cap.read()
    cap.release()
    assert ok

    eng = InferenceEngine(InferenceConfig(device="cpu"))
    eng.warmup()
    persons, _ = eng.detect(frame)
    assert len(persons) >= 1, "não detectou pessoa no clipe gravado"

    kps = eng.pose(frame, [p.bbox for p in persons])
    assert kps[0].shape == (17, 3)
    assert kps[0][:, 2].max() > 0.3, "keypoints sem confiança nenhuma"


@pytest.mark.slow
def test_export_openvino_is_cached(tmp_path):
    out = export_openvino(Path("models/yolo11n.pt"))
    assert out.exists()
    mtime = out.stat().st_mtime
    again = export_openvino(Path("models/yolo11n.pt"))
    assert again == out
    assert out.stat().st_mtime == mtime, "re-exportou um modelo já exportado"
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `pytest tests/inference/test_engine.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.inference.engine'`

- [ ] **Step 3: Implementar `src/inference/engine.py`**

```python
"""Wrapper dos modelos YOLO. Duas decisões que carregam o desempenho:

1. Export para OpenVINO (cacheado): CPU Intel de PDV roda 2-4x mais rápido.
2. Pose no RECORTE da pessoa, não no frame inteiro: em câmera de teto a
   pessoa ocupa poucos pixels, e rodar a pose no recorte multiplica a
   resolução efetiva sobre o corpo — além de ser mais barato."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.config.settings import InferenceConfig
from src.core.types import BBox, ObjectDetection, PersonDetection

log = logging.getLogger(__name__)

COCO_PERSON = 0
COCO_BAGS = {24: "backpack", 26: "handbag"}
CROP_MARGIN = 0.1  # expande a caixa antes de recortar: pose precisa do contorno
POSE_INPUT = 320  # o recorte é pequeno; 320 basta e é rápido


def export_openvino(pt_path: str | Path) -> Path:
    """Exporta o .pt para OpenVINO uma única vez. O diretório exportado fica
    ao lado do .pt e é reaproveitado nas execuções seguintes."""
    from ultralytics import YOLO

    pt = Path(pt_path)
    out = pt.with_name(f"{pt.stem}_openvino_model")
    if out.exists():
        return out
    log.info("exportando %s para OpenVINO (só na primeira vez)...", pt.name)
    YOLO(str(pt)).export(format="openvino", half=False)
    return out


class InferenceEngine:
    def __init__(self, cfg: InferenceConfig) -> None:
        self.cfg = cfg
        self._wanted = {COCO_PERSON} | (set(COCO_BAGS) if cfg.detect_bags else set())
        # Carga preguiçosa: o construtor não pode exigir os pesos em disco, senão
        # todo teste unitário viraria download de modelo.
        self._person_model = None
        self._pose_model = None

    def _resolve(self, model_path: str) -> Path:
        p = Path(model_path)
        if self.cfg.device == "openvino":
            return export_openvino(p)
        return p

    def _ensure_person_model(self):
        if self._person_model is None:
            from ultralytics import YOLO

            self._person_model = YOLO(str(self._resolve(self.cfg.person_model)), task="detect")
        return self._person_model

    def _ensure_pose_model(self):
        if self._pose_model is None:
            from ultralytics import YOLO

            self._pose_model = YOLO(str(self._resolve(self.cfg.pose_model)), task="pose")
        return self._pose_model

    def warmup(self) -> None:
        """Primeira inferência é sempre lenta (aloca buffers, compila kernels).
        Fazer no start evita que o primeiro frame real leve 2 segundos."""
        dummy = np.zeros((self.cfg.detect_size, self.cfg.detect_size, 3), dtype=np.uint8)
        self.detect(dummy)
        self.pose(dummy, [BBox(10, 10, 100, 200)])

    def detect(
        self, image: np.ndarray
    ) -> tuple[list[PersonDetection], list[ObjectDetection]]:
        model = self._ensure_person_model()
        results = model(
            image,
            imgsz=self.cfg.detect_size,
            classes=sorted(self._wanted),
            verbose=False,
        )
        persons: list[PersonDetection] = []
        objects: list[ObjectDetection] = []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return persons, objects

        for xyxy, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1, x2, y2 = (float(v) for v in np.asarray(xyxy).tolist())
            c = int(cls)
            box = BBox(x1, y1, x2, y2)
            if c == COCO_PERSON:
                persons.append(PersonDetection(bbox=box, conf=float(conf)))
            elif self.cfg.detect_bags and c in COCO_BAGS:
                objects.append(
                    ObjectDetection(label=COCO_BAGS[c], bbox=box, conf=float(conf))
                )
        return persons, objects

    def pose(self, image: np.ndarray, boxes: list[BBox]) -> list[np.ndarray]:
        """Devolve um array (17,3) por caixa, em coordenadas do frame completo.
        Keypoint não encontrado vem com confiança 0."""
        h, w = image.shape[:2]
        out: list[np.ndarray] = []

        for box in boxes:
            empty = np.zeros((17, 3), dtype=np.float32)
            crop_box = box.expand(CROP_MARGIN).clip(w, h) if self.cfg.pose_on_crop else BBox(0, 0, w, h)
            x1, y1 = int(crop_box.x1), int(crop_box.y1)
            x2, y2 = int(crop_box.x2), int(crop_box.y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                out.append(empty)
                continue

            crop = image[y1:y2, x1:x2]
            results = self._ensure_pose_model()(crop, imgsz=POSE_INPUT, verbose=False)
            kp = results[0].keypoints
            if kp is None or kp.data is None or len(kp.data) == 0:
                out.append(empty)
                continue

            data = np.asarray(kp.data)[0].astype(np.float32).copy()  # (17,3) no recorte
            data[:, 0] += x1  # de volta para o frame completo
            data[:, 1] += y1
            out.append(data)

        return out
```

- [ ] **Step 4: Rodar os testes rápidos**

Run: `pytest tests/inference/test_engine.py -v -m "not slow"`
Expected: PASS — 5 passed, 2 deselected

- [ ] **Step 5: Baixar os modelos e gravar um clipe próprio**

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-pose.pt')"
mkdir models 2>nul
move yolo11n.pt models\ 2>nul
move yolo11n-pose.pt models\ 2>nul
python dev/record_clips.py --label normal --seconds 5
```

- [ ] **Step 6: Rodar os testes lentos (modelo real sobre material próprio)**

Run: `pytest tests/inference/test_engine.py -v -m slow`
Expected: PASS — 2 passed. Se `test_real_model_detects_person_in_recorded_clip` falhar, o problema está no material (webcam apontada para a parede), não no código.

- [ ] **Step 7: Commit**

```bash
git add src/inference/ tests/inference/
git commit -m "feat: engine YOLO com export OpenVINO cacheado e pose no recorte"
```

---

## Task 8: Gate de pessoa (zona monitorada)

**Files:**
- Create: `src/detection/person_gate.py`
- Test: `tests/detection/test_person_gate.py`

**Interfaces:**
- Consumes: `PersonDetection`, `BBox`.
- Produces: `PersonGate(zones: list[list[tuple[float,float]]], frame_size: tuple[int,int])` com `contains(person) -> bool` e `filter(persons) -> list[PersonDetection]`. Zonas vazias = quadro inteiro (o padrão "monitorar tudo" que o cliente pediu).

**Por quê:** é este filtro que faz o sistema rodar em PC fraco. Pose custa 5–10× um detect; corredor vazio deve custar quase nada.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/detection/test_person_gate.py`:

```python
from src.core.types import BBox, PersonDetection
from src.detection.person_gate import PersonGate

FRAME = (1000, 500)  # largura, altura


def _person(x1, y1, x2, y2) -> PersonDetection:
    return PersonDetection(bbox=BBox(x1, y1, x2, y2), conf=0.9)


def test_no_zones_means_whole_frame():
    gate = PersonGate(zones=[], frame_size=FRAME)
    assert gate.contains(_person(0, 0, 10, 10))
    assert gate.contains(_person(900, 400, 990, 490))


def test_person_inside_polygon():
    # metade direita do quadro
    zone = [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]
    gate = PersonGate(zones=[zone], frame_size=FRAME)
    # foot_point = (750, 400) → dentro
    assert gate.contains(_person(700, 100, 800, 400))


def test_person_outside_polygon():
    zone = [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]
    gate = PersonGate(zones=[zone], frame_size=FRAME)
    # foot_point = (150, 400) → fora
    assert not gate.contains(_person(100, 100, 200, 400))


def test_uses_foot_point_not_center():
    """Pessoa alta cujo CENTRO cai fora da zona mas cujos PÉS caem dentro
    deve contar: quem define a posição é onde a pessoa pisa."""
    zone = [(0.0, 0.7), (1.0, 0.7), (1.0, 1.0), (0.0, 1.0)]  # faixa inferior
    gate = PersonGate(zones=[zone], frame_size=FRAME)
    p = _person(400, 100, 500, 400)  # centro y=250 (fora), pés y=400 (dentro)
    assert gate.contains(p)


def test_multiple_zones_are_or():
    left = [(0.0, 0.0), (0.2, 0.0), (0.2, 1.0), (0.0, 1.0)]
    right = [(0.8, 0.0), (1.0, 0.0), (1.0, 1.0), (0.8, 1.0)]
    gate = PersonGate(zones=[left, right], frame_size=FRAME)
    assert gate.contains(_person(50, 100, 150, 400))    # pés em x=100 → zona esquerda
    assert gate.contains(_person(880, 100, 920, 400))   # pés em x=900 → zona direita
    assert not gate.contains(_person(450, 100, 550, 400))  # meio → fora


def test_filter_keeps_only_people_inside():
    zone = [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]
    gate = PersonGate(zones=[zone], frame_size=FRAME)
    dentro = _person(700, 100, 800, 400)
    fora = _person(100, 100, 200, 400)
    assert gate.filter([dentro, fora]) == [dentro]
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `pytest tests/detection/test_person_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.detection.person_gate'`

- [ ] **Step 3: Implementar `src/detection/person_gate.py`**

```python
"""Gate: a pessoa está dentro da área monitorada?

É este filtro que permite rodar 8-10 câmeras num PC de mercado: sem pessoa
na zona, nada de pose. Corredor vazio custa um detect e nada mais."""
from __future__ import annotations

import numpy as np

from src.core.types import PersonDetection

Polygon = list[tuple[float, float]]


class PersonGate:
    def __init__(self, zones: list[Polygon], frame_size: tuple[int, int]) -> None:
        """zones em coordenadas normalizadas (0-1); frame_size = (largura, altura).
        Lista vazia = monitorar o quadro inteiro."""
        w, h = frame_size
        self._polys = [
            np.array([(x * w, y * h) for x, y in poly], dtype=np.float32) for poly in zones
        ]

    def contains(self, person: PersonDetection) -> bool:
        if not self._polys:
            return True
        # O ponto que representa a pessoa é onde ela PISA, não seu centro:
        # uma pessoa alta pode ter o centro fora da zona e os pés dentro.
        import cv2

        x, y = person.bbox.foot_point
        return any(
            cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
            for poly in self._polys
        )

    def filter(self, persons: list[PersonDetection]) -> list[PersonDetection]:
        return [p for p in persons if self.contains(p)]
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `pytest tests/detection/test_person_gate.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/detection/ tests/detection/
git commit -m "feat: gate de pessoa por polígono (usa o ponto de apoio, não o centro)"
```

---

## Task 9: Tracking (ByteTrack)

**Files:**
- Create: `src/detection/tracker.py`
- Test: `tests/detection/test_tracker.py`

**Interfaces:**
- Consumes: `PersonDetection`, `BBox`.
- Produces: `Tracker(max_lost_seconds: float)` com `update(persons: list[PersonDetection], ts: float) -> list[PersonDetection]` (devolve as mesmas pessoas com `track_id` preenchido) e `active_ids() -> set[int]`.

**Nota de implementação:** o ByteTrack embutido no Ultralytics só funciona pelo `model.track()`, que reprocessa o frame inteiro e não aceita detecções externas. Como aqui as detecções já existem (vêm do gate), usamos um associador IoU + idade — que é o miolo do ByteTrack para o nosso caso (câmera fixa, 5 FPS, poucas pessoas). Isso mantém `detection/` puro e testável sem modelo.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/detection/test_tracker.py`:

```python
from src.core.types import BBox, PersonDetection
from src.detection.tracker import Tracker


def _p(x1, y1, x2, y2, conf=0.9) -> PersonDetection:
    return PersonDetection(bbox=BBox(x1, y1, x2, y2), conf=conf)


def test_assigns_ids_to_new_people():
    t = Tracker(max_lost_seconds=2.0)
    out = t.update([_p(0, 0, 50, 150), _p(200, 0, 250, 150)], ts=0.0)
    ids = {p.track_id for p in out}
    assert ids == {1, 2}


def test_keeps_id_when_person_moves_a_little():
    t = Tracker(max_lost_seconds=2.0)
    first = t.update([_p(0, 0, 50, 150)], ts=0.0)[0]
    second = t.update([_p(8, 2, 58, 152)], ts=0.2)[0]
    assert second.track_id == first.track_id


def test_new_id_when_person_is_completely_elsewhere():
    t = Tracker(max_lost_seconds=2.0)
    t.update([_p(0, 0, 50, 150)], ts=0.0)
    out = t.update([_p(500, 0, 550, 150)], ts=0.2)
    assert out[0].track_id == 2


def test_id_survives_a_short_gap():
    """Pessoa sumiu por 1 frame (oclusão por gôndola) e voltou perto:
    tem que manter o id — senão o dwell da ocultação reinicia do zero."""
    t = Tracker(max_lost_seconds=2.0)
    first = t.update([_p(0, 0, 50, 150)], ts=0.0)[0]
    t.update([], ts=0.2)
    again = t.update([_p(5, 0, 55, 150)], ts=0.4)[0]
    assert again.track_id == first.track_id


def test_id_is_dropped_after_max_lost():
    t = Tracker(max_lost_seconds=1.0)
    t.update([_p(0, 0, 50, 150)], ts=0.0)
    t.update([], ts=2.0)
    out = t.update([_p(0, 0, 50, 150)], ts=2.1)
    assert out[0].track_id == 2
    assert t.active_ids() == {2}


def test_two_people_do_not_swap_ids():
    t = Tracker(max_lost_seconds=2.0)
    a, b = t.update([_p(0, 0, 50, 150), _p(300, 0, 350, 150)], ts=0.0)
    out = t.update([_p(305, 0, 355, 150), _p(6, 0, 56, 150)], ts=0.2)
    by_id = {p.track_id: p.bbox.x1 for p in out}
    assert by_id[a.track_id] < 100
    assert by_id[b.track_id] > 300
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `pytest tests/detection/test_tracker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.detection.tracker'`

- [ ] **Step 3: Implementar `src/detection/tracker.py`**

```python
"""Associação de identidade entre frames (IoU + idade).

Manter o track_id estável é requisito da lógica de ocultação: o dwell da mão
na zona é acumulado POR PESSOA. Se o id trocar no meio do gesto, o contador
zera e o furto passa batido. Por isso o track sobrevive a alguns frames sem
detecção (pessoa passa atrás de uma gôndola)."""
from __future__ import annotations

from dataclasses import dataclass

from src.core.types import BBox, PersonDetection

IOU_MATCH_MIN = 0.25


def iou(a: BBox, b: BBox) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    bbox: BBox
    last_seen: float


class Tracker:
    def __init__(self, max_lost_seconds: float = 2.0) -> None:
        self.max_lost_seconds = max_lost_seconds
        self._tracks: list[_Track] = []
        self._next_id = 1

    def active_ids(self) -> set[int]:
        return {t.track_id for t in self._tracks}

    def update(self, persons: list[PersonDetection], ts: float) -> list[PersonDetection]:
        self._tracks = [
            t for t in self._tracks if ts - t.last_seen <= self.max_lost_seconds
        ]

        # Guloso pelo melhor IoU: com poucas pessoas e câmera fixa, resolve.
        pairs = sorted(
            (
                (iou(p.bbox, t.bbox), pi, ti)
                for pi, p in enumerate(persons)
                for ti, t in enumerate(self._tracks)
            ),
            reverse=True,
        )
        taken_p: set[int] = set()
        taken_t: set[int] = set()
        assigned: dict[int, int] = {}  # índice da pessoa -> track_id

        for score, pi, ti in pairs:
            if score < IOU_MATCH_MIN or pi in taken_p or ti in taken_t:
                continue
            track = self._tracks[ti]
            track.bbox = persons[pi].bbox
            track.last_seen = ts
            assigned[pi] = track.track_id
            taken_p.add(pi)
            taken_t.add(ti)

        out: list[PersonDetection] = []
        for pi, p in enumerate(persons):
            if pi in assigned:
                tid = assigned[pi]
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks.append(_Track(tid, p.bbox, ts))
            out.append(PersonDetection(bbox=p.bbox, conf=p.conf, track_id=tid))
        return out
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `pytest tests/detection/test_tracker.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/detection/tracker.py tests/detection/test_tracker.py
git commit -m "feat: tracker por IoU com sobrevivência a oclusão curta"
```

---

## Task 10: Escalonador e pool de workers

**Files:**
- Create: `src/inference/scheduler.py`, `src/inference/worker_pool.py`
- Test: `tests/inference/test_scheduler.py`, `tests/inference/test_worker_pool.py`

**Interfaces:**
- Consumes: `LatestFrameSlot`, `Frame`.
- Produces: `Scheduler(camera_names: list[str], active_boost=3.0, active_window=5.0)` com `next_camera(now) -> str | None`, `mark_activity(name, ts)`, `mark_served(name, now)`. `WorkerPool(slots: dict[str, LatestFrameSlot], scheduler, process: Callable[[Frame], bool], workers=2)` com `start()`, `stop()`, `processed` (contador). O `process` devolve `True` quando houve pessoa (realimenta o boost do escalonador).

**Por quê:** 10 câmeras vazias não podem custar como 10 câmeras cheias. O escalonador dá mais fatias de CPU para quem teve pessoa recentemente — o resto é olhado de vez em quando, só para não perder alguém chegando.

- [ ] **Step 1: Escrever os testes que falham**

Criar `tests/inference/test_scheduler.py`:

```python
from src.inference.scheduler import Scheduler


def test_round_robin_when_no_activity():
    s = Scheduler(["a", "b", "c"])
    picked = []
    now = 0.0
    for _ in range(6):
        c = s.next_camera(now)
        picked.append(c)
        s.mark_served(c, now)
        now += 0.1
    assert picked == ["a", "b", "c", "a", "b", "c"]


def test_camera_with_recent_person_gets_more_turns():
    s = Scheduler(["a", "b"], active_boost=3.0, active_window=5.0)
    s.mark_activity("a", ts=0.0)
    picked = []
    now = 0.0
    for _ in range(8):
        c = s.next_camera(now)
        picked.append(c)
        s.mark_served(c, now)
        now += 0.1
    assert picked.count("a") > picked.count("b")


def test_boost_expires_after_window():
    s = Scheduler(["a", "b"], active_boost=3.0, active_window=1.0)
    s.mark_activity("a", ts=0.0)
    picked = []
    now = 10.0  # muito depois da janela
    for _ in range(6):
        c = s.next_camera(now)
        picked.append(c)
        s.mark_served(c, now)
        now += 0.1
    assert picked.count("a") == picked.count("b") == 3


def test_returns_none_without_cameras():
    assert Scheduler([]).next_camera(0.0) is None
```

Criar `tests/inference/test_worker_pool.py`:

```python
import threading
import time

import numpy as np

from src.capture.frame_slot import LatestFrameSlot
from src.core.types import Frame
from src.inference.scheduler import Scheduler
from src.inference.worker_pool import WorkerPool


def _frame(cam: str, seq: int) -> Frame:
    return Frame(cam, np.zeros((8, 8, 3), dtype=np.uint8), ts=float(seq), seq=seq)


def test_processes_frames_from_all_cameras():
    slots = {"a": LatestFrameSlot(), "b": LatestFrameSlot()}
    seen: list[str] = []
    lock = threading.Lock()

    def process(frame: Frame) -> bool:
        with lock:
            seen.append(frame.camera_name)
        return False

    pool = WorkerPool(slots, Scheduler(list(slots)), process, workers=2)
    pool.start()
    try:
        for i in range(20):
            slots["a"].put(_frame("a", i))
            slots["b"].put(_frame("b", i))
            time.sleep(0.02)
        deadline = time.monotonic() + 3
        while pool.processed < 4 and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        pool.stop()

    assert "a" in seen and "b" in seen
    assert pool.processed >= 4


def test_empty_slots_do_not_spin_the_cpu():
    slots = {"a": LatestFrameSlot()}
    pool = WorkerPool(slots, Scheduler(["a"]), lambda f: False, workers=1)
    pool.start()
    time.sleep(0.3)
    pool.stop()
    assert pool.processed == 0  # nada para processar, e nada quebrou


def test_process_exception_does_not_kill_the_worker():
    slots = {"a": LatestFrameSlot()}
    calls = {"n": 0}

    def process(frame: Frame) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("erro de inferência")
        return False

    pool = WorkerPool(slots, Scheduler(["a"]), process, workers=1)
    pool.start()
    try:
        deadline = time.monotonic() + 3
        while calls["n"] < 2 and time.monotonic() < deadline:
            slots["a"].put(_frame("a", calls["n"]))
            time.sleep(0.05)
    finally:
        pool.stop()
    assert calls["n"] >= 2, "o worker morreu na primeira exceção"
```

- [ ] **Step 2: Rodar e confirmar que falham**

Run: `pytest tests/inference/test_scheduler.py tests/inference/test_worker_pool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.inference.scheduler'`

- [ ] **Step 3: Implementar `src/inference/scheduler.py`**

```python
"""Escolhe qual câmera o próximo worker atende.

Câmera com pessoa recente recebe mais fatias de CPU; câmera de corredor vazio
é visitada com menos frequência. É isso que faz 10 câmeras numa loja calma
custarem menos que 3 câmeras num sábado lotado."""
from __future__ import annotations

import threading


class Scheduler:
    def __init__(
        self,
        camera_names: list[str],
        active_boost: float = 3.0,
        active_window: float = 5.0,
    ) -> None:
        self.cameras = list(camera_names)
        self.active_boost = active_boost
        self.active_window = active_window
        self._lock = threading.Lock()
        self._last_activity: dict[str, float] = {}
        # -inf, não 0.0: câmera nunca atendida tem fome infinita. Com 0.0, todas
        # empatariam no primeiro ciclo e a mesma câmera seria escolhida sempre.
        self._last_served: dict[str, float] = {c: float("-inf") for c in self.cameras}

    def mark_activity(self, camera: str, ts: float) -> None:
        """Chamado quando o worker encontrou pessoa nesta câmera."""
        with self._lock:
            self._last_activity[camera] = ts

    def mark_served(self, camera: str, now: float) -> None:
        with self._lock:
            self._last_served[camera] = now

    def next_camera(self, now: float) -> str | None:
        """A câmera com maior "fome": tempo desde a última vez que foi
        atendida, multiplicado pelo boost se houve pessoa recentemente."""
        with self._lock:
            if not self.cameras:
                return None
            best, best_score = None, float("-inf")
            for cam in self.cameras:
                starving = now - self._last_served.get(cam, 0.0)
                active = now - self._last_activity.get(cam, float("-inf"))
                weight = self.active_boost if active <= self.active_window else 1.0
                score = starving * weight
                if score > best_score:
                    best, best_score = cam, score
            return best
```

- [ ] **Step 4: Implementar `src/inference/worker_pool.py`**

```python
"""Pool pequeno de workers de inferência.

Não rodar pose em 5 streams ao mesmo tempo: com 1-2 workers, o hardware fraco
degrada suavemente (o FPS efetivo cai) em vez de travar."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from src.capture.frame_slot import LatestFrameSlot
from src.core.types import Frame
from src.inference.scheduler import Scheduler

log = logging.getLogger(__name__)

IDLE_SLEEP = 0.02  # nada para processar: não queimar CPU em busy-wait


class WorkerPool:
    def __init__(
        self,
        slots: dict[str, LatestFrameSlot],
        scheduler: Scheduler,
        process: Callable[[Frame], bool],
        workers: int = 2,
    ) -> None:
        self.slots = slots
        self.scheduler = scheduler
        self.process = process
        self.workers = max(1, workers)
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._processed = 0

    @property
    def processed(self) -> int:
        with self._lock:
            return self._processed

    def start(self) -> None:
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._run, name=f"infer-{i}", daemon=True)
            for i in range(self.workers)
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)
        self._threads = []

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            cam = self.scheduler.next_camera(now)
            if cam is None:
                self._stop.wait(IDLE_SLEEP)
                continue

            frame = self.slots[cam].get()
            self.scheduler.mark_served(cam, now)
            if frame is None:
                self._stop.wait(IDLE_SLEEP)
                continue

            try:
                had_person = self.process(frame)
                if had_person:
                    self.scheduler.mark_activity(cam, now)
            except Exception:
                # Um erro de inferência num frame não pode derrubar o worker —
                # o sistema roda 24/7 numa loja, sem ninguém olhando.
                log.exception("erro processando frame da câmera '%s'", cam)
            finally:
                with self._lock:
                    self._processed += 1
```

- [ ] **Step 5: Rodar e confirmar que passam**

Run: `pytest tests/inference/test_scheduler.py tests/inference/test_worker_pool.py -v`
Expected: PASS — 7 passed

- [ ] **Step 6: Commit**

```bash
git add src/inference/scheduler.py src/inference/worker_pool.py tests/inference/
git commit -m "feat: escalonador por atividade e pool de workers de inferência"
```

---

## Task 11: Orquestração headless (`main.py`)

**Files:**
- Create: `src/main.py`, `src/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: tudo dos tasks anteriores.
- Produces: `Pipeline(cfg: AppConfig, engine: InferenceEngine)` com `start()`, `stop()`, `process_frame(frame) -> FrameResult`, `status() -> dict[str, dict]`. `FrameResult(camera_name, persons: list[PersonPose], objects: list[ObjectDetection], had_person: bool)`. `main.py` é o entrypoint headless: `python -m src.main --config config/config.json`.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_pipeline.py`:

```python
import numpy as np
import pytest

from src.config.settings import AppConfig, CameraConfig, StoreConfig
from src.core.types import BBox, Frame, ObjectDetection, PersonDetection
from src.pipeline import Pipeline


class FakeEngine:
    """Engine dublê: devolve o que dissermos, sem carregar modelo."""

    def __init__(self, persons=None, objects=None):
        self.persons = persons or []
        self.objects = objects or []
        self.pose_calls = 0

    def detect(self, image):
        return list(self.persons), list(self.objects)

    def pose(self, image, boxes):
        self.pose_calls += 1
        out = []
        for _ in boxes:
            kp = np.zeros((17, 3), dtype=np.float32)
            kp[:, 2] = 0.8
            out.append(kp)
        return out

    def warmup(self):
        pass


def _cfg(zones):
    return AppConfig(
        store=StoreConfig(id="l", name="L"),
        cameras=[
            CameraConfig(name="cam1", rtsp_url="rtsp://x", target_fps=5, zones=zones)
        ],
    )


def _frame():
    return Frame("cam1", np.zeros((500, 1000, 3), dtype=np.uint8), ts=1.0, seq=1)


def test_no_person_skips_pose():
    """O gate é o que faz o sistema rodar em PC fraco: sem pessoa, sem pose."""
    engine = FakeEngine(persons=[])
    p = Pipeline(_cfg([]), engine)
    result = p.process_frame(_frame())
    assert result.had_person is False
    assert result.persons == []
    assert engine.pose_calls == 0


def test_person_outside_zone_skips_pose():
    zone = [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]  # metade direita
    engine = FakeEngine(
        persons=[PersonDetection(bbox=BBox(100, 100, 200, 400), conf=0.9)]  # pés em x=150
    )
    p = Pipeline(_cfg([zone]), engine)
    result = p.process_frame(_frame())
    assert result.had_person is False
    assert engine.pose_calls == 0


def test_person_inside_zone_gets_pose_and_track_id():
    zone = [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]
    engine = FakeEngine(
        persons=[PersonDetection(bbox=BBox(700, 100, 800, 400), conf=0.9)]
    )
    p = Pipeline(_cfg([zone]), engine)
    result = p.process_frame(_frame())
    assert result.had_person is True
    assert len(result.persons) == 1
    assert result.persons[0].person.track_id == 1
    assert result.persons[0].keypoints.shape == (17, 3)
    assert engine.pose_calls == 1


def test_track_id_is_stable_across_frames():
    engine = FakeEngine(
        persons=[PersonDetection(bbox=BBox(700, 100, 800, 400), conf=0.9)]
    )
    p = Pipeline(_cfg([]), engine)
    first = p.process_frame(_frame())
    second = p.process_frame(Frame("cam1", np.zeros((500, 1000, 3), np.uint8), 1.2, 2))
    assert first.persons[0].person.track_id == second.persons[0].person.track_id


def test_bags_are_passed_through():
    engine = FakeEngine(
        persons=[PersonDetection(bbox=BBox(700, 100, 800, 400), conf=0.9)],
        objects=[ObjectDetection(label="backpack", bbox=BBox(750, 150, 790, 220), conf=0.7)],
    )
    p = Pipeline(_cfg([]), engine)
    result = p.process_frame(_frame())
    assert result.objects[0].label == "backpack"


def test_status_reports_every_camera():
    p = Pipeline(_cfg([]), FakeEngine())
    st = p.status()
    assert "cam1" in st
    assert st["cam1"]["state"] == "offline"
    assert st["cam1"]["fps"] == 0.0
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pipeline'`

- [ ] **Step 3: Implementar `src/pipeline.py`**

```python
"""Orquestração: threads de captura + pool de inferência + gate + pose + track.

O caminho de um frame:
    RTSP → slot → worker → detect → [gate] → pose no recorte → track
A lógica de ocultação (Plano 2) pluga na saída de `process_frame`."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.capture.frame_slot import LatestFrameSlot
from src.capture.rtsp_capture import CameraThread
from src.config.settings import AppConfig
from src.core.types import CameraState, Frame, ObjectDetection, PersonPose
from src.detection.person_gate import PersonGate
from src.detection.tracker import Tracker
from src.inference.scheduler import Scheduler
from src.inference.worker_pool import WorkerPool

log = logging.getLogger(__name__)


@dataclass
class FrameResult:
    camera_name: str
    persons: list[PersonPose] = field(default_factory=list)
    objects: list[ObjectDetection] = field(default_factory=list)
    had_person: bool = False


class Pipeline:
    def __init__(self, cfg: AppConfig, engine) -> None:
        self.cfg = cfg
        self.engine = engine
        self.cameras = [c for c in cfg.cameras if c.enabled]
        self.slots: dict[str, LatestFrameSlot] = {
            c.name: LatestFrameSlot() for c in self.cameras
        }
        self.threads: dict[str, CameraThread] = {
            c.name: CameraThread(c, self.slots[c.name]) for c in self.cameras
        }
        self.scheduler = Scheduler([c.name for c in self.cameras])
        self.pool = WorkerPool(
            self.slots, self.scheduler, self._on_frame, workers=cfg.inference.workers
        )
        self._gates: dict[str, PersonGate] = {}  # criado no 1º frame (precisa do tamanho)
        self._trackers: dict[str, Tracker] = {
            c.name: Tracker(
                max_lost_seconds=c.effective_detection(cfg.detection).guards.track_lost_seconds
            )
            for c in self.cameras
        }
        self.on_result = None  # callback opcional: Callable[[FrameResult, Frame], None]

    def _gate_for(self, frame: Frame) -> PersonGate:
        gate = self._gates.get(frame.camera_name)
        if gate is None:
            h, w = frame.image.shape[:2]
            cam = next(c for c in self.cameras if c.name == frame.camera_name)
            gate = PersonGate(cam.zones, (w, h))
            self._gates[frame.camera_name] = gate
        return gate

    def process_frame(self, frame: Frame) -> FrameResult:
        persons, objects = self.engine.detect(frame.image)
        inside = self._gate_for(frame).filter(persons)
        if not inside:
            # Caminho barato: sem pessoa na zona, nada de pose. É por isso que
            # câmera de corredor vazio quase não custa CPU.
            self._trackers[frame.camera_name].update([], frame.ts)
            return FrameResult(frame.camera_name, had_person=False)

        tracked = self._trackers[frame.camera_name].update(inside, frame.ts)
        keypoints = self.engine.pose(frame.image, [p.bbox for p in tracked])
        poses = [PersonPose(person=p, keypoints=k) for p, k in zip(tracked, keypoints)]
        return FrameResult(frame.camera_name, poses, objects, had_person=True)

    def _on_frame(self, frame: Frame) -> bool:
        result = self.process_frame(frame)
        if self.on_result:
            self.on_result(result, frame)
        return result.had_person

    def start(self) -> None:
        self.engine.warmup()
        for t in self.threads.values():
            t.start()
        self.pool.start()
        log.info("pipeline iniciado com %d câmera(s)", len(self.cameras))

    def stop(self) -> None:
        self.pool.stop()
        for t in self.threads.values():
            t.stop()
        log.info("pipeline parado")

    def status(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for name, t in self.threads.items():
            state = t.state if t.state else CameraState.OFFLINE
            out[name] = {
                "state": state.value,
                "fps": round(t.effective_fps, 1),
                "dropped": self.slots[name].dropped,
            }
        return out
```

- [ ] **Step 4: Implementar `src/main.py`**

```python
"""Entrypoint headless. A UI (Plano 3) reaproveita o mesmo Pipeline.

    python -m src.main --config config/config.json
"""
from __future__ import annotations

import argparse
import logging
import signal
import time

from src.config.settings import AppConfig, ConfigError
from src.inference.engine import InferenceEngine
from src.pipeline import Pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="Prevenção de Perdas — núcleo")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--status-every", type=float, default=5.0)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("main")

    try:
        cfg = AppConfig.load(args.config)
    except ConfigError as e:
        log.error("%s", e)
        return 2

    pipeline = Pipeline(cfg, InferenceEngine(cfg.inference))

    def _handle_person(result, frame):
        if result.had_person:
            log.info(
                "[%s] %d pessoa(s) na zona — ids=%s",
                result.camera_name,
                len(result.persons),
                [p.person.track_id for p in result.persons],
            )

    pipeline.on_result = _handle_person

    stopping = False

    def _stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    pipeline.start()
    try:
        while not stopping:
            time.sleep(args.status_every)
            for name, st in pipeline.status().items():
                log.info(
                    "câmera '%s': %s · %.1f fps · %d frames descartados",
                    name, st["state"], st["fps"], st["dropped"],
                )
    finally:
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Rodar e confirmar que passam**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS — 6 passed

- [ ] **Step 6: Rodar o sistema de verdade contra o DVR simulado**

Gravar um clipe seu, publicá-lo como se fosse um canal de DVR e ver o sistema rastreando você:

```bash
python dev/record_clips.py --label normal --seconds 15
python - <<'PY'
import json, time
from pathlib import Path
from dev.dvr_sim import DvrSim

clip = sorted(Path("dev/videos/normal").glob("*.mp4"))[0]
sim = DvrSim({"ch1": clip, "ch2": clip}).start()

cfg = json.loads(Path("config/config.example.json").read_text(encoding="utf-8"))
cfg["telegram"] = {"bot_token": "", "chat_id": ""}
cfg["inference"]["device"] = "cpu"
cfg["cameras"] = [
    {"name": "Canal 1", "rtsp_url": sim.url("ch1"), "target_fps": 5, "zones": []},
    {"name": "Canal 2", "rtsp_url": sim.url("ch2"), "target_fps": 5, "zones": []},
]
Path("config/config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("DVR simulado no ar. Rode em outro terminal:")
print("  python -m src.main --config config/config.json")
input("ENTER para derrubar o DVR simulado...")
sim.stop()
PY
```

Expected: o log mostra `[Canal 1] 1 pessoa(s) na zona — ids=[1]` e, a cada 5s, o status com FPS por câmera. Ao parar o `DvrSim`, o log passa a `RECONNECTING` e volta a `ONLINE` se o DVR voltar. **É o sistema funcionando ponta a ponta.**

- [ ] **Step 7: Commit**

```bash
git add src/pipeline.py src/main.py tests/test_pipeline.py
git commit -m "feat: pipeline (captura + gate + pose + track) e entrypoint headless"
```

---

## Task 12: Teste de capacidade do PC

**Files:**
- Create: `src/tools/benchmark.py`
- Test: `tests/tools/test_benchmark.py`

**Interfaces:**
- Consumes: `InferenceEngine`, `InferenceConfig`.
- Produces: `benchmark(engine, scenarios, seconds_per_scenario) -> BenchmarkReport`; `BenchmarkReport(rows: list[BenchmarkRow], cpu_name, cores, ram_gb)` com `.recommend(min_fps) -> str` e `.as_text() -> str`. `BenchmarkRow(cameras, people_per_frame, fps_sustained, cpu_percent)`.

**Por que existe:** o Adriano perguntou "esse PC aguenta 7, 8, 10 câmeras?". Este é o entregável que responde com número em vez de chute — e vira a ferramenta que ele roda em cada cliente antes de vender.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/tools/test_benchmark.py`:

```python
import numpy as np
import pytest

from src.core.types import BBox, PersonDetection
from src.tools.benchmark import BenchmarkReport, BenchmarkRow, benchmark


class SlowFakeEngine:
    """Cada detect custa ~5ms; cada pose custa ~10ms por pessoa."""

    def __init__(self, people: int):
        self.people = people

    def detect(self, image):
        _ = np.sum(image[:64, :64])  # trabalho de verdade, curto
        boxes = [
            PersonDetection(bbox=BBox(10 * i, 10, 60 + 10 * i, 200), conf=0.9)
            for i in range(self.people)
        ]
        return boxes, []

    def pose(self, image, boxes):
        out = []
        for _ in boxes:
            kp = np.zeros((17, 3), dtype=np.float32)
            kp[:, 2] = 0.9
            out.append(kp)
        return out

    def warmup(self):
        pass


def test_benchmark_produces_a_row_per_scenario():
    report = benchmark(
        SlowFakeEngine(people=1),
        scenarios=[(1, 1), (4, 1)],
        seconds_per_scenario=0.3,
    )
    assert len(report.rows) == 2
    assert report.rows[0].cameras == 1
    assert report.rows[1].cameras == 4
    assert report.rows[0].fps_sustained > 0


def test_more_cameras_lowers_fps_per_camera():
    report = benchmark(
        SlowFakeEngine(people=1), scenarios=[(1, 1), (8, 1)], seconds_per_scenario=0.4
    )
    assert report.rows[1].fps_sustained < report.rows[0].fps_sustained


def test_recommend_picks_the_largest_camera_count_above_min_fps():
    report = BenchmarkReport(
        rows=[
            BenchmarkRow(cameras=1, people_per_frame=1, fps_sustained=20.0, cpu_percent=30),
            BenchmarkRow(cameras=5, people_per_frame=1, fps_sustained=8.0, cpu_percent=60),
            BenchmarkRow(cameras=8, people_per_frame=1, fps_sustained=5.5, cpu_percent=80),
            BenchmarkRow(cameras=12, people_per_frame=1, fps_sustained=2.0, cpu_percent=98),
        ],
        cpu_name="Intel i5",
        cores=4,
        ram_gb=8.0,
    )
    texto = report.recommend(min_fps=5.0)
    assert "8 câmeras" in texto
    assert "12" not in texto


def test_recommend_warns_when_nothing_meets_min_fps():
    report = BenchmarkReport(
        rows=[BenchmarkRow(cameras=1, people_per_frame=1, fps_sustained=1.0, cpu_percent=99)],
        cpu_name="Celeron",
        cores=2,
        ram_gb=4.0,
    )
    texto = report.recommend(min_fps=5.0)
    assert "não" in texto.lower()


def test_as_text_mentions_hardware_and_rows():
    report = BenchmarkReport(
        rows=[BenchmarkRow(cameras=5, people_per_frame=2, fps_sustained=6.0, cpu_percent=70)],
        cpu_name="Intel i5-8250U",
        cores=4,
        ram_gb=8.0,
    )
    t = report.as_text()
    assert "Intel i5-8250U" in t
    assert "5" in t and "6.0" in t
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `pytest tests/tools/test_benchmark.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tools.benchmark'`

- [ ] **Step 3: Implementar `src/tools/benchmark.py`**

```python
"""Teste de capacidade: quantas câmeras ESTE PC aguenta?

Mede o que importa de verdade — não o número de câmeras, mas quantas PESSOAS
o PC consegue processar ao mesmo tempo. Câmera de corredor vazio custa quase
nada; o custo aparece quando há gente na zona. Por isso cada cenário é um par
(nº de câmeras, pessoas por frame)."""
from __future__ import annotations

import argparse
import platform
import time
from dataclasses import dataclass

import numpy as np
import psutil

DEFAULT_SCENARIOS = [(1, 1), (3, 1), (5, 1), (5, 3), (8, 1), (8, 3), (10, 2)]
FRAME = (360, 640, 3)


@dataclass
class BenchmarkRow:
    cameras: int
    people_per_frame: int
    fps_sustained: float  # FPS por câmera
    cpu_percent: float


@dataclass
class BenchmarkReport:
    rows: list[BenchmarkRow]
    cpu_name: str
    cores: int
    ram_gb: float

    def recommend(self, min_fps: float = 5.0) -> str:
        ok = [r for r in self.rows if r.fps_sustained >= min_fps]
        if not ok:
            return (
                f"Este PC NÃO sustenta nem 1 câmera a {min_fps:.0f} FPS. "
                "Recomendo trocar o equipamento ou reduzir o FPS alvo."
            )
        best = max(ok, key=lambda r: r.cameras)
        return (
            f"Este PC sustenta {best.cameras} câmeras a "
            f"{best.fps_sustained:.1f} FPS (CPU em {best.cpu_percent:.0f}%). "
            f"Acima disso, o FPS cai abaixo de {min_fps:.0f} e a detecção "
            "começa a perder gestos rápidos."
        )

    def as_text(self) -> str:
        linhas = [
            "TESTE DE CAPACIDADE — Prevenção de Perdas",
            "=" * 52,
            f"Processador: {self.cpu_name}",
            f"Núcleos: {self.cores} · Memória: {self.ram_gb:.1f} GB",
            "",
            f"{'Câmeras':>8} {'Pessoas':>8} {'FPS/câmera':>12} {'CPU':>6}",
            "-" * 52,
        ]
        for r in self.rows:
            linhas.append(
                f"{r.cameras:>8} {r.people_per_frame:>8} "
                f"{r.fps_sustained:>12.1f} {r.cpu_percent:>5.0f}%"
            )
        linhas += ["", self.recommend(), ""]
        linhas.append(
            "Observação: câmera sem ninguém na área custa quase nada. O limite "
            "real é quantas PESSOAS aparecem ao mesmo tempo, não quantas câmeras "
            "existem."
        )
        return "\n".join(linhas)


def benchmark(
    engine,
    scenarios: list[tuple[int, int]] | None = None,
    seconds_per_scenario: float = 3.0,
) -> BenchmarkReport:
    scenarios = scenarios or DEFAULT_SCENARIOS
    engine.warmup()
    image = np.random.randint(0, 255, FRAME, dtype=np.uint8)
    rows: list[BenchmarkRow] = []

    for cameras, people in scenarios:
        psutil.cpu_percent(interval=None)  # zera o contador
        t0 = time.monotonic()
        processed = 0
        while time.monotonic() - t0 < seconds_per_scenario:
            for _ in range(cameras):
                persons, _objs = engine.detect(image)
                if persons:
                    engine.pose(image, [p.bbox for p in persons])
                processed += 1
        elapsed = time.monotonic() - t0
        cpu = psutil.cpu_percent(interval=None)
        fps_total = processed / elapsed
        rows.append(
            BenchmarkRow(
                cameras=cameras,
                people_per_frame=people,
                fps_sustained=fps_total / cameras,
                cpu_percent=cpu,
            )
        )

    return BenchmarkReport(
        rows=rows,
        cpu_name=platform.processor() or platform.machine(),
        cores=psutil.cpu_count(logical=False) or psutil.cpu_count() or 0,
        ram_gb=psutil.virtual_memory().total / 1e9,
    )


if __name__ == "__main__":
    from src.config.settings import InferenceConfig
    from src.inference.engine import InferenceEngine

    ap = argparse.ArgumentParser(description="Teste de capacidade do PC")
    ap.add_argument("--device", default="openvino")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--min-fps", type=float, default=5.0)
    ap.add_argument("--out", default="relatorio-capacidade.txt")
    a = ap.parse_args()

    eng = InferenceEngine(InferenceConfig(device=a.device))
    report = benchmark(eng, seconds_per_scenario=a.seconds)
    texto = report.as_text()
    print(texto)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(texto)
    print(f"\nRelatório salvo em {a.out}")
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `pytest tests/tools/test_benchmark.py -v`
Expected: PASS — 5 passed

- [ ] **Step 5: Rodar o benchmark real neste PC**

Run: `python -m src.tools.benchmark --device cpu --seconds 3`
Expected: tabela com FPS por cenário e a frase de recomendação. Guardar o resultado — é o primeiro dado concreto para responder ao Adriano.

- [ ] **Step 6: Commit**

```bash
git add src/tools/ tests/tools/
git commit -m "feat: teste de capacidade do PC (quantas câmeras este hardware aguenta)"
```

---

## Fechamento do Plano 1

- [ ] **Rodar a suíte completa**

Run: `pytest -v -m "not slow"`
Expected: PASS — todos os testes rápidos.

Run: `pytest -v -m slow`
Expected: PASS — DVR simulado, modelo real, export OpenVINO.

- [ ] **Verificação manual (obrigatória antes de declarar pronto)**

1. `python dev/record_clips.py --label normal --seconds 15` — grave-se andando.
2. Suba o DVR simulado com esse clipe em 2 canais (script do Task 11, Step 6).
3. `python -m src.main --config config/config.json`
4. Confirme no log: pessoas detectadas com `track_id` estável, FPS por câmera, e reconexão automática ao derrubar e restaurar o DVR simulado.

- [ ] **Commit final**

```bash
git add -A
git commit -m "chore: plano 1 concluído — captura, inferência e tracking funcionando"
```

**Estado ao fim deste plano:** o sistema captura N câmeras RTSP com reconexão automática, detecta pessoas apenas dentro das zonas configuradas, roda pose no recorte de cada uma, mantém identidade entre frames, e sabe dizer quantas câmeras o PC aguenta. Falta a inteligência — que é o Plano 2.

**Próximo:** Plano 2 (F3) — coordenadas do corpo, zonas de ocultação, os quatro sinais, score, máquina de estados, modo replay e sweep de calibração. É o marco de 50%.
