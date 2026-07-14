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
