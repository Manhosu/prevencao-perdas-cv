"""Configuração da aplicação. Toda constante de detecção mora aqui —
calibrar o sistema é editar JSON, nunca código."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ConfigError(Exception):
    """Erro de configuração legível para o usuário final (vai para a UI)."""


# Tradução dos tipos de erro mais comuns do pydantic para uma explicação em
# português, sem jargão (nada de "extra_forbidden" nem link para a doc do
# pydantic). Quem lê essa mensagem é um técnico instalador, não programador.
_PYDANTIC_ERROR_TRANSLATIONS: dict[str, str] = {
    "extra_forbidden": "campo desconhecido",
    "missing": "campo obrigatório ausente",
    "float_parsing": "valor deve ser um número",
    "int_parsing": "valor deve ser um número",
    "bool_parsing": "valor deve ser verdadeiro ou falso",
    "greater_than": "valor fora da faixa permitida",
    "greater_than_equal": "valor fora da faixa permitida",
    "less_than": "valor fora da faixa permitida",
    "less_than_equal": "valor fora da faixa permitida",
}


def _format_error_loc(loc: tuple) -> str:
    """Formata o caminho de um erro (ex.: ('guards', 'min_person_px')) como
    'guards.min_person_px', legível para quem não é programador."""
    parts: list[str] = []
    for item in loc:
        if isinstance(item, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{item}]"
            else:
                parts.append(f"[{item}]")
        else:
            parts.append(str(item))
    return ".".join(parts)


def format_validation_error(exc: ValidationError) -> str:
    """Converte um ValidationError do pydantic (a lista estruturada de
    exc.errors(), não o __str__ cru) em texto em português, uma linha por
    erro, no formato 'campo: explicação'. Nunca deixa vazar jargão do
    pydantic (tipos como 'extra_forbidden') nem o link para a doc."""
    lines: list[str] = []
    for err in exc.errors():
        field = _format_error_loc(err.get("loc", ()))
        err_type = err.get("type", "")
        if err_type == "value_error":
            # Erro de um @field_validator com raise ValueError(...) — a
            # mensagem do próprio validador já está em português.
            explanation = str(err.get("msg", "")).removeprefix("Value error, ")
        else:
            explanation = _PYDANTIC_ERROR_TRANSLATIONS.get(
                err_type, "valor inválido"
            )
        lines.append(f"{field}: {explanation}" if field else explanation)
    return "\n".join(lines)


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
    rate_limit_per_min: int = Field(default=15, gt=0)


class InferenceConfig(_Strict):
    device: str = "openvino"  # openvino | onnx | cpu
    person_model: str = "models/yolo11n.pt"
    pose_model: str = "models/yolo11n-pose.pt"
    detect_size: int = Field(default=640, ge=32)
    pose_on_crop: bool = True
    workers: int = Field(default=2, ge=1)
    detect_bags: bool = True


class Weights(_Strict):
    dwell: float = Field(default=0.70, ge=0.0, le=1.0)
    approach: float = Field(default=0.25, ge=0.0, le=1.0)
    vanish: float = Field(default=0.55, ge=0.0, le=1.0)
    retract: float = Field(default=0.15, ge=0.0, le=1.0)


class ZoneWeights(_Strict):
    waist: float = Field(default=1.00, ge=0.0, le=2.0)
    torso: float = Field(default=0.95, ge=0.0, le=2.0)
    back_waist: float = Field(default=1.05, ge=0.0, le=2.0)
    bag: float = Field(default=1.00, ge=0.0, le=2.0)


class Geometry(_Strict):
    waist_y: tuple[float, float] = (-0.45, 0.25)
    waist_x: tuple[float, float] = (0.10, 0.85)
    torso_y: tuple[float, float] = (0.15, 0.85)
    torso_x_max: float = 0.55
    reach_y_min: float = 0.9
    reach_x_min: float = 0.95


class Guards(_Strict):
    kp_conf_min: float = Field(default=0.35, ge=0.0, le=1.0)
    pose_quality_min: float = Field(default=0.40, ge=0.0, le=1.0)
    min_person_px: int = Field(default=120, gt=0)
    vanish_grace_seconds: float = Field(default=0.4, gt=0)
    vanish_max_seconds: float = Field(default=3.0, gt=0)
    gap_frames: int = Field(default=2, ge=0)
    track_lost_seconds: float = Field(default=2.0, gt=0)


class DetectionConfig(_Strict):
    threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    dwell_seconds: float = Field(default=1.2, gt=0)
    window_seconds: float = Field(default=3.0, gt=0)
    cooldown_seconds: float = Field(default=30.0, gt=0)
    weights: Weights = Field(default_factory=Weights)
    zone_weights: ZoneWeights = Field(default_factory=ZoneWeights)
    geometry: Geometry = Field(default_factory=Geometry)
    guards: Guards = Field(default_factory=Guards)


class EvidenceConfig(_Strict):
    dir: str = "evidence"
    retention_days: int = Field(default=30, ge=1)
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
    target_fps: float = Field(default=5.0, gt=0, le=30)
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
        é isso que torna a calibração por câmera possível.

        Sempre devolve um DetectionConfig independente: mesmo sem overrides,
        devolve uma cópia profunda de `base`, nunca o objeto global por
        referência — do contrário, código que mutasse o retorno corromperia
        silenciosamente o default compartilhado por todas as câmeras."""
        if not self.overrides:
            return base.model_copy(deep=True)
        merged = _deep_merge(base.model_dump(), self.overrides)
        try:
            return DetectionConfig(**merged)
        except ValidationError as e:
            raise ConfigError(
                f"overrides inválidos na câmera '{self.name}':\n"
                f"{format_validation_error(e)}"
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
            cfg = cls(**data)
        except ValidationError as e:
            raise ConfigError(
                f"configuração inválida em {p}:\n{format_validation_error(e)}"
            ) from e
        # Um typo em `overrides` (ex.: "threshhold" em vez de "threshold")
        # não pode sobreviver ao load: se não fosse validado aqui, o sistema
        # só quebraria muito depois, quando algum módulo chamasse
        # effective_detection() — e o técnico instalador já teria ido embora.
        for camera in cfg.cameras:
            camera.effective_detection(cfg.detection)
        return cfg

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
