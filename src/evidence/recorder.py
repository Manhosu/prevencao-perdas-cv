"""Gravador de evidência: foto anotada + clipe curto + registro no banco.

Ordem importa: o registro no banco vem PRIMEIRO. Se o disco estiver cheio ou o
arquivo travado, o evento não pode simplesmente sumir — ele fica registrado, só
sem a mídia."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.config.settings import EvidenceConfig, StoreConfig
from src.detection.concealment import ConcealmentEvent
from src.storage.db import Database

log = logging.getLogger(__name__)


def _slug(nome: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", nome).strip("_") or "camera"


@dataclass
class EvidenceResult:
    """O que `record()` gravou para ESTE evento — quem chama usa isso direto,
    nunca re-pergunta ao banco 'qual foi o ultimo evento' (com varias câmeras
    concorrentes, o ultimo pode ser de outra câmera)."""
    event_id: int
    image_path: str | None
    clip_path: str | None
    ts_local: datetime


class EvidenceRecorder:
    def __init__(self, db: Database, cfg: EvidenceConfig, store: StoreConfig) -> None:
        self.db = db
        self.cfg = cfg
        self.store = store

    def annotate(self, frame: np.ndarray, event: ConcealmentEvent) -> np.ndarray:
        """Marca o frame do alerta: barra vermelha + zona + score."""
        w = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 255), -1)
        cv2.putText(frame, f"OCULTACAO  {event.zone}  score {event.score:.2f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def _dir_for(self, camera_name: str, agora: datetime) -> Path:
        d = Path(self.cfg.dir) / _slug(camera_name) / agora.strftime("%Y-%m-%d")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def record(self, event: ConcealmentEvent, camera_name: str,
               frame_bgr: np.ndarray, clip_buffer=None) -> EvidenceResult:
        agora = datetime.now()
        agora_utc = datetime.now(timezone.utc)

        # 1) registra primeiro (o evento nao pode se perder por falha de disco)
        event_id = self.db.insert_event(
            store_id=self.store.id, camera_name=camera_name,
            ts_utc=agora_utc.isoformat(), ts_local=agora.isoformat(),
            track_id=event.track_id, score=event.score, zone=event.zone,
            signals=event.signals, image_path=None, clip_path=None,
        )

        base = f"{agora.strftime('%H-%M-%S')}_id{event.track_id}"
        image_path = clip_path = None

        # 2) foto anotada
        try:
            d = self._dir_for(camera_name, agora)
            p = d / f"{base}.jpg"
            if cv2.imwrite(str(p), self.annotate(frame_bgr.copy(), event)):
                image_path = str(p)
        except Exception:
            log.exception("falha ao salvar a foto da evidencia (evento %s registrado mesmo assim)",
                          event_id)

        # 3) clipe curto (o "antes" vem do buffer circular)
        if clip_buffer is not None:
            try:
                clip_path = self._save_clip(clip_buffer, event, camera_name, agora, base)
            except Exception:
                log.exception("falha ao salvar o clipe da evidencia")

        if image_path or clip_path:
            self.db.update_event_paths(event_id, image_path, clip_path)
        return EvidenceResult(event_id=event_id, image_path=image_path,
                              clip_path=clip_path, ts_local=agora)

    def _save_clip(self, buf, event, camera_name, agora, base) -> str | None:
        t0 = event.ts - self.cfg.clip_pre_seconds
        t1 = event.ts + self.cfg.clip_post_seconds
        frames = buf.frames_between(t0, t1)
        if len(frames) < 2:
            return None
        h, w = frames[0][1].shape[:2]
        span = frames[-1][0] - frames[0][0]
        fps = max(1.0, (len(frames) - 1) / span) if span > 0 else 5.0
        p = self._dir_for(camera_name, agora) / f"{base}.mp4"
        wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        try:
            for _, img in frames:
                wr.write(img)
        finally:
            wr.release()
        return str(p)
