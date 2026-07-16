"""Orquestração: threads de captura + pool de inferência + gate + pose + track.

O caminho de um frame:
    RTSP → slot → worker → detect → [gate] → pose no recorte → track
A lógica de ocultação (Plano 2) pluga na saída de `process_frame`."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from src.capture.frame_slot import LatestFrameSlot
from src.capture.rtsp_capture import CameraThread
from src.config.settings import AppConfig
from src.core.types import CameraState, Frame, ObjectDetection, PersonPose
from src.detection.concealment import ConcealmentAnalyzer, ConcealmentEvent
from src.detection.person_gate import PersonGate
from src.detection.tracker import Tracker
from src.evidence.clip_buffer import ClipBuffer
from src.inference.scheduler import Scheduler
from src.inference.worker_pool import WorkerPool

log = logging.getLogger(__name__)


@dataclass
class FrameResult:
    camera_name: str
    persons: list[PersonPose] = field(default_factory=list)
    objects: list[ObjectDetection] = field(default_factory=list)
    had_person: bool = False
    events: list[ConcealmentEvent] = field(default_factory=list)


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
        self._gates_lock = threading.Lock()
        self._trackers: dict[str, Tracker] = {
            c.name: Tracker(
                max_lost_seconds=c.effective_detection(cfg.detection).guards.track_lost_seconds
            )
            for c in self.cameras
        }
        self._analyzers: dict[str, ConcealmentAnalyzer] = {
            c.name: ConcealmentAnalyzer(
                c.effective_detection(cfg.detection),
                fps_hint=c.target_fps,
            )
            for c in self.cameras
        }
        # "antes" do clipe de evidência: alimentado a cada frame, mesmo sem
        # pessoa na zona — é o que garante o gesto inteiro no clipe salvo.
        self.clip_buffers: dict[str, ClipBuffer] = {
            c.name: ClipBuffer(
                seconds=cfg.evidence.clip_pre_seconds + cfg.evidence.clip_post_seconds + 1.0,
                fps_hint=c.target_fps,
            )
            for c in self.cameras
        }
        self.on_result = None  # callback opcional: Callable[[FrameResult, Frame], None]

    def _gate_for(self, frame: Frame) -> PersonGate:
        # Double-checked locking: leitura sem lock, depois re-checa dentro do lock
        gate = self._gates.get(frame.camera_name)
        if gate is None:
            with self._gates_lock:
                # Re-checa após adquirir o lock (outra thread pode ter criado)
                gate = self._gates.get(frame.camera_name)
                if gate is None:
                    h, w = frame.image.shape[:2]
                    cam = next(c for c in self.cameras if c.name == frame.camera_name)
                    gate = PersonGate(cam.zones, (w, h))
                    self._gates[frame.camera_name] = gate
        return gate

    def process_frame(self, frame: Frame) -> FrameResult:
        # alimenta o buffer ANTES do gate: o clipe precisa do "antes" mesmo
        # nos frames em que ninguém está na zona monitorada.
        buf = self.clip_buffers.get(frame.camera_name)
        if buf is not None:
            buf.add(frame.image, frame.ts)

        persons, objects = self.engine.detect(frame.image)
        inside = self._gate_for(frame).filter(persons)
        if not inside:
            # Caminho barato: sem pessoa na zona, nada de pose. É por isso que
            # câmera de corredor vazio quase não custa CPU.
            self._trackers[frame.camera_name].update([], frame.ts)
            self._analyzers[frame.camera_name].update([], [], frame.ts)
            result = FrameResult(frame.camera_name, had_person=False)
        else:
            tracked = self._trackers[frame.camera_name].update(inside, frame.ts)
            keypoints = self.engine.pose(frame.image, [p.bbox for p in tracked])
            poses = [PersonPose(person=p, keypoints=k) for p, k in zip(tracked, keypoints)]
            events = self._analyzers[frame.camera_name].update(poses, objects, frame.ts)
            result = FrameResult(frame.camera_name, poses, objects, had_person=True, events=events)

        self._emit(result, frame)
        return result

    def _emit(self, result: FrameResult, frame: Frame) -> None:
        """Chama o callback de operação (evidência/alerta), se houver. Uma
        falha do callback (ex.: disco cheio, banco travado) não pode derrubar
        o processamento de frames — por isso o erro é só logado."""
        if self.on_result is None:
            return
        try:
            self.on_result(result, frame)
        except Exception:
            log.exception("erro no callback on_result da câmera '%s'", frame.camera_name)

    def _on_frame(self, frame: Frame) -> bool:
        result = self.process_frame(frame)
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
