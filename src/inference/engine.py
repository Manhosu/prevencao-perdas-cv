"""Wrapper dos modelos YOLO. Duas decisões que carregam o desempenho:

1. Export para OpenVINO (cacheado): CPU Intel de PDV roda 2-4x mais rápido.
2. Pose no RECORTE da pessoa, não no frame inteiro: em câmera de teto a
   pessoa ocupa poucos pixels, e rodar a pose no recorte multiplica a
   resolução efetiva sobre o corpo — além de ser mais barato."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

from src.config.settings import InferenceConfig
from src.core.types import BBox, ObjectDetection, PersonDetection

log = logging.getLogger(__name__)

COCO_PERSON = 0
COCO_BAGS = {24: "backpack", 26: "handbag"}
CROP_MARGIN = 0.1  # expande a caixa antes de recortar: pose precisa do contorno
POSE_INPUT = 320  # o recorte é pequeno; 320 basta e é rápido

# Grava-se ao final de um export bem-sucedido. Um export interrompido (ex.:
# queda de energia num PDV) não deixa o marcador, então o cache incompleto
# é detectado como inválido e reexportado na próxima chamada.
_EXPORT_MARKER = ".export_ok"


def _openvino_export_is_valid(out: Path, stem: str) -> bool:
    """Cache só é confiável se: o diretório existir, tiver os arquivos
    essenciais do OpenVINO (.xml e .bin do modelo) E o marcador de export
    bem-sucedido com dynamic=True. Falta qualquer um desses três (export
    interrompido, diretório corrompido, ou cache antigo exportado sem
    dynamic=True) e o cache é considerado inválido."""
    return (
        out.is_dir()
        and (out / f"{stem}.xml").is_file()
        and (out / f"{stem}.bin").is_file()
        and (out / _EXPORT_MARKER).is_file()
    )


def export_openvino(pt_path: str | Path) -> Path:
    """Exporta o .pt para OpenVINO uma única vez, com dynamic=True (entrada de
    resolução variável). Sem dynamic=True o modelo exportado fica travado na
    resolução estática 640x640: detect() (imgsz=640) funciona, mas pose()
    roda no recorte com imgsz=POSE_INPUT (320) e quebra a partir da 2a
    chamada com RuntimeError de shape incompatível — e como warmup() só
    chama pose() uma vez, esse bug passava despercebido no boot (falso
    verde) e só estourava no primeiro frame real com pessoa.

    O diretório exportado fica ao lado do .pt e é reaproveitado nas
    execuções seguintes, mas só se passar em _openvino_export_is_valid;
    caso contrário é apagado e reexportado do zero."""
    from ultralytics import YOLO

    pt = Path(pt_path)
    out = pt.with_name(f"{pt.stem}_openvino_model")
    if _openvino_export_is_valid(out, pt.stem):
        return out
    if out.exists():
        log.warning(
            "cache OpenVINO em %s incompleto, corrompido ou exportado sem "
            "dynamic=True: apagando e reexportando",
            out,
        )
        shutil.rmtree(out)
    log.info("exportando %s para OpenVINO (só na primeira vez)...", pt.name)
    YOLO(str(pt)).export(format="openvino", half=False, dynamic=True)
    # Só grava o marcador depois do export completar: se o processo morrer
    # no meio (ex.: queda de energia), nenhum marcador fica gravado e o
    # cache incompleto é reexportado automaticamente na próxima chamada.
    (out / _EXPORT_MARKER).write_text("dynamic=True\n", encoding="utf-8")
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
