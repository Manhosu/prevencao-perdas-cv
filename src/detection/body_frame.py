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
