"""Associação de identidade entre frames (IoU + idade + predição de movimento).

Manter o track_id estável é requisito da lógica de ocultação: o dwell da mão
na zona é acumulado POR PESSOA. Se o id trocar no meio do gesto, o contador
zera e o furto passa batido. Por isso o track sobrevive a alguns frames sem
detecção (pessoa passa atrás de uma gôndola).

Duas correções pós-revisão em relação à primeira versão (guloso + posição
parada):

1. Predição de movimento: cada track guarda uma velocidade estimada (px/s)
   e a associação compara a detecção nova com a posição PREVISTA do track
   (bbox anterior extrapolado pela velocidade), não com a última posição
   parada. Isso é o que impede a troca de id quando duas pessoas se cruzam
   andando em direções opostas — a posição prevista de cada uma continua se
   afastando da outra mesmo no instante em que as caixas reais se sobrepõem.

2. Associação ótima (maximiza a soma dos IoUs) em vez de gulosa (pega o
   maior IoU isolado). Ver `_optimal_assignment` para a justificativa de não
   usar scipy.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

from src.core.types import BBox, PersonDetection

IOU_MATCH_MIN = 0.25

# Acima deste tamanho (nº de pessoas ou de tracks, o que for maior) a busca
# exaustiva por permutações (custo N!) fica cara demais para rodar por
# frame; caímos de volta para o pareamento guloso só nesse caso extremo de
# loja muito cheia. Em cena normal de varejo o número de pessoas é pequeno
# (a revisão testou < 8), então o caminho ótimo é o caso comum.
_MAX_EXACT_ASSIGNMENT_SIZE = 8

# Amortecimento da velocidade estimada: em vez de confiar 100% na medição
# mais recente (deslocamento entre os dois últimos frames / Δt), mistura
# com a velocidade anterior. Isso evita que ruído de detecção (bbox
# tremendo de frame a frame) faça a predição "disparar" para um lado.
_VELOCITY_DAMPING = 0.5


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
    # Velocidade em px/segundo de cada canto da bbox: (vx1, vy1, vx2, vy2).
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    has_velocity: bool = False

    def predicted_bbox(self, ts: float) -> BBox:
        """Extrapola a posição do track para o instante `ts` usando a
        velocidade estimada. No primeiro frame do track (ainda sem
        velocidade, pois é preciso de duas amostras para medir deslocamento)
        usa a posição parada — não há outra opção nesse instante."""
        if not self.has_velocity:
            return self.bbox
        dt = ts - self.last_seen
        vx1, vy1, vx2, vy2 = self.velocity
        return BBox(
            self.bbox.x1 + vx1 * dt,
            self.bbox.y1 + vy1 * dt,
            self.bbox.x2 + vx2 * dt,
            self.bbox.y2 + vy2 * dt,
        )

    def apply_detection(self, new_bbox: BBox, ts: float) -> None:
        """Registra a nova posição observada, atualizando a velocidade
        estimada (com amortecimento) antes de sobrescrever a bbox."""
        dt = ts - self.last_seen
        if dt > 0:
            measured = (
                (new_bbox.x1 - self.bbox.x1) / dt,
                (new_bbox.y1 - self.bbox.y1) / dt,
                (new_bbox.x2 - self.bbox.x2) / dt,
                (new_bbox.y2 - self.bbox.y2) / dt,
            )
            if self.has_velocity:
                old = self.velocity
                self.velocity = tuple(
                    old[i] * _VELOCITY_DAMPING + measured[i] * (1 - _VELOCITY_DAMPING)
                    for i in range(4)
                )
            else:
                self.velocity = measured
                self.has_velocity = True
        self.bbox = new_bbox
        self.last_seen = ts


def _optimal_assignment(score_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """Resolve o problema de atribuição maximizando a soma dos escores de
    IoU (matriz detecções x tracks), respeitando IOU_MATCH_MIN.

    Por que não algoritmo húngaro: com o número de pessoas em cena
    tipicamente pequeno (a revisão testou < 8), a busca exaustiva sobre
    todas as permutações possíveis é mais simples de escrever e de auditar
    do que uma implementação do húngaro à mão — e é ótima por construção
    (exame de todas as atribuições, não uma heurística). scipy não está
    disponível (não pode ser adicionado como dependência). Acima de
    `_MAX_EXACT_ASSIGNMENT_SIZE` o fatorial explode e o chamador cai para o
    pareamento guloso (ver `_greedy_assignment`) como salvaguarda.

    IMPORTANTE: os escores abaixo de IOU_MATCH_MIN já chegam zerados nesta
    matriz (ver `Tracker.update`) — do contrário, vários pares fracos
    poderiam se combinar (soma) e "vencer" um único par forte e válido na
    otimização, mesmo que nenhum deles isoladamente pudesse ser associado.
    Zerar antes torna um par abaixo do limiar equivalente a não casar com
    ninguém (mesma contribuição da linha/coluna fantasma do preenchimento).
    """
    n_rows = len(score_matrix)
    n_cols = len(score_matrix[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        return []
    n = max(n_rows, n_cols)
    # Preenche até virar quadrada: "casar" com uma linha/coluna fantasma
    # equivale a não casar (contribui 0 à soma), permitindo que a busca por
    # permutações trate matches parciais (nem toda pessoa tem track, nem
    # todo track tem pessoa) como um caso particular do problema quadrado.
    padded = [
        [score_matrix[r][c] if r < n_rows and c < n_cols else 0.0 for c in range(n)]
        for r in range(n)
    ]
    best_perm: tuple[int, ...] = tuple(range(n))
    best_score = -1.0
    for perm in permutations(range(n)):
        total = sum(padded[r][perm[r]] for r in range(n))
        if total > best_score:
            best_score = total
            best_perm = perm
    return [(r, c) for r, c in enumerate(best_perm) if r < n_rows and c < n_cols]


def _greedy_assignment(score_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """Pareamento guloso pelo maior IoU isolado — salvaguarda usada apenas
    quando o número de pessoas/tracks excede `_MAX_EXACT_ASSIGNMENT_SIZE`
    (busca exaustiva ficaria cara demais). Não é o caminho principal."""
    n_rows = len(score_matrix)
    n_cols = len(score_matrix[0]) if n_rows else 0
    candidates = sorted(
        (
            (score_matrix[r][c], r, c)
            for r in range(n_rows)
            for c in range(n_cols)
        ),
        reverse=True,
    )
    taken_r: set[int] = set()
    taken_c: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for score, r, c in candidates:
        if r in taken_r or c in taken_c:
            continue
        pairs.append((r, c))
        taken_r.add(r)
        taken_c.add(c)
    return pairs


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

        n_p = len(persons)
        n_t = len(self._tracks)
        assigned: dict[int, int] = {}  # índice da pessoa -> track_id

        if n_p and n_t:
            # Compara cada detecção com a posição PREVISTA de cada track
            # (bbox extrapolado pela velocidade estimada), não com a última
            # posição parada — ver docstring do módulo e `_Track.predicted_bbox`.
            # Escores abaixo do limiar já entram zerados (ver docstring de
            # `_optimal_assignment` para o porquê).
            def _clipped_iou(p: PersonDetection, track: _Track) -> float:
                score = iou(p.bbox, track.predicted_bbox(ts))
                return score if score >= IOU_MATCH_MIN else 0.0

            score_matrix = [
                [_clipped_iou(p, self._tracks[ti]) for ti in range(n_t)]
                for p in persons
            ]

            size = max(n_p, n_t)
            if size <= _MAX_EXACT_ASSIGNMENT_SIZE:
                pairs = _optimal_assignment(score_matrix)
            else:
                pairs = _greedy_assignment(score_matrix)

            for pi, ti in pairs:
                if score_matrix[pi][ti] < IOU_MATCH_MIN:
                    continue
                track = self._tracks[ti]
                track.apply_detection(persons[pi].bbox, ts)
                assigned[pi] = track.track_id

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
