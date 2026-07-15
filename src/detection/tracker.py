"""Associação de identidade entre frames (IoU + idade + predição de movimento).

Manter o track_id estável é requisito da lógica de ocultação: o dwell da mão
na zona é acumulado POR PESSOA. Se o id trocar no meio do gesto, o contador
zera e o furto passa batido. Por isso o track sobrevive a alguns frames sem
detecção (pessoa passa atrás de uma gôndola).

Três correções pós-revisão em relação à primeira versão (guloso + posição
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

3. Gating por proximidade para pessoa rápida: só a predição de movimento
   (item 1) não é suficiente em velocidade alta — duas caixas do mesmo
   tamanho deslocadas mais que ~0.6x a largura já têm IoU abaixo de
   IOU_MATCH_MIN mesmo quando a predição acerta a posição exata (é
   geometria de interseção, não erro de predição). Em termos de pessoa: com
   caixa de ~50px de largura, isso é ~30px por frame — a 5 FPS já é ritmo de
   caminhada normal, não de corrida. Por isso a associação complementa o
   IoU com um segundo critério: se o centro da detecção está perto do
   centro PREVISTO do track (dentro de um raio proporcional ao tamanho do
   track), aceita a associação mesmo com IoU abaixo do limiar, com um
   escore propositalmente menor que qualquer IoU real (para nunca vencer um
   par com IoU genuíno na atribuição ótima — ver `_proximity_gate_score` e
   `_optimal_assignment`). Isso só vale para tracks que já têm velocidade
   estimada, para não colar dois tracks parados e próximos (fila de caixa)
   que nunca se moveram.

   Faixa suportada na prática (medida com caixa 50x150, ver
   `tests/detection/test_tracker.py`): uma vez que o track já tem uma
   velocidade estimada (após o primeiro casamento bem-sucedido), o gating
   sustenta associação com deslocamentos de dezenas de pixels por frame
   acima do que o IoU puro cobre sozinho — 40 px/frame mantém o id do
   início ao fim nos testes. O caso que o gating NÃO cobre é a primeira
   transição de um track recém-criado (ainda sem velocidade estimada,
   apenas uma amostra de posição): se a pessoa já aparece se deslocando
   mais que ~0.6x a largura da caixa entre a primeira e a segunda detecção,
   essa transição específica ainda perde o id (ver teste que documenta esse
   limite conhecido). Na prática isso é raro — a maioria das pessoas entra
   em cena e é detectada por 1-2 frames antes de atingir velocidade de
   corrida.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

from src.core.types import BBox, PersonDetection

IOU_MATCH_MIN = 0.25

# Acima deste tamanho (nº de pessoas ou de tracks, o que for maior) a busca
# exaustiva por permutações (custo N!) fica cara demais para rodar por
# frame; caímos de volta para o pareamento guloso só nesse caso extremo de
# loja muito cheia. Corte deliberadamente baixo (6! = 720, < 1ms) para não
# encostar no penhasco fatorial: 7! já é 5.040 e 8! é 40.320 (~43ms medidos
# por update() — caro demais a 5 FPS com o pool de inferência compartilhado
# entre câmeras, onde a associação é só uma fatia do orçamento de 200ms por
# frame). Em cena normal de varejo o número de pessoas é pequeno, então o
# caminho ótimo continua sendo o caso comum mesmo com o corte em 6.
_MAX_EXACT_ASSIGNMENT_SIZE = 6

# Alcance do gating por proximidade (ver docstring do módulo, item 3):
# distância entre o centro da detecção e o centro PREVISTO do track,
# expressa como múltiplo do maior lado da caixa do track. 0.8-1.0 dá margem
# para deslocamento rápido sem abrir demais (evita colar em uma pessoa
# muito mais distante que a própria caixa).
GATE_FACTOR = 0.9

# Teto do escore de gating por proximidade: precisa ficar estritamente
# abaixo de IOU_MATCH_MIN para que um par com IoU real genuíno SEMPRE vença
# um par de gating na atribuição ótima — inclusive somado: como a
# atribuição é uma permutação (bijeção), trocar k pares reais por k pares de
# gating nunca aumenta a soma total, porque k * _GATE_SCORE_CEILING <
# k * IOU_MATCH_MIN <= soma dos k pares reais removidos. Isso preserva a
# garantia da matriz zerada abaixo do limiar (ver docstring de
# `_optimal_assignment`) mesmo com o gating misturado na mesma matriz.
_GATE_SCORE_CEILING = IOU_MATCH_MIN * 0.5

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


def _proximity_gate_score(detection_bbox: BBox, track: _Track, predicted: BBox) -> float:
    """Complementa o IoU para pessoa rápida (ver item 3 da docstring do
    módulo): quando o IoU real entre `detection_bbox` e `predicted` fica
    abaixo de IOU_MATCH_MIN, mas o centro da detecção está perto do centro
    PREVISTO do track, devolve um escore pequeno e positivo para permitir a
    associação mesmo assim — sempre menor que IOU_MATCH_MIN
    (`_GATE_SCORE_CEILING`), para que um IoU real genuíno tenha sempre
    prioridade na atribuição ótima.

    Só se aplica a tracks que já têm velocidade estimada (`has_velocity`):
    sem isso, dois tracks parados e próximos (fila de caixa) poderiam
    "colar" um no outro mesmo nunca tendo se movido — a velocidade
    estimada é o sinal de que o track é de fato um alvo em movimento, não
    apenas vizinho de outro track parado.
    """
    if not track.has_velocity:
        return 0.0
    dx = detection_bbox.center[0] - predicted.center[0]
    dy = detection_bbox.center[1] - predicted.center[1]
    dist = (dx * dx + dy * dy) ** 0.5
    radius = GATE_FACTOR * max(track.bbox.width, track.bbox.height)
    if radius <= 0 or dist > radius:
        return 0.0
    proximity = 1.0 - dist / radius
    return _GATE_SCORE_CEILING * proximity


def _optimal_assignment(score_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """Resolve o problema de atribuição maximizando a soma dos escores
    (matriz detecções x tracks) — escore é IoU quando IoU >= IOU_MATCH_MIN,
    ou um escore de gating por proximidade menor (ver `_proximity_gate_score`)
    quando não é.

    Por que não algoritmo húngaro: com o número de pessoas em cena
    tipicamente pequeno (a revisão testou < 8, e `_MAX_EXACT_ASSIGNMENT_SIZE`
    corta em 6 para conter o custo fatorial — ver comentário da constante), a
    busca exaustiva sobre todas as permutações possíveis é mais simples de
    escrever e de auditar do que uma implementação do húngaro à mão — e é
    ótima por construção (exame de todas as atribuições, não uma
    heurística). scipy não está disponível (não pode ser adicionado como
    dependência). Acima de `_MAX_EXACT_ASSIGNMENT_SIZE` o fatorial explode e
    o chamador cai para o pareamento guloso (ver `_greedy_assignment`) como
    salvaguarda.

    IMPORTANTE: os escores que não são nem IoU válido (>= IOU_MATCH_MIN) nem
    gating válido já chegam zerados nesta matriz (ver `Tracker.update`) — do
    contrário, vários pares fracos poderiam se combinar (soma) e "vencer" um
    único par forte e válido na otimização, mesmo que nenhum deles
    isoladamente pudesse ser associado. Zerar antes torna um par inválido
    equivalente a não casar com ninguém (mesma contribuição da linha/coluna
    fantasma do preenchimento). O escore de gating por proximidade, quando
    presente, é estritamente menor que IOU_MATCH_MIN (`_GATE_SCORE_CEILING`)
    justamente para preservar essa garantia: mesmo somados, k pares de
    gating nunca superam k pares de IoU real (ver comentário de
    `_GATE_SCORE_CEILING`), então nenhum par com IoU genuíno é sacrificado
    pela otimização em favor de gating.
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
            # Escore é o IoU quando >= IOU_MATCH_MIN; caso contrário, tenta o
            # gating por proximidade (pessoa rápida — ver item 3 da docstring
            # do módulo e `_proximity_gate_score`), que devolve um escore
            # menor que IOU_MATCH_MIN ou 0.0 se nem o gating se aplica. Pares
            # que não atingem nenhum dos dois critérios entram zerados na
            # matriz (ver docstring de `_optimal_assignment` para o porquê).
            def _match_score(p: PersonDetection, track: _Track) -> float:
                predicted = track.predicted_bbox(ts)
                score = iou(p.bbox, predicted)
                if score >= IOU_MATCH_MIN:
                    return score
                return _proximity_gate_score(p.bbox, track, predicted)

            score_matrix = [
                [_match_score(p, self._tracks[ti]) for ti in range(n_t)]
                for p in persons
            ]

            size = max(n_p, n_t)
            if size <= _MAX_EXACT_ASSIGNMENT_SIZE:
                pairs = _optimal_assignment(score_matrix)
            else:
                pairs = _greedy_assignment(score_matrix)

            for pi, ti in pairs:
                if score_matrix[pi][ti] <= 0.0:
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
