import time

import pytest

from src.core.types import BBox, PersonDetection
from src.detection.tracker import Tracker, iou


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


# ---------------------------------------------------------------------------
# Correção pós-revisão: predição de movimento + associação ótima.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("step", [8.0, 15.0, 25.0])
def test_ids_survive_two_people_crossing_paths(step):
    """Duas pessoas andando uma em direção à outra (8, 15 e 25px/frame —
    velocidades usadas na re-revisão da rodada 2 para reconfirmar que o
    cruzamento continua resolvido) se cruzam perto de uma prateleira —
    cenário comum de loja, não caso raro. Sem predição de movimento, o
    associador guloso compara a detecção nova com a última posição PARADA
    do track; no frame do cruzamento, a posição de A cai sobre onde B
    estava (e vice-versa), o guloso pareia pela maior sobreposição e troca
    os ids permanentemente (sem autocorreção).

    Confirmado manualmente contra o tracker guloso anterior (sem predição):
    a pessoa A (id 1, começa à esquerda) e a pessoa B (id 2, começa à
    direita) trocam de id a partir do frame em que as caixas se cruzam
    (ax1=210, bx1=190) e o erro persiste em todos os frames seguintes — este
    teste falha nessa implementação e só passa com a predição de movimento.
    """
    t = Tracker(max_lost_seconds=2.0)

    ax1, bx1 = 0.0, 400.0
    first = t.update([_p(ax1, 0, ax1 + 50, 150), _p(bx1, 0, bx1 + 50, 150)], ts=0.0)
    id_a, id_b = first[0].track_id, first[1].track_id
    assert id_a != id_b

    for frame in range(1, 40):
        ts = float(frame)
        ax1 += step
        bx1 -= step
        out = t.update([_p(ax1, 0, ax1 + 50, 150), _p(bx1, 0, bx1 + 50, 150)], ts=ts)
        # A pessoa que começou à esquerda (sempre passada primeiro na lista
        # de detecções, na sua posição real atual) mantém o id_a; idem para
        # B com id_b — inclusive no frame em que as caixas se sobrepõem.
        assert out[0].track_id == id_a, f"id_a trocou no frame {frame} (ax1={ax1}, bx1={bx1})"
        assert out[1].track_id == id_b, f"id_b trocou no frame {frame} (ax1={ax1}, bx1={bx1})"


def test_optimal_assignment_avoids_unnecessary_new_id():
    """Duas pessoas próximas (fila de caixa): o guloso pega o par de maior
    IoU isolado (A com o track de B) e deixa o outro sem par acima do
    limiar, criando um id novo desnecessário e zerando o dwell de quem já
    estava sendo rastreado.

    Matriz de IoU nesta configuração (A=pessoa que ficou perto do track 1,
    B=pessoa que ficou perto do track 2): A-T1=0.587, A-T2=0.754 (o maior
    IoU isolado da matriz), B-T1=0.111 (abaixo do limiar), B-T2=0.429.
    Guloso: pega A-T2 primeiro (maior IoU global) -> só resta B-T1, que é
    abaixo do limiar -> B vira id novo (confirmado manualmente: guloso
    produz ids {1 (para A), 3 (novo para B)}, descartando o id 2).
    Ótimo: soma da diagonal (A-T1 + B-T2 = 0.587+0.429=1.016) é maior que a
    soma fora da diagonal (A-T2 + 0 = 0.754) -> preserva os dois ids
    existentes, nenhum id novo é criado.
    """
    t = Tracker(max_lost_seconds=2.0)
    first = t.update([_p(0, 0, 50, 150), _p(20, 0, 70, 150)], ts=0.0)
    id1, id2 = first[0].track_id, first[1].track_id

    out = t.update([_p(13, 0, 63, 150), _p(40, 0, 90, 150)], ts=0.2)

    ids = {p.track_id for p in out}
    assert ids == {id1, id2}, f"associação ótima deveria preservar {{id1, id2}}, obteve {ids}"
    assert t.active_ids() == {id1, id2}


@pytest.mark.parametrize("gap_frames", [2, 3, 4])
def test_id_survives_multiple_empty_frames_within_max_lost(gap_frames):
    """Pessoa some por vários frames seguidos (oclusão mais longa) mas
    volta dentro de max_lost_seconds: tem que manter o id — o dwell não
    pode reiniciar só porque a oclusão durou mais de 1 frame."""
    t = Tracker(max_lost_seconds=2.0)
    first = t.update([_p(0, 0, 50, 150)], ts=0.0)[0]

    ts = 0.0
    for _ in range(gap_frames):
        ts += 0.2
        t.update([], ts=ts)

    ts += 0.2
    again = t.update([_p(5, 0, 55, 150)], ts=ts)[0]
    assert again.track_id == first.track_id


@pytest.mark.parametrize("gap_frames", [2, 3, 4])
def test_id_dropped_after_multiple_empty_frames_beyond_max_lost(gap_frames):
    """Mesmo cenário de gap de vários frames, mas o tempo decorrido excede
    max_lost_seconds: o id tem que ser descartado (limite inclusivo, <=,
    já coberto por test_id_is_dropped_after_max_lost para gap de 1 frame;
    aqui confirmamos que generaliza para vários frames vazios)."""
    max_lost = 1.0
    t = Tracker(max_lost_seconds=max_lost)
    t.update([_p(0, 0, 50, 150)], ts=0.0)

    ts = 0.0
    step = (max_lost + 0.5) / gap_frames
    for _ in range(gap_frames):
        ts += step
        t.update([], ts=ts)

    ts += 0.1
    out = t.update([_p(0, 0, 50, 150)], ts=ts)
    assert out[0].track_id == 2
    assert t.active_ids() == {2}


# ---------------------------------------------------------------------------
# Segunda correção pós-revisão: gating por proximidade para pessoa rápida
# (Important 2) e corte do custo da associação exaustiva (Important 1).
# ---------------------------------------------------------------------------


def _walk_at_constant_speed(
    tracker: Tracker, start_x1: float, speed: float, n_frames: int, settle_frames: int = 1
) -> list[int]:
    """Simula uma única pessoa (caixa 50x150): aparece, fica parada por
    `settle_frames` frame(s) — isso é o que estabelece a velocidade inicial
    do track (mesmo que zero); sem essa amostra extra o track recém-criado
    nunca tem uma velocidade estimada para o gating usar, ver docstring do
    módulo — e então anda a `speed` px/frame constante por `n_frames`
    frames. Devolve a lista de track_ids observados, na ordem (1 entrada
    para o aparecimento + `settle_frames` + `n_frames`)."""
    w, h = 50.0, 150.0
    ts = 0.0
    x1 = start_x1
    ids = [tracker.update([_p(x1, 0, x1 + w, h)], ts=ts)[0].track_id]
    for _ in range(settle_frames):
        ts += 1.0
        ids.append(tracker.update([_p(x1, 0, x1 + w, h)], ts=ts)[0].track_id)
    for _ in range(n_frames):
        ts += 1.0
        x1 += speed
        ids.append(tracker.update([_p(x1, 0, x1 + w, h)], ts=ts)[0].track_id)
    return ids


def test_fast_walker_keeps_id_at_40px_per_frame():
    """Pessoa única andando rápido a 40px/frame por vários frames mantém o
    MESMO id do início ao fim. Acima de ~0.6x a largura da caixa por frame
    (aqui: caixa de 50px de largura -> ~30px), o IoU entre a detecção e a
    posição PREVISTA cai abaixo de IOU_MATCH_MIN mesmo quando a predição
    acerta a posição exata — é geometria de interseção de duas caixas
    deslocadas, não erro de predição (ver item 3 da docstring do módulo).
    Só a predição de movimento (correção da rodada 1) não resolve isso.

    Confirmado manualmente contra a implementação da rodada 1 (predição de
    movimento + associação ótima, sem o gating por proximidade desta
    correção): a partir do frame em que a caminhada rápida começa, cada
    frame subsequente cria um id novo e o id nunca se estabiliza (ids
    observados: [1, 1, 2, 3, 4, 5, 6, ...], sempre crescendo) — este teste
    falha nessa implementação e só passa com o gating por proximidade.
    """
    t = Tracker(max_lost_seconds=2.0)
    ids = _walk_at_constant_speed(t, start_x1=0.0, speed=40.0, n_frames=20)
    assert len(set(ids)) == 1, f"id não ficou estável: {ids}"


def test_fast_walker_keeps_id_at_60px_per_frame_fleeing():
    """Documenta o comportamento a 60px/frame ('fuga', mais rápido que
    caminhada) com o track já estabelecido (mesmo cenário de
    `test_fast_walker_keeps_id_at_40px_per_frame`, só que mais rápido):
    idealmente o id se mantém, e de fato se mantém — o raio do gating por
    proximidade (`GATE_FACTOR * max(largura, altura)` da caixa do track,
    150px de altura para a caixa 50x150 usada aqui) ainda cobre folgadamente
    essa velocidade. Ver `test_known_limit_speed_without_established_velocity`
    para o teto residual que de fato existe (não este)."""
    t = Tracker(max_lost_seconds=2.0)
    ids = _walk_at_constant_speed(t, start_x1=0.0, speed=60.0, n_frames=20)
    assert len(set(ids)) == 1, f"id não ficou estável: {ids}"


def test_known_limit_speed_without_established_velocity():
    """Teto residual conhecido, documentado e travado por este teste (não é
    uma regressão desta correção, é um limite inerente ao design: o gating
    só vale para tracks que JÁ têm velocidade estimada — ver docstring do
    módulo e `_proximity_gate_score`). Um track recém-criado (uma única
    amostra de posição, velocidade ainda desconhecida) depende só do IoU
    puro no seu primeiro casamento. Se a pessoa já aparece se deslocando
    mais que ~0.6x a largura da caixa entre a primeira e a segunda detecção
    — aqui 32px/frame com caixa de 50px de largura, o mesmo valor citado na
    re-revisão ("a 32-40px/frame o id se perde a cada frame") — essa
    primeira transição falha, e como o track substituto também nasce sem
    velocidade, o id nunca se estabiliza (sem `settle_frames`, ao contrário
    dos dois testes acima). Trava esse comportamento conhecido para que uma
    mudança futura no bootstrap não regrida silenciosamente."""
    t = Tracker(max_lost_seconds=2.0)
    ids = _walk_at_constant_speed(
        t, start_x1=0.0, speed=32.0, n_frames=10, settle_frames=0
    )
    assert len(set(ids)) == len(ids), (
        f"comportamento conhecido mudou (esperava um id novo a cada frame): {ids}"
    )


def test_stationary_nearby_tracks_do_not_fuse_via_proximity_gating():
    """Guarda de regressão específica do gating por proximidade: duas
    pessoas paradas e próximas (fila de caixa) — mesmo depois de ambas
    terem velocidade estimada (zero, de ficarem paradas) e portanto ambas
    elegíveis ao gating — não trocam id nem se fundem quando uma pequena
    oscilação simultânea (35px, abaixo do limiar de IoU puro mas dentro do
    raio de gating de cada uma) as move uma em direção à outra. O escore de
    proximidade favorece o track mais próximo (35px de distância própria vs
    55px do vizinho), então a atribuição ótima preserva o pareamento
    correto — não é vácuo: uma implementação que gatasse por "qualquer
    track com velocidade estimada dentro do raio, sem preferir o mais
    próximo" trocaria os ids aqui."""
    t = Tracker(max_lost_seconds=2.0)
    # Duas pessoas paradas, com 40px de vão entre as caixas (sem overlap
    # entre si) -- estabelece velocidade zero para as duas.
    first = t.update([_p(0, 0, 50, 150), _p(90, 0, 140, 150)], ts=0.0)
    id1, id2 = first[0].track_id, first[1].track_id
    t.update([_p(0, 0, 50, 150), _p(90, 0, 140, 150)], ts=1.0)

    # Movimento simultâneo: T1 anda 35px para a direita, T2 anda 35px para
    # a esquerda (uma em direção à outra) -- abaixo do limiar de IoU puro
    # (35 > 0.6*50=30) para as duas, exigindo gating.
    out = t.update([_p(35, 0, 85, 150), _p(55, 0, 105, 150)], ts=2.0)

    assert out[0].track_id == id1, f"pessoa 1 trocou de id: esperava {id1}, obteve {out[0].track_id}"
    assert out[1].track_id == id2, f"pessoa 2 trocou de id: esperava {id2}, obteve {out[1].track_id}"
    assert t.active_ids() == {id1, id2}


def test_update_with_six_people_is_fast():
    """Custo da associação ótima com N=6 (novo teto de
    `_MAX_EXACT_ASSIGNMENT_SIZE`, baixado de 8 para conter o custo fatorial
    -- 8! = 40.320 permutações mediu ~43ms por update() antes desta
    correção; 6! = 720 é sub-milissegundo). Não é um benchmark de precisão
    (roda em ambiente de CI/dev compartilhado), só uma rede de segurança
    generosa contra reintroduzir o penhasco fatorial por engano."""
    t = Tracker(max_lost_seconds=2.0)
    people = [_p(i * 100.0, 0, i * 100.0 + 50, 150) for i in range(6)]
    t.update(people, ts=0.0)  # cria os 6 tracks

    moved = [_p(i * 100.0 + 5, 0, i * 100.0 + 55, 150) for i in range(6)]
    t0 = time.perf_counter()
    t.update(moved, ts=0.2)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"\n[tracker] update() com N=6: {elapsed_ms:.3f} ms")

    assert elapsed_ms < 50, f"update() com N=6 muito lento: {elapsed_ms:.3f} ms"


class TestIoU:
    """Testes isolados de `iou()`, sem passar pelo Tracker."""

    def test_identical_boxes_have_iou_one(self):
        box = BBox(10, 20, 60, 170)
        assert iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes_have_iou_zero(self):
        a = BBox(0, 0, 50, 150)
        b = BBox(1000, 0, 1050, 150)
        assert iou(a, b) == pytest.approx(0.0)

    def test_touching_but_not_overlapping_boxes_have_iou_zero(self):
        a = BBox(0, 0, 50, 150)
        b = BBox(50, 0, 100, 150)  # encostam na borda, sem área de sobreposição
        assert iou(a, b) == pytest.approx(0.0)

    def test_known_partial_overlap_exact_value(self):
        # a: 10x10 em (0,0)-(10,10); b: 10x10 em (5,5)-(15,15).
        # Interseção: 5x5=25. União: 100+100-25=175. IoU = 25/175 = 1/7.
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 15, 15)
        assert iou(a, b) == pytest.approx(1 / 7)

    def test_known_partial_overlap_exact_value_horizontal(self):
        # a: 10x10 em (0,0)-(10,10); b: 10x10 em (5,0)-(15,10).
        # Interseção: 5x10=50. União: 100+100-50=150. IoU = 50/150 = 1/3.
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 0, 15, 10)
        assert iou(a, b) == pytest.approx(1 / 3)
