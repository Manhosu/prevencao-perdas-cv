"""Testes do benchmark reprojetado.

O design antigo tinha dois defeitos Critical comprovados por revisão:
1. pose() nunca rodava (só era chamada `if persons:`, e sobre ruído
   aleatório o YOLO real nunca detecta ninguém).
2. o número era `throughput_serial / N`, sem medir concorrência real.

Os dublês aqui existem para exercitar exatamente essas duas dimensões:
- `FakeEngine.pose()` tem custo PROPORCIONAL ao número de caixas recebidas
  (não depende de "achar" ninguém — as caixas vêm prontas), então mais
  pessoas por câmera tem que reduzir a capacidade recomendada.
- `FakeEngineConcorrenciaBoa` / `FakeEngineConcorrenciaRuim` têm
  comportamentos de concorrência DIFERENTES e MEDÍVEIS, para provar que o
  fator de concorrência é medido (não assumido) e entra na conta.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from src.core.types import BBox
from src.tools.benchmark import (
    BenchmarkReport,
    BenchmarkRow,
    benchmark,
    measure_concurrency_factor,
    measure_detect_cost,
    measure_pose_cost_per_person,
)


class FakeEngine:
    """detect() custa um tempo fixo; pose() custa tempo PROPORCIONAL ao
    número de caixas recebidas — cada caixa é um pedaço de trabalho de CPU
    real e fixo. Ao contrário do dublê da versão antiga (que dependia do
    YOLO real "achar" pessoas em ruído aleatório), aqui pose() roda com as
    caixas que a chamada recebe diretamente, então SEMPRE é exercitada."""

    DETECT_UNITS = 4_000
    POSE_UNITS_PER_BOX = 12_000

    def __init__(self):
        pass

    @staticmethod
    def _work(units: int) -> float:
        # Trabalho de CPU real (não sleep): soma numérica pura em Python.
        x = 0.0
        for i in range(units):
            x += (i % 7) * 1.0000001
        return x

    def detect(self, image):
        self._work(self.DETECT_UNITS)
        return [], []

    def pose(self, image, boxes):
        self._work(self.POSE_UNITS_PER_BOX * max(1, len(boxes)))
        return [np.zeros((17, 3), dtype=np.float32) for _ in boxes]

    def warmup(self):
        self.detect(np.zeros((2, 2, 3), dtype=np.uint8))
        self.pose(np.zeros((2, 2, 3), dtype=np.uint8), [BBox(0, 0, 10, 10)])


class FakeEngineConcorrenciaBoa:
    """detect() custa um tempo fixo via time.sleep (libera o GIL). Como o
    sleep não segura o GIL, N threads dormindo ao mesmo tempo terminam
    juntas — concorrência real ajuda (fator > 1), o caso em que o backend
    tem folga para processar mais de uma chamada ao mesmo tempo."""

    def __init__(self, base=0.006):
        self.base = base

    def detect(self, image):
        time.sleep(self.base)
        return [], []

    def pose(self, image, boxes):
        time.sleep(self.base * max(1, len(boxes)))
        return [np.zeros((17, 3), dtype=np.float32) for _ in boxes]

    def warmup(self):
        self.detect(np.zeros((2, 2, 3), dtype=np.uint8))
        self.pose(np.zeros((2, 2, 3), dtype=np.uint8), [BBox(0, 0, 10, 10)])


class FakeEngineConcorrenciaRuim:
    """Simula um backend que já satura o hardware numa única chamada
    (o achado real desta máquina com OpenVINO em modo LATENCY): se mais de
    uma chamada estiver em andamento ao mesmo tempo, cada uma paga uma
    penalidade extra. Mais threads concorrentes PIORAM o throughput
    agregado — fator de concorrência < 1."""

    def __init__(self, base=0.006, penalty=0.05):
        self.base = base
        self.penalty = penalty
        self._lock = threading.Lock()
        self._in_flight = 0

    def detect(self, image):
        with self._lock:
            self._in_flight += 1
            concurrent = self._in_flight
        try:
            time.sleep(self.base + (self.penalty if concurrent > 1 else 0.0))
        finally:
            with self._lock:
                self._in_flight -= 1
        return [], []

    def pose(self, image, boxes):
        time.sleep(self.base * max(1, len(boxes)))
        return [np.zeros((17, 3), dtype=np.float32) for _ in boxes]

    def warmup(self):
        self.detect(np.zeros((2, 2, 3), dtype=np.uint8))
        self.pose(np.zeros((2, 2, 3), dtype=np.uint8), [BBox(0, 0, 10, 10)])


# ---------------------------------------------------------------------------
# Custos medidos isoladamente
# ---------------------------------------------------------------------------


def test_measure_detect_cost_is_positive():
    custo = measure_detect_cost(FakeEngine(), repeats=5)
    assert custo > 0


def test_measure_pose_cost_per_person_scales_with_boxes_not_with_call_count():
    # pose() com 1 caixa vs. 4 caixas: custo por pessoa deve ficar ~igual,
    # porque o dublê cobra proporcional ao número de caixas.
    engine = FakeEngine()
    custo_1 = measure_pose_cost_per_person(engine, n_people=1, repeats=8)
    custo_4 = measure_pose_cost_per_person(engine, n_people=4, repeats=8)
    assert custo_1 > 0 and custo_4 > 0
    # tolerância generosa (contagem de instrução, não sleep, tem algum ruído)
    assert custo_4 == custo_1 or abs(custo_4 - custo_1) / custo_1 < 0.6


def test_measure_pose_cost_forces_pose_to_actually_run():
    """Este é o teste que ataca o Critical 1 diretamente: pose() precisa
    rodar mesmo que 'ninguém tenha sido detectado' — a medição não passa
    por detect() nenhuma vez, só chama pose() com caixas sintéticas."""
    calls = []

    class SpyEngine(FakeEngine):
        def pose(self, image, boxes):
            calls.append(len(boxes))
            return super().pose(image, boxes)

    measure_pose_cost_per_person(SpyEngine(), n_people=3, repeats=4)
    assert len(calls) == 5  # 1 warmup + 4 repeats
    assert all(n == 3 for n in calls)


# ---------------------------------------------------------------------------
# Fator de concorrência: medido, não assumido
# ---------------------------------------------------------------------------


def test_concurrency_factor_below_one_when_backend_contends():
    fator = measure_concurrency_factor(
        FakeEngineConcorrenciaRuim(), workers=4, seconds=0.25
    )
    assert fator < 1.0


def test_concurrency_factor_above_one_when_backend_has_headroom():
    fator = measure_concurrency_factor(
        FakeEngineConcorrenciaBoa(), workers=4, seconds=0.25
    )
    assert fator > 1.3


def test_concurrency_factor_of_contending_backend_is_lower_than_healthy_one():
    fator_ruim = measure_concurrency_factor(
        FakeEngineConcorrenciaRuim(), workers=4, seconds=0.25
    )
    fator_bom = measure_concurrency_factor(
        FakeEngineConcorrenciaBoa(), workers=4, seconds=0.25
    )
    assert fator_ruim < fator_bom


# ---------------------------------------------------------------------------
# O buraco do Critical 1: mais pessoas por câmera tem que reduzir a capacidade
# ---------------------------------------------------------------------------


def test_more_people_per_camera_lowers_recommended_capacity():
    """Este é o teste que a versão antiga NÃO passaria: lá, pose() só
    rodava sobre ruído aleatório (onde o YOLO nunca acha ninguém), então o
    número de pessoas do cenário nunca influenciava o resultado medido —
    era decorativo. Aqui, o dublê cobra de verdade por pessoa em pose(), e
    a capacidade recomendada tem que cair quando o cenário tem mais gente."""
    engine = FakeEngine()
    kwargs = dict(
        workers=2,
        detect_repeats=6,
        pose_repeats=6,
        pose_n_people=4,
        concurrency_seconds=0.2,
    )
    report_1_pessoa = benchmark(engine, pessoas_por_camera_ativa=1, **kwargs)
    report_3_pessoas = benchmark(engine, pessoas_por_camera_ativa=3, **kwargs)

    assert report_3_pessoas.melhor_n_cameras < report_1_pessoa.melhor_n_cameras


def test_more_people_per_camera_lowers_sustained_throughput():
    engine = FakeEngine()
    kwargs = dict(
        workers=2,
        detect_repeats=6,
        pose_repeats=6,
        pose_n_people=4,
        concurrency_seconds=0.2,
    )
    report_1_pessoa = benchmark(engine, pessoas_por_camera_ativa=1, **kwargs)
    report_3_pessoas = benchmark(engine, pessoas_por_camera_ativa=3, **kwargs)

    tp_1 = report_1_pessoa.throughput_sustentavel(report_1_pessoa.pessoas_medias_do_cenario)
    tp_3 = report_3_pessoas.throughput_sustentavel(report_3_pessoas.pessoas_medias_do_cenario)
    assert tp_3 < tp_1


# ---------------------------------------------------------------------------
# Fator de concorrência entra na conta da capacidade final
# ---------------------------------------------------------------------------


def test_bad_concurrency_factor_reduces_recommended_capacity():
    kwargs = dict(
        detect_repeats=6,
        pose_repeats=6,
        pose_n_people=4,
        concurrency_seconds=0.25,
        pessoas_por_camera_ativa=1,
    )
    report_bom = benchmark(FakeEngineConcorrenciaBoa(), workers=4, **kwargs)
    report_ruim = benchmark(FakeEngineConcorrenciaRuim(), workers=4, **kwargs)

    assert report_ruim.fator_concorrencia < report_bom.fator_concorrencia
    assert report_ruim.melhor_n_cameras < report_bom.melhor_n_cameras


# ---------------------------------------------------------------------------
# benchmark() de ponta a ponta: formato do relatório
# ---------------------------------------------------------------------------


def test_benchmark_measures_all_three_costs_and_builds_rows():
    report = benchmark(
        FakeEngine(),
        workers=2,
        camera_counts=[1, 4, 10],
        detect_repeats=5,
        pose_repeats=5,
        pose_n_people=3,
        concurrency_seconds=0.15,
    )
    assert report.custo_detect > 0
    assert report.custo_pose_por_pessoa > 0
    assert report.fator_concorrencia > 0
    assert len(report.rows) == 3
    assert [r.cameras for r in report.rows] == [1, 4, 10]
    assert all(isinstance(r, BenchmarkRow) for r in report.rows)


# ---------------------------------------------------------------------------
# recommend() explica as premissas em português
# ---------------------------------------------------------------------------


def test_recommend_explains_target_fps_and_fracao_cameras_com_pessoa():
    report = BenchmarkReport(
        custo_detect=0.01,
        custo_pose_por_pessoa=0.02,
        fator_concorrencia=1.0,
        workers=2,
        target_fps=5.0,
        fracao_cameras_com_pessoa=0.5,
        pessoas_por_camera_ativa=2.0,
        cpu_name="Intel i5",
        cores=4,
        ram_gb=8.0,
    )
    texto = report.recommend()
    assert "5" in texto  # FPS alvo
    assert "50%" in texto  # fração de câmeras com pessoa
    assert "câmera" in texto.lower()


def test_recommend_warns_when_nothing_fits():
    report = BenchmarkReport(
        custo_detect=5.0,  # custo absurdo: nem 1 câmera cabe
        custo_pose_por_pessoa=5.0,
        fator_concorrencia=1.0,
        workers=2,
        target_fps=5.0,
        fracao_cameras_com_pessoa=0.5,
        pessoas_por_camera_ativa=2.0,
        cpu_name="Celeron",
        cores=2,
        ram_gb=4.0,
    )
    assert report.melhor_n_cameras == 0
    texto = report.recommend()
    assert "não" in texto.lower()


# ---------------------------------------------------------------------------
# as_text(): hardware, tabela e ausência de leitura causal do CPU%
# ---------------------------------------------------------------------------


def test_as_text_mentions_hardware_and_rows():
    report = BenchmarkReport(
        custo_detect=0.01,
        custo_pose_por_pessoa=0.02,
        fator_concorrencia=1.0,
        workers=2,
        target_fps=5.0,
        fracao_cameras_com_pessoa=0.5,
        pessoas_por_camera_ativa=2.0,
        rows=[
            BenchmarkRow(
                cameras=5,
                target_fps=5.0,
                demanda_fps=25.0,
                throughput_sustentavel_fps=30.0,
                cabe=True,
            )
        ],
        cpu_name="Intel i5-8250U",
        cores=4,
        ram_gb=8.0,
        cpu_percent=26.0,
    )
    t = report.as_text()
    assert "Intel i5-8250U" in t
    assert "5" in t
    assert "26" in t


def test_as_text_does_not_present_cpu_percent_as_a_causal_ceiling():
    """A revisão apontou que '3 câmeras, CPU 26%' induzia uma leitura causal
    falsa (como se 26% de CPU fosse o teto ligado ao nº de câmeras). O texto
    precisa deixar explícito que é utilização do sistema inteiro, não um
    teto vinculado às câmeras."""
    report = BenchmarkReport(
        custo_detect=0.01,
        custo_pose_por_pessoa=0.02,
        fator_concorrencia=1.0,
        workers=2,
        target_fps=5.0,
        fracao_cameras_com_pessoa=0.5,
        pessoas_por_camera_ativa=2.0,
        rows=[
            BenchmarkRow(
                cameras=3,
                target_fps=5.0,
                demanda_fps=15.0,
                throughput_sustentavel_fps=30.0,
                cabe=True,
            )
        ],
        cpu_name="Intel i5-8250U",
        cores=4,
        ram_gb=8.0,
        cpu_percent=26.0,
    )
    t = report.as_text().lower()
    assert "sistema inteiro" in t or "máquina inteira" in t
    assert "não é um teto" in t or "não deve ser lido" in t
