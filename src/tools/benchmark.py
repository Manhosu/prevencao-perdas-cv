"""Teste de capacidade: quantas câmeras ESTE PC aguenta?

Reprojetado após uma revisão de código reprovar a primeira versão por dois
defeitos Critical, ambos comprovados empiricamente nesta máquina:

1. A pose NUNCA rodava. O benchmark antigo chamava `engine.detect()` sobre
   ruído aleatório (`np.random.randint`), no qual o YOLO real não detecta
   nenhuma pessoa em praticamente nenhum frame — e o loop só chamava
   `engine.pose()` dentro de um `if persons:`. Como pose (recorte + segundo
   modelo) é a operação mais cara do sistema, e o próprio texto da ferramenta
   já dizia que "o limite real é quantas pessoas aparecem ao mesmo tempo",
   o número antigo superestimava a capacidade exatamente no cenário de loja
   cheia, que é o que importa para a venda.

2. O número era `throughput_serial / N`, não uma medida de concorrência. O
   loop antigo rodava tudo serializado num único thread Python e só dividia
   o throughput de 1 stream por N câmeras — nunca exercitou o `WorkerPool`
   real, que roda `workers` threads concorrentes (default 2) contra o mesmo
   engine. Medido de verdade nesta máquina, com OpenVINO em modo LATENCY
   (que já usa todos os núcleos numa única chamada), rodar mais threads
   concorrentes PIORA o throughput agregado — o oposto do que "escala com
   núcleos" assumiria. Por isso este módulo MEDE o fator de concorrência em
   vez de assumir que ele é 1 ou que escala linearmente.

Modelo novo: mede três custos, separadamente, forçando cada operação a
executar de verdade — não depende de o YOLO "achar" alguém na imagem — e
compõe a capacidade a partir deles."""
from __future__ import annotations

import argparse
import platform
import statistics
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import psutil

from src.core.types import BBox

FRAME = (360, 640, 3)  # HxWx3, resolução típica de câmera IP reduzida

# Premissas de referência para o cenário de loja usado por recommend().
# Nada disso é "mágico": são parâmetros do negócio, não do hardware, e por
# isso ficam configuráveis por flag no CLI (--target-fps etc.).
DEFAULT_TARGET_FPS = 5.0  # default de CameraConfig.target_fps
DEFAULT_FRACAO_CAMERAS_COM_PESSOA = 0.5  # metade das câmeras com gente ao mesmo tempo
DEFAULT_PESSOAS_POR_CAMERA_ATIVA = 2.0  # pessoas numa câmera que tem gente

# Contagens de câmeras usadas só para desenhar a tabela do relatório —
# recommend() calcula o número exato por fórmula, não varre esta lista.
DEFAULT_CAMERA_COUNTS = [1, 2, 3, 5, 8, 10, 15, 20, 30]

DEFAULT_DETECT_REPEATS = 30
DEFAULT_POSE_REPEATS = 15
DEFAULT_POSE_N_PEOPLE = 4
DEFAULT_CONCURRENCY_SECONDS = 2.0


def _synthetic_boxes(n: int, frame_shape: tuple[int, int, int] = FRAME) -> list[BBox]:
    """N caixas plausíveis (80x200px, tamanho de uma pessoa inteira num
    frame 640x360), espalhadas horizontalmente dentro do frame. Passadas
    direto para `engine.pose()` — não dependem de `detect()` achar alguém,
    o que é exatamente o que FORÇA a pose a rodar de verdade."""
    h, w = frame_shape[0], frame_shape[1]
    box_w, box_h = 80, 200
    y1 = float(max(0, h - box_h - 10))
    boxes = []
    usable_w = max(1, w - box_w - 10)
    for i in range(n):
        x1 = float(10 + (i * (box_w + 15)) % usable_w)
        boxes.append(BBox(x1, y1, x1 + box_w, y1 + box_h))
    return boxes


def measure_detect_cost(engine, repeats: int = DEFAULT_DETECT_REPEATS) -> float:
    """Tempo médio de `engine.detect()` sobre um frame, em segundos.

    O frame pode ser ruído — detect() roda de qualquer forma, então não há
    o problema do Critical 1 aqui (esse defeito era sobre pose, que só
    rodava condicionalmente). Descarta a primeira chamada (warmup: aloca
    buffers, compila kernel do backend) para não distorcer a média."""
    image = np.random.randint(0, 255, FRAME, dtype=np.uint8)
    engine.detect(image)  # warmup, descartado
    times = [_time_call(engine.detect, image) for _ in range(repeats)]
    return statistics.mean(times)


def measure_pose_cost_per_person(
    engine,
    n_people: int = DEFAULT_POSE_N_PEOPLE,
    repeats: int = DEFAULT_POSE_REPEATS,
) -> float:
    """Tempo médio de `engine.pose()` por pessoa, em segundos.

    Passa caixas sintéticas diretamente para `engine.pose()` — não passa
    por `engine.detect()` e não depende de haver "detecção" nenhuma. É essa
    chamada direta que corrige o Critical 1: a versão antiga só chamava
    pose() `if persons:`, e sobre ruído aleatório o YOLO real não detecta
    ninguém, então pose() nunca era exercitada e seu custo (o mais caro do
    sistema) nunca entrava no número final."""
    image = np.random.randint(0, 255, FRAME, dtype=np.uint8)
    boxes = _synthetic_boxes(n_people)
    engine.pose(image, boxes)  # warmup, descartado
    times = [_time_call(engine.pose, image, boxes) for _ in range(repeats)]
    return statistics.mean(times) / n_people


def _time_call(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def measure_concurrency_factor(
    engine, workers: int, seconds: float = DEFAULT_CONCURRENCY_SECONDS
) -> float:
    """Fator de concorrência real do pool: roda `workers` threads chamando
    `engine.detect()` concorrentemente contra o MESMO engine por `seconds`
    segundos, e compara o throughput agregado com o de 1 thread só pelo
    mesmo período.

        fator_concorrencia = throughput_agregado_com_workers / throughput_1_thread

    Isso reproduz o `WorkerPool` de verdade (`src/inference/worker_pool.py`):
    `workers` threads Python chamando o engine ao mesmo tempo. O número
    pode dar < 1 (contenção: o backend já satura os núcleos numa única
    chamada, e threads extras só atrapalham — o caso medido nesta máquina
    com OpenVINO em modo LATENCY), = 1 (concorrência não ajuda nem atrapalha)
    ou > 1 (o backend tem folga e mais threads realmente processam mais por
    segundo). Por isso é MEDIDO aqui, nunca assumido."""
    workers = max(1, workers)
    image = np.random.randint(0, 255, FRAME, dtype=np.uint8)
    engine.detect(image)  # warmup

    def _throughput(n_threads: int) -> float:
        counters = [0] * n_threads
        stop = threading.Event()

        def _worker(idx: int) -> None:
            while not stop.is_set():
                engine.detect(image)
                counters[idx] += 1

        threads = [
            threading.Thread(target=_worker, args=(i,), daemon=True)
            for i in range(n_threads)
        ]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        stop.wait(seconds)
        stop.set()
        for t in threads:
            t.join(timeout=max(5.0, seconds * 4))
        elapsed = time.perf_counter() - t0
        total = sum(counters)
        return total / elapsed if elapsed > 0 else 0.0

    throughput_1 = _throughput(1)
    throughput_n = _throughput(workers)
    if throughput_1 <= 0:
        return 1.0
    return throughput_n / throughput_1


@dataclass
class BenchmarkRow:
    """Uma linha da tabela do relatório: para N câmeras, no cenário de
    referência do relatório (fração de câmeras com pessoa e pessoas por
    câmera ativa), quanto o PC suporta e se cabe."""

    cameras: int
    target_fps: float
    demanda_fps: float
    throughput_sustentavel_fps: float
    cabe: bool


@dataclass
class BenchmarkReport:
    # Custos medidos, forçando cada operação a rodar de verdade (segundos).
    custo_detect: float
    custo_pose_por_pessoa: float
    fator_concorrencia: float
    workers: int

    # Premissas do cenário de referência (parâmetros de negócio, não de hardware).
    target_fps: float
    fracao_cameras_com_pessoa: float
    pessoas_por_camera_ativa: float

    rows: list[BenchmarkRow] = field(default_factory=list)

    # Hardware.
    cpu_name: str = ""
    cores: int = 0
    ram_gb: float = 0.0
    cpu_percent: float = 0.0

    def custo_por_frame(self, pessoas_medias_por_frame: float) -> float:
        """Custo esperado (segundos) de UM frame processado, dado quantas
        pessoas em média incidem sobre ele. `pessoas_medias_por_frame` já
        deve vir ponderada pela fração de câmeras com gente (ver
        `pessoas_medias_do_cenario`) — não é "pessoas numa câmera ativa",
        é a média sobre TODOS os frames do fluxo, incluindo os de câmeras
        vazias, que custam só o detect."""
        return self.custo_detect + pessoas_medias_por_frame * self.custo_pose_por_pessoa

    def throughput_sustentavel(self, pessoas_medias_por_frame: float) -> float:
        """FPS agregado (frames/s, somando todas as câmeras) que a máquina
        sustenta para um fluxo cujo custo médio por frame é
        `custo_por_frame(pessoas_medias_por_frame)`.

        Formulação escolhida (das duas propostas na revisão, esta é a "mais
        simples e defensável"):

            throughput_sustentavel = fator_concorrencia / custo_por_frame

        Por quê: 1 thread rodando sem parar entrega, por definição, 1
        segundo de "trabalho do engine" a cada 1 segundo de relógio. Se
        `fator_concorrencia` threads concorrentes entregam
        `fator_concorrencia` vezes esse trabalho por segundo de relógio
        (podendo ser < 1 por contenção), o orçamento de trabalho disponível
        por segundo é `fator_concorrencia` segundos-de-engine. Dividir esse
        orçamento pelo custo de UM frame dá quantos frames/s cabem. Isso é
        matematicamente equivalente a somar a demanda de trabalho bruta
        (N*target_fps*custo_detect + N*fração*target_fps*p*custo_pose) e
        compará-la ao orçamento — só que já dividido por N*target_fps, o
        que evita carregar duas fórmulas paralelas (uma "bruta" de detect e
        uma "de demanda" de pose) no relatório."""
        custo = self.custo_por_frame(pessoas_medias_por_frame)
        if custo <= 0:
            return float("inf")
        return self.fator_concorrencia / custo

    @property
    def pessoas_medias_do_cenario(self) -> float:
        """Amortiza o custo de pose sobre TODOS os frames do cenário: só
        uma fração das câmeras (`fracao_cameras_com_pessoa`) tem gente num
        dado instante, e mesmo essas só têm `pessoas_por_camera_ativa`
        pessoas. Câmera de corredor vazio custa quase nada (só detect); o
        custo de pose só aparece quando há gente — é essa premissa que o
        relatório precisa deixar explícita para o Adriano."""
        return self.fracao_cameras_com_pessoa * self.pessoas_por_camera_ativa

    @property
    def melhor_n_cameras(self) -> int:
        """Maior N de câmeras cujo throughput sustentável ainda cobre a
        demanda (N * target_fps), no cenário de referência do relatório.
        Cresce N só aumenta a demanda (linear); o throughput sustentável não
        depende de N (é uma taxa por frame do fluxo médio) — por isso não
        há ambiguidade de "qual N escolher": é o maior que ainda cabe."""
        throughput = self.throughput_sustentavel(self.pessoas_medias_do_cenario)
        if throughput <= 0:
            return 0
        if throughput == float("inf"):
            return 10_000  # custo por frame ~0 (dublê degenerado): sem teto medido
        return max(0, int(throughput // self.target_fps))

    def recommend(self) -> str:
        n = self.melhor_n_cameras
        premissas = (
            f"considerando FPS alvo de {self.target_fps:.0f} por câmera, "
            f"{self.fracao_cameras_com_pessoa:.0%} das câmeras com gente ao "
            f"mesmo tempo e {self.pessoas_por_camera_ativa:.0f} pessoa(s) "
            "por câmera ativa"
        )
        if n <= 0:
            return (
                f"Este PC NÃO sustenta nem 1 câmera a {self.target_fps:.0f} FPS "
                f"{premissas}. Recomendo trocar o equipamento, reduzir o FPS "
                "alvo ou rever o cenário de movimento esperado na loja."
            )
        return (
            f"Este PC sustenta até {n} câmera(s), {premissas}.\n"
            "Esse número NÃO é fixo: depende do quanto a loja do cliente "
            "costuma ter gente em várias câmeras ao mesmo tempo. Loja mais "
            "cheia (mais câmeras com gente simultaneamente, ou mais pessoas "
            "por câmera) reduz esse número; loja mais vazia aumenta."
        )

    def as_text(self) -> str:
        linhas = [
            "TESTE DE CAPACIDADE — Prevenção de Perdas",
            "=" * 64,
            f"Processador: {self.cpu_name}",
            f"Núcleos: {self.cores} · Memória: {self.ram_gb:.1f} GB",
            "",
            "Custos medidos (cada operação forçada a rodar de verdade,",
            "não estimada por uma imagem onde ninguém é detectado):",
            f"  detect (por frame, qualquer câmera):     {self.custo_detect * 1000:7.1f} ms",
            f"  pose (por pessoa, quando há gente):      {self.custo_pose_por_pessoa * 1000:7.1f} ms",
            f"  fator de concorrência ({self.workers} workers vs. 1 thread): {self.fator_concorrencia:.2f}x",
            "",
            f"{'Câmeras':>8} {'Demanda(fps)':>13} {'Suporta(fps)':>13} {'Cabe?':>7}",
            "-" * 64,
        ]
        for r in self.rows:
            linhas.append(
                f"{r.cameras:>8} {r.demanda_fps:>13.1f} "
                f"{r.throughput_sustentavel_fps:>13.1f} {'sim' if r.cabe else 'não':>7}"
            )
        linhas += ["", self.recommend(), ""]
        linhas.append(
            f"CPU do sistema durante o teste: {self.cpu_percent:.0f}%. Isso é a "
            "utilização da MÁQUINA INTEIRA enquanto o benchmark rodava — não é "
            "um teto que determina quantas câmeras cabem, e não deve ser lido "
            "como causa do número de câmeras acima. O teto real vem dos custos "
            "de detect/pose e do fator de concorrência medidos acima; CPU% "
            "baixo pode simplesmente significar que a inferência não usa todos "
            "os núcleos disponíveis, não que sobra capacidade proporcional."
        )
        linhas.append(
            "Premissa importante: nem toda câmera tem gente o tempo todo. Este "
            f"relatório assume {self.fracao_cameras_com_pessoa:.0%} das câmeras "
            f"com gente simultaneamente, {self.pessoas_por_camera_ativa:.0f} "
            "pessoa(s) em cada câmera ativa. Mude esses parâmetros (flags "
            "--fracao-cameras-com-pessoa e --pessoas-por-camera) para refletir "
            "o movimento real da loja do cliente."
        )
        return "\n".join(linhas)


def benchmark(
    engine,
    *,
    workers: int | None = None,
    target_fps: float = DEFAULT_TARGET_FPS,
    fracao_cameras_com_pessoa: float = DEFAULT_FRACAO_CAMERAS_COM_PESSOA,
    pessoas_por_camera_ativa: float = DEFAULT_PESSOAS_POR_CAMERA_ATIVA,
    camera_counts: list[int] | None = None,
    detect_repeats: int = DEFAULT_DETECT_REPEATS,
    pose_repeats: int = DEFAULT_POSE_REPEATS,
    pose_n_people: int = DEFAULT_POSE_N_PEOPLE,
    concurrency_seconds: float = DEFAULT_CONCURRENCY_SECONDS,
) -> BenchmarkReport:
    """Mede os três custos reais (detect, pose por pessoa, fator de
    concorrência) e compõe a capacidade para uma lista de cenários (nº de
    câmeras), no cenário de referência (fração de câmeras com pessoa,
    pessoas por câmera ativa, FPS alvo)."""
    if workers is None:
        from src.config.settings import InferenceConfig

        workers = InferenceConfig().workers

    engine.warmup()
    psutil.cpu_percent(interval=None)  # zera o contador do processo/sistema

    custo_detect = measure_detect_cost(engine, repeats=detect_repeats)
    custo_pose = measure_pose_cost_per_person(
        engine, n_people=pose_n_people, repeats=pose_repeats
    )
    fator_concorrencia = measure_concurrency_factor(
        engine, workers=workers, seconds=concurrency_seconds
    )

    cpu_percent = psutil.cpu_percent(interval=None)

    report = BenchmarkReport(
        custo_detect=custo_detect,
        custo_pose_por_pessoa=custo_pose,
        fator_concorrencia=fator_concorrencia,
        workers=workers,
        target_fps=target_fps,
        fracao_cameras_com_pessoa=fracao_cameras_com_pessoa,
        pessoas_por_camera_ativa=pessoas_por_camera_ativa,
        cpu_name=platform.processor() or platform.machine(),
        cores=psutil.cpu_count(logical=False) or psutil.cpu_count() or 0,
        ram_gb=psutil.virtual_memory().total / 1e9,
        cpu_percent=cpu_percent,
    )

    pessoas_medias = report.pessoas_medias_do_cenario
    throughput = report.throughput_sustentavel(pessoas_medias)
    for n in camera_counts or DEFAULT_CAMERA_COUNTS:
        demanda = n * target_fps
        report.rows.append(
            BenchmarkRow(
                cameras=n,
                target_fps=target_fps,
                demanda_fps=demanda,
                throughput_sustentavel_fps=throughput,
                cabe=throughput >= demanda,
            )
        )

    return report


if __name__ == "__main__":
    from src.config.settings import InferenceConfig
    from src.inference.engine import InferenceEngine

    ap = argparse.ArgumentParser(description="Teste de capacidade do PC")
    ap.add_argument("--device", default="openvino")
    ap.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_CONCURRENCY_SECONDS,
        help="segundos por thread para medir o fator de concorrência",
    )
    ap.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS)
    ap.add_argument(
        "--fracao-cameras-com-pessoa",
        type=float,
        default=DEFAULT_FRACAO_CAMERAS_COM_PESSOA,
    )
    ap.add_argument(
        "--pessoas-por-camera", type=float, default=DEFAULT_PESSOAS_POR_CAMERA_ATIVA
    )
    ap.add_argument("--out", default="relatorio-capacidade.txt")
    a = ap.parse_args()

    cfg = InferenceConfig(device=a.device)
    eng = InferenceEngine(cfg)
    report = benchmark(
        eng,
        workers=cfg.workers,
        target_fps=a.target_fps,
        fracao_cameras_com_pessoa=a.fracao_cameras_com_pessoa,
        pessoas_por_camera_ativa=a.pessoas_por_camera,
        concurrency_seconds=a.seconds,
    )
    texto = report.as_text()
    print(texto)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(texto)
    print(f"\nRelatório salvo em {a.out}")
