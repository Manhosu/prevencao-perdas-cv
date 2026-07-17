# Plano 4 — Interface, Editor de Zonas e Instalador

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dar cara de produto ao sistema: uma **tela** onde o revendedor aponta as câmeras, **desenha a área monitorada** (o que mais derruba alarme falso), vê os eventos e marca falso positivo — e um **instalador Windows** para rodar na loja com um clique.

**Architecture:** A decisão central é **separar a lógica do Qt**. Toda a matemática das zonas (pontos normalizados, arrastar vértice, hit-testing, salvar no config) vive em `src/ui/zone_model.py` — **puro, sem Qt, 100% testável**. Os widgets PySide6 são cascas finas em cima disso. É o que permite testar a parte que importa sem depender de tela.

**Tech Stack:** Python 3.13, PySide6 6.11, OpenCV (snapshot da câmera), pytest. Testes de Qt rodam com `QT_QPA_PLATFORM=offscreen`.

## Global Constraints

- **`src/ui/zone_model.py` não importa Qt.** Só matemática e tipos. É a regra que torna o plano testável.
- **`src/detection/` continua núcleo puro** — a UI não entra lá.
- **A UI NUNCA bloqueia:** nada de inferência na thread do Qt. A UI lê estado (status, eventos) e desenha; o pipeline roda nas threads dele. Comunicação por sinal Qt/polling curto.
- **Zonas em coordenadas normalizadas (0–1)** — sobrevivem a troca de resolução/substream. É o contrato que o `PersonGate` já espera.
- Textos da interface em **português** (o usuário é o lojista/revendedor); nomes de código em inglês; commits em português com prefixo convencional.
- Rode os testes com `PYTHONPATH=. .venv/Scripts/python.exe -m pytest`. Testes de widget: `QT_QPA_PLATFORM=offscreen`.
- Todo teste de Qt que exigir display real deve ser marcado `@pytest.mark.slow`.

## Interfaces já existentes (NÃO modificar)

- `src.config.settings`: `AppConfig.load(path)`, `.save(path)`, `CameraConfig(name, rtsp_url, enabled, target_fps, zones: list[list[tuple[float,float]]], overrides)`.
- `src.storage.db.Database`: `.list_events(limit, since) -> list[Row]` (colunas: id, camera_name, ts_local, score, zone, image_path, clip_path, sent_telegram, feedback), `.set_feedback(event_id, "true_positive"|"false_positive")`, `.get_camera_status(name)`.
- `src.pipeline.Pipeline`: `.status() -> dict[str, dict]` (por câmera: `state`, `fps`, `dropped`), `.threads`, `.slots`.
- `src.capture.frame_slot.LatestFrameSlot.peek() -> Frame | None` (para o preview ao vivo).
- `src.detection.person_gate.PersonGate(zones, frame_size)`.

---

## Estrutura de arquivos deste plano

| Arquivo | Responsabilidade |
|---|---|
| `src/ui/zone_model.py` | **lógica pura** das zonas (sem Qt): pontos, arrastar, hit-test, normalizar |
| `src/ui/zone_editor.py` | widget: desenha o polígono sobre o snapshot |
| `src/ui/camera_form.py` | assistente: montar URL RTSP por marca + testar conexão |
| `src/ui/event_log.py` | aba de eventos: lista, miniatura, marcar falso positivo |
| `src/ui/live_view.py` | grade de câmeras com status e FPS |
| `src/ui/app.py` | janela principal com abas |
| `scripts/build_installer.py` | PyInstaller |
| `installer/setup.iss` | Inno Setup |

---

## Task 1: Lógica das zonas (pura, sem Qt)

**Files:**
- Create: `src/ui/zone_model.py`
- Test: `tests/ui/test_zone_model.py`

**Interfaces:**
- `ZoneModel(zones: list[Polygon] | None = None)` — polígonos em coords normalizadas (0–1).
- `.add_point(x_n, y_n)` — adiciona vértice ao polígono corrente.
- `.finish_polygon()` — fecha o polígono corrente (mínimo 3 pontos) e começa outro.
- `.hit_test(x_n, y_n, tol) -> tuple[int, int] | None` — (índice do polígono, índice do vértice) sob o cursor.
- `.move_point(poly_i, pt_i, x_n, y_n)` — arrasta um vértice (clampado em 0–1).
- `.remove_point(poly_i, pt_i)` / `.remove_polygon(poly_i)`.
- `.to_config() -> list[Polygon]` — só os polígonos válidos (≥3 pontos).
- `.from_pixels(x, y, w, h) -> (x_n, y_n)` e `.to_pixels(x_n, y_n, w, h) -> (x, y)` — conversão de/para a tela.
- `.covers_whole_frame -> bool` — sem zonas = monitorar o quadro inteiro (o padrão que o cliente pediu).

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/ui/test_zone_model.py`:

```python
import pytest

from src.ui.zone_model import ZoneModel


def test_starts_empty_and_covers_whole_frame():
    m = ZoneModel()
    assert m.to_config() == []
    assert m.covers_whole_frame is True


def test_add_points_and_finish_polygon():
    m = ZoneModel()
    m.add_point(0.1, 0.1)
    m.add_point(0.9, 0.1)
    m.add_point(0.9, 0.9)
    m.finish_polygon()
    cfg = m.to_config()
    assert len(cfg) == 1
    assert len(cfg[0]) == 3
    assert m.covers_whole_frame is False


def test_polygon_with_less_than_three_points_is_discarded():
    m = ZoneModel()
    m.add_point(0.1, 0.1)
    m.add_point(0.5, 0.5)
    m.finish_polygon()
    assert m.to_config() == []  # 2 pontos nao formam area


def test_pixel_conversion_roundtrip():
    m = ZoneModel()
    x_n, y_n = m.from_pixels(320, 120, w=640, h=480)
    assert x_n == pytest.approx(0.5)
    assert y_n == pytest.approx(0.25)
    x, y = m.to_pixels(x_n, y_n, w=640, h=480)
    assert (round(x), round(y)) == (320, 120)


def test_hit_test_finds_vertex():
    m = ZoneModel([[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8)]])
    assert m.hit_test(0.21, 0.21, tol=0.03) == (0, 0)
    assert m.hit_test(0.79, 0.21, tol=0.03) == (0, 1)
    assert m.hit_test(0.5, 0.5, tol=0.03) is None


def test_move_point_updates_and_clamps():
    m = ZoneModel([[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8)]])
    m.move_point(0, 0, 0.35, 0.4)
    assert m.to_config()[0][0] == (0.35, 0.4)
    m.move_point(0, 0, 1.5, -0.3)  # fora do quadro
    assert m.to_config()[0][0] == (1.0, 0.0)  # clampado


def test_remove_point_and_polygon():
    m = ZoneModel([[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]])
    m.remove_point(0, 0)
    assert len(m.to_config()[0]) == 3
    m.remove_polygon(0)
    assert m.to_config() == []


def test_removing_point_below_three_drops_the_polygon():
    m = ZoneModel([[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]])
    m.remove_point(0, 0)  # sobra 2 -> nao e mais area valida
    assert m.to_config() == []


def test_multiple_polygons():
    m = ZoneModel()
    for p in [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3)]:
        m.add_point(*p)
    m.finish_polygon()
    for p in [(0.6, 0.6), (0.9, 0.6), (0.9, 0.9)]:
        m.add_point(*p)
    m.finish_polygon()
    assert len(m.to_config()) == 2


def test_config_roundtrip_matches_person_gate_contract():
    """O formato tem que ser exatamente o que o PersonGate ja consome."""
    from src.core.types import BBox, PersonDetection
    from src.detection.person_gate import PersonGate

    m = ZoneModel()
    for p in [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]:
        m.add_point(*p)
    m.finish_polygon()

    gate = PersonGate(m.to_config(), frame_size=(1000, 500))
    dentro = PersonDetection(bbox=BBox(700, 100, 800, 400), conf=0.9)   # pes em x=750
    fora = PersonDetection(bbox=BBox(100, 100, 200, 400), conf=0.9)     # pes em x=150
    assert gate.contains(dentro) and not gate.contains(fora)
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/ui/test_zone_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ui.zone_model'`

- [ ] **Step 3: Implementar `src/ui/zone_model.py`**

Criar `src/ui/__init__.py` e `tests/ui/__init__.py` vazios.

```python
"""Lógica das zonas monitoradas — SEM Qt.

Separar isto do widget é o que torna a parte que importa (a geometria) testável
sem depender de tela. O widget PySide6 é uma casca fina em cima disto.

Zonas ficam em coordenadas normalizadas (0–1) sobre o quadro: sobrevivem a troca
de resolução ou substream do DVR. É o mesmo contrato que o PersonGate consome."""
from __future__ import annotations

Point = tuple[float, float]
Polygon = list[Point]

MIN_POINTS = 3  # menos que isso não é área


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


class ZoneModel:
    def __init__(self, zones: list[Polygon] | None = None) -> None:
        self.polygons: list[Polygon] = [list(p) for p in (zones or [])]
        self._current: Polygon = []

    # --- construção ---
    def add_point(self, x_n: float, y_n: float) -> None:
        self._current.append((_clamp01(x_n), _clamp01(y_n)))

    def finish_polygon(self) -> None:
        if len(self._current) >= MIN_POINTS:
            self.polygons.append(self._current)
        self._current = []

    @property
    def current(self) -> Polygon:
        return list(self._current)

    # --- edição ---
    def hit_test(self, x_n: float, y_n: float, tol: float = 0.02) -> tuple[int, int] | None:
        """Qual vértice está sob o cursor (para arrastar)."""
        for pi, poly in enumerate(self.polygons):
            for vi, (px, py) in enumerate(poly):
                if abs(px - x_n) <= tol and abs(py - y_n) <= tol:
                    return pi, vi
        return None

    def move_point(self, poly_i: int, pt_i: int, x_n: float, y_n: float) -> None:
        self.polygons[poly_i][pt_i] = (_clamp01(x_n), _clamp01(y_n))

    def remove_point(self, poly_i: int, pt_i: int) -> None:
        poly = self.polygons[poly_i]
        del poly[pt_i]
        if len(poly) < MIN_POINTS:
            del self.polygons[poly_i]

    def remove_polygon(self, poly_i: int) -> None:
        del self.polygons[poly_i]

    def clear(self) -> None:
        self.polygons = []
        self._current = []

    # --- config ---
    def to_config(self) -> list[Polygon]:
        return [list(p) for p in self.polygons if len(p) >= MIN_POINTS]

    @property
    def covers_whole_frame(self) -> bool:
        """Sem zona = monitorar o quadro inteiro (o padrão que o cliente pediu
        para reduzir o trabalho de configuração por loja)."""
        return not self.to_config()

    # --- tela <-> normalizado ---
    @staticmethod
    def from_pixels(x: float, y: float, w: int, h: int) -> Point:
        return (_clamp01(x / w) if w else 0.0, _clamp01(y / h) if h else 0.0)

    @staticmethod
    def to_pixels(x_n: float, y_n: float, w: int, h: int) -> tuple[float, float]:
        return (x_n * w, y_n * h)
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/ui/test_zone_model.py -v`
Expected: PASS — 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/ui/ tests/ui/
git commit -m "feat: logica das zonas monitoradas (pura, sem Qt, testavel)"
```

---

## Task 2: Aba de eventos (lista + marcar falso positivo)

**Files:**
- Create: `src/ui/event_log.py`
- Test: `tests/ui/test_event_log.py`

**Interfaces:**
- `EventLogModel(db)` — camada entre o banco e a tela, **sem Qt**:
  - `.load(limit=100) -> list[dict]` (id, camera, hora legível, score, zona em português, caminho da foto, enviado, feedback)
  - `.mark_false_positive(event_id)` / `.mark_true_positive(event_id)`
  - `.stats() -> dict` (total, enviados, falsos marcados)
- `EventLogWidget(model)` — a tabela Qt (casca fina).

**Por quê:** o botão "isso foi falso alarme" é o que realimenta a calibração — cada marcação vira material de ajuste. É a feature que faz o sistema melhorar com o uso.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/ui/test_event_log.py`:

```python
from datetime import datetime, timezone

import pytest

from src.storage.db import Database
from src.ui.event_log import EventLogModel


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "e.db")
    d.init_schema()
    yield d
    d.close()


def _ev(db, camera="Caixa 01", zone="waist", score=0.8):
    agora = datetime.now(timezone.utc)
    return db.insert_event(store_id="l", camera_name=camera, ts_utc=agora.isoformat(),
                           ts_local=datetime.now().isoformat(), track_id=1, score=score,
                           zone=zone, signals={}, image_path="/tmp/a.jpg", clip_path=None)


def test_load_returns_rows_for_the_screen(db):
    _ev(db)
    m = EventLogModel(db)
    linhas = m.load()
    assert len(linhas) == 1
    r = linhas[0]
    assert r["camera"] == "Caixa 01"
    assert r["score"] == 0.8
    assert ":" in r["hora"]          # hora legivel
    assert r["zona"] != "waist"      # traduzido p/ o lojista
    assert "cintura" in r["zona"].lower() or "bolso" in r["zona"].lower()


def test_mark_false_positive_persists(db):
    eid = _ev(db)
    m = EventLogModel(db)
    m.mark_false_positive(eid)
    assert m.load()[0]["feedback"] == "false_positive"


def test_mark_true_positive_persists(db):
    eid = _ev(db)
    m = EventLogModel(db)
    m.mark_true_positive(eid)
    assert m.load()[0]["feedback"] == "true_positive"


def test_stats_counts(db):
    a = _ev(db)
    _ev(db)
    db.mark_sent(a)
    m = EventLogModel(db)
    m.mark_false_positive(a)
    s = m.stats()
    assert s["total"] == 2
    assert s["enviados"] == 1
    assert s["falsos"] == 1


def test_load_is_newest_first(db):
    _ev(db, camera="Antiga")
    _ev(db, camera="Nova")
    linhas = EventLogModel(db).load()
    assert linhas[0]["camera"] == "Nova"
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/ui/test_event_log.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/ui/event_log.py`**

O `EventLogModel` NÃO importa Qt (só o `EventLogWidget`, que virá com a janela). Traduza a zona com o mesmo dicionário do Telegram (`src.alerts.telegram_alert.ZONA_PT`) — reuso, não duplicação.

```python
"""Aba de eventos: o histórico e o botão 'isso foi falso alarme'.

O model é separado do widget e não importa Qt — a lógica é testável sem tela.
A marcação de falso positivo é o que realimenta a calibração: cada marcação vira
material de ajuste, e o sistema melhora com o uso."""
from __future__ import annotations

from datetime import datetime

from src.alerts.telegram_alert import ZONA_PT
from src.storage.db import Database


class EventLogModel:
    def __init__(self, db: Database) -> None:
        self.db = db

    def load(self, limit: int = 100) -> list[dict]:
        out = []
        for r in self.db.list_events(limit=limit):
            try:
                hora = datetime.fromisoformat(r["ts_local"]).strftime("%d/%m/%Y %H:%M:%S")
            except Exception:
                hora = r["ts_local"]
            out.append({
                "id": r["id"],
                "camera": r["camera_name"],
                "hora": hora,
                "score": round(r["score"], 2),
                "zona": ZONA_PT.get(r["zone"], r["zone"]),
                "foto": r["image_path"],
                "clipe": r["clip_path"],
                "enviado": bool(r["sent_telegram"]),
                "feedback": r["feedback"],
            })
        return out

    def mark_false_positive(self, event_id: int) -> None:
        self.db.set_feedback(event_id, "false_positive")

    def mark_true_positive(self, event_id: int) -> None:
        self.db.set_feedback(event_id, "true_positive")

    def stats(self) -> dict:
        linhas = self.db.list_events(limit=10000)
        return {
            "total": len(linhas),
            "enviados": sum(1 for r in linhas if r["sent_telegram"]),
            "falsos": sum(1 for r in linhas if r["feedback"] == "false_positive"),
        }
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/ui/test_event_log.py -v`
Expected: PASS — 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/ui/event_log.py tests/ui/test_event_log.py
git commit -m "feat: model da aba de eventos (historico + marcar falso positivo)"
```

---

## Task 3: Montador de URL RTSP por marca

**Files:**
- Create: `src/ui/camera_form.py`
- Test: `tests/ui/test_camera_form.py`

**Interfaces:**
- `build_rtsp_url(marca, ip, usuario, senha, canal, substream=True, porta=554) -> str`
- `MARCAS: dict[str, str]` — os padrões suportados (Intelbras/Dahua, Hikvision, Genérico).
- `parse_rtsp_url(url) -> dict | None` — extrai ip/usuario/canal de uma URL existente (para editar).
- `test_connection(url, timeout=10) -> tuple[bool, str]` — abre a câmera e devolve (ok, mensagem em português).

**Por quê:** o revendedor instala em várias lojas. Ele escolhe a marca, digita IP/usuário/senha/canal, e a URL sai pronta — sem precisar saber o que é RTSP. É o que torna a configuração "alguns minutos por câmera" que foi prometido.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/ui/test_camera_form.py`:

```python
import pytest

from src.ui.camera_form import MARCAS, build_rtsp_url, parse_rtsp_url


def test_intelbras_url():
    u = build_rtsp_url("Intelbras/Dahua", "192.168.0.11", "admin", "s3nh@", 8)
    assert u == "rtsp://admin:s3nh%40@192.168.0.11:554/cam/realmonitor?channel=8&subtype=1"


def test_intelbras_mainstream_when_substream_false():
    u = build_rtsp_url("Intelbras/Dahua", "192.168.0.11", "admin", "x", 8, substream=False)
    assert "subtype=0" in u


def test_hikvision_url():
    u = build_rtsp_url("Hikvision", "192.168.0.20", "admin", "x", 3)
    assert u == "rtsp://admin:x@192.168.0.20:554/Streaming/Channels/302"


def test_hikvision_mainstream():
    u = build_rtsp_url("Hikvision", "192.168.0.20", "admin", "x", 3, substream=False)
    assert u.endswith("/Streaming/Channels/301")


def test_senha_com_caracter_especial_e_escapada():
    """Senha com @ ou : quebra a URL se nao for escapada."""
    u = build_rtsp_url("Intelbras/Dahua", "10.0.0.1", "admin", "a@b:c", 1)
    assert "a%40b%3Ac" in u
    assert u.count("@") == 1  # so o separador usuario@host


def test_porta_customizada():
    u = build_rtsp_url("Intelbras/Dahua", "10.0.0.1", "admin", "x", 1, porta=8554)
    assert ":8554/" in u


def test_marcas_disponiveis():
    assert "Intelbras/Dahua" in MARCAS
    assert "Hikvision" in MARCAS


def test_parse_intelbras_url():
    d = parse_rtsp_url("rtsp://admin:x@192.168.0.11:554/cam/realmonitor?channel=8&subtype=1")
    assert d["ip"] == "192.168.0.11"
    assert d["usuario"] == "admin"
    assert d["canal"] == 8
    assert d["substream"] is True


def test_parse_hikvision_url():
    d = parse_rtsp_url("rtsp://admin:x@192.168.0.20:554/Streaming/Channels/302")
    assert d["ip"] == "192.168.0.20"
    assert d["canal"] == 3
    assert d["substream"] is True


def test_parse_url_invalida():
    assert parse_rtsp_url("http://nao-e-rtsp") is None
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/ui/test_camera_form.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/ui/camera_form.py`**

Sem Qt neste arquivo (só as funções). A senha DEVE ser escapada com `urllib.parse.quote` — senha com `@` é comum e quebra a URL. O `test_connection` usa `cv2.VideoCapture` com timeout e devolve mensagem em português (ex.: "Conectou! A câmera está respondendo." / "Não consegui conectar — confira IP, usuário, senha e canal.").

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/ui/test_camera_form.py -v`
Expected: PASS — 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/ui/camera_form.py tests/ui/test_camera_form.py
git commit -m "feat: montador de URL RTSP por marca (Intelbras/Dahua, Hikvision)"
```

---

## Task 4: Widget do editor de zonas

**Files:**
- Create: `src/ui/zone_editor.py`
- Test: `tests/ui/test_zone_editor.py` (com `QT_QPA_PLATFORM=offscreen`)

**Interfaces:**
- `ZoneEditor(QWidget)` — recebe um snapshot (`np.ndarray` BGR) e um `ZoneModel`.
  - clique esquerdo: adiciona vértice; arrastar sobre um vértice: move; duplo clique: fecha o polígono; botão direito sobre vértice: remove.
  - `.set_snapshot(img)`, `.zones() -> list[Polygon]`, sinal `zonesChanged`.
  - Desenha o polígono translúcido + os vértices sobre a imagem.
- Toda a matemática vem do `ZoneModel` — o widget só traduz eventos de mouse em chamadas do model.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/ui/test_zone_editor.py`. Use `QT_QPA_PLATFORM=offscreen` via fixture, `QApplication` única, e simule cliques chamando os handlers com `QMouseEvent` (ou testando os métodos de tradução de coordenada). Foque no que é lógica: um clique em (320,120) num widget 640×480 vira o ponto normalizado (0.5, 0.25) no model; arrastar move o vértice; o `zones()` devolve o que o `ZoneModel` tem.

- [ ] **Step 2–5:** implementar, rodar, commitar (`feat: editor visual de zonas sobre o snapshot`).

---

## Task 5: Grade de câmeras ao vivo

**Files:**
- Create: `src/ui/live_view.py`
- Test: `tests/ui/test_live_view.py`

**Interfaces:**
- `LiveViewModel(pipeline)` — **sem Qt**: `.snapshot(camera) -> np.ndarray | None` (via `slots[cam].peek()`), `.status() -> dict` (do `pipeline.status()`), `.overlay_zones(img, zones)` (desenha a zona no preview).
- `LiveViewWidget(model)` — a grade Qt, com badge de estado (verde/vermelho) e FPS por câmera, atualizando por `QTimer` (nunca bloqueando).

- [ ] Testes: o model devolve o status certo; o snapshot vem do slot; o overlay desenha. O widget só é exercitado offscreen.

---

## Task 6: Janela principal

**Files:**
- Create: `src/ui/app.py`
- Modify: `src/main.py` (flag `--ui` para abrir a janela; sem a flag, segue headless)

**Interfaces:**
- `MainWindow(pipeline, db, cfg, config_path)` com abas: **Ao vivo**, **Câmeras & Zonas**, **Eventos**, **Configuração**.
- A aba Configuração tem o botão **Teste de Capacidade** (roda o `src/tools/benchmark.py` numa thread e mostra o relatório).
- Salvar as zonas grava no `config.json` via `AppConfig.save`.
- **A UI nunca roda inferência.** Ela lê `pipeline.status()` e os slots por `QTimer`.

- [ ] Teste: a janela abre offscreen sem exceção; salvar zonas persiste no config e o `AppConfig.load` relê igual (round-trip).

---

## Task 7: Instalador Windows

**Files:**
- Create: `scripts/build_installer.py`, `installer/setup.iss`
- Test: `tests/tools/test_build_installer.py` (só a lógica de montagem dos argumentos; o build real é `@pytest.mark.slow`)

**Requisitos:**
- PyInstaller em modo `onedir` (abre mais rápido e atualiza melhor que `onefile`).
- **Embarcar os modelos** (`models/*.pt` e o cache OpenVINO) — a loja pode não ter internet liberada, e baixar modelo na primeira execução seria uma falha em campo.
- Inno Setup: atalho na área de trabalho, opção **iniciar com o Windows**, e desinstalador.
- O instalador NÃO pode embarcar `config.json` com token — só o `config.example.json`.

- [ ] Teste: os argumentos do PyInstaller incluem os modelos e excluem `config.json`; o `.iss` referencia a pasta certa.
- [ ] **Verificação manual:** rodar o build, instalar numa máquina limpa, abrir o programa e confirmar que ele sobe sem Python instalado.

---

## Fechamento do Plano 4

- [ ] Suíte completa (`-m "not slow"` e `-m slow`), zero regressões.
- [ ] **Verificação manual obrigatória:** abrir a UI, cadastrar uma câmera pelo assistente (montador de URL), desenhar uma zona, salvar, e confirmar que o `PersonGate` passa a respeitar a zona (o sistema só detecta dentro dela).
- [ ] Revisão final da branch — foco em: a UI nunca bloqueia; a zona salva bate com o que o gate consome; nenhum segredo no instalador.
- [ ] Merge para `master`.

**Estado ao fim:** o sistema tem cara de produto — o revendedor instala com um clique, aponta as câmeras num assistente, desenha a área monitorada, vê os eventos e marca os falsos alarmes. Falta só a calibração fina em campo.
