# Plano 3 — Operação: Evidência, Telegram e Watchdog

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transformar o `ConcealmentEvent` (que o Plano 2 já emite) em **valor operacional**: evidência salva (foto anotada + clipe curto + registro), **alerta chegando no Telegram do lojista** com foto e hora, e **aviso de câmera offline**. É o que faz o sistema deixar de ser um detector e virar um produto que a loja usa.

**Architecture:** O `Pipeline` já emite eventos por câmera. Adicionamos consumidores desacoplados: um **buffer circular de frames** por câmera (para ter o "antes" do clipe), um **EvidenceRecorder** (JPEG anotado + MP4 + linha no SQLite) e uma **AlertQueue** em thread própria (Telegram lento ou fora do ar NUNCA trava a inferência), com retry, backoff e rate-limit. Um **Watchdog** vigia o heartbeat de cada `CameraThread` e alerta quando uma câmera cai ou volta. A retenção purga evidência antiga.

**Tech Stack:** Python 3.13, OpenCV (escrita de JPEG/MP4), requests (Telegram), SQLite (já existe), pytest.

## Global Constraints

- **`src/detection/` continua núcleo puro** — nada deste plano entra lá. Evidência/alertas/watchdog são camadas de fora que consomem eventos.
- **O Telegram NUNCA pode travar o pipeline.** Envio é sempre assíncrono, em thread própria, com fila. Falha de rede não pode derrubar nada nem perder o evento (o registro no banco acontece independente do envio).
- **Toda constante vem do config** (`EvidenceConfig`, `TelegramConfig`, `WatchdogConfig` já existem no `settings.py`).
- **Nenhum teste pode fazer request real ao Telegram.** Use dublê/mock do endpoint. O teste de integração real com token fica manual, documentado.
- **O sistema roda 24/7 sem supervisão:** nenhuma exceção de I/O (disco cheio, rede fora, arquivo travado) pode derrubar uma thread.
- Nomes de código em inglês; comentários/logs e o texto do alerta em **português** (é o lojista que lê); commits em português com prefixo convencional; `pathlib`.
- Rode os testes com `PYTHONPATH=. .venv/Scripts/python.exe -m pytest`.

## Interfaces já existentes (NÃO modificar)

- `src.detection.concealment.ConcealmentEvent(track_id, score, zone, signals: dict, ts)`.
- `src.pipeline.Pipeline` / `FrameResult(camera_name, persons, objects, had_person, events)`; `Pipeline.on_result` é um callback opcional `(FrameResult, Frame) -> None`.
- `src.storage.db.Database`: `insert_event(store_id, camera_name, ts_utc, ts_local, track_id, score, zone, signals: dict, image_path, clip_path) -> int`, `mark_sent(event_id)`, `purge_older_than(days) -> list[Path]`, `upsert_camera_status(camera_name, state: CameraState, last_frame_ts)`.
- `src.config.settings`: `EvidenceConfig(dir, retention_days, clip_pre_seconds, clip_post_seconds)`, `TelegramConfig(bot_token, chat_id, send_photo, send_clip, rate_limit_per_min)`, `WatchdogConfig(offline_after_seconds, notify)`, `StoreConfig(id, name)`.
- `src.capture.rtsp_capture.CameraThread`: `.state -> CameraState`, `.last_frame_ts -> float | None` (monotonic), `.effective_fps`.
- `src.core.types.CameraState` (`ONLINE`/`OFFLINE`/`RECONNECTING`), `Frame(camera_name, image, ts, seq)`.

---

## Estrutura de arquivos deste plano

| Arquivo | Responsabilidade |
|---|---|
| `src/evidence/clip_buffer.py` | buffer circular de frames por câmera (o "antes" do clipe) |
| `src/evidence/recorder.py` | JPEG anotado + clipe MP4 + registro no SQLite |
| `src/evidence/retention.py` | purga periódica de evidência antiga |
| `src/alerts/telegram_alert.py` | envio `sendPhoto`/`sendVideo` (só o HTTP) |
| `src/alerts/alert_queue.py` | fila em thread + retry + backoff + rate-limit |
| `src/watchdog/monitor.py` | câmera offline/recuperada a partir do heartbeat |
| `src/main.py` | (modificar) liga tudo |

---

## Task 1: Buffer circular de frames

**Files:**
- Create: `src/evidence/clip_buffer.py`
- Test: `tests/evidence/test_clip_buffer.py`

**Interfaces:**
- Produces: `ClipBuffer(seconds: float, fps_hint: float)` com `.add(frame_bgr, ts)`, `.frames_between(t0, t1) -> list[tuple[float, np.ndarray]]`, `.newest_ts -> float | None`, `.__len__`.

**Por quê:** o clipe de evidência precisa mostrar os segundos ANTES do alerta (o gesto inteiro), mas só sabemos que houve alerta depois. Guardamos os últimos N segundos de frames por câmera; quando dispara, o "antes" já está na mão. O buffer é limitado por tempo — memória não cresce.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/evidence/test_clip_buffer.py`:

```python
import numpy as np

from src.evidence.clip_buffer import ClipBuffer


def _img(v):
    return np.full((4, 4, 3), v, dtype=np.uint8)


def test_keeps_only_the_window():
    buf = ClipBuffer(seconds=2.0, fps_hint=5)
    for i in range(20):  # 4s a 5fps
        buf.add(_img(i), ts=i * 0.2)
    # janela de 2s termina no ts mais novo (3.8) -> mantem >= 1.8
    assert all(ts >= 1.8 - 1e-6 for ts, _ in buf.frames_between(0, 99))
    assert buf.newest_ts == 3.8


def test_frames_between_returns_window_in_order():
    buf = ClipBuffer(seconds=10.0, fps_hint=5)
    for i in range(10):
        buf.add(_img(i), ts=i * 0.2)
    got = buf.frames_between(0.4, 1.0)
    assert [round(ts, 2) for ts, _ in got] == [0.4, 0.6, 0.8, 1.0]
    assert got[0][1][0, 0, 0] == 2  # o frame de ts=0.4 e o i=2


def test_empty_buffer():
    buf = ClipBuffer(seconds=2.0, fps_hint=5)
    assert buf.newest_ts is None
    assert buf.frames_between(0, 1) == []
    assert len(buf) == 0


def test_add_copies_frame_so_caller_can_reuse_buffer():
    """A thread de captura reusa o array do frame; o buffer PRECISA copiar,
    senão o clipe sai todo com a mesma imagem."""
    buf = ClipBuffer(seconds=2.0, fps_hint=5)
    img = _img(1)
    buf.add(img, ts=0.0)
    img[:] = 99  # o chamador mexe no array depois
    _, guardado = buf.frames_between(0, 1)[0]
    assert guardado[0, 0, 0] == 1


def test_memory_is_bounded():
    buf = ClipBuffer(seconds=1.0, fps_hint=5)
    for i in range(1000):
        buf.add(_img(i % 255), ts=i * 0.2)
    assert len(buf) <= 8  # ~1s a 5fps, com folga
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/evidence/test_clip_buffer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evidence.clip_buffer'`

- [ ] **Step 3: Implementar `src/evidence/clip_buffer.py`**

Criar também `src/evidence/__init__.py` e `tests/evidence/__init__.py` vazios.

```python
"""Buffer circular de frames por câmera.

O clipe de evidência precisa dos segundos ANTES do alerta (o gesto inteiro),
mas só sabemos do alerta depois que ele acontece. Guardamos uma janela curta
dos frames recentes; quando dispara, o "antes" já está na mão. Limitado por
tempo — a memória não cresce."""
from __future__ import annotations

import threading
from collections import deque

import numpy as np


class ClipBuffer:
    def __init__(self, seconds: float, fps_hint: float = 5.0) -> None:
        self.seconds = seconds
        # teto de itens com folga, para o deque nunca crescer sem limite
        maxlen = max(2, int(seconds * fps_hint * 2) + 2)
        self._buf: deque[tuple[float, np.ndarray]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, frame_bgr: np.ndarray, ts: float) -> None:
        # copia: a thread de captura reusa o array, e sem cópia o clipe sairia
        # todo com a mesma imagem
        with self._lock:
            self._buf.append((ts, frame_bgr.copy()))
            self._drop_old()

    def _drop_old(self) -> None:
        if not self._buf:
            return
        cutoff = self._buf[-1][0] - self.seconds
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def frames_between(self, t0: float, t1: float) -> list[tuple[float, np.ndarray]]:
        with self._lock:
            return [(ts, img) for ts, img in self._buf if t0 <= ts <= t1]

    @property
    def newest_ts(self) -> float | None:
        with self._lock:
            return self._buf[-1][0] if self._buf else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/evidence/test_clip_buffer.py -v`
Expected: PASS — 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/evidence/ tests/evidence/
git commit -m "feat: buffer circular de frames (o antes do clipe de evidencia)"
```

---

## Task 2: Gravador de evidência

**Files:**
- Create: `src/evidence/recorder.py`
- Test: `tests/evidence/test_recorder.py`

**Interfaces:**
- Consumes: `ConcealmentEvent`, `Database`, `EvidenceConfig`, `StoreConfig`, `ClipBuffer`.
- Produces: `EvidenceRecorder(db, evidence_cfg, store_cfg)` com:
  - `record(event, camera_name, frame_bgr, clip_buffer=None) -> int` (devolve o `event_id` do banco).
  - Salva `evidence/<camera>/<YYYY-MM-DD>/<hora>_<track>.jpg` (anotado) e, se houver buffer e `send_clip`, o `.mp4`.
  - `annotate(frame, event) -> np.ndarray` desenha a marca do alerta no JPEG.

**Nota:** o registro no banco acontece SEMPRE, mesmo se a escrita do arquivo falhar (disco cheio) — o evento não pode ser perdido.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/evidence/test_recorder.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from src.config.settings import EvidenceConfig, StoreConfig
from src.detection.concealment import ConcealmentEvent
from src.evidence.clip_buffer import ClipBuffer
from src.evidence.recorder import EvidenceRecorder
from src.storage.db import Database


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "app.db")
    d.init_schema()
    yield d
    d.close()


def _event(ts=1.0):
    return ConcealmentEvent(track_id=7, score=0.82, zone="waist",
                            signals={"dwell": 1.0, "approach": 0.5,
                                     "vanish": 0.0, "retract": 0.0}, ts=ts)


def _img():
    return np.full((120, 160, 3), 40, dtype=np.uint8)


def test_record_saves_jpeg_and_row(db, tmp_path):
    rec = EvidenceRecorder(db, EvidenceConfig(dir=str(tmp_path / "ev")),
                           StoreConfig(id="l1", name="Loja 1"))
    eid = rec.record(_event(), "Caixa 01", _img())

    assert eid > 0
    row = db.list_events(limit=1)[0]
    assert row["camera_name"] == "Caixa 01"
    assert row["zone"] == "waist"
    assert row["score"] == pytest.approx(0.82)
    assert row["store_id"] == "l1"
    img_path = Path(row["image_path"])
    assert img_path.exists() and img_path.suffix == ".jpg"


def test_record_saves_clip_when_buffer_given(db, tmp_path):
    buf = ClipBuffer(seconds=6.0, fps_hint=5)
    for i in range(30):  # 6s de frames
        buf.add(_img(), ts=i * 0.2)
    rec = EvidenceRecorder(db, EvidenceConfig(dir=str(tmp_path / "ev"),
                                              clip_pre_seconds=2.0, clip_post_seconds=0.0),
                           StoreConfig(id="l1", name="Loja 1"))
    eid = rec.record(_event(ts=4.0), "Caixa 01", _img(), clip_buffer=buf)

    row = db.list_events(limit=1)[0]
    assert row["clip_path"]
    assert Path(row["clip_path"]).exists()


def test_row_is_saved_even_if_file_write_fails(db, tmp_path, monkeypatch):
    """Disco cheio nao pode fazer o evento sumir — o registro vem primeiro."""
    rec = EvidenceRecorder(db, EvidenceConfig(dir=str(tmp_path / "ev")),
                           StoreConfig(id="l1", name="Loja 1"))
    import cv2
    monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: (_ for _ in ()).throw(OSError("disco cheio")))

    eid = rec.record(_event(), "Caixa 01", _img())

    assert eid > 0  # o evento existe no banco
    row = db.list_events(limit=1)[0]
    assert row["image_path"] in (None, "")  # sem arquivo, mas registrado


def test_annotate_marks_the_frame(db, tmp_path):
    rec = EvidenceRecorder(db, EvidenceConfig(dir=str(tmp_path / "ev")),
                           StoreConfig(id="l1", name="Loja 1"))
    original = _img()
    out = rec.annotate(original.copy(), _event())
    assert out.shape == original.shape
    assert not np.array_equal(out, original)  # desenhou alguma coisa


def test_paths_are_organized_by_camera_and_date(db, tmp_path):
    rec = EvidenceRecorder(db, EvidenceConfig(dir=str(tmp_path / "ev")),
                           StoreConfig(id="l1", name="Loja 1"))
    rec.record(_event(), "Corredor 3", _img())
    row = db.list_events(limit=1)[0]
    p = Path(row["image_path"])
    # .../ev/<camera>/<data>/<arquivo>.jpg
    assert p.parent.parent.name.replace("_", " ") == "Corredor 3"
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/evidence/test_recorder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evidence.recorder'`

- [ ] **Step 3: Implementar `src/evidence/recorder.py`**

```python
"""Gravador de evidência: foto anotada + clipe curto + registro no banco.

Ordem importa: o registro no banco vem PRIMEIRO. Se o disco estiver cheio ou o
arquivo travado, o evento não pode simplesmente sumir — ele fica registrado, só
sem a mídia."""
from __future__ import annotations

import logging
import re
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
               frame_bgr: np.ndarray, clip_buffer=None) -> int:
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
            self._update_paths(event_id, image_path, clip_path)
        return event_id

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

    def _update_paths(self, event_id: int, image_path, clip_path) -> None:
        with self.db._conn:  # noqa: SLF001 — mesma camada de persistencia
            self.db._conn.execute(
                "UPDATE events SET image_path=?, clip_path=? WHERE id=?",
                (image_path, clip_path, event_id),
            )
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/evidence/test_recorder.py -v`
Expected: PASS — 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/evidence/recorder.py tests/evidence/test_recorder.py
git commit -m "feat: gravador de evidencia (foto anotada + clipe + registro)"
```

---

## Task 3: Envio ao Telegram

**Files:**
- Create: `src/alerts/telegram_alert.py`
- Test: `tests/alerts/test_telegram_alert.py`

**Interfaces:**
- Consumes: `TelegramConfig`, `requests`.
- Produces: `TelegramSender(cfg: TelegramConfig, session=None)` com:
  - `.send_photo(image_path, caption) -> bool`
  - `.send_video(video_path, caption) -> bool`
  - `.send_message(text) -> bool`
  - `.caption_for(store_name, camera_name, ts_local, zone) -> str` (texto em português)
  - `.configured -> bool` (False se token/chat_id vazios → não tenta enviar)

**Nenhum teste faz request real.** A `session` é injetável (dublê).

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/alerts/test_telegram_alert.py`:

```python
from datetime import datetime

import pytest

from src.alerts.telegram_alert import TelegramSender
from src.config.settings import TelegramConfig


class FakeResp:
    def __init__(self, ok=True, status=200):
        self.status_code = status
        self._ok = ok

    def json(self):
        return {"ok": self._ok}


class FakeSession:
    def __init__(self, resp=None, boom=False):
        self.resp = resp or FakeResp()
        self.boom = boom
        self.calls = []

    def post(self, url, data=None, files=None, timeout=None):
        self.calls.append({"url": url, "data": data, "files": files})
        if self.boom:
            raise ConnectionError("rede fora")
        return self.resp


def _cfg(**kw):
    base = dict(bot_token="123:ABC", chat_id="-100999")
    base.update(kw)
    return TelegramConfig(**base)


def test_not_configured_when_token_missing():
    s = TelegramSender(TelegramConfig(bot_token="", chat_id=""), session=FakeSession())
    assert s.configured is False


def test_configured_with_token_and_chat():
    assert TelegramSender(_cfg(), session=FakeSession()).configured is True


def test_send_photo_posts_to_sendphoto(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    sess = FakeSession()
    s = TelegramSender(_cfg(), session=sess)

    assert s.send_photo(img, "legenda") is True
    assert sess.calls[0]["url"].endswith("/sendPhoto")
    assert "123:ABC" in sess.calls[0]["url"]
    assert sess.calls[0]["data"]["chat_id"] == "-100999"
    assert sess.calls[0]["data"]["caption"] == "legenda"


def test_send_video_posts_to_sendvideo(tmp_path):
    v = tmp_path / "a.mp4"
    v.write_bytes(b"x")
    sess = FakeSession()
    s = TelegramSender(_cfg(), session=sess)
    assert s.send_video(v, "leg") is True
    assert sess.calls[0]["url"].endswith("/sendVideo")


def test_returns_false_on_network_error(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    s = TelegramSender(_cfg(), session=FakeSession(boom=True))
    assert s.send_photo(img, "x") is False  # nao levanta, so reporta


def test_returns_false_when_telegram_says_not_ok(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    s = TelegramSender(_cfg(), session=FakeSession(resp=FakeResp(ok=False, status=400)))
    assert s.send_photo(img, "x") is False


def test_does_not_send_when_not_configured(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    sess = FakeSession()
    s = TelegramSender(TelegramConfig(bot_token="", chat_id=""), session=sess)
    assert s.send_photo(img, "x") is False
    assert sess.calls == []  # nem tentou


def test_caption_is_portuguese_and_has_the_facts():
    s = TelegramSender(_cfg(), session=FakeSession())
    cap = s.caption_for("Mercado Central", "Corredor 3",
                        datetime(2026, 7, 16, 14, 35, 9), "waist")
    assert "Mercado Central" in cap
    assert "Corredor 3" in cap
    assert "14:35" in cap
    assert "16/07/2026" in cap
    # a zona aparece em portugues, nao o codigo interno
    assert "waist" not in cap
    assert any(p in cap.lower() for p in ("cintura", "bolso"))
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/alerts/test_telegram_alert.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.alerts.telegram_alert'`

- [ ] **Step 3: Implementar `src/alerts/telegram_alert.py`**

Criar `src/alerts/__init__.py` e `tests/alerts/__init__.py` vazios.

```python
"""Envio ao Telegram (só o HTTP). A fila/retry fica no alert_queue.

Nunca levanta exceção para o chamador: devolve True/False. Uma falha de rede não
pode derrubar quem chamou."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import requests

from src.config.settings import TelegramConfig

log = logging.getLogger(__name__)

API = "https://api.telegram.org"
TIMEOUT = 20

# O lojista lê isso — nada de código interno na legenda.
ZONA_PT = {
    "waist": "mão na cintura/bolso",
    "torso": "mão sob a roupa",
    "back_waist": "mão na cintura (de costas)",
    "bag": "mão na bolsa/mochila",
}


class TelegramSender:
    def __init__(self, cfg: TelegramConfig, session=None) -> None:
        self.cfg = cfg
        self._session = session or requests.Session()

    @property
    def configured(self) -> bool:
        return bool(self.cfg.bot_token and self.cfg.chat_id)

    def caption_for(self, store_name: str, camera_name: str,
                    ts_local: datetime, zone: str) -> str:
        gesto = ZONA_PT.get(zone, "ocultação de produto")
        return (f"⚠️ Possível ocultação de produto\n"
                f"🏪 {store_name}\n"
                f"📷 {camera_name}\n"
                f"🕒 {ts_local.strftime('%d/%m/%Y %H:%M:%S')}\n"
                f"👀 {gesto}")

    def _post(self, metodo: str, campo: str, caminho: Path, caption: str) -> bool:
        if not self.configured:
            return False
        url = f"{API}/bot{self.cfg.bot_token}/{metodo}"
        try:
            with open(caminho, "rb") as f:
                r = self._session.post(
                    url,
                    data={"chat_id": self.cfg.chat_id, "caption": caption},
                    files={campo: f},
                    timeout=TIMEOUT,
                )
            ok = r.status_code == 200 and r.json().get("ok", False)
            if not ok:
                log.warning("Telegram recusou o envio (%s): %s", metodo, r.status_code)
            return bool(ok)
        except Exception as e:
            log.warning("falha ao enviar para o Telegram (%s): %s", metodo, e)
            return False

    def send_photo(self, image_path, caption: str) -> bool:
        return self._post("sendPhoto", "photo", Path(image_path), caption)

    def send_video(self, video_path, caption: str) -> bool:
        return self._post("sendVideo", "video", Path(video_path), caption)

    def send_message(self, text: str) -> bool:
        if not self.configured:
            return False
        url = f"{API}/bot{self.cfg.bot_token}/sendMessage"
        try:
            r = self._session.post(
                url, data={"chat_id": self.cfg.chat_id, "text": text}, timeout=TIMEOUT
            )
            return r.status_code == 200 and r.json().get("ok", False)
        except Exception as e:
            log.warning("falha ao enviar mensagem ao Telegram: %s", e)
            return False
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/alerts/test_telegram_alert.py -v`
Expected: PASS — 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/alerts/ tests/alerts/
git commit -m "feat: envio ao Telegram (sendPhoto/sendVideo) com legenda em portugues"
```

---

## Task 4: Fila de alertas (thread, retry, rate-limit)

**Files:**
- Create: `src/alerts/alert_queue.py`
- Test: `tests/alerts/test_alert_queue.py`

**Interfaces:**
- Consumes: `TelegramSender`, `Database`, `TelegramConfig`.
- Produces: `AlertQueue(sender, db, rate_limit_per_min=15, max_retries=3, backoff_base=1.0)` com:
  - `.start()`, `.stop()` (drena o que dá e encerra),
  - `.enqueue(event_id, image_path, clip_path, caption)`,
  - `.enqueue_system(text)` (alerta de câmera offline; sem mídia),
  - `.sent_count`, `.failed_count`, `.pending`.
- Ao enviar com sucesso um alerta de evento, chama `db.mark_sent(event_id)`.

**Por quê:** o Telegram é lento e cai. Se o envio fosse síncrono no pipeline, um timeout de 20s travaria a inferência de todas as câmeras. A fila isola isso. E o rate-limit local evita o bloqueio do Telegram (~20 msg/min).

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/alerts/test_alert_queue.py`:

```python
import time

import pytest

from src.alerts.alert_queue import AlertQueue
from src.storage.db import Database


class FakeSender:
    configured = True

    def __init__(self, falhas_ate=0):
        self.falhas_ate = falhas_ate
        self.tentativas = 0
        self.fotos = []
        self.mensagens = []

    def send_photo(self, path, caption):
        self.tentativas += 1
        if self.tentativas <= self.falhas_ate:
            return False
        self.fotos.append((str(path), caption))
        return True

    def send_video(self, path, caption):
        return True

    def send_message(self, text):
        self.mensagens.append(text)
        return True


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "a.db")
    d.init_schema()
    yield d
    d.close()


def _evento(db, tmp_path):
    from datetime import datetime, timezone
    img = tmp_path / "e.jpg"
    img.write_bytes(b"x")
    eid = db.insert_event(store_id="l", camera_name="c", ts_utc=datetime.now(timezone.utc).isoformat(),
                          ts_local=datetime.now().isoformat(), track_id=1, score=0.9, zone="waist",
                          signals={}, image_path=str(img), clip_path=None)
    return eid, img


def _drena(q, ate=3.0):
    fim = time.monotonic() + ate
    while q.pending > 0 and time.monotonic() < fim:
        time.sleep(0.02)


def test_sends_photo_and_marks_as_sent(db, tmp_path):
    eid, img = _evento(db, tmp_path)
    s = FakeSender()
    q = AlertQueue(s, db, rate_limit_per_min=600)
    q.start()
    try:
        q.enqueue(eid, img, None, "legenda")
        _drena(q)
    finally:
        q.stop()
    assert s.fotos and s.fotos[0][1] == "legenda"
    assert db.list_events(limit=1)[0]["sent_telegram"] == 1
    assert q.sent_count == 1


def test_retries_then_succeeds(db, tmp_path):
    eid, img = _evento(db, tmp_path)
    s = FakeSender(falhas_ate=2)  # falha 2x, acerta na 3a
    q = AlertQueue(s, db, rate_limit_per_min=600, backoff_base=0.01)
    q.start()
    try:
        q.enqueue(eid, img, None, "x")
        _drena(q, ate=5.0)
    finally:
        q.stop()
    assert s.tentativas == 3
    assert q.sent_count == 1
    assert db.list_events(limit=1)[0]["sent_telegram"] == 1


def test_gives_up_after_max_retries_but_keeps_event(db, tmp_path):
    eid, img = _evento(db, tmp_path)
    s = FakeSender(falhas_ate=99)  # sempre falha
    q = AlertQueue(s, db, rate_limit_per_min=600, max_retries=2, backoff_base=0.01)
    q.start()
    try:
        q.enqueue(eid, img, None, "x")
        _drena(q, ate=5.0)
    finally:
        q.stop()
    assert q.failed_count == 1
    # o evento continua no banco, marcado como NAO enviado (permite reenvio)
    assert db.list_events(limit=1)[0]["sent_telegram"] == 0


def test_rate_limit_delays_second_send(db, tmp_path):
    eid, img = _evento(db, tmp_path)
    s = FakeSender()
    q = AlertQueue(s, db, rate_limit_per_min=60)  # 1 por segundo
    q.start()
    try:
        t0 = time.monotonic()
        q.enqueue(eid, img, None, "1")
        q.enqueue(eid, img, None, "2")
        _drena(q, ate=5.0)
        dt = time.monotonic() - t0
    finally:
        q.stop()
    assert len(s.fotos) == 2
    assert dt >= 0.9  # o segundo esperou o rate-limit


def test_system_alert_uses_message(db, tmp_path):
    s = FakeSender()
    q = AlertQueue(s, db, rate_limit_per_min=600)
    q.start()
    try:
        q.enqueue_system("Camera 'Caixa 01' esta offline")
        _drena(q)
    finally:
        q.stop()
    assert any("offline" in m for m in s.mensagens)


def test_sender_exception_does_not_kill_the_thread(db, tmp_path):
    eid, img = _evento(db, tmp_path)

    class Explode:
        configured = True
        def __init__(self):
            self.n = 0
        def send_photo(self, p, c):
            self.n += 1
            raise RuntimeError("boom")
        def send_message(self, t):
            return True

    s = Explode()
    q = AlertQueue(s, db, rate_limit_per_min=600, max_retries=1, backoff_base=0.01)
    q.start()
    try:
        q.enqueue(eid, img, None, "x")
        _drena(q, ate=3.0)
        q.enqueue_system("ainda vivo")
        _drena(q, ate=3.0)
    finally:
        q.stop()
    assert s.n >= 1  # tentou e a thread sobreviveu
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/alerts/test_alert_queue.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.alerts.alert_queue'`

- [ ] **Step 3: Implementar `src/alerts/alert_queue.py`**

```python
"""Fila de alertas em thread própria.

O Telegram é lento e cai. Se o envio fosse síncrono no pipeline, um timeout de
20s travaria a inferência de TODAS as câmeras. A fila isola isso: o pipeline só
enfileira e segue. Retry com backoff, rate-limit local (o Telegram bloqueia
acima de ~20 msg/min) e — importante — o evento já está no banco, então uma
falha de envio nunca perde a evidência."""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class _Item:
    event_id: int | None
    image_path: Path | None
    clip_path: Path | None
    caption: str
    text: str | None = None
    tentativas: int = 0


class AlertQueue:
    def __init__(self, sender, db, rate_limit_per_min: int = 15,
                 max_retries: int = 3, backoff_base: float = 1.0) -> None:
        self.sender = sender
        self.db = db
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._min_interval = 60.0 / max(1, rate_limit_per_min)
        self._q: queue.Queue[_Item] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_send = 0.0
        self._lock = threading.Lock()
        self._sent = 0
        self._failed = 0

    @property
    def sent_count(self) -> int:
        with self._lock:
            return self._sent

    @property
    def failed_count(self) -> int:
        with self._lock:
            return self._failed

    @property
    def pending(self) -> int:
        return self._q.qsize()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="alertas", daemon=True)
        self._thread.start()

    def stop(self, drain_seconds: float = 3.0) -> None:
        fim = time.monotonic() + drain_seconds
        while self._q.qsize() > 0 and time.monotonic() < fim:
            time.sleep(0.02)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def enqueue(self, event_id, image_path, clip_path, caption: str) -> None:
        self._q.put(_Item(event_id, Path(image_path) if image_path else None,
                          Path(clip_path) if clip_path else None, caption))

    def enqueue_system(self, text: str) -> None:
        self._q.put(_Item(None, None, None, "", text=text))

    def _respeita_rate_limit(self) -> None:
        espera = self._min_interval - (time.monotonic() - self._last_send)
        if espera > 0:
            self._stop.wait(espera)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._processa(item)
            except Exception:
                # nenhuma falha de envio pode derrubar a thread de alertas
                log.exception("erro inesperado ao processar alerta")
            finally:
                self._q.task_done()

    def _processa(self, item: _Item) -> None:
        self._respeita_rate_limit()
        ok = False
        try:
            if item.text is not None:
                ok = bool(self.sender.send_message(item.text))
            elif item.image_path is not None:
                ok = bool(self.sender.send_photo(item.image_path, item.caption))
            elif item.clip_path is not None:
                ok = bool(self.sender.send_video(item.clip_path, item.caption))
        except Exception as e:
            log.warning("envio falhou: %s", e)
            ok = False
        self._last_send = time.monotonic()

        if ok:
            with self._lock:
                self._sent += 1
            if item.event_id is not None:
                try:
                    self.db.mark_sent(item.event_id)
                except Exception:
                    log.exception("falha ao marcar evento %s como enviado", item.event_id)
            return

        item.tentativas += 1
        if item.tentativas <= self.max_retries:
            self._stop.wait(self.backoff_base * (2 ** (item.tentativas - 1)))
            self._q.put(item)  # re-enfileira para nova tentativa
        else:
            with self._lock:
                self._failed += 1
            log.warning("desisti de enviar o alerta apos %s tentativas "
                        "(o evento continua salvo no banco)", item.tentativas)
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/alerts/test_alert_queue.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/alerts/alert_queue.py tests/alerts/test_alert_queue.py
git commit -m "feat: fila de alertas em thread (retry, backoff, rate-limit)"
```

---

## Task 5: Watchdog de câmera offline

**Files:**
- Create: `src/watchdog/monitor.py`
- Test: `tests/watchdog/test_monitor.py`

**Interfaces:**
- Consumes: `CameraThread` (só `.state` e `.last_frame_ts`), `Database`, `WatchdogConfig`, `AlertQueue`.
- Produces: `Watchdog(threads: dict[str, CameraThread], db, cfg: WatchdogConfig, alert_queue=None, clock=time.monotonic)` com `.start()`, `.stop()`, `.check_once()`, `.states -> dict[str, CameraState]`.
- Ao detectar queda: `db.upsert_camera_status(...OFFLINE...)` + alerta "câmera offline" (uma vez, sem repetir). Ao voltar: status ONLINE + alerta de recuperação.

**Por quê:** é o pior tipo de falha — o DVR cai, o sistema para de vigiar, e ninguém percebe. Sem isso, a loja acha que está protegida e não está.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/watchdog/test_monitor.py`:

```python
import pytest

from src.config.settings import WatchdogConfig
from src.core.types import CameraState
from src.storage.db import Database
from src.watchdog.monitor import Watchdog


class FakeCam:
    def __init__(self, state=CameraState.ONLINE, last=0.0):
        self.state = state
        self.last_frame_ts = last


class FakeQueue:
    def __init__(self):
        self.msgs = []

    def enqueue_system(self, text):
        self.msgs.append(text)


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "w.db")
    d.init_schema()
    yield d
    d.close()


def test_detects_offline_after_timeout(db):
    cam = FakeCam(last=0.0)
    q = FakeQueue()
    agora = [0.0]
    w = Watchdog({"cam1": cam}, db, WatchdogConfig(offline_after_seconds=30),
                 alert_queue=q, clock=lambda: agora[0])

    agora[0] = 10.0
    w.check_once()
    assert w.states["cam1"] == CameraState.ONLINE
    assert q.msgs == []

    agora[0] = 40.0  # 40s sem frame > 30s
    w.check_once()
    assert w.states["cam1"] == CameraState.OFFLINE
    assert len(q.msgs) == 1
    assert "offline" in q.msgs[0].lower() and "cam1" in q.msgs[0]
    assert db.get_camera_status("cam1")["state"] == "offline"


def test_does_not_repeat_offline_alert(db):
    cam = FakeCam(last=0.0)
    q = FakeQueue()
    agora = [40.0]
    w = Watchdog({"cam1": cam}, db, WatchdogConfig(offline_after_seconds=30),
                 alert_queue=q, clock=lambda: agora[0])
    w.check_once()
    agora[0] = 80.0
    w.check_once()
    agora[0] = 120.0
    w.check_once()
    assert len(q.msgs) == 1  # avisa uma vez, nao fica spammando


def test_alerts_on_recovery(db):
    cam = FakeCam(last=0.0)
    q = FakeQueue()
    agora = [40.0]
    w = Watchdog({"cam1": cam}, db, WatchdogConfig(offline_after_seconds=30),
                 alert_queue=q, clock=lambda: agora[0])
    w.check_once()  # offline
    cam.last_frame_ts = 41.0  # voltou a receber frame
    agora[0] = 42.0
    w.check_once()
    assert w.states["cam1"] == CameraState.ONLINE
    assert len(q.msgs) == 2
    assert "voltou" in q.msgs[1].lower() or "recuper" in q.msgs[1].lower()
    assert db.get_camera_status("cam1")["state"] == "online"


def test_camera_that_never_sent_a_frame_is_offline(db):
    cam = FakeCam(last=None)
    q = FakeQueue()
    w = Watchdog({"cam1": cam}, db, WatchdogConfig(offline_after_seconds=30),
                 alert_queue=q, clock=lambda: 100.0)
    w.check_once()
    assert w.states["cam1"] == CameraState.OFFLINE


def test_notify_false_does_not_alert(db):
    cam = FakeCam(last=0.0)
    q = FakeQueue()
    w = Watchdog({"cam1": cam}, db, WatchdogConfig(offline_after_seconds=30, notify=False),
                 alert_queue=q, clock=lambda: 40.0)
    w.check_once()
    assert w.states["cam1"] == CameraState.OFFLINE  # registra
    assert q.msgs == []                              # mas nao avisa


def test_works_without_alert_queue(db):
    cam = FakeCam(last=0.0)
    w = Watchdog({"cam1": cam}, db, WatchdogConfig(offline_after_seconds=30),
                 alert_queue=None, clock=lambda: 40.0)
    w.check_once()  # nao pode explodir
    assert w.states["cam1"] == CameraState.OFFLINE
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/watchdog/test_monitor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.watchdog.monitor'`

- [ ] **Step 3: Implementar `src/watchdog/monitor.py`**

Criar `src/watchdog/__init__.py` e `tests/watchdog/__init__.py` vazios.

```python
"""Watchdog: vigia o heartbeat de cada câmera.

É o pior tipo de falha — o DVR reinicia, um cabo solta, a câmera para de
mandar frame, e o sistema segue "rodando" sem vigiar nada. Sem este aviso, a
loja acha que está protegida e não está."""
from __future__ import annotations

import logging
import threading
import time

from src.config.settings import WatchdogConfig
from src.core.types import CameraState
from src.storage.db import Database

log = logging.getLogger(__name__)

CHECK_INTERVAL = 2.0


class Watchdog:
    def __init__(self, threads: dict, db: Database, cfg: WatchdogConfig,
                 alert_queue=None, clock=time.monotonic) -> None:
        self.threads = threads
        self.db = db
        self.cfg = cfg
        self.alert_queue = alert_queue
        self.clock = clock
        self.states: dict[str, CameraState] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def check_once(self) -> None:
        agora = self.clock()
        for nome, cam in self.threads.items():
            last = getattr(cam, "last_frame_ts", None)
            vivo = last is not None and (agora - last) <= self.cfg.offline_after_seconds
            novo = CameraState.ONLINE if vivo else CameraState.OFFLINE
            anterior = self.states.get(nome)
            if novo == anterior:
                continue  # sem mudanca: nao registra nem avisa de novo

            self.states[nome] = novo
            try:
                self.db.upsert_camera_status(nome, novo, str(last) if last else None)
            except Exception:
                log.exception("falha ao registrar status da camera '%s'", nome)

            # primeira leitura: registra o estado sem alarmar quem acabou de subir
            if anterior is None and novo == CameraState.ONLINE:
                continue
            self._avisa(nome, novo, anterior)

    def _avisa(self, nome: str, novo: CameraState, anterior) -> None:
        if not self.cfg.notify or self.alert_queue is None:
            return
        if novo == CameraState.OFFLINE:
            texto = (f"🔴 Câmera '{nome}' está OFFLINE — o sistema parou de vigiar "
                     f"esta câmera. Verifique o DVR, o cabo ou a rede.")
        else:
            texto = f"🟢 Câmera '{nome}' voltou ao normal."
        try:
            self.alert_queue.enqueue_system(texto)
        except Exception:
            log.exception("falha ao enfileirar aviso de camera")

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.check_once()
            except Exception:
                log.exception("erro no watchdog")  # nunca derruba a thread
            self._stop.wait(CHECK_INTERVAL)
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/watchdog/test_monitor.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/watchdog/ tests/watchdog/
git commit -m "feat: watchdog de camera offline (avisa queda e recuperacao)"
```

---

## Task 6: Retenção de evidência

**Files:**
- Create: `src/evidence/retention.py`
- Test: `tests/evidence/test_retention.py`

**Interfaces:**
- Consumes: `Database.purge_older_than(days)`, `EvidenceConfig.retention_days`.
- Produces: `RetentionJob(db, cfg, interval_seconds=6*3600, clock=time.monotonic)` com `.start()`, `.stop()`, `.run_once() -> int` (nº de arquivos removidos).

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/evidence/test_retention.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest

from src.config.settings import EvidenceConfig
from src.evidence.retention import RetentionJob
from src.storage.db import Database


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "r.db")
    d.init_schema()
    yield d
    d.close()


def _evento(db, tmp_path, dias_atras, nome):
    p = tmp_path / nome
    p.write_bytes(b"x")
    ts = (datetime.now(timezone.utc) - timedelta(days=dias_atras)).isoformat()
    db.insert_event(store_id="l", camera_name="c", ts_utc=ts,
                    ts_local=ts, track_id=1, score=0.9, zone="waist",
                    signals={}, image_path=str(p), clip_path=None)
    return p


def test_run_once_removes_old_and_keeps_recent(db, tmp_path):
    velho = _evento(db, tmp_path, 40, "velho.jpg")
    novo = _evento(db, tmp_path, 1, "novo.jpg")
    job = RetentionJob(db, EvidenceConfig(retention_days=30))

    n = job.run_once()

    assert n == 1
    assert not velho.exists()
    assert novo.exists()
    assert len(db.list_events(limit=10)) == 1


def test_run_once_is_safe_when_nothing_to_purge(db, tmp_path):
    _evento(db, tmp_path, 1, "novo.jpg")
    job = RetentionJob(db, EvidenceConfig(retention_days=30))
    assert job.run_once() == 0


def test_error_does_not_propagate(db, tmp_path, monkeypatch):
    """A rotina de manutencao nunca pode derrubar o sistema."""
    job = RetentionJob(db, EvidenceConfig(retention_days=30))
    monkeypatch.setattr(db, "purge_older_than",
                        lambda days: (_ for _ in ()).throw(OSError("banco travado")))
    assert job.run_once() == 0  # engole, loga, segue
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/evidence/test_retention.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evidence.retention'`

- [ ] **Step 3: Implementar `src/evidence/retention.py`**

```python
"""Purga periódica da evidência antiga, para o disco da loja não encher.

É rotina de manutenção: nunca pode derrubar o sistema. Qualquer erro é logado
e engolido — a próxima passada tenta de novo."""
from __future__ import annotations

import logging
import threading
import time

from src.config.settings import EvidenceConfig
from src.storage.db import Database

log = logging.getLogger(__name__)


class RetentionJob:
    def __init__(self, db: Database, cfg: EvidenceConfig,
                 interval_seconds: float = 6 * 3600) -> None:
        self.db = db
        self.cfg = cfg
        self.interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_once(self) -> int:
        try:
            removidos = self.db.purge_older_than(self.cfg.retention_days)
            if removidos:
                log.info("limpeza: %d arquivo(s) de evidencia com mais de %d dias removido(s)",
                         len(removidos), self.cfg.retention_days)
            return len(removidos)
        except Exception:
            log.exception("falha na limpeza de evidencias (tentara de novo depois)")
            return 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="retencao", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.interval)
```

- [ ] **Step 4: Rodar e confirmar que passam**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/evidence/test_retention.py -v`
Expected: PASS — 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/evidence/retention.py tests/evidence/test_retention.py
git commit -m "feat: purga periodica de evidencia antiga"
```

---

## Task 7: Ligar tudo no `main.py`

**Files:**
- Modify: `src/main.py`
- Modify: `src/pipeline.py` (alimentar o ClipBuffer por câmera)
- Test: `tests/test_operacao_integracao.py`

**Interfaces:**
- `Pipeline` ganha `clip_buffers: dict[str, ClipBuffer]` e alimenta o buffer da câmera a cada frame processado (antes do gate, para ter o "antes" mesmo sem pessoa).
- `main.py` monta: `Database` → `EvidenceRecorder` → `TelegramSender` → `AlertQueue` → `Watchdog` → `RetentionJob`, e o `on_result` do pipeline grava evidência e enfileira o alerta.

- [ ] **Step 1: Escrever o teste de integração que falha**

Criar `tests/test_operacao_integracao.py`:

```python
import time
from datetime import datetime

import numpy as np

from src.alerts.alert_queue import AlertQueue
from src.config.settings import AppConfig, CameraConfig, EvidenceConfig, StoreConfig
from src.core.types import BBox, Frame, KP, PersonDetection
from src.evidence.recorder import EvidenceRecorder
from src.pipeline import Pipeline
from src.storage.db import Database


class ScriptedEngine:
    def __init__(self, script):
        self.script = script
        self.i = 0

    def detect(self, image):
        return [PersonDetection(bbox=BBox(80, 60, 120, 300), conf=0.9)], []

    def pose(self, image, boxes):
        wx, wy, wc = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        kp = np.zeros((17, 3), dtype=np.float32)
        for nome, xy in (("left_shoulder", (90, 100)), ("right_shoulder", (110, 100)),
                         ("left_hip", (92, 200)), ("right_hip", (108, 200)),
                         ("nose", (100, 80)), ("left_eye", (96, 78)), ("right_eye", (104, 78))):
            kp[KP[nome]] = [xy[0], xy[1], 0.9]
        kp[KP["right_wrist"]] = [wx, wy, wc]
        return [kp]

    def warmup(self):
        pass


class FakeSender:
    configured = True

    def __init__(self):
        self.fotos = []

    def send_photo(self, path, caption):
        self.fotos.append((str(path), caption))
        return True

    def send_video(self, path, caption):
        return True

    def send_message(self, text):
        return True


def test_evento_vira_evidencia_e_alerta(tmp_path):
    """Ponta a ponta: gesto de ocultacao -> evento -> foto salva + linha no
    banco + alerta enfileirado e enviado."""
    db = Database(tmp_path / "app.db")
    db.init_schema()
    store = StoreConfig(id="l1", name="Mercado Teste")
    rec = EvidenceRecorder(db, EvidenceConfig(dir=str(tmp_path / "ev")), store)
    sender = FakeSender()
    fila = AlertQueue(sender, db, rate_limit_per_min=600)
    fila.start()

    cfg = AppConfig(store=store,
                    cameras=[CameraConfig(name="cam1", rtsp_url="rtsp://x",
                                          target_fps=5, zones=[])])
    # gesto: vem do reach (200,90) e some na cintura (130,205)
    script = [(200, 90, 0.9)] * 3 + [(130, 205, 0.9)] * 2 + [(130, 205, 0.05)] * 10
    p = Pipeline(cfg, ScriptedEngine(script))

    def on_result(result, frame):
        for ev in result.events:
            eid = rec.record(ev, result.camera_name, frame.image,
                             clip_buffer=p.clip_buffers.get(result.camera_name))
            row = db.list_events(limit=1)[0]
            fila.enqueue(eid, row["image_path"], row["clip_path"],
                         f"{store.name} / {result.camera_name}")

    p.on_result = on_result
    t = 0.0
    for _ in range(len(script)):
        p.process_frame(Frame("cam1", np.zeros((360, 200, 3), np.uint8), t, 1))
        t += 0.2

    fim = time.monotonic() + 3
    while fila.pending > 0 and time.monotonic() < fim:
        time.sleep(0.02)
    fila.stop()

    linhas = db.list_events(limit=10)
    assert len(linhas) >= 1, "o evento nao virou registro"
    assert linhas[0]["image_path"], "a foto da evidencia nao foi salva"
    assert linhas[0]["sent_telegram"] == 1, "o alerta nao foi marcado como enviado"
    assert sender.fotos, "o Telegram nao recebeu a foto"
    db.close()


def test_pipeline_alimenta_clip_buffer(tmp_path):
    cfg = AppConfig(store=StoreConfig(id="l", name="L"),
                    cameras=[CameraConfig(name="cam1", rtsp_url="rtsp://x",
                                          target_fps=5, zones=[])])
    p = Pipeline(cfg, ScriptedEngine([(160, 120, 0.9)] * 5))
    for i in range(5):
        p.process_frame(Frame("cam1", np.zeros((360, 200, 3), np.uint8), i * 0.2, i))
    assert len(p.clip_buffers["cam1"]) == 5
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_operacao_integracao.py -v`
Expected: FAIL — `AttributeError: 'Pipeline' object has no attribute 'clip_buffers'`

- [ ] **Step 3: Modificar `src/pipeline.py`**

Adicionar o import:
```python
from src.evidence.clip_buffer import ClipBuffer
```

No `__init__`, criar um buffer por câmera (o "antes" do clipe):
```python
        self.clip_buffers: dict[str, ClipBuffer] = {
            c.name: ClipBuffer(
                seconds=cfg.evidence.clip_pre_seconds + cfg.evidence.clip_post_seconds + 1.0,
                fps_hint=c.target_fps,
            )
            for c in self.cameras
        }
```

No início de `process_frame`, ANTES do gate (o buffer precisa do "antes" mesmo sem pessoa na zona):
```python
        buf = self.clip_buffers.get(frame.camera_name)
        if buf is not None:
            buf.add(frame.image, frame.ts)
```

- [ ] **Step 4: Modificar `src/main.py`**

Depois de montar o pipeline, montar a camada de operação e ligar o `on_result`:

```python
    db = Database("data/app.db")
    db.init_schema()
    recorder = EvidenceRecorder(db, cfg.evidence, cfg.store)
    sender = TelegramSender(cfg.telegram)
    alerts = AlertQueue(sender, db, rate_limit_per_min=cfg.telegram.rate_limit_per_min)
    retention = RetentionJob(db, cfg.evidence)
    watchdog = Watchdog(pipeline.threads, db, cfg.watchdog, alert_queue=alerts)

    if not sender.configured:
        log.warning("Telegram sem token/chat_id no config — os alertas ficam so "
                    "registrados no banco, sem envio.")

    def _on_result(result, frame):
        for ev in result.events:
            event_id = recorder.record(ev, result.camera_name, frame.image,
                                       clip_buffer=pipeline.clip_buffers.get(result.camera_name))
            row = db.list_events(limit=1)[0]
            caption = sender.caption_for(cfg.store.name, result.camera_name,
                                         datetime.now(), ev.zone)
            alerts.enqueue(event_id, row["image_path"], row["clip_path"], caption)
            log.info("OCULTACAO em '%s' (zona %s, score %.2f) — evidencia #%s",
                     result.camera_name, ev.zone, ev.score, event_id)

    pipeline.on_result = _on_result
```

E no ciclo de vida: `alerts.start()`, `retention.start()`, `watchdog.start()` junto com `pipeline.start()`; e no `finally`, parar todos (`watchdog.stop()`, `retention.stop()`, `alerts.stop()`, `pipeline.stop()`, `db.close()`).

- [ ] **Step 5: Rodar e confirmar que passam**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_operacao_integracao.py -v`
Expected: PASS — 2 passed

- [ ] **Step 6: Rodar a suíte inteira**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest -q -m "not slow"`
Expected: PASS — 186 (planos 1-2) + ~35 novos, zero regressões.

- [ ] **Step 7: Commit**

```bash
git add src/main.py src/pipeline.py tests/test_operacao_integracao.py
git commit -m "feat: liga evidencia, alertas, watchdog e retencao no sistema"
```

---

## Fechamento do Plano 3

- [ ] **Suíte completa** — `-m "not slow"` e `-m slow`, zero regressões.

- [ ] **Teste manual com Telegram REAL** (quando o token chegar do cliente)

1. Preencher `bot_token` e `chat_id` no config.
2. Rodar `python -m src.main --config config/config.json` contra o DVR simulado.
3. Confirmar: a foto anotada chega no grupo com a legenda em português (loja, câmera, hora, gesto).
4. Derrubar o DVR simulado e confirmar o aviso "🔴 Câmera offline"; restaurar e confirmar o "🟢 voltou ao normal".
Documentar o resultado (é a prova de que o alerta funciona ponta a ponta).

- [ ] **Revisão final da branch** (subagent-driven-development exige) — foco em: nenhuma exceção derruba thread; o Telegram nunca trava o pipeline; o evento nunca se perde por falha de envio/disco.

- [ ] **Merge para `master`** e seguir para o Plano 4 (UI PySide6 + editor de zonas + instalador).

**Estado ao fim deste plano:** o gesto de ocultação vira **foto anotada + clipe + registro no banco + alerta no Telegram do lojista**, e a loja é avisada se uma câmera cair. É o sistema fazendo o trabalho dele.
