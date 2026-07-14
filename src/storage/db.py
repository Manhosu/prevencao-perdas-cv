"""Persistência local. `signals_json` guarda a decomposição do score —
é o que torna um falso positivo em campo depurável.
`store_id` custa zero hoje e evita reescrever o schema no dia em que o
cliente quiser um painel com várias lojas."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.core.types import CameraState

logger = logging.getLogger(__name__)

VALID_FEEDBACK = {"true_positive", "false_positive"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY,
  store_id TEXT NOT NULL,
  camera_name TEXT NOT NULL,
  ts_utc TEXT NOT NULL,
  ts_local TEXT NOT NULL,
  track_id INTEGER,
  score REAL NOT NULL,
  zone TEXT NOT NULL,
  signals_json TEXT NOT NULL,
  image_path TEXT,
  clip_path TEXT,
  sent_telegram INTEGER NOT NULL DEFAULT 0,
  feedback TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts_utc);

CREATE TABLE IF NOT EXISTS camera_status (
  camera_name TEXT PRIMARY KEY,
  state TEXT NOT NULL,
  last_frame_ts TEXT,
  since TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_meta (key TEXT PRIMARY KEY, value TEXT);
"""


def _ensure_ts_utc_has_timezone(ts_utc: str) -> None:
    """`list_events` depende de comparação lexicográfica de strings
    ISO-8601 em UTC para ordenar por data. Um `ts_utc` naive (sem fuso,
    ex.: `datetime.now().isoformat()`) quebra essa ordenação em silêncio —
    por isso é rejeitado aqui, na entrada."""
    try:
        parsed = datetime.fromisoformat(ts_utc)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "ts_utc precisa ser uma string ISO-8601 válida com fuso "
            f"horário (ex.: '2026-07-14T10:00:00+00:00'); recebido: {ts_utc!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise ValueError(
            "ts_utc precisa incluir informação de fuso horário explícita "
            f"(ex.: '+00:00' ou 'Z'); recebido sem fuso: {ts_utc!r}"
        )


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        # `check_same_thread=False` só desliga a checagem do Python — não
        # torna o uso concorrente do mesmo objeto Connection seguro. A
        # thread de inferência insere eventos, a UI lê e o watchdog atualiza
        # status, todos sobre este mesmo objeto Database: um único RLock por
        # instância serializa todo acesso a `self._conn`, inclusive leituras
        # (um cursor sendo consumido em uma thread enquanto outra escreve é
        # o que produz `another row available` / `bad parameter or other
        # API misuse`).
        self._lock = threading.RLock()

    def init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)

    def insert_event(
        self,
        *,
        store_id: str,
        camera_name: str,
        ts_utc: str,
        ts_local: str,
        track_id: int | None,
        score: float,
        zone: str,
        signals: dict,
        image_path: str | None,
        clip_path: str | None,
    ) -> int:
        _ensure_ts_utc_has_timezone(ts_utc)
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    """INSERT INTO events (store_id, camera_name, ts_utc, ts_local,
                           track_id, score, zone, signals_json, image_path, clip_path)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        store_id,
                        camera_name,
                        ts_utc,
                        ts_local,
                        track_id,
                        score,
                        zone,
                        json.dumps(signals),
                        image_path,
                        clip_path,
                    ),
                )
                event_id = int(cur.lastrowid)
        return event_id

    def mark_sent(self, event_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE events SET sent_telegram=1 WHERE id=?", (event_id,)
            )

    def set_feedback(self, event_id: int, value: str) -> None:
        if value not in VALID_FEEDBACK:
            raise ValueError(f"feedback inválido: {value}")
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE events SET feedback=? WHERE id=?", (value, event_id)
            )

    def list_events(self, limit: int = 100, since: str | None = None) -> list[sqlite3.Row]:
        with self._lock:
            if since:
                return list(
                    self._conn.execute(
                        "SELECT * FROM events WHERE ts_utc >= ? ORDER BY ts_utc DESC LIMIT ?",
                        (since, limit),
                    )
                )
            return list(
                self._conn.execute(
                    "SELECT * FROM events ORDER BY ts_utc DESC LIMIT ?", (limit,)
                )
            )

    def upsert_camera_status(
        self, camera_name: str, state: CameraState, last_frame_ts: str | None
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO camera_status (camera_name, state, last_frame_ts, since)
                   VALUES (?,?,?,?)
                   ON CONFLICT(camera_name) DO UPDATE SET
                     state=excluded.state,
                     last_frame_ts=excluded.last_frame_ts,
                     since=CASE WHEN camera_status.state != excluded.state
                                THEN excluded.since ELSE camera_status.since END""",
                (camera_name, state.value, last_frame_ts, now),
            )

    def get_camera_status(self, camera_name: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM camera_status WHERE camera_name=?", (camera_name,)
            ).fetchone()

    def purge_older_than(self, days: int) -> list[Path]:
        """Apaga eventos mais velhos que `days`, tentando remover os arquivos
        de evidência associados. Nunca propaga exceção de I/O: se um arquivo
        não puder ser removido agora (antivírus escaneando, Explorer gerando
        thumbnail, visualizador aberto), o evento correspondente é mantido
        no banco para que a próxima purga tente de novo — em vez de apagar
        a linha e deixar o arquivo órfão para sempre."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._lock:
            rows = list(
                self._conn.execute(
                    "SELECT id, image_path, clip_path FROM events WHERE ts_utc < ?",
                    (cutoff,),
                )
            )

        removed_files: list[Path] = []
        # path -> True (removido ou já inexistente) / False (falhou e deve
        # ser reconsiderado na próxima purga). Deduplicado: cada caminho só
        # é tentado uma vez, mesmo que dois eventos apontem para ele.
        path_ok: dict[str, bool] = {}

        def try_remove(p: str) -> bool:
            if p in path_ok:
                return path_ok[p]
            path_obj = Path(p)
            if not path_obj.exists():
                path_ok[p] = True
                return True
            try:
                path_obj.unlink()
                removed_files.append(path_obj)
                path_ok[p] = True
            except OSError:
                logger.warning(
                    "Não foi possível remover o arquivo de evidência %s "
                    "(pode estar bloqueado por outro processo); o evento "
                    "correspondente será mantido no banco para nova "
                    "tentativa na próxima purga.",
                    p,
                )
                path_ok[p] = False
            return path_ok[p]

        ids_to_delete = []
        for row in rows:
            paths = [p for p in (row["image_path"], row["clip_path"]) if p]
            if all(try_remove(p) for p in paths):
                ids_to_delete.append(row["id"])

        if ids_to_delete:
            with self._lock:
                with self._conn:
                    self._conn.executemany(
                        "DELETE FROM events WHERE id=?",
                        [(i,) for i in ids_to_delete],
                    )

        return removed_files

    def close(self) -> None:
        with self._lock:
            self._conn.close()
