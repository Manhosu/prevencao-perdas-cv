"""Persistência local. `signals_json` guarda a decomposição do score —
é o que torna um falso positivo em campo depurável.
`store_id` custa zero hoje e evita reescrever o schema no dia em que o
cliente quiser um painel com várias lojas."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.core.types import CameraState

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


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")

    def init_schema(self) -> None:
        with self._conn:
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
        return int(cur.lastrowid)

    def mark_sent(self, event_id: int) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE events SET sent_telegram=1 WHERE id=?", (event_id,)
            )

    def set_feedback(self, event_id: int, value: str) -> None:
        if value not in VALID_FEEDBACK:
            raise ValueError(f"feedback inválido: {value}")
        with self._conn:
            self._conn.execute(
                "UPDATE events SET feedback=? WHERE id=?", (value, event_id)
            )

    def list_events(self, limit: int = 100, since: str | None = None) -> list[sqlite3.Row]:
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
        with self._conn:
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
        return self._conn.execute(
            "SELECT * FROM camera_status WHERE camera_name=?", (camera_name,)
        ).fetchone()

    def purge_older_than(self, days: int) -> list[Path]:
        """Apaga eventos mais velhos que `days` e devolve os arquivos de
        evidência que devem ser removidos do disco."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = list(
            self._conn.execute(
                "SELECT image_path, clip_path FROM events WHERE ts_utc < ?", (cutoff,)
            )
        )
        files = [
            Path(p)
            for r in rows
            for p in (r["image_path"], r["clip_path"])
            if p and Path(p).exists()
        ]
        with self._conn:
            self._conn.execute("DELETE FROM events WHERE ts_utc < ?", (cutoff,))
        for f in files:
            f.unlink(missing_ok=True)
        return files

    def close(self) -> None:
        self._conn.close()
