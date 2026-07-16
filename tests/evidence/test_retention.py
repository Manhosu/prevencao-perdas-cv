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
