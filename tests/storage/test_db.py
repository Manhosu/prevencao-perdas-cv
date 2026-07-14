import json
from datetime import datetime, timedelta, timezone

import pytest

from src.core.types import CameraState
from src.storage.db import Database


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "app.db")
    d.init_schema()
    yield d
    d.close()


def _insert(db, **kw):
    defaults = dict(
        store_id="loja1",
        camera_name="Caixa 01",
        ts_utc=datetime.now(timezone.utc).isoformat(),
        ts_local=datetime.now().isoformat(),
        track_id=7,
        score=0.81,
        zone="waist",
        signals={"dwell": 1.0, "approach": 1.0, "vanish": 0.0, "retract": 0.0},
        image_path="evidence/a.jpg",
        clip_path=None,
    )
    defaults.update(kw)
    return db.insert_event(**defaults)


def test_insert_and_list_event(db):
    eid = _insert(db)
    assert eid > 0
    rows = db.list_events(limit=10)
    assert len(rows) == 1
    assert rows[0]["camera_name"] == "Caixa 01"
    assert rows[0]["score"] == pytest.approx(0.81)
    assert rows[0]["sent_telegram"] == 0
    assert rows[0]["feedback"] is None
    # a decomposição do score é preservada: sem isso, falso positivo em campo
    # é indepurável
    assert json.loads(rows[0]["signals_json"])["dwell"] == 1.0


def test_mark_sent(db):
    eid = _insert(db)
    db.mark_sent(eid)
    assert db.list_events(limit=1)[0]["sent_telegram"] == 1


def test_set_feedback(db):
    eid = _insert(db)
    db.set_feedback(eid, "false_positive")
    assert db.list_events(limit=1)[0]["feedback"] == "false_positive"


def test_set_feedback_rejects_invalid_value(db):
    eid = _insert(db)
    with pytest.raises(ValueError):
        db.set_feedback(eid, "talvez")


def test_list_events_is_newest_first(db):
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    _insert(db, ts_utc=old.isoformat(), camera_name="Antiga")
    _insert(db, camera_name="Nova")
    rows = db.list_events(limit=10)
    assert rows[0]["camera_name"] == "Nova"


def test_camera_status_upsert(db):
    db.upsert_camera_status("Caixa 01", CameraState.ONLINE, "2026-07-14T10:00:00")
    db.upsert_camera_status("Caixa 01", CameraState.OFFLINE, "2026-07-14T10:05:00")
    row = db.get_camera_status("Caixa 01")
    assert row["state"] == "offline"
    assert row["last_frame_ts"] == "2026-07-14T10:05:00"


def test_purge_removes_old_events_and_returns_files(db, tmp_path):
    img = tmp_path / "old.jpg"
    img.write_bytes(b"x")
    old = datetime.now(timezone.utc) - timedelta(days=40)
    _insert(db, ts_utc=old.isoformat(), image_path=str(img))
    _insert(db)  # recente

    removed = db.purge_older_than(days=30)

    assert [str(p) for p in removed] == [str(img)]
    assert len(db.list_events(limit=10)) == 1
