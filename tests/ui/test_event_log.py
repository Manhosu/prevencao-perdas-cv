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
