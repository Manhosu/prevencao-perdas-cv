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
    res = rec.record(_event(), "Caixa 01", _img())

    assert res.event_id > 0
    row = db.list_events(limit=1)[0]
    assert row["camera_name"] == "Caixa 01"
    assert row["zone"] == "waist"
    assert row["score"] == pytest.approx(0.82)
    assert row["store_id"] == "l1"
    img_path = Path(row["image_path"])
    assert img_path.exists() and img_path.suffix == ".jpg"
    # o record() devolve o mesmo caminho que gravou no banco — quem chama usa
    # isso direto, sem re-consultar o banco (CRITICAL 1).
    assert res.image_path == row["image_path"]


def test_record_saves_clip_when_buffer_given(db, tmp_path):
    buf = ClipBuffer(seconds=6.0, fps_hint=5)
    for i in range(30):  # 6s de frames
        buf.add(_img(), ts=i * 0.2)
    rec = EvidenceRecorder(db, EvidenceConfig(dir=str(tmp_path / "ev"),
                                              clip_pre_seconds=2.0, clip_post_seconds=0.0),
                           StoreConfig(id="l1", name="Loja 1"))
    res = rec.record(_event(ts=4.0), "Caixa 01", _img(), clip_buffer=buf)

    row = db.list_events(limit=1)[0]
    assert row["clip_path"]
    assert Path(row["clip_path"]).exists()
    assert res.clip_path == row["clip_path"]


def test_row_is_saved_even_if_file_write_fails(db, tmp_path, monkeypatch):
    """Disco cheio nao pode fazer o evento sumir — o registro vem primeiro."""
    rec = EvidenceRecorder(db, EvidenceConfig(dir=str(tmp_path / "ev")),
                           StoreConfig(id="l1", name="Loja 1"))
    import cv2
    monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: (_ for _ in ()).throw(OSError("disco cheio")))

    res = rec.record(_event(), "Caixa 01", _img())

    assert res.event_id > 0  # o evento existe no banco
    assert res.image_path is None  # sem arquivo — quem chama sabe na hora, sem reconsulta
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
