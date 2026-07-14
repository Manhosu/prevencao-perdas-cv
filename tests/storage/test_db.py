import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


def test_concurrent_inserts_never_lose_events_or_raise(db):
    """16 threads martelando insert/upsert/list ao mesmo tempo — reproduz o
    cenário de produção: thread de inferência, UI e watchdog no mesmo
    objeto Database. Nenhuma exceção pode escapar e nenhuma linha pode
    sumir."""
    n_insert_threads = 8
    inserts_per_thread = 100
    n_status_threads = 4
    n_list_threads = 4
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def record_error(exc: BaseException) -> None:
        with errors_lock:
            errors.append(exc)

    def insert_worker(idx: int) -> None:
        try:
            for i in range(inserts_per_thread):
                _insert(db, camera_name=f"Cam{idx}", track_id=i)
        except BaseException as exc:  # noqa: BLE001 - queremos capturar tudo
            record_error(exc)

    def status_worker(idx: int) -> None:
        try:
            for i in range(50):
                state = CameraState.ONLINE if i % 2 == 0 else CameraState.OFFLINE
                db.upsert_camera_status(f"Status{idx}", state, None)
        except BaseException as exc:  # noqa: BLE001
            record_error(exc)

    def list_worker() -> None:
        try:
            for _ in range(50):
                db.list_events(limit=20)
        except BaseException as exc:  # noqa: BLE001
            record_error(exc)

    threads = (
        [threading.Thread(target=insert_worker, args=(i,)) for i in range(n_insert_threads)]
        + [threading.Thread(target=status_worker, args=(i,)) for i in range(n_status_threads)]
        + [threading.Thread(target=list_worker) for _ in range(n_list_threads)]
    )

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"exceções levantadas durante acesso concorrente: {errors!r}"
    rows = db.list_events(limit=100000)
    assert len(rows) == n_insert_threads * inserts_per_thread


def test_camera_status_since_unchanged_while_state_stays_the_same(db):
    """`since` é o "desde quando esta câmera está neste estado" que
    aparece na UI e no alerta do Telegram. Atualizações repetidas com o
    mesmo state (ex.: novos frames chegando enquanto a câmera segue
    ONLINE) não podem empurrar o `since` pra frente."""
    db.upsert_camera_status("Caixa 01", CameraState.ONLINE, "2026-07-14T10:00:00")
    since_first = db.get_camera_status("Caixa 01")["since"]

    db.upsert_camera_status("Caixa 01", CameraState.ONLINE, "2026-07-14T10:05:00")
    since_after_same_state = db.get_camera_status("Caixa 01")["since"]

    assert since_after_same_state == since_first


def test_camera_status_since_changes_when_state_changes(db):
    db.upsert_camera_status("Caixa 01", CameraState.ONLINE, "2026-07-14T10:00:00")
    since_online = db.get_camera_status("Caixa 01")["since"]

    db.upsert_camera_status("Caixa 01", CameraState.OFFLINE, "2026-07-14T10:05:00")
    since_offline = db.get_camera_status("Caixa 01")["since"]

    assert since_offline != since_online


def test_purge_removes_old_events_and_returns_files(db, tmp_path):
    img = tmp_path / "old.jpg"
    img.write_bytes(b"x")
    old = datetime.now(timezone.utc) - timedelta(days=40)
    _insert(db, ts_utc=old.isoformat(), image_path=str(img))
    _insert(db)  # recente

    removed = db.purge_older_than(days=30)

    assert [str(p) for p in removed] == [str(img)]
    assert len(db.list_events(limit=10)) == 1


def test_purge_with_missing_file_on_disk_still_removes_event(db, tmp_path):
    """O arquivo já não existe no disco (foi apagado manualmente, por
    exemplo) — a purga não pode quebrar por isso, e o evento deve ser
    apagado normalmente."""
    missing = tmp_path / "sumiu.jpg"
    old = datetime.now(timezone.utc) - timedelta(days=40)
    _insert(db, ts_utc=old.isoformat(), image_path=str(missing))

    removed = db.purge_older_than(days=30)

    assert removed == []
    assert len(db.list_events(limit=10)) == 0


def test_purge_deduplicates_shared_file_path(db, tmp_path):
    """Dois eventos apontando pro mesmo arquivo (ex.: image_path de um e
    clip_path de outro) só devem gerar uma remoção na lista devolvida."""
    shared = tmp_path / "shared.jpg"
    shared.write_bytes(b"x")
    old = datetime.now(timezone.utc) - timedelta(days=40)
    _insert(db, ts_utc=old.isoformat(), image_path=str(shared), clip_path=None)
    _insert(db, ts_utc=old.isoformat(), image_path=str(shared), clip_path=None)

    removed = db.purge_older_than(days=30)

    assert [str(p) for p in removed] == [str(shared)]
    assert len(db.list_events(limit=10)) == 0


def test_purge_keeps_event_when_file_cannot_be_removed(db, tmp_path, monkeypatch):
    """Arquivo bloqueado (antivírus, Explorer, visualizador aberto no
    Windows): unlink levanta PermissionError. A purga não pode propagar a
    exceção, nem apagar a linha do banco — senão o arquivo vira órfão para
    sempre. Uma segunda purga, com o unlink funcionando de novo, deve
    remover tudo."""
    locked = tmp_path / "locked.jpg"
    locked.write_bytes(b"x")
    old = datetime.now(timezone.utc) - timedelta(days=40)
    _insert(db, ts_utc=old.isoformat(), image_path=str(locked))

    original_unlink = Path.unlink

    def fake_unlink(self, *args, **kwargs):
        if self == locked:
            raise PermissionError("arquivo em uso por outro processo")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fake_unlink)

    removed = db.purge_older_than(days=30)

    assert removed == []
    assert len(db.list_events(limit=10)) == 1  # evento mantido, não órfão
    assert locked.exists()

    monkeypatch.setattr(Path, "unlink", original_unlink)

    removed_second = db.purge_older_than(days=30)

    assert [str(p) for p in removed_second] == [str(locked)]
    assert len(db.list_events(limit=10)) == 0


def test_insert_event_rejects_naive_ts_utc(db):
    """`list_events` ordena por comparação lexicográfica de ISO-8601 em
    UTC. Um `ts_utc` naive (sem fuso) quebra essa ordenação em silêncio —
    por isso precisa ser rejeitado explicitamente na entrada."""
    naive = datetime.now().isoformat()  # sem tzinfo, ex.: '2026-07-14T10:00:00'
    with pytest.raises(ValueError):
        _insert(db, ts_utc=naive)


def test_insert_event_accepts_ts_utc_with_utc_offset(db):
    eid = _insert(db, ts_utc=datetime.now(timezone.utc).isoformat())
    assert eid > 0


def test_insert_event_accepts_ts_utc_with_z_suffix(db):
    eid = _insert(db, ts_utc="2026-07-14T10:00:00Z")
    assert eid > 0
