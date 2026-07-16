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
