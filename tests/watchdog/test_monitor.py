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
