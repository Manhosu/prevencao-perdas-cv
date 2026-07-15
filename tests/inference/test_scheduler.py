from src.inference.scheduler import Scheduler


def test_round_robin_when_no_activity():
    s = Scheduler(["a", "b", "c"])
    picked = []
    now = 0.0
    for _ in range(6):
        c = s.next_camera(now)
        picked.append(c)
        s.mark_served(c, now)
        now += 0.1
    assert picked == ["a", "b", "c", "a", "b", "c"]


def test_camera_with_recent_person_gets_more_turns():
    s = Scheduler(["a", "b"], active_boost=3.0, active_window=5.0)
    s.mark_activity("a", ts=0.0)
    picked = []
    now = 0.0
    for _ in range(8):
        c = s.next_camera(now)
        picked.append(c)
        s.mark_served(c, now)
        now += 0.1
    assert picked.count("a") > picked.count("b")


def test_boost_expires_after_window():
    s = Scheduler(["a", "b"], active_boost=3.0, active_window=1.0)
    s.mark_activity("a", ts=0.0)
    picked = []
    now = 10.0  # muito depois da janela
    for _ in range(6):
        c = s.next_camera(now)
        picked.append(c)
        s.mark_served(c, now)
        now += 0.1
    assert picked.count("a") == picked.count("b") == 3


def test_returns_none_without_cameras():
    assert Scheduler([]).next_camera(0.0) is None
