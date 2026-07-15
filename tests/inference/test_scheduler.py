from src.inference.scheduler import Scheduler


def _pick_serve_release(s: Scheduler, now: float) -> str | None:
    """Simula o ciclo completo de um worker: reivindica a câmera, atende, e
    libera — igual ao que `WorkerPool._run` faz de verdade. Os testes de
    priorização abaixo testam a ORDEM de escolha entre ciclos completos, não
    a exclusividade em si (isso é coberto por `test_next_camera_...` e por
    `tests/inference/test_worker_pool_race.py`), então cada iteração deve
    liberar a câmera antes da próxima chamada a `next_camera`."""
    c = s.next_camera(now)
    if c is not None:
        s.mark_served(c, now)
        s.release(c)
    return c


def test_round_robin_when_no_activity():
    s = Scheduler(["a", "b", "c"])
    picked = []
    now = 0.0
    for _ in range(6):
        picked.append(_pick_serve_release(s, now))
        now += 0.1
    assert picked == ["a", "b", "c", "a", "b", "c"]


def test_camera_with_recent_person_gets_more_turns():
    s = Scheduler(["a", "b"], active_boost=3.0, active_window=5.0)
    s.mark_activity("a", ts=0.0)
    picked = []
    now = 0.0
    for _ in range(8):
        picked.append(_pick_serve_release(s, now))
        now += 0.1
    assert picked.count("a") > picked.count("b")


def test_boost_expires_after_window():
    s = Scheduler(["a", "b"], active_boost=3.0, active_window=1.0)
    s.mark_activity("a", ts=0.0)
    picked = []
    now = 10.0  # muito depois da janela
    for _ in range(6):
        picked.append(_pick_serve_release(s, now))
        now += 0.1
    assert picked.count("a") == picked.count("b") == 3


def test_returns_none_without_cameras():
    assert Scheduler([]).next_camera(0.0) is None


def test_next_camera_does_not_return_an_in_flight_camera():
    """O cerne da correção: uma câmera já reivindicada (in-flight) não pode
    ser entregue a um segundo worker — é isso que impede dois workers
    chamando Tracker.update() da mesma câmera ao mesmo tempo."""
    s = Scheduler(["a", "b"])
    first = s.next_camera(0.0)
    assert first in ("a", "b")
    second = s.next_camera(0.0)
    assert second is not None
    assert second != first, "a mesma câmera não pode ser entregue duas vezes sem release()"


def test_next_camera_returns_none_when_all_cameras_in_flight():
    s = Scheduler(["a", "b"])
    s.next_camera(0.0)
    s.next_camera(0.0)
    assert s.next_camera(0.0) is None, "todas in-flight: nada disponível agora"


def test_release_makes_camera_available_again():
    s = Scheduler(["a"])
    claimed = s.next_camera(0.0)
    assert claimed == "a"
    assert s.next_camera(0.1) is None, "única câmera já está in-flight"
    s.release("a")
    assert s.next_camera(0.2) == "a", "liberada, volta a ser elegível"
