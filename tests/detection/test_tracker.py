from src.core.types import BBox, PersonDetection
from src.detection.tracker import Tracker


def _p(x1, y1, x2, y2, conf=0.9) -> PersonDetection:
    return PersonDetection(bbox=BBox(x1, y1, x2, y2), conf=conf)


def test_assigns_ids_to_new_people():
    t = Tracker(max_lost_seconds=2.0)
    out = t.update([_p(0, 0, 50, 150), _p(200, 0, 250, 150)], ts=0.0)
    ids = {p.track_id for p in out}
    assert ids == {1, 2}


def test_keeps_id_when_person_moves_a_little():
    t = Tracker(max_lost_seconds=2.0)
    first = t.update([_p(0, 0, 50, 150)], ts=0.0)[0]
    second = t.update([_p(8, 2, 58, 152)], ts=0.2)[0]
    assert second.track_id == first.track_id


def test_new_id_when_person_is_completely_elsewhere():
    t = Tracker(max_lost_seconds=2.0)
    t.update([_p(0, 0, 50, 150)], ts=0.0)
    out = t.update([_p(500, 0, 550, 150)], ts=0.2)
    assert out[0].track_id == 2


def test_id_survives_a_short_gap():
    """Pessoa sumiu por 1 frame (oclusão por gôndola) e voltou perto:
    tem que manter o id — senão o dwell da ocultação reinicia do zero."""
    t = Tracker(max_lost_seconds=2.0)
    first = t.update([_p(0, 0, 50, 150)], ts=0.0)[0]
    t.update([], ts=0.2)
    again = t.update([_p(5, 0, 55, 150)], ts=0.4)[0]
    assert again.track_id == first.track_id


def test_id_is_dropped_after_max_lost():
    t = Tracker(max_lost_seconds=1.0)
    t.update([_p(0, 0, 50, 150)], ts=0.0)
    t.update([], ts=2.0)
    out = t.update([_p(0, 0, 50, 150)], ts=2.1)
    assert out[0].track_id == 2
    assert t.active_ids() == {2}


def test_two_people_do_not_swap_ids():
    t = Tracker(max_lost_seconds=2.0)
    a, b = t.update([_p(0, 0, 50, 150), _p(300, 0, 350, 150)], ts=0.0)
    out = t.update([_p(305, 0, 355, 150), _p(6, 0, 56, 150)], ts=0.2)
    by_id = {p.track_id: p.bbox.x1 for p in out}
    assert by_id[a.track_id] < 100
    assert by_id[b.track_id] > 300
