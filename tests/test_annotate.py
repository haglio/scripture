"""Tests for sparse ground-truth labeling sessions."""

from scripture.annotate import next_scheduled, schedule_frames


def test_schedule_frames_strides_through_range():
    assert schedule_frames(100, 400, stride=100) == [100, 200, 300, 400]


def test_next_scheduled_returns_first_unannotated_after_current():
    sched = [100, 200, 300, 400]
    assert next_scheduled(sched, annotated={200}, after=150) == 300


def test_next_scheduled_wraps_to_earliest_unannotated():
    sched = [100, 200, 300]
    assert next_scheduled(sched, annotated={300}, after=350) == 100


def test_next_scheduled_none_when_all_done():
    sched = [100, 200]
    assert next_scheduled(sched, annotated={100, 200}, after=0) is None


def test_collect_labels_flattens_explicit_entries_with_pos():
    from scripture.annotate import collect_labels
    ground_truth = {
        1: {
            300: {"tip": (50, 20), "base": (50, 120), "contact": (50, 70),
                  "is_action": False},
            360: {"tip": (50, 20), "base": (50, 120), "contact": None,
                  "is_action": False},
        },
    }
    rows = collect_labels(ground_truth)
    assert len(rows) == 2
    mid = next(r for r in rows if r["frame"] == 300)
    assert mid["scene"] == 1 and abs(mid["pos"] - 50.0) < 1e-6
    none = next(r for r in rows if r["frame"] == 360)
    assert none["pos"] is None and none["contact"] is None
