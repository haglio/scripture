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

