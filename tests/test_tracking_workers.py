from __future__ import annotations

import time

import numpy as np

from scripture.motion_tracker import AxisDefinition, TrackingResult
from scripture.scene import Scene
from scripture.tracking_workers import AutoProcessWorker, ProcessWorker

_JOBS = [(0, Scene(0, 5), AxisDefinition(tip=(80, 20), base=(80, 100)))]


def _everything_said_by(worker, *signals):
    said = []
    for name in signals:
        getattr(worker, name).connect(lambda *args, name=name: said.append((name, args)))
    return said


def _frames_until_stopped(on_frame):
    frame = 0
    while True:
        on_frame(frame)
        frame += 1
        time.sleep(0.001)


def test_a_stopped_scene_run_ends_at_the_next_frame_saying_nothing(qt_app):
    worker = ProcessWorker("example clip.mp4", _JOBS, 30.0,
                           track=lambda *_args, on_frame: _frames_until_stopped(on_frame))
    said = _everything_said_by(worker, "frame_progress", "scene_done", "error", "finished")
    worker.ask_to_stop()

    worker.run()

    assert said == []


def test_a_stopped_automatic_run_ends_at_the_next_frame_saying_nothing(qt_app):
    worker = AutoProcessWorker("example clip.mp4",
                               track=lambda _path, on_frame: _frames_until_stopped(on_frame))
    said = _everything_said_by(worker, "frame_progress", "done", "error")
    worker.ask_to_stop()

    worker.run()

    assert said == []


def test_stopping_a_running_thread_lets_the_tracker_tidy_up(qt_app):
    tidied = []

    def tracker(_path, on_frame):
        try:
            _frames_until_stopped(on_frame)
        finally:
            tidied.append(True)

    worker = AutoProcessWorker("example clip.mp4", track=tracker)
    worker.frame_progress.connect(lambda _frame: None)
    worker.start()

    worker.stop()

    assert worker.isFinished()
    assert tidied == [True]


def test_a_tracker_that_will_not_reach_a_frame_is_ended_anyway(qt_app):
    ended = []

    class StuckBetweenFrames(AutoProcessWorker):
        def wait(self, *_args):
            return bool(ended)

        def terminate(self):
            ended.append(True)

    worker = StuckBetweenFrames("example clip.mp4")
    worker.frame_progress.connect(lambda _frame: None)

    worker.stop()

    assert ended == [True]


def test_a_scene_run_reports_its_frames_with_each_scenes_offset_then_its_result(qt_app):
    tracked = TrackingResult(timestamps_ms=np.arange(5) * 33.0, positions=np.full(5, 0.5))

    def tracker(*_args, on_frame):
        on_frame(2)
        return tracked

    jobs = [*_JOBS, (1, Scene(5, 10), _JOBS[0][2])]
    worker = ProcessWorker("example clip.mp4", jobs, 30.0, track=tracker)
    said = _everything_said_by(worker, "frame_progress", "scene_done", "finished")

    worker.run()

    assert [entry[0] for entry in said] == [
        "frame_progress", "scene_done", "frame_progress", "scene_done", "finished"]
    assert said[0][1] == (2,)
    assert said[2][1] == (7,)


def test_a_scene_that_fails_says_so_and_the_run_goes_on(qt_app):
    def tracker(*_args, on_frame):
        raise ValueError("an invented failure")

    worker = ProcessWorker("example clip.mp4", _JOBS, 30.0, track=tracker)
    said = _everything_said_by(worker, "error", "finished")

    worker.run()

    assert said == [("error", (0, "an invented failure")), ("finished", ())]


def test_an_automatic_run_hands_over_what_it_found(qt_app):
    worker = AutoProcessWorker("example clip.mp4", track=lambda _path, on_frame: "what it found")
    said = _everything_said_by(worker, "done")

    worker.run()

    assert said == [("done", ("what it found",))]


def test_an_automatic_run_that_fails_says_so(qt_app):
    def tracker(_path, on_frame):
        raise ValueError("an invented failure")

    worker = AutoProcessWorker("example clip.mp4", track=tracker)
    said = _everything_said_by(worker, "error")

    worker.run()

    assert said == [("error", ("an invented failure",))]
