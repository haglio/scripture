from __future__ import annotations

import numpy as np

from scripture import video
from scripture.video import NO_VIDEO, VideoSource


class FakeCapture:

    def __init__(self, opened=True, frames=3, fps=30.0):
        self._opened = opened
        self._frames = frames
        self._props = {video.cv2.CAP_PROP_FPS: fps,
                       video.cv2.CAP_PROP_FRAME_COUNT: frames}
        self.sought = None
        self.released = False

    def isOpened(self):
        return self._opened

    def get(self, prop):
        return self._props.get(prop, 0)

    def set(self, _prop, value):
        self.sought = value
        return True

    def read(self):
        if self.sought is None or self.sought >= self._frames:
            return False, None
        return True, np.zeros((2, 2, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def _source(monkeypatch, capture):
    monkeypatch.setattr(video.cv2, "VideoCapture", lambda _path: capture)
    return VideoSource.opened("example clip.mp4")


class TestOpening:

    def test_a_video_that_opens_reports_its_length_and_frame_rate(self, monkeypatch):
        source = _source(monkeypatch, FakeCapture(frames=900, fps=25.0))

        assert (source.fps, source.total_frames) == (25.0, 900)

    def test_a_video_that_will_not_open_is_no_source_and_is_let_go(self, monkeypatch):
        refusing = FakeCapture(opened=False)

        assert _source(monkeypatch, refusing) is None
        assert refusing.released

    def test_a_video_reporting_no_frame_rate_is_read_at_thirty(self, monkeypatch):
        assert _source(monkeypatch, FakeCapture(fps=0)).fps == 30.0


class TestReadingAFrame:

    def test_the_frame_asked_for_is_the_one_sought(self, monkeypatch):
        source = _source(monkeypatch, capture := FakeCapture(frames=10))

        assert source.frame_at(4) is not None
        assert capture.sought == 4

    def test_a_frame_the_video_cannot_give_reads_as_nothing(self, monkeypatch):
        assert _source(monkeypatch, FakeCapture(frames=2)).frame_at(5) is None


class TestLettingGo:

    def test_releasing_hands_the_file_back(self, monkeypatch):
        source = _source(monkeypatch, capture := FakeCapture())

        source.release()

        assert capture.released


def test_with_no_video_there_are_no_frames_to_show_and_nothing_to_let_go():
    assert NO_VIDEO.total_frames == 0
    assert NO_VIDEO.frame_at(0) is None
    NO_VIDEO.release()
