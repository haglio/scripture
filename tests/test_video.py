"""The open video: its shape, the frame at an index, and refusing to open.

The window read four properties off a raw cv2 capture in two places, seeked and
read in a third, and released the old capture by hand in both of the first two --
so "a video that cannot be opened is not a video" was a check at one site and a
missing one at the other, which is the bug tests/test_gui_loading.py records.
"""
from __future__ import annotations

import numpy as np

from scripture import video
from scripture.video import VideoSource


class FakeCapture:
    """A cv2 capture over a fixed number of frames."""

    def __init__(self, opened=True, frames=3, fps=30.0, width=640, height=480):
        self._opened = opened
        self._frames = frames
        self._props = {video.cv2.CAP_PROP_FPS: fps,
                       video.cv2.CAP_PROP_FRAME_COUNT: frames,
                       video.cv2.CAP_PROP_FRAME_WIDTH: width,
                       video.cv2.CAP_PROP_FRAME_HEIGHT: height}
        self.sought = None
        self.released = False

    def isOpened(self):  # noqa: N802 - cv2 spells it this way
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

    def test_a_video_that_opens_reports_its_shape(self, monkeypatch):
        source = _source(monkeypatch, FakeCapture(frames=900, fps=25.0,
                                                  width=1920, height=1080))

        assert (source.fps, source.total_frames) == (25.0, 900)
        assert (source.width, source.height) == (1920, 1080)

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
