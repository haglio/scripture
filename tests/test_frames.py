from __future__ import annotations

import numpy as np

from scripture.frames import frame_of_ms, half_frame_ms, ms_of_frame


def test_a_frame_sits_at_its_index_over_the_frame_rate():
    assert ms_of_frame(90, 30.0) == 3000.0


def test_every_frame_of_an_array_gets_its_own_time():
    np.testing.assert_allclose(ms_of_frame(np.array([0, 25, 50]), 25.0), [0.0, 1000.0, 2000.0])


def test_a_time_falls_on_the_nearest_frame():
    assert frame_of_ms(3010, 30.0) == 90
    assert frame_of_ms(3020, 30.0) == 91


def test_every_frame_comes_back_from_its_own_time():
    for fps in (23.976, 25.0, 29.97, 30.0, 60.0):
        assert [frame_of_ms(ms_of_frame(f, fps), fps) for f in range(500)] == list(range(500))


def test_half_a_frame_of_time():
    assert half_frame_ms(25.0) == 20.0
