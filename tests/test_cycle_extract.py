from __future__ import annotations

import numpy as np
import pytest

from scripture.cycle_extract import (
    _adaptive_prominence,
    _enforce_alternating,
    extract_cycles,
    remove_drift,
    smooth_signal,
)


class TestSmoothSignal:

    def test_short_signal_unchanged(self):
        signal = np.array([0.0, 1.0, 0.0])
        result = smooth_signal(signal, window=15)
        np.testing.assert_array_equal(result, signal)

    def test_smoothing_reduces_noise(self):
        np.random.seed(42)
        clean = np.sin(np.linspace(0, 4 * np.pi, 200))
        noisy = clean + np.random.normal(0, 0.3, 200)
        smoothed = smooth_signal(noisy, window=15, polyorder=3)
        noisy_error = np.mean((noisy - clean) ** 2)
        smooth_error = np.mean((smoothed - clean) ** 2)
        assert smooth_error < noisy_error


class TestExtractCycles:

    def test_simple_sine_finds_peaks_and_valleys(self):
        t = np.linspace(0, 4 * np.pi, 400)
        positions = (np.sin(t) + 1) / 2
        timestamps_ms = np.linspace(0, 4000, 400)
        actions = extract_cycles(positions, timestamps_ms, max_prominence=0.3)
        assert len(actions) >= 3
        for a in actions:
            assert 0 <= a["pos"] <= 100
            assert a["at"] >= 0

    def test_flat_signal_no_cycles(self):
        positions = np.full(100, 0.5)
        timestamps_ms = np.linspace(0, 1000, 100)
        actions = extract_cycles(positions, timestamps_ms)
        assert actions == []

    def test_drifting_signal_detects_cycles(self):
        drift_that_leaves_the_cycles_under_the_default_prominence_once_normalized = np.linspace(0, 15, 1000)
        cycles = np.sin(np.linspace(0, 20 * np.pi, 1000))
        raw = drift_that_leaves_the_cycles_under_the_default_prominence_once_normalized + cycles
        positions = (raw - raw.min()) / (raw.max() - raw.min())
        timestamps_ms = np.linspace(0, 33333, 1000)
        actions = extract_cycles(positions, timestamps_ms, fps=30)
        assert len(actions) >= 12

    def test_alternating_extrema(self):
        t = np.linspace(0, 10 * np.pi, 500)
        positions = (np.sin(t) + 1) / 2
        timestamps_ms = np.linspace(0, 5000, 500)
        actions = extract_cycles(positions, timestamps_ms, fps=30)
        assert len(actions) >= 4
        for i in range(1, len(actions)):
            if actions[i - 1]["pos"] > 50:
                assert actions[i]["pos"] < 50, "Two consecutive peaks"
            else:
                assert actions[i]["pos"] > 50, "Two consecutive valleys"


class TestRemoveDrift:

    def test_flat_signal_stays_centered(self):
        signal = np.full(500, 0.5)
        result = remove_drift(signal)
        np.testing.assert_allclose(result, 0.5, atol=0.01)

    def test_removes_linear_trend(self):
        drift = np.linspace(0, 1, 500)
        cycles = 0.1 * np.sin(np.linspace(0, 20 * np.pi, 500))
        signal = np.clip(drift + cycles + 0.5, 0, 1)
        result = remove_drift(signal, cutoff_period_frames=101)
        assert np.std(result) > 0.02
        slope = np.polyfit(np.arange(len(result)), result, 1)[0]
        assert abs(slope) < 0.001

    def test_a_signal_shorter_than_the_window_is_only_recentered(self):
        signal = np.array([0.0, 0.5, 1.0])

        result = remove_drift(signal)

        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_an_even_window_is_the_odd_one_above_it_scipys_savitzky_golay_filter_needs_an_odd_one(self):
        signal = np.random.RandomState(42).rand(500)

        np.testing.assert_allclose(
            remove_drift(signal, cutoff_period_frames=300),
            remove_drift(signal, cutoff_period_frames=301))


class TestEnforceAlternating:

    def test_two_peaks_in_a_row_leave_only_the_higher(self):
        values = np.array([0.9, 0.7, 0.1])

        kept = _enforce_alternating(np.array([0, 1, 2]), values)

        assert list(kept) == [0, 2]

    def test_two_valleys_in_a_row_leave_only_the_deeper(self):
        values = np.array([0.2, 0.05, 0.9])

        kept = _enforce_alternating(np.array([0, 1, 2]), values)

        assert list(kept) == [1, 2]

    def test_extrema_that_already_alternate_are_all_kept(self):
        values = np.array([0.9, 0.1, 0.8])

        kept = _enforce_alternating(np.array([0, 1, 2]), values)

        assert list(kept) == [0, 1, 2]


class TestWhatCountsAsACycle:

    @staticmethod
    def _sine(period_ms, seconds, fps, *, level=0.5, swing=0.5):
        ms = np.arange(0, seconds * 1000, 1000 / fps)
        return level + swing * np.sin(2 * np.pi * ms / period_ms), ms

    def test_turnarounds_closer_than_200_ms_do_not_all_count(self):
        positions, ms = self._sine(100, 2, 1000)

        assert 0 < len(extract_cycles(positions, ms, fps=1000)) <= 24

    def test_the_prominence_needed_can_be_capped_below_what_the_signal_asks(self):
        ms = np.arange(0, 20000, 1000 / 30)
        wave = (0.5 + 0.4 * np.sin(2 * np.pi * ms / 8000)
                + 0.04 * np.sin(2 * np.pi * ms / 800))

        capped = extract_cycles(wave, ms, max_prominence=0.02, fps=30)

        assert len(capped) > len(extract_cycles(wave, ms, fps=30))

    def test_a_flat_signal_needs_only_the_floor(self):
        assert _adaptive_prominence(np.full(300, 0.5), fps=30) == 0.05

    def test_a_full_swing_needs_about_a_seventh_of_the_range(self):
        positions, _ms = self._sine(1000, 20, 30)

        assert _adaptive_prominence(positions, fps=30) == pytest.approx(0.143, abs=0.005)

    def test_turnarounds_all_on_one_side_of_the_middle_collapse_to_the_most_extreme(self):
        positions, ms = self._sine(2000, 30, 30, level=0.2, swing=0.1)

        assert len(extract_cycles(positions, ms, fps=30)) == 1

    def test_a_position_past_the_top_is_written_as_100(self):
        positions = np.zeros(60)
        positions[28:33] = 1.0
        ms = np.arange(60) * 1000 / 30

        assert max(a["pos"] for a in extract_cycles(positions, ms, fps=30)) == 100
