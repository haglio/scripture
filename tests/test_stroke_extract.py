import numpy as np

from scripture.stroke_extract import smooth_signal, extract_strokes, remove_drift


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
        # Smoothed should be closer to clean than noisy is
        noisy_error = np.mean((noisy - clean) ** 2)
        smooth_error = np.mean((smoothed - clean) ** 2)
        assert smooth_error < noisy_error


class TestExtractStrokes:

    def test_simple_sine_finds_peaks_and_valleys(self):
        t = np.linspace(0, 4 * np.pi, 400)
        positions = (np.sin(t) + 1) / 2  # 0 to 1 range
        timestamps_ms = np.linspace(0, 4000, 400)
        actions = extract_strokes(positions, timestamps_ms, min_stroke_height=0.3)
        # A 2-cycle sine should produce roughly 4 peaks + 4 valleys = ~4-5 extrema
        assert len(actions) >= 3
        # All positions should be in 0-100 range
        for a in actions:
            assert 0 <= a["pos"] <= 100
            assert a["at"] >= 0

    def test_flat_signal_no_strokes(self):
        positions = np.full(100, 0.5)
        timestamps_ms = np.linspace(0, 1000, 100)
        actions = extract_strokes(positions, timestamps_ms)
        assert actions == []


class TestRemoveDrift:

    def test_flat_signal_stays_centered(self):
        signal = np.full(500, 0.5)
        result = remove_drift(signal)
        np.testing.assert_allclose(result, 0.5, atol=0.01)

    def test_removes_linear_trend(self):
        drift = np.linspace(0, 1, 500)
        strokes = 0.1 * np.sin(np.linspace(0, 20 * np.pi, 500))
        signal = np.clip(drift + strokes + 0.5, 0, 1)
        result = remove_drift(signal, cutoff_period_frames=101)
        # Strokes should survive
        assert np.std(result) > 0.02
        # Drift slope should be removed: linear fit on result should be near-flat
        slope = np.polyfit(np.arange(len(result)), result, 1)[0]
        assert abs(slope) < 0.001

    def test_short_signal_no_crash(self):
        signal = np.array([0.0, 0.5, 1.0])
        result = remove_drift(signal)
        assert result.shape == signal.shape

    def test_even_cutoff_no_crash(self):
        signal = np.random.RandomState(42).rand(500)
        # Even and odd should both work without error
        result_even = remove_drift(signal, cutoff_period_frames=300)
        result_odd = remove_drift(signal, cutoff_period_frames=301)
        assert result_even.shape == signal.shape
        assert result_odd.shape == signal.shape


class TestExtractStrokesImproved:

    def test_drifting_signal_detects_strokes(self):
        """Globally-normalized drift drowns strokes when drift >> stroke amplitude.
        After normalization, strokes have prominence ~0.06, well below the
        default 0.15 threshold.  extract_strokes must still find them.
        """
        drift = np.linspace(0, 15, 1000)
        strokes = np.sin(np.linspace(0, 20 * np.pi, 1000))
        raw = drift + strokes
        # Simulate global normalization
        positions = (raw - raw.min()) / (raw.max() - raw.min())
        timestamps_ms = np.linspace(0, 33333, 1000)  # ~33s at 30fps
        actions = extract_strokes(positions, timestamps_ms, fps=30)
        # 10 cycles × 2 extrema = ~20 expected
        assert len(actions) >= 12

    def test_alternating_extrema(self):
        """Peaks and valleys must alternate — no two consecutive peaks."""
        t = np.linspace(0, 10 * np.pi, 500)
        positions = (np.sin(t) + 1) / 2
        timestamps_ms = np.linspace(0, 5000, 500)
        actions = extract_strokes(positions, timestamps_ms, fps=30)
        assert len(actions) >= 4
        for i in range(1, len(actions)):
            if actions[i - 1]["pos"] > 50:  # was a peak
                assert actions[i]["pos"] < 50, "Two consecutive peaks"
            else:  # was a valley
                assert actions[i]["pos"] > 50, "Two consecutive valleys"
