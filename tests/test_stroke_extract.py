import numpy as np

from scripture.stroke_extract import smooth_signal, extract_strokes


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
