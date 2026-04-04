import numpy as np
import pytest

from scripture.motion_tracker import AxisDefinition, TrackingResult


class TestTrackingResultCompat:

    def test_default_none_fields(self):
        result = TrackingResult(timestamps_ms=np.array([0.0]), positions=np.array([0.5]))
        assert result.tip_coords is None
        assert result.base_coords is None

    def test_with_coords(self):
        tips = np.array([[100, 50], [101, 51]], dtype=np.float64)
        bases = np.array([[100, 350], [101, 351]], dtype=np.float64)
        result = TrackingResult(
            timestamps_ms=np.array([0.0, 33.3]),
            positions=np.array([0.5, 0.6]),
            tip_coords=tips,
            base_coords=bases,
        )
        assert result.tip_coords.shape == (2, 2)
        assert result.base_coords.shape == (2, 2)
