from scripture.scene import Scene


class TestScene:

    def test_dataclass_fields(self):
        scene = Scene(start_frame=100, end_frame=500)
        assert scene.start_frame == 100
        assert scene.end_frame == 500


class TestScenesFromSplits:
    """Test the split-points-to-scenes derivation logic."""

    def test_no_splits_gives_one_scene(self):
        scenes = _scenes_from_splits([], 1000)
        assert len(scenes) == 1
        assert scenes[0] == Scene(0, 1000)

    def test_one_split(self):
        scenes = _scenes_from_splits([300], 1000)
        assert scenes == [Scene(0, 300), Scene(300, 1000)]

    def test_multiple_splits(self):
        scenes = _scenes_from_splits([100, 400, 700], 1000)
        assert scenes == [
            Scene(0, 100),
            Scene(100, 400),
            Scene(400, 700),
            Scene(700, 1000),
        ]

    def test_splits_are_sorted(self):
        scenes = _scenes_from_splits([700, 100, 400], 1000)
        assert scenes == [
            Scene(0, 100),
            Scene(100, 400),
            Scene(400, 700),
            Scene(700, 1000),
        ]


def _scenes_from_splits(splits: list[int], total_frames: int) -> list[Scene]:
    """Derive scenes from split points — extracted here so GUI can reuse."""
    from scripture.scene import scenes_from_splits
    return scenes_from_splits(splits, total_frames)
