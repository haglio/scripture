from scripture.scene import Scene, actions_by_scene


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


class TestActionsByScene:
    def test_actions_bucketed_by_frame_range(self):
        scenes = [Scene(0, 300), Scene(300, 900)]
        fps = 30.0
        actions = [
            {"at": 1000, "pos": 10},   # frame 30 -> scene 0
            {"at": 9000, "pos": 90},   # frame 270 -> scene 0
            {"at": 11000, "pos": 40},  # frame 330 -> scene 1
        ]
        buckets = actions_by_scene(actions, scenes, fps)
        assert [a["at"] for a in buckets[0]] == [1000, 9000]
        assert [a["at"] for a in buckets[1]] == [11000]

    def test_scenes_without_actions_are_absent(self):
        scenes = [Scene(0, 300), Scene(300, 900)]
        buckets = actions_by_scene([{"at": 100, "pos": 5}], scenes, 30.0)
        assert 1 not in buckets
