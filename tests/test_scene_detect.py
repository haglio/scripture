from scripture.scene_detect import Scene, filter_scenes


class TestScene:

    def test_scene_dataclass_fields(self):
        scene = Scene(
            start_frame=0, end_frame=100,
            is_content=True, representative_frame=50,
        )
        assert scene.start_frame == 0
        assert scene.end_frame == 100
        assert scene.is_content is True
        assert scene.representative_frame == 50


class TestFilterScenes:

    def test_removes_non_content_scenes(self):
        scenes = [
            Scene(0, 90, is_content=False, representative_frame=45),
            Scene(90, 5000, is_content=True, representative_frame=2500),
            Scene(5000, 5010, is_content=False, representative_frame=5005),
        ]
        result = filter_scenes(scenes)
        assert len(result) == 1
        assert result[0].start_frame == 90

    def test_removes_zero_length_scenes(self):
        scenes = [
            Scene(0, 0, is_content=True, representative_frame=0),
            Scene(0, 100, is_content=True, representative_frame=50),
        ]
        result = filter_scenes(scenes)
        assert len(result) == 1
        assert result[0].end_frame == 100

    def test_removes_very_short_scenes(self):
        scenes = [
            Scene(0, 5, is_content=True, representative_frame=2),
            Scene(5, 5000, is_content=True, representative_frame=2500),
        ]
        result = filter_scenes(scenes, min_frames=10)
        assert len(result) == 1
        assert result[0].start_frame == 5

    def test_keeps_all_valid_content(self):
        scenes = [
            Scene(0, 3000, is_content=True, representative_frame=1500),
            Scene(3000, 6000, is_content=True, representative_frame=4500),
        ]
        result = filter_scenes(scenes)
        assert len(result) == 2

    def test_empty_input(self):
        assert filter_scenes([]) == []
