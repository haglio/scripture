from scripture.scene_detect import Scene


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

    def test_non_content_scene(self):
        scene = Scene(
            start_frame=0, end_frame=30,
            is_content=False, representative_frame=15,
        )
        assert scene.is_content is False
