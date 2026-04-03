from scripture.project import save_project, load_project


class TestProjectPersistence:

    def test_round_trip(self, tmp_path):
        path = tmp_path / "test.scripture"
        state = {
            "video_path": "C:/videos/test.mp4",
            "splits": [100, 500, 1200],
            "axes": {
                "0": {"tip": [10, 20], "base": [30, 40], "frame": 50},
                "2": {"tip": [100, 200], "base": [300, 400], "frame": 1000},
            },
            "actions": {
                "0": [{"at": 1000, "pos": 50}, {"at": 2000, "pos": 100}],
            },
        }
        save_project(str(path), state)
        loaded = load_project(str(path))
        assert loaded == state

    def test_empty_project(self, tmp_path):
        path = tmp_path / "empty.scripture"
        state = {
            "video_path": "C:/videos/test.mp4",
            "splits": [],
            "axes": {},
            "actions": {},
        }
        save_project(str(path), state)
        loaded = load_project(str(path))
        assert loaded == state

    def test_backwards_compat_no_actions_key(self, tmp_path):
        """Old project files without 'actions' key should still load."""
        path = tmp_path / "old.scripture"
        state = {
            "video_path": "C:/videos/test.mp4",
            "splits": [],
            "axes": {},
        }
        save_project(str(path), state)
        loaded = load_project(str(path))
        # actions key absent is fine — GUI uses .get("actions", {})
        assert "video_path" in loaded
