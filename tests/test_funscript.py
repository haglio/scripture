import json

from scripture.funscript import build_funscript, save_funscript


class TestBuildFunscript:

    def test_empty_actions(self):
        result = build_funscript([], 100)
        assert result["actions"] == []
        assert result["version"] == "1.0"
        assert result["range"] == 100
        assert result["metadata"]["duration"] == 100

    def test_actions_sorted_by_timestamp(self):
        actions = [
            {"at": 3000, "pos": 50},
            {"at": 1000, "pos": 100},
            {"at": 2000, "pos": 0},
        ]
        result = build_funscript(actions, 10)
        timestamps = [a["at"] for a in result["actions"]]
        assert timestamps == [1000, 2000, 3000]


class TestSaveFunscript:

    def test_the_file_reads_back_as_the_funscript_that_was_written(self, tmp_path):
        """Replacing this function's body with `return None` left the suite at
        its exact count, because nothing here had ever opened what it wrote."""
        funscript = build_funscript([{"at": 0, "pos": 50}], 12)
        path = tmp_path / "example clip.funscript"

        save_funscript(funscript, str(path))

        assert json.loads(path.read_text(encoding="utf-8")) == funscript
