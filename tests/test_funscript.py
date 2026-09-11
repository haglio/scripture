from __future__ import annotations

import json

from app_support.funscript import write as write_funscript

from scripture.funscript import CREATOR, build_funscript


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

    def test_the_script_says_which_app_authored_it(self):
        """The credit is all this app adds to the family's document, and it is
        what an outside player shows for a script it did not make."""
        assert build_funscript([], 100)["metadata"]["creator"] == CREATOR


class TestExport:

    def test_the_file_reads_back_as_the_funscript_that_was_written(self, tmp_path):
        """Replacing the writer's body with `return None` left the suite at its
        exact count, because nothing here had ever opened what it wrote."""
        funscript = build_funscript([{"at": 0, "pos": 50}], 12)
        path = tmp_path / "example clip.funscript"

        write_funscript(path, funscript)

        assert json.loads(path.read_text(encoding="utf-8")) == funscript
