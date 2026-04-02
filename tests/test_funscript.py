from scripture.funscript import build_funscript


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
