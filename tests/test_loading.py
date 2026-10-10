"""What Scripture shows while it opens: the family's loading screen, walking
the steps the launch takes, the slow one first."""

from __future__ import annotations

from unittest.mock import patch

from scripture import main


class _Loading:
    def __init__(self):
        self.said: list[str] = []

    def say(self, step: str) -> None:
        self.said.append(step)


def test_the_launch_says_each_step_before_it_takes_it():
    loading = _Loading()
    with patch.object(main, "_report_tracker_environment") as checked, \
         patch("scripture.gui.App") as window_class:
        window = main.open_scripture(loading, preview=None, dress=lambda: None)

    assert loading.said == [step for step, _weight in main.STEPS]
    checked.assert_called_once_with()
    window_class.assert_called_once_with(preview=None)
    assert window is window_class.return_value


def test_the_tracker_check_is_the_long_step_so_the_bar_spends_most_of_its_time_there():
    (first, weight), *rest = main.STEPS

    assert first == "Checking the tracker..."
    assert weight > sum(w for _step, w in rest)
