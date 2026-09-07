"""What the overlay reads off an action list for one frame.

`_interpolate_actions` had no test and three lines that cannot run: its only
caller reaches it under `elif actions:`, so the empty-list guard could never
fire. Deleting an unreachable guard is only safe once the reachable branches
are pinned, which is what these do.
"""

from __future__ import annotations

from scripture.gui import App

_ACTIONS = [{"at": 0, "pos": 20}, {"at": 1000, "pos": 80}, {"at": 2000, "pos": 40}]
_HALF_FRAME_MS = 16.0


def _at(frame_ms):
    return App._interpolate_actions(_ACTIONS, frame_ms, _HALF_FRAME_MS)


def test_a_frame_on_an_action_takes_its_position_and_is_marked_one():
    assert _at(1010) == (80, True)


def test_a_frame_before_the_first_action_holds_its_position():
    assert _at(-500) == (20, False)


def test_a_frame_after_the_last_action_holds_its_position():
    assert _at(5000) == (40, False)


def test_a_frame_between_two_actions_lands_on_the_line_between_them():
    assert _at(500) == (50, False)
    assert _at(1500) == (60, False)
