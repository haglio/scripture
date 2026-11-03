from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from scripture.motion_tracker import AxisDefinition, TrackingResult

Label = dict[str, object]


@dataclass
class SceneAnnotation:

    axis: AxisDefinition | None = None
    actions: list[dict] | None = None
    tracking: TrackingResult | None = None
    labels: dict[int, Label] = field(default_factory=dict)


class SceneAnnotations:

    def __init__(self) -> None:
        self._by_scene: dict[int, SceneAnnotation] = {}

    def _record(self, scene: int) -> SceneAnnotation:
        return self._by_scene.setdefault(scene, SceneAnnotation())

    def set_axis(self, scene: int, axis: AxisDefinition) -> None:
        record = self._record(scene)
        record.axis = axis
        record.actions = None
        record.tracking = None

    def axis_of(self, scene: int) -> AxisDefinition | None:
        record = self._by_scene.get(scene)
        return record.axis if record else None

    def drop_axis(self, scene: int) -> AxisDefinition | None:
        record = self._by_scene.get(scene)
        if record is None:
            return None
        axis, record.axis = record.axis, None
        record.actions = None
        record.tracking = None
        self._prune(scene)
        return axis

    def set_result(self, scene: int, actions: list[dict],
                   tracking: TrackingResult | None) -> None:
        record = self._record(scene)
        record.actions = actions
        record.tracking = tracking

    def discard_result(self, scene: int) -> None:
        record = self._by_scene.get(scene)
        if record is None:
            return
        record.actions = None
        record.tracking = None
        self._prune(scene)

    def set_label(self, scene: int, frame: int, label: Label) -> None:
        self._record(scene).labels[frame] = label

    def label_at(self, scene: int, frame: int) -> Label | None:
        return self.labels_of(scene).get(frame)

    def labels_of(self, scene: int) -> dict[int, Label]:
        record = self._by_scene.get(scene)
        return record.labels if record else {}

    def drop_label(self, scene: int, frame: int) -> None:
        record = self._by_scene.get(scene)
        if record is None:
            return
        record.labels.pop(frame, None)
        self._prune(scene)

    def drop_labels(self, scene: int) -> None:
        record = self._by_scene.get(scene)
        if record is None:
            return
        record.labels = {}
        self._prune(scene)

    def reindex(self, scene_of_frame: Callable[[int], int]) -> None:
        moved: dict[int, SceneAnnotation] = {}
        for record in self._by_scene.values():
            if record.axis is None:
                continue
            landing = moved.setdefault(scene_of_frame(record.axis.frame), SceneAnnotation())
            landing.axis = record.axis
            landing.actions = record.actions
            landing.tracking = record.tracking
        for record in self._by_scene.values():
            for frame, label in record.labels.items():
                moved.setdefault(scene_of_frame(frame), SceneAnnotation()).labels[frame] = label
        self._by_scene = moved

    def replace_actions(self, actions: dict[int, list[dict]]) -> None:
        for record in self._by_scene.values():
            record.actions = None
        for scene, entries in actions.items():
            self._record(scene).actions = entries
        for scene in list(self._by_scene):
            self._prune(scene)

    def clear_tracking(self) -> None:
        for record in self._by_scene.values():
            record.tracking = None
        for scene in list(self._by_scene):
            self._prune(scene)

    def load(self, *, axes: dict[int, AxisDefinition], actions: dict[int, list[dict]],
             tracking: dict[int, TrackingResult],
             labels: dict[int, dict[int, Label]]) -> None:
        for scene, axis in axes.items():
            self._record(scene).axis = axis
        for scene, entries in actions.items():
            self._record(scene).actions = entries
        for scene, result in tracking.items():
            self._record(scene).tracking = result
        for scene, frames in labels.items():
            self._record(scene).labels = frames

    def clear(self) -> None:
        self._by_scene.clear()

    def _prune(self, scene: int) -> None:
        record = self._by_scene[scene]
        if record == SceneAnnotation():
            del self._by_scene[scene]

    @property
    def axes(self) -> dict[int, AxisDefinition]:
        return {scene: r.axis for scene, r in self._by_scene.items() if r.axis is not None}

    @property
    def actions(self) -> dict[int, list[dict]]:
        return {scene: r.actions for scene, r in self._by_scene.items()
                if r.actions is not None}

    @property
    def tracking(self) -> dict[int, TrackingResult]:
        return {scene: r.tracking for scene, r in self._by_scene.items()
                if r.tracking is not None}

    @property
    def labels(self) -> dict[int, dict[int, Label]]:
        return {scene: r.labels for scene, r in self._by_scene.items() if r.labels}
