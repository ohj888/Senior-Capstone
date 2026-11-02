"""Choose existing detector evidence once per object/frame, without GT or box edits."""
import math


POLICY = "unique_frame_nested_box_v1"


def _box(view):
    values = tuple(float(value) for value in view["bbox"])
    if (len(values) != 4 or not all(math.isfinite(value) for value in values)
            or values[2] <= values[0] or values[3] <= values[1]):
        raise ValueError("Observation has an invalid detector box")
    return values


def _area(box):
    return (box[2]-box[0]) * (box[3]-box[1])


def select_observations(views):
    """Keep one actual observation for each object/image.

    When an object has a broad and a substantially tighter nested detection in
    the same frame, prefer the tighter candidate. A 25% minimum area ratio
    prevents a tiny contained fragment from replacing a full-object detection.
    This never compares/merges different map objects or clips boxes by hand.
    """
    grouped = {}
    for view in views:
        grouped.setdefault((str(view["rgb"]), int(view["frame_id"])), []).append(view)
    selected = []
    for candidates in grouped.values():
        boxes = [_box(view) for view in candidates]
        if len(candidates) == 1 and candidates[0].get("observation_selection") == POLICY:
            selected.append(dict(candidates[0]))
            continue
        viable = []
        for index, outer in enumerate(boxes):
            broad = False
            for inner in boxes:
                ratio = _area(inner) / _area(outer)
                overlap = (max(0, min(outer[2], inner[2])-max(outer[0], inner[0]))
                           * max(0, min(outer[3], inner[3])-max(outer[1], inner[1])))
                if .25 <= ratio <= .90 and overlap / _area(inner) >= .98:
                    broad = True
                    break
            if not broad:
                viable.append(index)
        chosen = min(viable, key=lambda index: (
            -float(candidates[index].get("confidence") or 0), _area(boxes[index]), boxes[index]))
        view = dict(candidates[chosen])
        view.update(observation_selection=POLICY, same_frame_alternatives=len(candidates),
                    bbox_selection="tighter_nested_detection" if len(viable) < len(candidates) else "confidence",
                    alternate_bboxes=[list(box) for index, box in enumerate(boxes) if index != chosen])
        selected.append(view)
    return selected
