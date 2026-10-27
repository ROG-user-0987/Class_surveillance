"""Extra per-frame cues from keypoints (all orange tier), and the grey
"cannot assess" state. Measurements only; risk.py times them with
EventTrackers and decides when to raise a flag for HUMAN review.

    neighbour_peek   The head is turned (nose off the midpoint of the ears,
                     or of the eyes if the ears are hidden) and a horizontal
                     ray from the nose in that direction reaches another
                     tracked person's box within cues.peek_max_distance
                     shoulder widths. Returns that neighbour's track_id.
    object_passing   A wrist of one student is within cues.pass_distance
                     shoulder widths of a wrist of another student.
    hands_hidden     The head is pitched down (nose close to / below the
                     shoulder line, or lower than the student's own baseline)
                     AND both wrists are below cues.kp_conf.
    low_visibility   Fewer than cues.min_visible_keypoints keypoints reach
                     cues.kp_conf: posture can't be judged, so risk.py marks
                     the student grey ("cannot assess") and pauses posture
                     alerts (not phone alerts) while it lasts.

Each cue has its own switch under features: in config.yaml. Every function
returns None when the cue can't be measured in this frame (e.g. keypoints
hidden), which counts as a gap, not as "no".

Direction words ("left"/"right") are IMAGE directions.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import TYPE_CHECKING, Any

from src.pose_rules import head_pitch_ratio
from src.utils import (
    LEFT_EAR,
    LEFT_EYE,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_EAR,
    RIGHT_EYE,
    RIGHT_SHOULDER,
    WRIST_KEYPOINTS,
)

if TYPE_CHECKING:  # type hints only; importing tracking here for real would be circular
    from src.tracking import FrameRecord


def _visible(record: FrameRecord, index: int, kp_conf: float) -> bool:
    return bool(record.keypoints[index, 2] >= kp_conf)


def shoulder_width(record: FrameRecord, cfg: dict[str, Any]) -> float:
    """Shoulder width in pixels; falls back to a fraction of the box width."""
    cues = cfg["cues"]
    if _visible(record, LEFT_SHOULDER, cues["kp_conf"]) and _visible(record, RIGHT_SHOULDER, cues["kp_conf"]):
        width = abs(float(record.keypoints[LEFT_SHOULDER, 0] - record.keypoints[RIGHT_SHOULDER, 0]))
        if width >= 1:
            return width
    box_width = float(record.person_box[2] - record.person_box[0])
    return max(box_width * cues["shoulder_fallback_box_fraction"], 1.0)


def head_direction(record: FrameRecord, cfg: dict[str, Any]) -> tuple[int, float] | None:
    """Horizontal head direction as (+1 = image right / -1 = image left,
    strength). Strength = nose offset from the ear midpoint (eye midpoint if
    an ear is hidden) in half ear-distances: ~0 facing the camera, ~1 when
    the nose is above one ear. None if it can't be measured."""
    kp_conf = cfg["cues"]["kp_conf"]
    if not _visible(record, NOSE, kp_conf):
        return None
    k = record.keypoints
    for left, right in ((LEFT_EAR, RIGHT_EAR), (LEFT_EYE, RIGHT_EYE)):
        if _visible(record, left, kp_conf) and _visible(record, right, kp_conf):
            half_distance = abs(float(k[left, 0] - k[right, 0])) / 2
            if half_distance < 1:
                continue
            offset = (float(k[NOSE, 0]) - float(k[left, 0] + k[right, 0]) / 2) / half_distance
            return (1 if offset > 0 else -1), abs(offset)
    return None


def peek_target(
    record: FrameRecord, others: list[FrameRecord], cfg: dict[str, Any]
) -> tuple[float | None, int | None]:
    """(signal, neighbour_id) for neighbour_peek in this frame.

    signal is 1.0 when the head is turned toward a neighbour's box within
    reach, 0.0 when it isn't, None when head direction can't be measured.
    """
    cues = cfg["cues"]
    direction = head_direction(record, cfg)
    if direction is None:
        return None, None
    side, strength = direction
    if strength < cues["peek_min_turn"]:
        return 0.0, None

    nose_x, nose_y = float(record.keypoints[NOSE, 0]), float(record.keypoints[NOSE, 1])
    reach = cues["peek_max_distance"] * shoulder_width(record, cfg)
    best_id, best_gap = None, math.inf
    for other in others:
        if other.track_id == record.track_id:
            continue
        x1, y1, x2, y2 = (float(v) for v in other.person_box)
        if not y1 <= nose_y <= y2:
            continue  # a horizontal ray from the nose misses this box
        if ((x1 + x2) / 2 - nose_x) * side <= 0:
            continue  # the box is on the other side
        gap = max(0.0, x1 - nose_x) if side > 0 else max(0.0, nose_x - x2)
        if gap <= reach and gap < best_gap:
            best_id, best_gap = other.track_id, gap
    return (1.0 if best_id is not None else 0.0), best_id


def visible_wrists(record: FrameRecord, kp_conf: float) -> list[tuple[float, float]]:
    """(x, y) of each wrist at or above kp_conf."""
    return [(float(record.keypoints[k, 0]), float(record.keypoints[k, 1]))
            for k in WRIST_KEYPOINTS if _visible(record, k, kp_conf)]


def passing_pairs(records: list[FrameRecord], cfg: dict[str, Any]) -> dict[tuple[int, int], float]:
    """object_passing signal per pair of students (smaller id first): 1.0 if
    a wrist of one is within cues.pass_distance shoulder widths (average of
    the two) of a wrist of the other, else 0.0. Pairs where either student
    has no visible wrist are left out (= not measurable this frame)."""
    cues = cfg["cues"]
    signals: dict[tuple[int, int], float] = {}
    for a, b in combinations(records, 2):
        wrists_a, wrists_b = visible_wrists(a, cues["kp_conf"]), visible_wrists(b, cues["kp_conf"])
        if not wrists_a or not wrists_b:
            continue
        scale = (shoulder_width(a, cfg) + shoulder_width(b, cfg)) / 2
        closest = min(math.hypot(xa - xb, ya - yb) for xa, ya in wrists_a for xb, yb in wrists_b)
        key = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
        signals[key] = 1.0 if closest <= cues["pass_distance"] * scale else 0.0
    return signals


def hands_hidden_signal(
    record: FrameRecord, cfg: dict[str, Any], pitch_baseline: float | None = None
) -> tuple[float | None, dict[str, Any]]:
    """(signal, details) for hands_hidden: 1.0 when the head is down AND
    both wrists are hidden, 0.0 otherwise, None if head pitch can't be measured.

    "Head down" uses the student's own pitch baseline when one exists
    (cues.head_down_delta lower than normal), else the absolute
    cues.head_down_pitch.
    """
    cues, pose = cfg["cues"], cfg["pose"]
    pitch = head_pitch_ratio(record, min_conf=pose["keypoint_conf"],
                             min_shoulder_box_ratio=pose["min_shoulder_box_ratio"])
    if pitch is None:
        return None, {}
    if pitch_baseline is not None:
        head_down = pitch - pitch_baseline >= cues["head_down_delta"]
    else:
        head_down = pitch >= cues["head_down_pitch"]
    hidden = not visible_wrists(record, cues["kp_conf"])
    return (1.0 if head_down and hidden else 0.0), {"pitch": pitch, "baseline": pitch_baseline}


def visible_keypoint_count(record: FrameRecord, kp_conf: float) -> int:
    """How many of the 17 keypoints reach kp_conf."""
    return int((record.keypoints[:, 2] >= kp_conf).sum())


def low_visibility_signal(record: FrameRecord, cfg: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """(signal, details) for the grey state: 1.0 when fewer than
    cues.min_visible_keypoints keypoints are visible, else 0.0."""
    cues = cfg["cues"]
    count = visible_keypoint_count(record, cues["kp_conf"])
    return (1.0 if count < cues["min_visible_keypoints"] else 0.0), {"visible_keypoints": count}
