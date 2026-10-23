"""Posture measurements from FrameRecord keypoints: head turn and lean.

These are measurements, not judgements. risk.py decides how long a posture
must last before a human is asked to look.

    head turn = how far the nose is to the side of the shoulders' midpoint,
                in shoulder widths. ~0 facing the same way as the body,
                ~0.25 full profile. Needs nose + both shoulders visible, and
                is skipped when the person is side-on to the camera.
    lean      = angle of the hips-to-shoulders line away from vertical, in
                degrees. Needs both shoulders AND both hips visible, so it is
                often unavailable when desks hide the hips.

Known limits: a camera at the side of the hall makes everyone look slightly
"turned". Run this module on a sample clip to see typical values before
setting pose.head_turn_threshold.

Run on its own (prints per-person posture statistics, for tuning):
    python -m src.pose_rules --video data/sample.mp4
"""

from __future__ import annotations

import argparse
import math
from typing import TYPE_CHECKING, Any

from src.utils import (
    LEFT_HIP,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    load_config,
    resolve_input_video,
)

if TYPE_CHECKING:  # type hints only; importing tracking here for real would be circular
    from src.tracking import FrameRecord


def head_turn_ratio(
    record: FrameRecord, *, min_conf: float, min_shoulder_box_ratio: float
) -> float | None:
    """Sideways nose offset in shoulder widths, or None if it can't be measured."""
    k = record.keypoints
    nose, left, right = k[NOSE], k[LEFT_SHOULDER], k[RIGHT_SHOULDER]
    if min(nose[2], left[2], right[2]) < min_conf:
        return None

    shoulder_width = abs(left[0] - right[0])
    box_width = record.person_box[2] - record.person_box[0]
    if box_width <= 0 or shoulder_width < box_width * min_shoulder_box_ratio:
        return None  # side-on to the camera: shoulders overlap, ratio is meaningless

    shoulder_mid_x = (left[0] + right[0]) / 2
    return float(abs(nose[0] - shoulder_mid_x) / shoulder_width)


def lean_angle_deg(record: FrameRecord, *, min_conf: float) -> float | None:
    """Torso angle away from vertical in degrees, or None if hips/shoulders hidden."""
    k = record.keypoints
    if min(k[LEFT_SHOULDER, 2], k[RIGHT_SHOULDER, 2], k[LEFT_HIP, 2], k[RIGHT_HIP, 2]) < min_conf:
        return None

    shoulder_x, shoulder_y = (k[LEFT_SHOULDER, :2] + k[RIGHT_SHOULDER, :2]) / 2
    hip_x, hip_y = (k[LEFT_HIP, :2] + k[RIGHT_HIP, :2]) / 2
    up = hip_y - shoulder_y  # image y grows downwards, so upright = positive
    if up <= 0:
        return None  # keypoints don't make sense (shoulders below hips)
    return math.degrees(math.atan2(abs(shoulder_x - hip_x), up))


def posture_flags(record: FrameRecord, cfg: dict[str, Any]) -> dict[str, bool]:
    """Which postures are past their threshold in this one frame."""
    pose = cfg["pose"]
    turn = head_turn_ratio(record, min_conf=pose["keypoint_conf"],
                           min_shoulder_box_ratio=pose["min_shoulder_box_ratio"])
    lean = lean_angle_deg(record, min_conf=pose["keypoint_conf"])
    return {
        "head_turn": turn is not None and turn >= pose["head_turn_threshold"],
        "lean": lean is not None and lean >= pose["lean_threshold_deg"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-person head-turn / lean statistics (for tuning).")
    parser.add_argument("--video", help="Input video (default: paths.input_video in config.yaml).")
    parser.add_argument("--config", help="Config file (default: config.yaml at the repo root).")
    args = parser.parse_args()

    from src.tracking import run_on_video  # imported here to avoid a circular import

    cfg = load_config(args.config)
    video = resolve_input_video(args.video, cfg)
    records, _ = run_on_video(video, cfg)
    pose = cfg["pose"]

    # Collect measurements per track_id.
    per_person: dict[int, dict[str, list[float]]] = {}
    for record in records:
        stats = per_person.setdefault(record.track_id, {"frames": [], "turn": [], "lean": []})
        stats["frames"].append(record.timestamp)
        turn = head_turn_ratio(record, min_conf=pose["keypoint_conf"],
                               min_shoulder_box_ratio=pose["min_shoulder_box_ratio"])
        lean = lean_angle_deg(record, min_conf=pose["keypoint_conf"])
        if turn is not None:
            stats["turn"].append(turn)
        if lean is not None:
            stats["lean"].append(lean)

    print(f"\nhead_turn_threshold = {pose['head_turn_threshold']}, "
          f"lean_threshold_deg = {pose['lean_threshold_deg']}")
    print(f"{'ID':>4} {'frames':>6} | {'turn measured':>13} {'over thr':>8} {'median':>6} {'max':>5} | "
          f"{'lean measured':>13} {'over thr':>8} {'max':>5}")
    for track_id in sorted(per_person):
        s = per_person[track_id]
        n = len(s["frames"])
        turns, leans = sorted(s["turn"]), s["lean"]
        turn_over = sum(t >= pose["head_turn_threshold"] for t in turns)
        lean_over = sum(a >= pose["lean_threshold_deg"] for a in leans)
        median = f"{turns[len(turns) // 2]:.2f}" if turns else "-"
        print(f"{track_id:>4} {n:>6} | {len(turns) / n:>13.0%} {turn_over:>8} {median:>6} "
              f"{max(turns, default=0):>5.2f} | {len(leans) / n:>13.0%} {lean_over:>8} "
              f"{max(leans, default=0):>5.1f}")


if __name__ == "__main__":
    main()
