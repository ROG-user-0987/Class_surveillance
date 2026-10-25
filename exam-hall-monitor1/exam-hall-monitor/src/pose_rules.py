"""Posture measurements from FrameRecord keypoints: head turn, lean, pitch.

These are measurements, not judgements. risk.py decides how long a posture
must last before a human is asked to look.

    head turn  = how far the nose is to the side of the shoulders' midpoint,
                 in shoulder widths. ~0 facing the same way as the body,
                 ~0.25 full profile. Needs nose + both shoulders visible, and
                 is skipped when the person is side-on to the camera.
    lean       = angle of the hips-to-shoulders line away from vertical, in
                 degrees. Needs both shoulders AND both hips visible, so it is
                 often unavailable when desks hide the hips.
    head pitch = nose height relative to the shoulder line, in shoulder
                 widths (negative = nose above shoulders; rises when the head
                 goes down). Collected for baselines; used by later cues.

The *_signed functions keep the direction (negative = towards image left /
up); head_turn_ratio and lean_angle_deg are their absolute values.

With features.baseline_calibration on, PostureBaseline learns each
student's normal values, and readings_from_signals() measures posture as
deviation from that normal (see PostureBaseline for limitations).

Known limits: a camera at the side of the hall makes everyone look slightly
"turned" (the baseline feature is meant to absorb this). Run this module on a
sample clip to see typical values before setting pose.head_turn_threshold.

Run on its own (prints per-person posture statistics, for tuning):
    python -m src.pose_rules --video data/sample.mp4
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
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


def _shoulder_frame(
    record: FrameRecord, *, min_conf: float, min_shoulder_box_ratio: float
) -> tuple[float, float, float] | None:
    """(shoulder_mid_x, shoulder_mid_y, shoulder_width) if nose + both
    shoulders are visible and the person is not side-on, else None."""
    k = record.keypoints
    nose, left, right = k[NOSE], k[LEFT_SHOULDER], k[RIGHT_SHOULDER]
    if min(nose[2], left[2], right[2]) < min_conf:
        return None

    shoulder_width = abs(left[0] - right[0])
    box_width = record.person_box[2] - record.person_box[0]
    if box_width <= 0 or shoulder_width < box_width * min_shoulder_box_ratio:
        return None  # side-on to the camera: shoulders overlap, ratio is meaningless
    return (left[0] + right[0]) / 2, (left[1] + right[1]) / 2, shoulder_width


def head_turn_signed(
    record: FrameRecord, *, min_conf: float, min_shoulder_box_ratio: float
) -> float | None:
    """Sideways nose offset in shoulder widths (negative = towards image left)."""
    frame = _shoulder_frame(record, min_conf=min_conf, min_shoulder_box_ratio=min_shoulder_box_ratio)
    if frame is None:
        return None
    shoulder_mid_x, _, shoulder_width = frame
    return float((record.keypoints[NOSE, 0] - shoulder_mid_x) / shoulder_width)


def head_turn_ratio(
    record: FrameRecord, *, min_conf: float, min_shoulder_box_ratio: float
) -> float | None:
    """Sideways nose offset in shoulder widths, or None if it can't be measured."""
    signed = head_turn_signed(record, min_conf=min_conf, min_shoulder_box_ratio=min_shoulder_box_ratio)
    return None if signed is None else abs(signed)


def head_pitch_ratio(
    record: FrameRecord, *, min_conf: float, min_shoulder_box_ratio: float
) -> float | None:
    """Nose height relative to the shoulder line, in shoulder widths.

    Negative = nose above the shoulders (normal); it rises towards 0 and
    above as the head goes down.
    """
    frame = _shoulder_frame(record, min_conf=min_conf, min_shoulder_box_ratio=min_shoulder_box_ratio)
    if frame is None:
        return None
    _, shoulder_mid_y, shoulder_width = frame
    return float((record.keypoints[NOSE, 1] - shoulder_mid_y) / shoulder_width)


def lean_angle_signed(record: FrameRecord, *, min_conf: float) -> float | None:
    """Torso angle from vertical in degrees (negative = shoulders towards image left)."""
    k = record.keypoints
    if min(k[LEFT_SHOULDER, 2], k[RIGHT_SHOULDER, 2], k[LEFT_HIP, 2], k[RIGHT_HIP, 2]) < min_conf:
        return None

    shoulder_x, shoulder_y = (k[LEFT_SHOULDER, :2] + k[RIGHT_SHOULDER, :2]) / 2
    hip_x, hip_y = (k[LEFT_HIP, :2] + k[RIGHT_HIP, :2]) / 2
    up = hip_y - shoulder_y  # image y grows downwards, so upright = positive
    if up <= 0:
        return None  # keypoints don't make sense (shoulders below hips)
    return math.degrees(math.atan2(shoulder_x - hip_x, up))


def lean_angle_deg(record: FrameRecord, *, min_conf: float) -> float | None:
    """Torso angle away from vertical in degrees, or None if hips/shoulders hidden."""
    signed = lean_angle_signed(record, min_conf=min_conf)
    return None if signed is None else abs(signed)


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


# ---------------------------------------------------------------------------
# Per-student baseline (features.baseline_calibration)
# ---------------------------------------------------------------------------

def posture_signals(record: FrameRecord, cfg: dict[str, Any]) -> dict[str, float | None]:
    """Signed head turn, lean and head pitch for one record (None = not measurable)."""
    pose = cfg["pose"]
    shoulder_args = {"min_conf": pose["keypoint_conf"],
                     "min_shoulder_box_ratio": pose["min_shoulder_box_ratio"]}
    return {
        "head_turn": head_turn_signed(record, **shoulder_args),
        "lean": lean_angle_signed(record, min_conf=pose["keypoint_conf"]),
        "head_pitch": head_pitch_ratio(record, **shoulder_args),
    }


class PostureBaseline:
    """Learns each student's normal posture during their first seconds on camera.

    For every track_id it collects the signed head-turn, lean and head-pitch
    values seen during the first `baseline_seconds` after that track first
    appeared. When the window ends, each signal's baseline is the MEDIAN of
    the collected values, if there were at least `min_samples` of them
    (otherwise that signal gets no baseline and absolute thresholds keep
    being used). Until then, absolute thresholds are used too.

    LIMITATIONS (worth stating in the report):
      - Baselines are keyed by track_id. If ByteTrack loses a student and
        gives them a NEW id, that id starts calibrating from scratch, and
        absolute thresholds apply meanwhile. Re-identifying people would need
        appearance features, which this project deliberately avoids.
      - Whatever a student does during calibration becomes their "normal".
        Absolute thresholds still apply during the window, so a clear turn in
        the first seconds can still be flagged, but a subtle one is absorbed.
    """

    SIGNALS = ("head_turn", "lean", "head_pitch")

    def __init__(self, baseline_seconds: float, min_samples: int) -> None:
        self.window = baseline_seconds
        self.min_samples = min_samples
        self.first_seen: dict[int, float] = {}
        self.samples: dict[int, dict[str, list[float]]] = {}   # tracks still calibrating
        self.baselines: dict[int, dict[str, float]] = {}       # tracks that finished

    def update(self, track_id: int, timestamp: float, values: dict[str, float | None]) -> None:
        """Feed one frame's signed signals for one student."""
        if track_id in self.baselines:
            return  # already calibrated; the baseline is frozen
        first = self.first_seen.setdefault(track_id, timestamp)
        if timestamp - first < self.window:
            samples = self.samples.setdefault(track_id, {s: [] for s in self.SIGNALS})
            for signal in self.SIGNALS:
                if values.get(signal) is not None:
                    samples[signal].append(values[signal])
            return
        # Window over: freeze the baseline (median of what was collected).
        samples = self.samples.pop(track_id, {})
        self.baselines[track_id] = {
            signal: statistics.median(vals) for signal, vals in samples.items()
            if len(vals) >= self.min_samples
        }

    def get(self, track_id: int, signal: str) -> float | None:
        """The student's baseline for `signal`, or None if not (yet) available."""
        return self.baselines.get(track_id, {}).get(signal)


@dataclass
class PostureReading:
    """One posture signal for one student in one frame."""

    value: float | None      # signed measurement (None = can't be measured this frame)
    baseline: float | None   # the student's own normal value, if calibrated
    threshold: float         # pose.* threshold, same units as value

    @property
    def deviation(self) -> float | None:
        """How far from normal: |value - baseline|, or |value| without a baseline."""
        if self.value is None:
            return None
        return abs(self.value - (self.baseline if self.baseline is not None else 0.0))

    @property
    def over_threshold(self) -> bool:
        deviation = self.deviation
        return deviation is not None and deviation >= self.threshold

    @property
    def score(self) -> float | None:
        """Deviation in units of the threshold (1.0 = exactly at the threshold)."""
        deviation = self.deviation
        return None if deviation is None else deviation / self.threshold

    @property
    def direction(self) -> str | None:
        """"left" or "right" (image direction) of the deviation, None if unknown."""
        if self.value is None:
            return None
        return "left" if self.value - (self.baseline or 0.0) < 0 else "right"


def readings_from_signals(
    track_id: int,
    signals: dict[str, float | None],
    cfg: dict[str, Any],
    baseline: PostureBaseline | None = None,
) -> dict[str, PostureReading]:
    """Head-turn and lean readings, relative to the student's baseline if one
    exists, else to the absolute thresholds (pose.* in config.yaml)."""
    pose = cfg["pose"]
    thresholds = {"head_turn": pose["head_turn_threshold"], "lean": pose["lean_threshold_deg"]}
    return {
        signal: PostureReading(
            value=signals[signal],
            baseline=baseline.get(track_id, signal) if baseline is not None else None,
            threshold=threshold,
        )
        for signal, threshold in thresholds.items()
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
