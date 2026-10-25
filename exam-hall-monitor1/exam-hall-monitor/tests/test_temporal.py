"""Tests for src/temporal.py (EventTracker) and the per-student baseline.

Synthetic signals only: no video, no models. Run either way:
    python -m pytest tests/test_temporal.py
    python tests/test_temporal.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Make "import src..." work when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.pose_rules import (  # noqa: E402
    PostureBaseline,
    posture_flags,
    posture_signals,
    readings_from_signals,
)
from src.temporal import EventTracker  # noqa: E402
from src.tracking import FrameRecord  # noqa: E402

FPS = 10  # synthetic frame rate: one processed frame every 0.1 s

POSE_CFG = {"pose": {"keypoint_conf": 0.5, "min_shoulder_box_ratio": 0.3,
                     "head_turn_threshold": 0.2, "lean_threshold_deg": 25}}


def phone_tracker(min_duration_s: float = 1.0) -> EventTracker:
    return EventTracker(ema_alpha=0.3, on_threshold=0.3, off_threshold=0.15,
                        max_gap_frames=3, min_duration_s=min_duration_s)


def posture_tracker(min_duration_s: float = 2.0) -> EventTracker:
    return EventTracker(ema_alpha=0.3, on_threshold=1.0, off_threshold=0.7,
                        max_gap_frames=3, min_duration_s=min_duration_s)


def run(tracker: EventTracker, values: list[float | None], track_id: int = 1) -> list[tuple[float, object]]:
    """Feed one value per frame (None = track missing); return (time, event) for each firing."""
    fired = []
    for i, value in enumerate(values):
        t = i / FPS
        frame_values = {} if value is None else {track_id: value}
        for event in tracker.update(t, frame_values):
            fired.append((t, event))
    return fired


def make_record(turn: float, t: float = 0.0, track_id: int = 1) -> FrameRecord:
    """A person facing the camera whose nose is `turn` shoulder widths to the side."""
    k = np.zeros((17, 3), np.float32)
    k[0] = [150 + turn * 100, 80, 0.9]           # nose
    k[5], k[6] = [100, 150, 0.9], [200, 150, 0.9]  # shoulders, 100 px apart
    k[11], k[12] = [110, 300, 0.9], [190, 300, 0.9]  # hips straight below
    return FrameRecord(frame_index=int(t * FPS), timestamp=t, track_id=track_id,
                       person_box=np.array([75, 50, 225, 400], np.float32), keypoints=k)


# ---------------------------------------------------------------------------
# EventTracker
# ---------------------------------------------------------------------------

def test_flicker_does_not_trigger() -> None:
    # Sparse single-frame spikes (a false detection every 4th frame) for 10 s.
    sparse = [0.6 if i % 4 == 0 else 0.0 for i in range(100)]
    assert run(phone_tracker(), sparse) == []

    # Short two-frame bursts may START an event, but it ends before 1 s.
    bursts = ([0.9, 0.9] + [0.0] * 8) * 10
    assert run(phone_tracker(), bursts) == []


def test_single_frame_dropout_does_not_end_event() -> None:
    for dropout in (None, 0.0):  # track missing for a frame / signal low for a frame
        values = [1.6] * 30
        values[15] = dropout
        fired = run(posture_tracker(min_duration_s=2.0), values)
        assert len(fired) == 1, f"dropout={dropout}: expected one event, got {len(fired)}"
        fired_at, event = fired[0]
        assert event.start < 1.5, "event restarted after the dropout"
        # The Event object keeps updating after it fires, so measure at the firing time.
        assert 2.0 <= fired_at - event.start < 2.0 + 1.5 / FPS


def test_sustained_signal_triggers_after_min_duration() -> None:
    fired = run(phone_tracker(min_duration_s=1.0), [0.7] * 50)
    assert len(fired) == 1, "a sustained event must fire exactly once"
    fired_at, event = fired[0]
    assert 1.0 <= fired_at - event.start < 1.0 + 1.5 / FPS
    assert abs(event.mean_value - 0.7) < 1e-9 and abs(event.max_value - 0.7) < 1e-9


def test_long_gap_ends_event_and_new_one_starts() -> None:
    values = [0.7] * 20 + [None] * 5 + [0.7] * 20  # 5 missing frames > max_gap_frames (3)
    fired = run(phone_tracker(min_duration_s=1.0), values)
    assert len(fired) == 2, "the long gap should end the first event"
    assert fired[1][1].start > 2.0


def test_hysteresis_needs_rise_above_on() -> None:
    # A steady signal between off (0.15) and on (0.3) never starts an event.
    assert run(phone_tracker(min_duration_s=0.5), [0.25] * 100) == []


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def test_baseline_subtraction() -> None:
    baseline = PostureBaseline(baseline_seconds=2.0, min_samples=5)
    # This student's camera angle makes them look 0.25 "turned" while facing forward.
    for i in range(int(2.0 * FPS) + 1):
        record = make_record(turn=0.25, t=i / FPS)
        baseline.update(record.track_id, record.timestamp, posture_signals(record, POSE_CFG))
    assert abs(baseline.get(1, "head_turn") - 0.25) < 1e-6

    # Facing forward again (0.27): absolute rule says "turned", baseline says normal.
    forward = posture_signals(make_record(turn=0.27, t=3.0), POSE_CFG)
    assert readings_from_signals(1, forward, POSE_CFG)["head_turn"].over_threshold
    adjusted = readings_from_signals(1, forward, POSE_CFG, baseline)["head_turn"]
    assert not adjusted.over_threshold and abs(adjusted.deviation - 0.02) < 1e-4

    # Turned the other way (-0.05): 0.30 away from this student's normal.
    other_way = readings_from_signals(1, posture_signals(make_record(turn=-0.05), POSE_CFG),
                                      POSE_CFG, baseline)["head_turn"]
    assert other_way.over_threshold and other_way.direction == "left"


def test_no_baseline_until_window_ends_or_too_few_samples() -> None:
    baseline = PostureBaseline(baseline_seconds=2.0, min_samples=5)
    baseline.update(1, 0.0, {"head_turn": 0.1, "lean": None, "head_pitch": None})
    assert baseline.get(1, "head_turn") is None  # still calibrating -> absolute thresholds
    baseline.update(1, 0.5, {"head_turn": 0.1, "lean": None, "head_pitch": None})
    baseline.update(1, 2.5, {"head_turn": 0.1, "lean": None, "head_pitch": None})
    assert baseline.get(1, "head_turn") is None  # only 2 samples < min_samples


def test_readings_without_baseline_match_original_rule() -> None:
    # With features off, risk.py uses readings_from_signals(); it must agree
    # exactly with the original posture_flags() on any keypoints.
    rng = random.Random(0)
    for _ in range(3000):
        k = np.array([[rng.uniform(0, 400), rng.uniform(0, 400), rng.uniform(0, 1)]
                      for _ in range(17)], np.float32)
        record = FrameRecord(0, 0.0, 1, np.array([50, 20, 350, 420], np.float32), k)
        readings = readings_from_signals(1, posture_signals(record, POSE_CFG), POSE_CFG)
        flags = posture_flags(record, POSE_CFG)
        assert readings["head_turn"].over_threshold == flags["head_turn"]
        assert readings["lean"].over_threshold == flags["lean"]


if __name__ == "__main__":
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"All {len(tests)} tests passed.")
