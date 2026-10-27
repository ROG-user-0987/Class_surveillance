"""Tests for the event matching rule and scoring in evaluate.py.
Run: python tests/test_evaluate.py  (or python -m pytest tests)"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from helpers import run_all

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import Clip, LabelledEvent, match_events, score  # noqa: E402

from src.risk import Alert  # noqa: E402


def alert(t: float, behaviour: str = "phone", duration: float = 1.0, tier: str = "red",
          combined: list[str] | None = None) -> Alert:
    signals = {"duration_s": duration}
    if combined:
        signals["combined"] = combined
    return Alert(timestamp=t, track_id=1, tier=tier, reason="", behaviour=behaviour,
                 signals=json.dumps(signals))


def test_overlap_with_tolerance_counts() -> None:
    event = LabelledEvent("phone", 10.0, 15.0)
    assert len(match_events([alert(15.5)], [event], tolerance=1.0)[0]) == 1    # 0.5 s late: OK
    assert len(match_events([alert(7.5, duration=1.0)], [event], 1.0)[0]) == 0  # ends 8.5 < 9.0
    assert len(match_events([alert(8.5, duration=1.0)], [event], 1.0)[0]) == 1  # ends 9.5 >= 9.0


def test_event_type_must_match() -> None:
    matches, false_positives, missed = match_events(
        [alert(11.0, behaviour="head_turn", tier="orange")], [LabelledEvent("phone", 10, 15)], 1.0)
    assert not matches and len(false_positives) == 1 and len(missed) == 1


def test_each_labelled_event_matches_at_most_once() -> None:
    matches, false_positives, missed = match_events(
        [alert(11.0), alert(13.0)], [LabelledEvent("phone", 10, 15)], 1.0)
    assert len(matches) == 1 and len(false_positives) == 1 and not missed


def test_combined_alert_matches_any_of_its_cues_and_grey_is_ignored() -> None:
    combined = alert(20.0, behaviour="combined", tier="orange", combined=["head_turn", "lean"])
    grey = alert(1.0, behaviour="low_visibility", tier="grey")
    matches, false_positives, _ = match_events([combined, grey], [LabelledEvent("lean", 19, 22)], 1.0)
    assert len(matches) == 1 and not false_positives


def test_false_alerts_per_hour_use_honest_clips() -> None:
    cheating = Clip("a.mp4", Path("a.mp4"), events=[LabelledEvent("phone", 10, 15)], duration_s=1800)
    honest = Clip("b.mp4", Path("b.mp4"), duration_s=1800)
    alerts = {"a.mp4": [alert(11.0), alert(100.0)], "b.mp4": [alert(50.0), alert(70.0)]}
    result = score([cheating, honest], alerts, tolerance=1.0)
    assert result["recall"] == 1.0 and result["fp"] == 3
    assert result["false_alerts_per_hour"] == 4.0 and result["false_alert_basis"] == "honest clips"
    # On the honest clip alone recall is undefined (no events), not 0.
    assert score([honest], alerts, tolerance=1.0)["recall"] is None


if __name__ == "__main__":
    run_all(globals())
