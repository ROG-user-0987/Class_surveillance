"""Tests for src/metrics.py (no video or models needed).
Run: python tests/test_metrics.py  (or python -m pytest tests)"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from helpers import make_cfg, run_all

from src.metrics import (StageTimer, alert_metrics, compute, read_alerts, review_metrics,  # noqa: E402
                         run_metrics, wilson_ci)


def turn(duration: float, mean: float, peak: float, baseline: bool) -> dict:
    return {"method": "hysteresis", "duration_s": duration, "mean": mean, "max": peak,
            "units": "shoulder_widths", "baseline_adjusted": baseline}


ROWS = [  # timestamp, track, tier, behaviour, signals, draft_verdict, draft_cause
    (10.0, 1, "orange", "head_turn", turn(2.0, 0.25, 0.4, True), "not_supported", "writing_no_turn"),
    (45.0, 1, "orange", "head_turn", turn(3.0, 0.5, 0.7, False), "supported", ""),
    (50.0, 2, "red", "phone", {"method": "hysteresis", "duration_s": 1.0, "units": "confidence"}, "supported", ""),
    (60.0, 3, "orange", "combined", {"method": "fusion", "duration_s": 2.0, "combined": ["head_turn", "neighbour_peek"],
                                     "parts": [turn(2.0, 0.22, 0.3, False), {"units": "present", "duration_s": 2.0}]},
     "unclear", "distant_view"),
    (90.0, 4, "grey", "low_visibility", {"method": "hysteresis", "duration_s": 2.5, "units": "present"}, "supported", ""),
]


def write_csv(path: Path, review_decision: bool = False) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alert_id", "timestamp", "track_id", "tier", "behaviour", "signals",
                    "review_decision", "draft_verdict", "draft_cause"])
        for i, (t, track, tier, behaviour, sig, verdict, cause) in enumerate(ROWS):
            decision = {"supported": "confirmed", "not_supported": "dismissed"}.get(verdict, "") if review_decision else ""
            w.writerow([f"A{i}", t, track, tier, behaviour, json.dumps(sig), decision, verdict, cause])
    return path


def test_wilson_interval() -> None:
    assert wilson_ci(0, 0) is None
    low, high = wilson_ci(2, 23)
    assert 0.02 < low < 0.087 < high < 0.3        # 2/23 = 8.7%, wide interval for small n
    assert wilson_ci(5, 5)[1] == 1.0 and wilson_ci(0, 5)[0] == 0.0


def test_alert_metrics_without_labels() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        rows = read_alerts(write_csv(Path(tmp) / "x_alerts.csv"))
    m = alert_metrics(rows, duration_s=120, cfg=make_cfg())
    assert m["alerts"] == 5 and m["alerts_per_minute"] == 2.5
    assert m["by_tier"] == {"orange": 3, "red": 1, "grey": 1}
    assert m["students_alerted"] == 4 and m["students_with_one_alert"] == 3
    h = m["head_turn"]  # two head-turn alerts + one inside the combined alert
    assert h["detections"] == 3 and h["baseline_adjusted"] == 1
    assert h["share_mean_below_1.5x_threshold"] == round(2 / 3, 3)  # 0.25 and 0.22 < 0.3
    assert m["min_same_cue_gap_s"] == 35.0          # student 1, head_turn at 10 s and 45 s
    assert m["share_at_minimum_duration"] == 0.8    # all but the 3.0 s head turn


def test_review_precision_excludes_grey_and_unclear() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        rows = read_alerts(write_csv(Path(tmp) / "x_alerts.csv"))
    r = review_metrics(rows)
    assert r["verdict_column"] == "draft_verdict"
    ro = r["red_and_orange"]  # grey is not a detection
    assert (ro["supported"], ro["not_supported"], ro["unclear"]) == (2, 1, 1)
    assert ro["precision"] == round(2 / 3, 4) and ro["ci95"] is not None
    assert r["grey_correct_abstentions"]["correct"] == 1
    assert r["causes_not_supported"] == {"writing_no_turn": 1}


def test_dashboard_decisions_take_priority() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        rows = read_alerts(write_csv(Path(tmp) / "x_alerts.csv", review_decision=True))
    assert review_metrics(rows)["verdict_column"] == "review_decision"


def test_run_metrics_and_compute() -> None:
    summary = {"device": "cpu", "video_duration_s": 120.0, "seconds": 60.0, "frames_processed": 1800,
               "every_nth": 2, "fps": 30.0, "pipeline_fps": 32.0, "ms_per_frame": {"models": 25.0},
               "people_seen": 14, "max_people_in_frame": 7, "mean_people_per_frame": 5.2}
    run = run_metrics(summary)
    assert run["ids_per_visible_person"] == 2.0 and run["speed_vs_real_time"] == 2.0
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = write_csv(Path(tmp) / "x_alerts.csv")
        json_path = Path(tmp) / "x_summary.json"
        json_path.write_text(json.dumps(summary))
        report = compute(csv_path, json_path, None, make_cfg())
    assert report["video_duration_s"] == 120.0 and "run" in report and "review" in report


def test_stage_timer() -> None:
    timer = StageTimer()
    with timer.time("models"):
        sum(range(10000))
    timer.frames = 1
    s = timer.summary()
    assert s["frames"] == 1 and "models" in s["ms_per_frame"] and s["pipeline_fps"] > 0


if __name__ == "__main__":
    run_all(globals())
