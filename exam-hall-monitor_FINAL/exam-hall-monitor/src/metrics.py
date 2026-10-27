"""Metrics for a scanned video: what the system did, and how good its alerts were.

evaluate.py scores alerts against LABELLED clips (precision / recall / F1,
false alerts per hour, sweep, ablation). This module covers what you can
measure WITHOUT labels, plus precision from reviewer decisions:

1. Run metrics (from <video>_summary.json, written by every scan):
   speed (processed frames/s and ms per frame for each stage), people per
   frame, track IDs issued vs people actually visible (identity stability).
2. Alert metrics (from <video>_alerts.csv):
   alerts per minute and per hour by tier and cue, students alerted, alerts per
   student, share of head-turn detections that used a personal baseline, how
   far above the threshold the signals were, events that fired at the minimum
   duration, and the smallest gap between two alerts of the same cue for one
   student (cooldown check).
3. Review metrics (from the same CSV once alerts are reviewed):
   precision = confirmed / (confirmed + dismissed), overall, per tier and per
   cue, each with a 95% Wilson confidence interval (honest for small numbers).
   Grey alerts are "cannot assess", not detections, so they are reported
   separately ("correct abstentions"), never mixed into precision.
   Accepted verdict columns: review_decision (confirmed / dismissed, filled in
   by the dashboard) or, if that is empty, draft_verdict (supported /
   not_supported / unclear) plus an optional draft_cause column.

Command (repo root):
    python -m src.metrics --alerts results/clip_alerts.csv
        [--summary results/clip_summary.json]  (found automatically if next to the CSV)
        [--duration-s 300]                      (only needed without a summary)
        [--plot]                                -> results/<name>_metrics.png
    -> results/<name>_metrics.json and a printed report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.utils import load_config

# Every verdict word we accept -> "tp" (behaviour was there), "fp" (it was not)
# or "unclear" (reviewer could not tell; left out of precision).
VERDICTS = {
    "confirmed": "tp", "supported": "tp",
    "dismissed": "fp", "not_supported": "fp",
    "unclear": "unclear", "": "unreviewed",
}
WILSON_Z = 1.96  # 95% confidence


# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = WILSON_Z) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion; None when n == 0.
    Unlike p +/- 1.96*sqrt(p(1-p)/n) it stays inside [0, 1] and is sensible
    for small n (e.g. 2 out of 23)."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


class StageTimer:
    """Wall-clock time per pipeline stage, for 'ms per frame' and FPS figures.

        timer = StageTimer()
        with timer.time("models"): ...
        with timer.time("rules"): ...
        timer.frames += 1
        timer.summary()  # {"frames": n, "ms_per_frame": {...}, "pipeline_fps": x}
    """

    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.frames = 0

    def time(self, stage: str):
        timer = self

        class _Stage:
            def __enter__(self):
                self.start = time.perf_counter()

            def __exit__(self, *exc):
                timer.totals[stage] += time.perf_counter() - self.start

        return _Stage()

    def summary(self) -> dict[str, Any]:
        n = max(self.frames, 1)
        total = sum(self.totals.values())
        return {"frames": self.frames,
                "ms_per_frame": {k: round(1000 * v / n, 2) for k, v in self.totals.items()},
                "pipeline_fps": round(n / total, 2) if total > 0 else None}


# ---------------------------------------------------------------------------
# Reading inputs
# ---------------------------------------------------------------------------

def read_alerts(path: str | Path) -> list[dict[str, Any]]:
    """Alerts CSV rows with numbers parsed and signals decoded."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["timestamp"] = float(row["timestamp"])
        row["track_id"] = int(float(row["track_id"]))
        try:
            row["signals"] = json.loads(row.get("signals") or "{}")
        except ValueError:
            row["signals"] = {}
        row["duration_s"] = row["signals"].get("duration_s")
    return sorted(rows, key=lambda r: r["timestamp"])


def signal_parts(signals: dict[str, Any]) -> list[dict[str, Any]]:
    """The per-cue measurements inside an alert (a combined alert has several)."""
    return signals.get("parts", []) if signals.get("method") == "fusion" else [signals]


def verdict(row: dict[str, Any], column: str) -> str:
    return VERDICTS.get(str(row.get(column) or "").strip().lower(), "unclear")


def pick_verdict_column(rows: list[dict[str, Any]]) -> str | None:
    """review_decision if anyone has reviewed, else draft_verdict if present."""
    if any((r.get("review_decision") or "").strip() for r in rows):
        return "review_decision"
    if any((r.get("draft_verdict") or "").strip() for r in rows):
        return "draft_verdict"
    return None


# ---------------------------------------------------------------------------
# 1. Run metrics
# ---------------------------------------------------------------------------

def run_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    """Speed and identity stability from a scan summary JSON."""
    out = {key: summary.get(key) for key in (
        "device", "video_duration_s", "frames_processed", "every_nth", "fps", "pipeline_fps",
        "ms_per_frame", "people_seen", "max_people_in_frame", "mean_people_per_frame")}
    if summary.get("video_duration_s") and summary.get("seconds"):
        out["speed_vs_real_time"] = round(summary["video_duration_s"] / summary["seconds"], 2)
    if summary.get("people_seen") and summary.get("max_people_in_frame"):
        # 1.0 = every person kept one ID all video; higher = IDs lost and re-issued
        out["ids_per_visible_person"] = round(summary["people_seen"] / summary["max_people_in_frame"], 2)
    return out


# ---------------------------------------------------------------------------
# 2. Alert metrics (no labels needed)
# ---------------------------------------------------------------------------

def alert_metrics(rows: list[dict[str, Any]], duration_s: float, cfg: dict[str, Any]) -> dict[str, Any]:
    """Volume, spread over students, baseline use and signal strength of alerts."""
    minutes = duration_s / 60 if duration_s else 0
    per_student = Counter(r["track_id"] for r in rows)
    out: dict[str, Any] = {
        "alerts": len(rows),
        "alerts_per_minute": round(len(rows) / minutes, 2) if minutes else None,
        "alerts_per_hour": round(len(rows) / minutes * 60, 1) if minutes else None,
        "by_tier": dict(Counter(r["tier"] for r in rows)),
        "by_cue": dict(Counter(r.get("behaviour", "") for r in rows)),
        "students_alerted": len(per_student),
        "alerts_per_student_max": max(per_student.values(), default=0),
        "students_with_one_alert": sum(n == 1 for n in per_student.values()),
        "highest_track_id": max((r["track_id"] for r in rows), default=0),
    }

    # Head-turn measurements (shoulder widths), from single and combined alerts.
    threshold = cfg["pose"]["head_turn_threshold"]
    near = cfg["metrics"]["near_threshold_factor"]
    turns = [p for r in rows for p in signal_parts(r["signals"]) if p.get("units") == "shoulder_widths"]
    if turns:
        means = [p["mean"] for p in turns]
        out["head_turn"] = {
            "detections": len(turns),
            "baseline_adjusted": sum(bool(p.get("baseline_adjusted")) for p in turns),
            "baseline_coverage": round(sum(bool(p.get("baseline_adjusted")) for p in turns) / len(turns), 3),
            "threshold": threshold,
            "median_mean_turn": round(statistics.median(means), 3),
            "median_peak_turn": round(statistics.median(p["max"] for p in turns), 3),
            f"share_mean_below_{near}x_threshold": round(sum(m < near * threshold for m in means) / len(means), 3),
        }

    # Events that fired the moment they reached their cue's minimum duration.
    by_cue: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["duration_s"] is not None:
            by_cue[r.get("behaviour", "")].append(r["duration_s"])
    at_min = sum(d <= min(ds) + 0.1 for ds in by_cue.values() for d in ds)
    total = sum(len(ds) for ds in by_cue.values())
    out["share_at_minimum_duration"] = round(at_min / total, 3) if total else None

    # Cooldown check: smallest gap between alerts of the same cue for one student.
    times: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        times[(r["track_id"], r.get("behaviour", ""))].append(r["timestamp"])
    gaps = [b - a for ts in times.values() for a, b in zip(sorted(ts), sorted(ts)[1:])]
    out["min_same_cue_gap_s"] = round(min(gaps), 2) if gaps else None
    return out


# ---------------------------------------------------------------------------
# 3. Review metrics
# ---------------------------------------------------------------------------

def _precision_block(rows: list[dict[str, Any]], column: str) -> dict[str, Any]:
    counts = Counter(verdict(r, column) for r in rows)
    tp, fp = counts["tp"], counts["fp"]
    return {"alerts": len(rows), "supported": tp, "not_supported": fp,
            "unclear": counts["unclear"], "unreviewed": counts["unreviewed"],
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "ci95": wilson_ci(tp, tp + fp)}


def review_metrics(rows: list[dict[str, Any]], column: str | None = None) -> dict[str, Any] | None:
    """Precision of red + orange alerts from reviewer verdicts; grey separately."""
    column = column or pick_verdict_column(rows)
    if column is None:
        return None
    detections = [r for r in rows if r["tier"] != "grey"]
    grey = [r for r in rows if r["tier"] == "grey"]
    out: dict[str, Any] = {
        "verdict_column": column,
        "red_and_orange": _precision_block(detections, column),
        "by_tier": {t: _precision_block([r for r in rows if r["tier"] == t], column)
                    for t in sorted({r["tier"] for r in rows})},
        "by_cue": {c: _precision_block([r for r in rows if r.get("behaviour") == c], column)
                   for c in sorted({r.get("behaviour", "") for r in rows})},
    }
    if grey:  # for grey, "supported" = the person really could not be assessed
        block = _precision_block(grey, column)
        out["grey_correct_abstentions"] = {"alerts": len(grey), "correct": block["supported"],
                                           "share": block["precision"], "ci95": block["ci95"]}
    if any(r.get("draft_cause") for r in rows):
        out["causes_not_supported"] = dict(Counter(r["draft_cause"] for r in detections
                                                   if verdict(r, column) == "fp"))
        out["causes_unclear"] = dict(Counter(r["draft_cause"] for r in rows if verdict(r, column) == "unclear"))
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _ci(ci: tuple[float, float] | None) -> str:
    return "" if ci is None else f"  (95% CI {_pct(ci[0])} - {_pct(ci[1])})"


def print_report(report: dict[str, Any]) -> None:
    run, alerts, review = report.get("run"), report["alerts"], report.get("review")
    print("\n=== Run ===")
    if run:
        print(f"Device {run['device']}: {run['fps']:.1f} processed frames/s "
              f"(1 frame in {run['every_nth']} processed), ms per frame {run['ms_per_frame']}")
        print(f"People: max {run['max_people_in_frame']} in one frame, "
              f"{run['people_seen']} track IDs issued -> {run.get('ids_per_visible_person')} IDs per visible person")
    else:
        print("(no summary JSON: run the scan again to get speed and people metrics)")
    print("\n=== Alerts ===")
    print(f"{alerts['alerts']} alerts ({alerts['alerts_per_minute']}/min, {alerts['alerts_per_hour']}/h)  "
          f"tiers {alerts['by_tier']}")
    print(f"cues {alerts['by_cue']}")
    print(f"{alerts['students_alerted']} students alerted, {alerts['students_with_one_alert']} only once, "
          f"max {alerts['alerts_per_student_max']} for one student")
    if "head_turn" in alerts:
        h = alerts["head_turn"]
        print(f"head turn: {h['baseline_adjusted']}/{h['detections']} baseline-adjusted, "
              f"median mean {h['median_mean_turn']} vs threshold {h['threshold']}")
    print(f"share at minimum duration {_pct(alerts['share_at_minimum_duration'])}, "
          f"smallest same-cue gap {alerts['min_same_cue_gap_s']} s")
    if review:
        print(f"\n=== Review ({review['verdict_column']}) ===")
        rows = [("red+orange", review["red_and_orange"])] + list(review["by_tier"].items())
        for name, b in rows:
            print(f"{name:11s} supported {b['supported']:3d}  not {b['not_supported']:3d}  "
                  f"unclear {b['unclear']:3d}  unreviewed {b['unreviewed']:3d}  "
                  f"precision {_pct(b['precision'])}{_ci(b['ci95'])}")
        if "causes_not_supported" in review:
            print(f"why not supported: {review['causes_not_supported']}")


def plot_report(report: dict[str, Any], path: Path) -> None:
    """Two small bar charts: alerts per cue, and review outcome per cue."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    review = report.get("review")
    fig, axes = plt.subplots(1, 2 if review else 1, figsize=(11 if review else 6, 3.6), squeeze=False)
    cues = report["alerts"]["by_cue"]
    names = sorted(cues, key=cues.get)
    axes[0][0].barh(names, [cues[n] for n in names], color="#2a78d6")
    axes[0][0].set_title("Alerts per cue")
    if review:
        ax = axes[0][1]
        left = [0] * len(names)
        for key, colour, label in (("supported", "#1baf7a", "supported"),
                                   ("not_supported", "#eb6834", "not supported"),
                                   ("unclear", "#c3c2b7", "unclear")):
            vals = [review["by_cue"].get(n, {}).get(key, 0) for n in names]
            ax.barh(names, vals, left=left, color=colour, label=label, edgecolor="white")
            left = [a + b for a, b in zip(left, vals)]
        ax.set_title("Review outcome per cue")
        ax.legend(frameon=False, fontsize=8)
    for row in axes:
        for ax in row:
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compute(alerts_csv: str | Path, summary_json: str | Path | None, duration_s: float | None,
            cfg: dict[str, Any]) -> dict[str, Any]:
    rows = read_alerts(alerts_csv)
    summary = json.loads(Path(summary_json).read_text()) if summary_json else None
    duration = duration_s or (summary or {}).get("video_duration_s")
    if not duration:
        raise SystemExit("Need the video length: pass --duration-s or a scan summary JSON.")
    report: dict[str, Any] = {"alerts_csv": str(alerts_csv), "video_duration_s": duration,
                              "alerts": alert_metrics(rows, duration, cfg)}
    if summary:
        report["run"] = run_metrics(summary)
    review = review_metrics(rows)
    if review:
        report["review"] = review
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Metrics for one scanned video (see module docstring).")
    parser.add_argument("--alerts", required=True, help="<video>_alerts.csv from a scan")
    parser.add_argument("--summary", help="<video>_summary.json (default: next to the alerts CSV)")
    parser.add_argument("--duration-s", type=float, help="video length, if there is no summary JSON")
    parser.add_argument("--config", help="Config file (default: config.yaml at the repo root).")
    parser.add_argument("--plot", action="store_true", help="also save a PNG chart")
    args = parser.parse_args()

    cfg = load_config(args.config)
    alerts_csv = Path(args.alerts)
    name = alerts_csv.stem.removesuffix("_alerts")
    summary = Path(args.summary) if args.summary else alerts_csv.with_name(f"{name}_summary.json")
    report = compute(alerts_csv, summary if summary.exists() else None, args.duration_s, cfg)

    out = alerts_csv.with_name(f"{name}_metrics.json")
    out.write_text(json.dumps(report, indent=2))
    print_report(report)
    print(f"\nWrote {out}")
    if args.plot:
        png = alerts_csv.with_name(f"{name}_metrics.png")
        plot_report(report, png)
        print(f"Wrote {png}")


if __name__ == "__main__":
    main()
