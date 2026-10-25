"""Evaluation for the report.

Commands (run from the repo root):
    python evaluate.py --video data/clip.mp4 --labels data/labels.csv
        score one clip against its labelled events (the default)
    python evaluate.py --clips-dir data/eval --labels data/labels.csv
        score every clip listed in labels.csv; add any of:
            --sweep     phone threshold sweep -> results/sweep.csv + results/sweep.png
            --ablation  features off / each on / all on -> results/ablation.csv
            --gallery   false positive / negative thumbnails -> results/gallery/
        results/per_condition.csv is written whenever labels.csv has
        angle / lighting columns.
    python evaluate.py --compare-modes --gt data/phone_set [--images DIR]
        full vs sahi vs crop phone detection on labelled images
        -> results/mode_comparison.csv (see src/detect_phone.py)

labels.csv: one row per labelled event (header required):
    clip,event_type,start_s,end_s,angle,lighting
    exam1.mp4,phone,12.0,18.5,front,bright
    exam1.mp4,head_turn,40.0,44.0,front,bright
    exam2.mp4,none,,,side,dim        <- honest clip: no events
  clip        file name inside --clips-dir (or the --video file name)
  event_type  phone, head_turn, lean, out_of_seat, neighbour_peek,
              object_passing, hands_hidden; "none" or empty = honest clip
  angle, lighting are optional (used for the per-condition breakdown).

MATCHING RULE (5a): a predicted alert is a TRUE POSITIVE if its interval
[start, start + duration] overlaps a labelled interval of the same event type
widened by evaluation.match_tolerance_s on each side. Alerts are taken in time
order and each one takes the earliest unmatched labelled event it overlaps;
each labelled event can be matched AT MOST ONCE, so a second alert on the
same event is a false positive. A combined alert matches any of its cue
types. Grey ("cannot assess") alerts are not detections and are ignored.
  precision = true positives / alerts
  recall    = matched labelled events / labelled events
  false alerts per hour = false positives per hour of HONEST clips (clips
              with no labelled events); if there are none, of all clips.

Detections (people + phone boxes) are cached per frame in results/cache/, so
sweeps and ablations replay the risk logic without re-running the models.
Phones are cached at the lowest confidence any run needs and filtered per run.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from src.detect_phone import MODES, PhoneDetection, compare_modes, detection_mode
from src.risk import Alert, RiskScorer, alert_duration, peak_time_of
from src.tracking import (
    ExamHallTracker,
    FrameOutput,
    Person,
    associate_phones,
    build_records,
    draw_frame,
    prepare_runtime,
)
from src.utils import (
    get_video_info,
    load_config,
    read_video,
    resize_to_width,
    resolve_path,
)

EVENT_TYPES = {"phone", "head_turn", "lean", "out_of_seat", "neighbour_peek", "object_passing", "hands_hidden"}


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

@dataclass
class LabelledEvent:
    event_type: str
    start: float
    end: float


@dataclass
class Clip:
    name: str
    path: Path
    events: list[LabelledEvent] = field(default_factory=list)
    conditions: dict[str, str] = field(default_factory=dict)  # e.g. {"angle": "front"}
    duration_s: float = 0.0

    @property
    def honest(self) -> bool:
        """No labelled events: every alert on this clip is a false alert."""
        return not self.events


def load_labels(labels_path: str | Path, clips_dir: Path, only: str | None = None) -> list[Clip]:
    """Clips and their labelled events from labels.csv (see the module docstring)."""
    clips: dict[str, Clip] = {}
    with open(labels_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("clip") or "").strip()
            if not name or (only and name != only):
                continue
            clip = clips.setdefault(name, Clip(name=name, path=clips_dir / name))
            for column in ("angle", "lighting"):
                if (row.get(column) or "").strip():
                    clip.conditions[column] = row[column].strip()
            event_type = (row.get("event_type") or "").strip().lower()
            if event_type in ("", "none"):
                continue
            if event_type not in EVENT_TYPES:
                raise SystemExit(f"Unknown event_type {event_type!r} in {labels_path}; "
                                 f"use one of {sorted(EVENT_TYPES)} or 'none'.")
            clip.events.append(LabelledEvent(event_type, float(row["start_s"]), float(row["end_s"])))
    return list(clips.values())


# ---------------------------------------------------------------------------
# Matching and scoring
# ---------------------------------------------------------------------------

def alert_types(alert: Alert) -> set[str]:
    """Event types an alert can match (a combined alert: any of its cues)."""
    if alert.behaviour == "combined":
        try:
            return set(json.loads(alert.signals)["combined"])
        except (ValueError, KeyError, TypeError):
            return set()
    return {alert.behaviour}


def match_events(
    alerts: list[Alert], events: list[LabelledEvent], tolerance: float
) -> tuple[list[tuple[Alert, LabelledEvent]], list[Alert], list[LabelledEvent]]:
    """Apply the matching rule (module docstring). Returns
    (matched (alert, event) pairs, false-positive alerts, missed events)."""
    unmatched = sorted(events, key=lambda e: e.start)
    matches, false_positives = [], []
    for alert in sorted(alerts, key=lambda a: a.timestamp):
        if alert.tier == "grey":
            continue  # "cannot assess" is not a detection
        start = alert.timestamp
        end = start + alert_duration(alert)
        types = alert_types(alert)
        hit = next((e for e in unmatched if e.event_type in types
                    and start <= e.end + tolerance and end >= e.start - tolerance), None)
        if hit is None:
            false_positives.append(alert)
        else:
            unmatched.remove(hit)
            matches.append((alert, hit))
    return matches, false_positives, unmatched


def score(
    clips: list[Clip], alerts_by_clip: dict[str, list[Alert]], tolerance: float,
    types: set[str] | None = None,
) -> dict[str, Any]:
    """Precision, recall, F1 and false alerts per hour over these clips
    (optionally only for the given event types)."""
    tp = fp = fn = honest_fp = 0
    honest_s = total_s = 0.0
    for clip in clips:
        alerts = [a for a in alerts_by_clip[clip.name] if types is None or alert_types(a) & types]
        events = [e for e in clip.events if types is None or e.event_type in types]
        matches, false_positives, missed = match_events(alerts, events, tolerance)
        tp, fp, fn = tp + len(matches), fp + len(false_positives), fn + len(missed)
        total_s += clip.duration_s
        if clip.honest:
            honest_fp += len(false_positives)
            honest_s += clip.duration_s

    # None = undefined (no alerts for precision, no labelled events for recall),
    # shown as "-" on the console and left blank in the CSVs.
    precision = round(tp / (tp + fp), 4) if tp + fp else None
    recall = round(tp / (tp + fn), 4) if tp + fn else None
    f1 = None
    if precision is not None and recall is not None:
        f1 = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0
    if honest_s > 0:
        per_hour, basis = honest_fp / (honest_s / 3600), "honest clips"
    else:
        per_hour, basis = (fp / (total_s / 3600) if total_s > 0 else 0.0), "all clips"
    return {"precision": precision, "recall": recall, "f1": f1,
            "false_alerts_per_hour": round(per_hour, 2), "tp": tp, "fp": fp, "fn": fn,
            "false_alert_basis": basis}


# ---------------------------------------------------------------------------
# Detection cache + replay
# ---------------------------------------------------------------------------

@dataclass
class CachedFrame:
    frame_index: int
    timestamp: float
    people: list[Person]
    phones: list[PhoneDetection]


def detection_conf(cfg: dict[str, Any]) -> float:
    """Lowest phone confidence any run (normal or sweep) needs."""
    return min([cfg["phone"]["conf_threshold"], *cfg["evaluation"]["sweep_conf"]])


def cache_detections(clip: Clip, cfg: dict[str, Any], cache_dir: Path) -> list[CachedFrame]:
    """People + phones for every processed frame of a clip, from the cache if
    the same clip was already run with the same detection settings."""
    phone = cfg["phone"]
    settings = {
        "clip": clip.name, "size": clip.path.stat().st_size, "mtime": clip.path.stat().st_mtime,
        "pose": cfg["paths"]["pose_weights"], "phone": cfg["paths"]["phone_weights"],
        "tracker": cfg["paths"]["tracker"], "mode": detection_mode(cfg), "class": phone["class_id"],
        "crop": [phone.get("crop_pad"), phone.get("crop_down_bias"), phone.get("crop_size")],
        "sahi": [phone.get("slice_size"), phone.get("slice_overlap")], "nms": phone.get("nms_iou"),
        "imgsz": cfg["runtime"]["imgsz"], "every_nth": cfg["runtime"]["every_nth_frame"],
        "person_conf": cfg["pose"]["person_conf"], "phone_conf": detection_conf(cfg),
    }
    key = hashlib.md5(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    path = cache_dir / f"{clip.path.stem}_{key}.pkl"
    if path.exists():
        print(f"  {clip.name}: using cached detections ({path.name})")
        with open(path, "rb") as f:  # only files this script wrote itself
            return pickle.load(f)

    print(f"  {clip.name}: running detection (cached afterwards)")
    tracker = ExamHallTracker(cfg)  # fresh tracker per clip (ByteTrack state)
    frames = []
    for frame_index, timestamp, frame in read_video(clip.path, cfg["runtime"]["every_nth_frame"],
                                                    cfg["runtime"]["fallback_fps"]):
        people, phones = tracker.detect(frame, phone_conf=detection_conf(cfg))
        frames.append(CachedFrame(frame_index, timestamp, people, phones))
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(frames, f)
    return frames


def frame_output(frame: CachedFrame, cfg: dict[str, Any]) -> FrameOutput:
    """Association + records for one cached frame, at this cfg's phone threshold."""
    phones = [p for p in frame.phones if p.conf >= cfg["phone"]["conf_threshold"]]
    assoc = cfg["association"]
    assignments = associate_phones(phones, frame.people, keypoint_conf=assoc["keypoint_conf"],
                                   max_wrist_distance=assoc["max_wrist_distance"],
                                   min_box_overlap=assoc["min_box_overlap"])
    records = build_records(frame.frame_index, frame.timestamp, frame.people, phones, assignments)
    return FrameOutput(frame.timestamp, records, frame.people, phones, assignments)


def replay(frames: list[CachedFrame], cfg: dict[str, Any]) -> list[Alert]:
    """Run the risk logic over cached detections (no models, no video)."""
    risk = RiskScorer(cfg)
    alerts: list[Alert] = []
    for frame in frames:
        alerts += risk.update(frame.timestamp, frame_output(frame, cfg).records)
    alerts += risk.finish(frames[-1].timestamp if frames else 0.0)
    return alerts


# ---------------------------------------------------------------------------
# 5b sweep, 5c ablation, 5d per condition, 5e gallery
# ---------------------------------------------------------------------------

def run_sweep(clips: list[Clip], frames: dict[str, list[CachedFrame]], cfg: dict[str, Any],
              out_dir: Path) -> list[dict[str, Any]]:
    """Phone event recall vs false alerts/hour for each phone.conf_threshold
    and each phone.min_duration_s value in evaluation.sweep_*."""
    evaluation = cfg["evaluation"]
    rows = []
    for parameter, values in (("conf_threshold", evaluation["sweep_conf"]),
                              ("min_duration_s", evaluation["sweep_min_duration"])):
        for value in values:
            run_cfg = copy.deepcopy(cfg)
            run_cfg["phone"][parameter] = value
            alerts = {clip.name: replay(frames[clip.name], run_cfg) for clip in clips}
            result = score(clips, alerts, evaluation["match_tolerance_s"], types={"phone"})
            rows.append({"parameter": parameter, "value": value, "recall": result["recall"],
                         "false_alerts_per_hour": result["false_alerts_per_hour"],
                         "precision": result["precision"],
                         "current": value == cfg["phone"][parameter],
                         "false_alert_basis": result["false_alert_basis"]})
    write_csv(out_dir / "sweep.csv", rows)
    plot_sweep(rows, out_dir / "sweep.png")
    return rows


def plot_sweep(rows: list[dict[str, Any]], path: Path) -> None:
    """Recall (y) vs false alerts/hour (x), one point per setting, current config ringed."""
    import matplotlib
    matplotlib.use("Agg")  # no screen needed
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ringed = False
    # Labels: "c=0.3" (confidence, above the point) and "d=1s" (duration, below it).
    for parameter, marker, label, prefix, unit, offset in (
            ("conf_threshold", "o", "phone confidence threshold (c)", "c=", "", (4, 5)),
            ("min_duration_s", "s", "phone minimum duration (d)", "d=", "s", (4, -11))):
        points = [r for r in rows if r["parameter"] == parameter and r["recall"] is not None]
        xs = [r["false_alerts_per_hour"] for r in points]
        ys = [r["recall"] for r in points]
        ax.plot(xs, ys, marker=marker, linestyle="-", alpha=0.7, label=label)
        same_spot: dict[tuple[float, float], list[str]] = {}  # settings that give the same point
        for r in points:
            same_spot.setdefault((r["false_alerts_per_hour"], r["recall"]), []).append(f"{r['value']:g}")
        for (x, y), values in same_spot.items():
            ax.annotate(f"{prefix}{','.join(values)}{unit}", (x, y),
                        textcoords="offset points", xytext=offset, fontsize=7)
        for r in points:  # ring the setting currently in config.yaml
            if r["current"]:
                ax.scatter([r["false_alerts_per_hour"]], [r["recall"]], s=220, facecolors="none",
                           edgecolors="red", linewidths=2, zorder=3,
                           label=None if ringed else "current config")
                ringed = True
    basis = rows[0]["false_alert_basis"] if rows else "honest clips"
    ax.set_xlabel(f"False alerts per hour ({basis})")
    ax.set_ylabel("Phone event recall")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Phone threshold sweep")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_ablation(clips: list[Clip], frames: dict[str, list[CachedFrame]], cfg: dict[str, Any],
                 out_dir: Path) -> list[dict[str, Any]]:
    """All features off, each feature on by itself, all on. (Detection mode is
    fixed by the cache; compare modes with --compare-modes.)"""
    names = list(cfg["features"])
    settings = [("all off", dict.fromkeys(names, False))]
    settings += [(f"only {n}", {m: m == n for m in names}) for n in names]
    settings += [("all on", dict.fromkeys(names, True))]
    rows = []
    for label, features in settings:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["features"] = features
        alerts = {clip.name: replay(frames[clip.name], run_cfg) for clip in clips}
        result = score(clips, alerts, cfg["evaluation"]["match_tolerance_s"])
        rows.append({"setting": label, "precision": result["precision"], "recall": result["recall"],
                     "f1": result["f1"], "false_alerts_per_hour": result["false_alerts_per_hour"]})
    write_csv(out_dir / "ablation.csv", rows)
    return rows


def per_condition(clips: list[Clip], alerts: dict[str, list[Alert]], cfg: dict[str, Any],
                  out_dir: Path) -> list[dict[str, Any]] | None:
    """Recall and false alerts/hour per angle / lighting value, if labelled."""
    columns = sorted({column for clip in clips for column in clip.conditions})
    if not columns:
        print("\nNote: labels.csv has no angle/lighting columns; skipping the per-condition breakdown.")
        return None
    rows = []
    for column in columns:
        for value in sorted({clip.conditions.get(column, "unknown") for clip in clips}):
            subset = [clip for clip in clips if clip.conditions.get(column, "unknown") == value]
            result = score(subset, alerts, cfg["evaluation"]["match_tolerance_s"])
            rows.append({"condition": column, "value": value, "clips": len(subset),
                         "recall": result["recall"], "false_alerts_per_hour": result["false_alerts_per_hour"]})
    write_csv(out_dir / "per_condition.csv", rows)
    return rows


def render_frame(clip: Clip, frames: list[CachedFrame], timestamp: float, cfg: dict[str, Any]):
    """The annotated, head-blurred frame closest to `timestamp`, or None."""
    if not frames:
        return None
    cached = min(frames, key=lambda f: abs(f.timestamp - timestamp))
    cap = cv2.VideoCapture(str(clip.path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, cached.frame_index)
    ok, image = cap.read()
    cap.release()
    if not ok:
        return None
    annotated = draw_frame(image, frame_output(cached, cfg), cfg)  # blurs heads first
    return resize_to_width(annotated, cfg["live"]["display_width"])


def make_gallery(clips: list[Clip], frames: dict[str, list[CachedFrame]], alerts: dict[str, list[Alert]],
                 cfg: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    """Up to evaluation.gallery_max false positives and false negatives as
    head-blurred JPEGs, plus gallery.csv with an empty `cause` column."""
    folder = out_dir / "gallery"
    folder.mkdir(parents=True, exist_ok=True)
    limit = cfg["evaluation"]["gallery_max"]
    false_positives, missed = [], []
    for clip in clips:
        _, fps, fns = match_events(alerts[clip.name], clip.events, cfg["evaluation"]["match_tolerance_s"])
        false_positives += [(clip, a) for a in fps]
        missed += [(clip, e) for e in fns]

    rows = []
    for kind, items in (("false_positive", false_positives[:limit]), ("false_negative", missed[:limit])):
        for number, (clip, item) in enumerate(items, start=1):
            if kind == "false_positive":
                time_s, tier, reason = peak_time_of(item), item.tier, item.reason
            else:
                time_s, tier = (item.start + item.end) / 2, ""
                reason = f"missed {item.event_type} (labelled {item.start:.1f}-{item.end:.1f} s)"
            image = render_frame(clip, frames[clip.name], time_s, cfg)
            name = f"{'fp' if kind == 'false_positive' else 'fn'}_{number:02d}.jpg"
            if image is not None:
                cv2.imwrite(str(folder / name), image)
            rows.append({"kind": kind, "clip": clip.name, "time_s": round(time_s, 2), "tier": tier,
                         "reason": reason, "image": name if image is not None else "", "cause": ""})
    write_csv(folder / "gallery.csv", rows,
              columns=["kind", "clip", "time_s", "tier", "reason", "image", "cause"])
    return rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def print_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """A plain aligned table on the console."""
    print(f"\n=== {title} ===")
    if not rows:
        print("(no rows)")
        return
    def cell(row: dict[str, Any], column: str) -> str:
        value = row.get(column, "")
        return "-" if value is None else str(value)

    widths = [max(len(c), *(len(cell(r, c)) for r in rows)) for c in columns]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    for r in rows:
        print("  ".join(cell(r, c).ljust(w) for c, w in zip(columns, widths)))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate alerts against labelled clips (see module docstring).")
    parser.add_argument("--video", help="One clip to evaluate.")
    parser.add_argument("--clips-dir", help="Folder with the clips listed in --labels.")
    parser.add_argument("--labels", help="labels.csv (format in the module docstring).")
    parser.add_argument("--config", help="Config file (default: config.yaml at the repo root).")
    parser.add_argument("--sweep", action="store_true", help="Phone threshold sweep -> results/sweep.*")
    parser.add_argument("--ablation", action="store_true", help="Feature ablation -> results/ablation.csv")
    parser.add_argument("--gallery", action="store_true", help="Error thumbnails -> results/gallery/")
    parser.add_argument("--compare-modes", action="store_true",
                        help=f"Compare phone detection modes {MODES} on labelled images (needs --gt).")
    parser.add_argument("--gt", help="--compare-modes: YOLO dataset folder or COCO JSON with phone boxes.")
    parser.add_argument("--images", help="--compare-modes with COCO JSON: folder with the images.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = resolve_path(cfg["paths"]["output_dir"])

    if args.compare_modes:
        if not args.gt:
            raise SystemExit("--compare-modes needs --gt (YOLO dataset folder or COCO JSON).")
        rows = compare_modes(cfg, args.gt, args.images)
        print_table("Phone detection modes", rows,
                    ["mode", "precision", "recall", "mAP50", "fps", "device", "model"])
        return

    if not args.labels or not (args.video or args.clips_dir):
        print("Nothing to evaluate: pass --labels labels.csv and --video CLIP or --clips-dir DIR.")
        print("The labels.csv format is described at the top of evaluate.py.")
        return

    video = Path(args.video) if args.video else None
    clips_dir = video.parent if video else Path(args.clips_dir)
    clips = load_labels(args.labels, clips_dir, only=video.name if video else None)
    if not clips:
        raise SystemExit(f"No rows for {'clip ' + video.name if video else 'any clip'} in {args.labels}.")

    cfg = prepare_runtime(cfg)  # resolve GPU/CPU once, so the cache key is stable
    print(f"Evaluating {len(clips)} clip(s); detection mode {detection_mode(cfg)} on {cfg['runtime']['device']}")
    frames: dict[str, list[CachedFrame]] = {}
    for clip in clips:
        if not clip.path.exists():
            raise SystemExit(f"Clip not found: {clip.path}")
        info = get_video_info(clip.path, cfg["runtime"]["fallback_fps"])
        clip.duration_s = info["frame_count"] / info["fps"] if info["fps"] else 0.0
        frames[clip.name] = cache_detections(clip, cfg, out_dir / "cache")

    tolerance = cfg["evaluation"]["match_tolerance_s"]
    alerts = {clip.name: replay(frames[clip.name], cfg) for clip in clips}

    rows = []
    for clip in clips:
        result = score([clip], alerts, tolerance)
        rows.append({"clip": clip.name, "events": len(clip.events), "alerts": result["tp"] + result["fp"],
                     **{k: result[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")}})
    overall = score(clips, alerts, tolerance)
    rows.append({"clip": "ALL", "events": sum(len(c.events) for c in clips),
                 "alerts": overall["tp"] + overall["fp"],
                 **{k: overall[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")}})
    print_table("Event evaluation (current config)", rows,
                ["clip", "events", "alerts", "tp", "fp", "fn", "precision", "recall", "f1"])
    print(f"False alerts per hour ({overall['false_alert_basis']}): {overall['false_alerts_per_hour']}")

    condition_rows = per_condition(clips, alerts, cfg, out_dir)
    if condition_rows:
        print_table("Per condition", condition_rows,
                    ["condition", "value", "clips", "recall", "false_alerts_per_hour"])
    if args.sweep:
        print_table("Threshold sweep (phone events)", run_sweep(clips, frames, cfg, out_dir),
                    ["parameter", "value", "recall", "false_alerts_per_hour", "precision", "current"])
    if args.ablation:
        print_table("Ablation", run_ablation(clips, frames, cfg, out_dir),
                    ["setting", "precision", "recall", "f1", "false_alerts_per_hour"])
    if args.gallery:
        print_table("Error gallery", make_gallery(clips, frames, alerts, cfg, out_dir),
                    ["kind", "clip", "time_s", "tier", "reason", "image"])


if __name__ == "__main__":
    main()
