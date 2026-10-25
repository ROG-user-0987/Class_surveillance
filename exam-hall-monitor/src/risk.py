"""Risk scoring: turns FrameRecords into Alerts for HUMAN review.

For every person (track_id) and every behaviour below, the scorer watches
how long the behaviour lasts:

    behaviour       present in a frame when ...                must last for
    phone           a phone is assigned to the person           phone.min_duration_s
    head_turn       head turned past pose.head_turn_threshold   pose.min_duration_s
    lean            torso past pose.lean_threshold_deg          pose.min_duration_s
    out_of_seat     box centre in no seat zone (zones set)      seats.max_outside_s
    neighbour_peek  head turned toward a neighbour's box        cues.peek_min_s
    object_passing  wrists of two students close together       cues.pass_min_s
    hands_hidden    head down + both wrists not visible         cues.hands_hidden_min_s

The last three (src/cues.py) and the grey state below each have their own
switch under features:, and always use EventTrackers.

Two ways of timing phone / head_turn / lean (features.temporal_hysteresis):
    off: a simple timer. Detections flicker, so a behaviour may disappear
         for up to alerts.gap_tolerance_s without restarting the timer.
    on:  src/temporal.py (EMA smoothing, hysteresis on/off thresholds, gap
         tolerance in frames). Phone follows the phone confidence; posture
         follows a score where 1.0 is exactly at the pose.* threshold.
         out_of_seat always keeps the simple timer.

With features.baseline_calibration on, head turn and lean (and head pitch
for hands_hidden) are measured relative to each student's own first seconds
(see pose_rules.PostureBaseline).

Grey "cannot assess" (features.low_visibility): while too few keypoints are
visible, POSTURE alerts for that student are paused (phone alerts are not),
and the period is logged as a grey-tier alert when it ends.

Combined alerts (features.cue_fusion): orange alerts are held for
cues.fusion_window_s. If a different orange cue fires for the same student in
that window, ONE combined orange alert is raised instead of two.
NOTE: the original code had no phone + posture fusion, so there is none to
keep; red (phone) alerts are never held back or combined.

Once a behaviour has lasted long enough, ONE Alert fires, with the tier from
alerts.tiers. The same person + behaviour cannot alert again until
alerts.cooldown_s has passed (for object passing: neither of the two students).

An Alert is a request for an invigilator to look at a clip, never a verdict.
Reason texts describe what was observed, not intent. With
features.evidence_alerts on they also give the side/direction, duration and
signal strength, e.g. "Phone near right wrist, 4.2 s, mean conf 0.71, max
0.88". The alerts CSV has alert_id, the raw numbers as JSON (signals), and
review_decision / reviewed_at columns filled in by the reviewer (app.py).

Run on its own (scans the video and lists the alerts):
    python -m src.risk --video data/sample.mp4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.cues import hands_hidden_signal, low_visibility_signal, passing_pairs, peek_target
from src.pose_rules import PostureBaseline, posture_signals, readings_from_signals
from src.seat_count import is_outside_seats, load_zones
from src.temporal import Event, EventTracker
from src.utils import LEFT_WRIST, RIGHT_WRIST, load_config, resolve_input_video

if TYPE_CHECKING:  # type hints only; importing tracking here for real would be circular
    from src.tracking import FrameRecord

# "red" = review first, "orange" = review soon, "grey" = could not assess.
# None of them is an accusation.
Tier = Literal["red", "orange", "grey"]

# Neutral wording shown to reviewers (full) and on the video (short).
DESCRIPTIONS = {
    "phone": "Phone-like object near hands",
    "head_turn": "Head turned to the side",
    "lean": "Leaning",
    "out_of_seat": "Outside all seat zones",
    "neighbour_peek": "Head turned toward a neighbour",
    "object_passing": "Possible object passing",
    "hands_hidden": "Head down with hands not visible",
    "low_visibility": "Cannot assess: too few keypoints visible",
    "combined": "Several cues together",
}
SHORT_LABELS = {"phone": "phone", "head_turn": "head turn", "lean": "lean", "out_of_seat": "out of seat",
                "neighbour_peek": "toward neighbour", "object_passing": "wrists close",
                "hands_hidden": "hands hidden", "low_visibility": "cannot assess"}
UNITS = {"phone": "confidence", "head_turn": "shoulder_widths", "lean": "degrees", "out_of_seat": "present",
         "neighbour_peek": "present", "object_passing": "present", "hands_hidden": "present",
         "low_visibility": "present"}

# Alerts paused for a student in the grey "cannot assess" state (phone is not).
POSTURE_BEHAVIOURS = {"head_turn", "lean", "neighbour_peek", "object_passing", "hands_hidden"}
CUES = ("neighbour_peek", "object_passing", "hands_hidden")

CSV_EXTRA_COLUMNS = ["review_decision", "reviewed_at"]


@dataclass(frozen=True)
class Alert:
    """One item in the human review queue.

    Fields:
        timestamp: seconds from the start of the video when the observed
            behaviour began.
        track_id: the FrameRecord.track_id it refers to (the lower id of the
            two students for object_passing).
        tier: "red" (review first), "orange" (review soon) or "grey" (could
            not assess this student for a while).
        reason: neutral description of what was observed, for the reviewer.
        clip_path: path to the saved review clip, or None if not saved.
        behaviour: "phone", "head_turn", "lean", "out_of_seat",
            "neighbour_peek", "object_passing", "hands_hidden",
            "low_visibility" or "combined".
        thumbnail_path: head-blurred JPEG of the peak frame, or None
            (only with features.evidence_alerts).
        alert_id: unique id, e.g. "exam1-A0003".
        signals: JSON string with the raw numbers behind the alert
            (duration, mean/max signal, peak time, baseline, side, method;
            for "combined": the behaviours and each part's numbers).
    """

    timestamp: float
    track_id: int
    tier: Tier
    reason: str
    clip_path: str | None = None
    behaviour: str = ""
    thumbnail_path: str | None = None
    alert_id: str = ""
    signals: str = ""


@dataclass
class _Streak:
    """How long one behaviour has been going on for one person (simple timer)."""

    start: float              # timestamp when it began
    last_seen: float          # latest timestamp it was present
    alerted: bool = False     # an alert was already raised for this streak
    count: int = 0            # frames present (for the mean)
    total: float = 0.0        # sum of the raw values (for the mean)
    max_value: float = 0.0    # highest raw value, e.g. phone confidence
    peak_time: float = 0.0
    peak_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Observation:
    """What one behaviour looks like for one person in one frame."""

    present: bool                  # past its threshold this frame (used by the simple timer)
    value: float | None            # strength in natural units (conf, shoulder widths, degrees)
    signal: float | None           # what an EventTracker follows (conf, or posture score)
    details: dict[str, Any] = field(default_factory=dict)  # side / direction / baseline


@dataclass
class _HeldAlerts:
    """Orange alerts for one student waiting for the fusion window to close."""

    release_at: float
    alerts: list[Alert]


def phone_side(record: FrameRecord, min_conf: float) -> str:
    """Which of the student's wrists is nearest the phone: "left wrist",
    "right wrist", or "hands (wrists not visible)". Left/right are the
    STUDENT's own sides (COCO keypoint convention), not the image's."""
    if record.phone_box is None:
        return "hands"
    cx = (record.phone_box[0] + record.phone_box[2]) / 2
    cy = (record.phone_box[1] + record.phone_box[3]) / 2
    best_name, best_distance = "hands (wrists not visible)", math.inf
    for index, name in ((LEFT_WRIST, "left wrist"), (RIGHT_WRIST, "right wrist")):
        x, y, conf = record.keypoints[index]
        if conf >= min_conf and math.hypot(x - cx, y - cy) < best_distance:
            best_name, best_distance = name, math.hypot(x - cx, y - cy)
    return best_name


def evidence_reason(
    behaviour: str, duration: float, mean: float, maximum: float, details: dict[str, Any]
) -> str:
    """Detailed, neutral reason text (features.evidence_alerts)."""
    if behaviour == "phone":
        return (f"Phone near {details.get('side', 'hands')}, {duration:.1f} s, "
                f"mean conf {mean:.2f}, max {maximum:.2f}")
    if behaviour == "out_of_seat":
        return f"{DESCRIPTIONS['out_of_seat']}, {duration:.1f} s"
    if behaviour == "neighbour_peek":
        neighbour = details.get("neighbour")
        target = f"student #{neighbour}" if neighbour is not None else "a neighbour"
        return f"Head turned toward {target}, {duration:.1f} s"
    if behaviour == "object_passing":
        a, b = details.get("students", ["?", "?"])
        return f"Wrists of students #{a} and #{b} close together (possible object passing), {duration:.1f} s"
    if behaviour == "low_visibility":
        return (f"Cannot assess: only {details.get('visible_keypoints', '?')} keypoints visible, "
                f"{duration:.1f} s")
    if behaviour == "hands_hidden":
        text = f"Head down with hands not visible, {duration:.1f} s"
    else:
        direction = details.get("direction")
        towards = f" toward image {direction}" if direction else " to the side"
        if behaviour == "head_turn":
            text = (f"Head turned{towards}, {duration:.1f} s, "
                    f"mean {mean:.2f}, max {maximum:.2f} shoulder widths")
        else:  # lean
            text = f"Leaning{towards}, {duration:.1f} s, mean {mean:.0f} deg, max {maximum:.0f} deg"
    if details.get("baseline") is not None:
        text += " (baseline-adjusted)"
    return text


def _plain(value: Any) -> Any:
    """Make a value JSON-friendly (numpy floats -> rounded Python floats)."""
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return round(float(value), 4)


def peak_time_of(alert: Alert) -> float:
    """When the alert's signal peaked (falls back to its start time)."""
    try:
        return float(json.loads(alert.signals)["peak_time"])
    except (ValueError, KeyError, TypeError):
        return alert.timestamp


def alert_duration(alert: Alert) -> float:
    """How long the observed behaviour lasted when the alert was raised."""
    try:
        return float(json.loads(alert.signals)["duration_s"])
    except (ValueError, KeyError, TypeError):
        return 0.0


class RiskScorer:
    """Consumes FrameRecords frame by frame and emits Alerts.

    Use one instance per video (or call reset() between videos), and call
    finish() at the end of the video to collect alerts still being held.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.features = dict(cfg.get("features") or {})
        self.use_hysteresis = bool(self.features.get("temporal_hysteresis", False))
        self.use_baseline = bool(self.features.get("baseline_calibration", False))
        self.evidence = bool(self.features.get("evidence_alerts", False))
        self.use_fusion = bool(self.features.get("cue_fusion", False))

        alerts = cfg["alerts"]
        self.gap_tolerance = alerts["gap_tolerance_s"]
        self.cooldown = alerts["cooldown_s"]
        self.tiers: dict[str, Tier] = alerts["tiers"]
        self.min_duration = {
            "phone": cfg["phone"]["min_duration_s"],
            "head_turn": cfg["pose"]["min_duration_s"],
            "lean": cfg["pose"]["min_duration_s"],
            "out_of_seat": cfg["seats"]["max_outside_s"],
        }
        self.thresholds = {"head_turn": cfg["pose"]["head_turn_threshold"],
                           "lean": cfg["pose"]["lean_threshold_deg"]}
        self.zones = load_zones(cfg)
        self.reset()

    def reset(self) -> None:
        """Forget all state (call before starting a new video)."""
        self.streaks: dict[tuple[int, str], _Streak] = {}
        self.last_alert: dict[tuple[int, str], float] = {}
        self.alert_count = 0
        self.held: dict[int, _HeldAlerts] = {}  # cue_fusion buffer per student

        self.baseline: PostureBaseline | None = None
        if self.use_baseline:
            b = self.cfg["baseline"]
            self.baseline = PostureBaseline(b["baseline_seconds"], b["min_samples"])

        # With hysteresis on, these behaviours are timed by EventTrackers
        # instead of the simple timer.
        self.event_trackers: dict[str, EventTracker] = {}
        if self.use_hysteresis:
            t = self.cfg["temporal"]
            common = {"ema_alpha": t["ema_alpha"], "max_gap_frames": t["max_gap_frames"]}
            self.event_trackers = {
                "phone": EventTracker(on_threshold=t["phone_on"], off_threshold=t["phone_off"],
                                      min_duration_s=self.min_duration["phone"], **common),
                "head_turn": EventTracker(on_threshold=t["posture_on"], off_threshold=t["posture_off"],
                                          min_duration_s=self.min_duration["head_turn"], **common),
                "lean": EventTracker(on_threshold=t["posture_on"], off_threshold=t["posture_off"],
                                     min_duration_s=self.min_duration["lean"], **common),
            }

        # Extra cues and the grey state: yes/no signals, always EventTrackers.
        self.cue_trackers: dict[str, EventTracker] = {}
        self.visibility_tracker: EventTracker | None = None
        if any(self.features.get(name) for name in (*CUES, "low_visibility")):
            t, c = self.cfg["temporal"], self.cfg["cues"]
            yes_no = {"ema_alpha": t["ema_alpha"], "on_threshold": t["cue_on"],
                      "off_threshold": t["cue_off"], "max_gap_frames": t["max_gap_frames"]}
            minimum = {"neighbour_peek": c["peek_min_s"], "object_passing": c["pass_min_s"],
                       "hands_hidden": c["hands_hidden_min_s"]}
            for name in CUES:
                if self.features.get(name):
                    self.cue_trackers[name] = EventTracker(min_duration_s=minimum[name], **yes_no)
            if self.features.get("low_visibility"):
                self.visibility_tracker = EventTracker(min_duration_s=c["low_vis_min_s"], **yes_no)

    # -- one frame ----------------------------------------------------------

    def observe(self, record: FrameRecord) -> dict[str, _Observation]:
        """How each original behaviour looks for this person in this one frame."""
        signals = posture_signals(record, self.cfg)
        if self.baseline is not None:
            self.baseline.update(record.track_id, record.timestamp, signals)
        readings = readings_from_signals(record.track_id, signals, self.cfg, self.baseline)

        observations = {
            "phone": _Observation(
                present=record.phone_conf > 0,
                value=record.phone_conf,
                signal=record.phone_conf,
                details={"side": phone_side(record, self.cfg["association"]["keypoint_conf"])}
                if record.phone_conf > 0 else {},
            ),
        }
        for behaviour in ("head_turn", "lean"):
            reading = readings[behaviour]
            observations[behaviour] = _Observation(
                present=reading.over_threshold,
                value=reading.deviation,
                signal=reading.score,
                details={"direction": reading.direction, "baseline": reading.baseline,
                         "raw": reading.value},
            )
        outside = is_outside_seats(record, self.zones) is True
        observations["out_of_seat"] = _Observation(present=outside, value=1.0 if outside else None,
                                                   signal=None)
        return observations

    def behaviours_present(self, record: FrameRecord) -> dict[str, bool]:
        """Which behaviours this person shows in this one frame."""
        return {behaviour: obs.present for behaviour, obs in self.observe(record).items()}

    def update(self, timestamp: float, records: list[FrameRecord]) -> list[Alert]:
        """Feed all FrameRecords of ONE processed frame; return any new Alerts.

        Call it for every processed frame, even when `records` is empty, so
        behaviours of people who left the view can time out.
        """
        produced: list[Alert] = []

        # --- 1. Grey "cannot assess" state first: it pauses posture alerts --
        grey = self._update_visibility(timestamp, records, produced)

        # --- 2. Original behaviours ------------------------------------------
        event_values: dict[str, dict[int, float | None]] = {b: {} for b in self.event_trackers}
        event_details: dict[str, dict[int, dict[str, Any]]] = {b: {} for b in self.event_trackers}

        for record in records:
            for behaviour, obs in self.observe(record).items():
                if behaviour in self.event_trackers:
                    event_values[behaviour][record.track_id] = obs.signal
                    event_details[behaviour][record.track_id] = obs.details
                    continue
                if not obs.present:
                    continue

                # simple timer
                key = (record.track_id, behaviour)
                streak = self.streaks.get(key)
                if streak is None or timestamp - streak.last_seen > self.gap_tolerance:
                    streak = self.streaks[key] = _Streak(start=timestamp, last_seen=timestamp)
                streak.last_seen = timestamp
                value = obs.value or 0.0
                streak.count += 1
                streak.total += value
                if streak.count == 1 or value > streak.max_value:
                    streak.max_value, streak.peak_time = value, timestamp
                    streak.peak_details = dict(obs.details)

                if behaviour in POSTURE_BEHAVIOURS and record.track_id in grey:
                    continue  # paused while this student can't be assessed
                alert = self._maybe_alert(key, streak)
                if alert is not None:
                    produced.append(alert)

        # Forget behaviours that stopped (or whose person disappeared).
        self.streaks = {
            key: s for key, s in self.streaks.items() if timestamp - s.last_seen <= self.gap_tolerance
        }

        # EMA + hysteresis events (features.temporal_hysteresis)
        for behaviour, tracker in self.event_trackers.items():
            for event in tracker.update(timestamp, event_values[behaviour], event_details[behaviour]):
                alert = None
                if not (behaviour in POSTURE_BEHAVIOURS and event.track_id in grey):
                    alert = self._alert_from_event(behaviour, event)
                if alert is None:
                    event.fired = False  # blocked (cooldown / grey): re-check on its next active frame
                else:
                    produced.append(alert)
            tracker.pop_ended()  # ended events aren't needed here

        # --- 3. Extra cues (src/cues.py) --------------------------------------
        produced += self._update_cues(timestamp, records, grey)

        # --- 4. Combine orange alerts (cue_fusion) and number them ------------
        return self._release(produced, timestamp)

    def _update_visibility(self, timestamp: float, records: list[FrameRecord], produced: list[Alert]) -> set[int]:
        """Update the grey state; log finished grey periods; return the
        track_ids whose posture alerts are paused right now.

        Posture is paused as soon as low visibility is detected (an event has
        started), not only after cues.low_vis_min_s: otherwise a posture cue
        with the same minimum duration could slip out first. The minimum
        duration only decides whether the period is logged as a grey alert."""
        if self.visibility_tracker is None:
            return set()
        values, details = {}, {}
        for record in records:
            values[record.track_id], details[record.track_id] = low_visibility_signal(record, self.cfg)
        self.visibility_tracker.update(timestamp, values, details)
        for event in self.visibility_tracker.pop_ended():
            if event.fired:  # lasted long enough to count as a grey period
                alert = self._grey_alert(event)
                if alert is not None:
                    produced.append(alert)
        return set(self.visibility_tracker.events)  # every started low-visibility event

    def _update_cues(self, timestamp: float, records: list[FrameRecord], grey: set[int]) -> list[Alert]:
        if not self.cue_trackers:
            return []
        values: dict[str, dict[Any, float | None]] = {name: {} for name in self.cue_trackers}
        details: dict[str, dict[Any, dict[str, Any]]] = {name: {} for name in self.cue_trackers}
        for record in records:
            if "neighbour_peek" in self.cue_trackers:
                signal, neighbour = peek_target(record, records, self.cfg)
                values["neighbour_peek"][record.track_id] = signal
                details["neighbour_peek"][record.track_id] = {"neighbour": neighbour}
            if "hands_hidden" in self.cue_trackers:
                pitch_baseline = self.baseline.get(record.track_id, "head_pitch") if self.baseline else None
                signal, info = hands_hidden_signal(record, self.cfg, pitch_baseline)
                values["hands_hidden"][record.track_id] = signal
                details["hands_hidden"][record.track_id] = info
        if "object_passing" in self.cue_trackers:
            values["object_passing"] = passing_pairs(records, self.cfg)

        alerts = []
        for name, tracker in self.cue_trackers.items():
            for event in tracker.update(timestamp, values[name], details[name]):
                students = event.track_id if isinstance(event.track_id, tuple) else (event.track_id,)
                alert = None
                if not any(s in grey for s in students):
                    alert = self._alert_from_cue(name, event, students)
                if alert is None:
                    event.fired = False  # blocked (cooldown / grey): re-check on its next active frame
                else:
                    alerts.append(alert)
            tracker.pop_ended()
        return alerts

    # -- alerts -------------------------------------------------------------

    def _maybe_alert(self, key: tuple[int, str], streak: _Streak) -> Alert | None:
        track_id, behaviour = key
        duration = streak.last_seen - streak.start
        if streak.alerted or duration < self.min_duration[behaviour]:
            return None
        if streak.last_seen - self.last_alert.get(key, -math.inf) < self.cooldown:
            return None  # same person + behaviour alerted too recently

        streak.alerted = True
        self.last_alert[key] = streak.last_seen
        mean = streak.total / streak.count if streak.count else 0.0
        return self._build_alert(behaviour, track_id, streak.start, duration, mean,
                                 streak.max_value, streak.peak_time, streak.peak_details, "timer")

    def _alert_from_event(self, behaviour: str, event: Event) -> Alert | None:
        key = (event.track_id, behaviour)
        if event.last_active - self.last_alert.get(key, -math.inf) < self.cooldown:
            return None  # same person + behaviour alerted too recently
        self.last_alert[key] = event.last_active

        # Posture events follow a score (1.0 = threshold); report natural units.
        scale = self.thresholds.get(behaviour, 1.0)
        return self._build_alert(behaviour, event.track_id, event.start, event.duration,
                                 event.mean_value * scale, event.max_value * scale,
                                 event.peak_time, event.peak_details, "hysteresis")

    def _alert_from_cue(self, name: str, event: Event, students: tuple[int, ...]) -> Alert | None:
        # Per-student cooldown: for a pair, neither student may have had this cue recently.
        if any(event.last_active - self.last_alert.get((s, name), -math.inf) < self.cooldown
               for s in students):
            return None
        for s in students:
            self.last_alert[(s, name)] = event.last_active
        details = dict(event.peak_details)
        if len(students) > 1:
            details["students"] = list(students)
        return self._build_alert(name, students[0], event.start, event.duration, event.mean_value,
                                 event.max_value, event.peak_time, details, "hysteresis")

    def _grey_alert(self, event: Event) -> Alert | None:
        key = (event.track_id, "low_visibility")
        if event.start - self.last_alert.get(key, -math.inf) < self.cooldown:
            return None
        self.last_alert[key] = event.last_active
        return self._build_alert("low_visibility", event.track_id, event.start, event.duration,
                                 event.mean_value, event.max_value, event.peak_time,
                                 event.peak_details, "hysteresis")

    def _build_alert(
        self, behaviour: str, track_id: int, start: float, duration: float, mean: float,
        maximum: float, peak_time: float, details: dict[str, Any], method: str,
    ) -> Alert:
        if self.evidence:
            reason = evidence_reason(behaviour, duration, mean, maximum, details)
        else:
            reason = f"{DESCRIPTIONS[behaviour]} for {duration:.1f}s"
            if behaviour == "phone":
                reason += f" (max confidence {maximum:.2f})"

        signals = {"method": method, "duration_s": duration, "mean": mean, "max": maximum,
                   "units": UNITS[behaviour], "peak_time": peak_time,
                   "baseline_adjusted": details.get("baseline") is not None, **details}
        return Alert(
            timestamp=start, track_id=track_id, tier=self.tiers[behaviour], reason=reason,
            behaviour=behaviour, signals=json.dumps({k: _plain(v) for k, v in signals.items()}),
        )

    def _release(self, produced: list[Alert], timestamp: float, flush: bool = False) -> list[Alert]:
        """Hold orange alerts for the fusion window (features.cue_fusion),
        release those whose window has closed (all of them if flush), and give
        every released alert its id."""
        out = []
        window = self.cfg["cues"]["fusion_window_s"] if self.use_fusion else 0.0
        for alert in produced:
            if not self.use_fusion or alert.tier != "orange":
                out.append(alert)
            elif alert.track_id in self.held:
                self.held[alert.track_id].alerts.append(alert)
            else:
                self.held[alert.track_id] = _HeldAlerts(release_at=timestamp + window, alerts=[alert])
        for track_id, group in list(self.held.items()):
            if flush or timestamp >= group.release_at:
                out.append(self._combine(group.alerts))
                del self.held[track_id]

        numbered = []
        for alert in out:
            self.alert_count += 1
            numbered.append(replace(alert, alert_id=f"A{self.alert_count:04d}"))
        return numbered

    def _combine(self, alerts: list[Alert]) -> Alert:
        """One alert, or ONE combined orange alert for several cues of one student."""
        if len(alerts) == 1:
            return alerts[0]
        start = min(a.timestamp for a in alerts)
        end = max(a.timestamp + alert_duration(a) for a in alerts)
        signals = {"method": "fusion", "combined": [a.behaviour for a in alerts],
                   "duration_s": round(end - start, 4), "peak_time": peak_time_of(alerts[0]),
                   "parts": [json.loads(a.signals) for a in alerts]}
        return Alert(timestamp=start, track_id=alerts[0].track_id, tier=self.tiers["combined"],
                     reason="Combined: " + "; ".join(a.reason for a in alerts),
                     behaviour="combined", signals=json.dumps(signals))

    def finish(self, timestamp: float) -> list[Alert]:
        """End of the video: log grey periods still open and release every
        held alert. Returns the alerts produced."""
        produced: list[Alert] = []
        if self.visibility_tracker is not None:
            for event in self.visibility_tracker.pop_ended() + self.visibility_tracker.end_all():
                if event.fired:
                    alert = self._grey_alert(event)
                    if alert is not None:
                        produced.append(alert)
        return self._release(produced, timestamp, flush=True)

    def active_flags(self) -> dict[int, list[tuple[Tier, str]]]:
        """People whose behaviour has lasted long enough to be flagged right
        now, as {track_id: [(tier, short label), ...]}, for drawing."""
        flags: dict[int, list[tuple[Tier, str]]] = {}

        def add(track_id: int, behaviour: str) -> None:
            flags.setdefault(track_id, []).append((self.tiers[behaviour], SHORT_LABELS[behaviour]))

        for (track_id, behaviour), s in self.streaks.items():
            if s.last_seen - s.start >= self.min_duration[behaviour]:
                add(track_id, behaviour)
        for behaviour, tracker in {**self.event_trackers, **self.cue_trackers}.items():
            for event in tracker.ongoing():
                students = event.track_id if isinstance(event.track_id, tuple) else (event.track_id,)
                for s in students:
                    add(s, behaviour)
        if self.visibility_tracker is not None:
            for event in self.visibility_tracker.ongoing():
                add(event.track_id, "low_visibility")
        return flags


def format_alert(alert: Alert) -> str:
    """One readable line, e.g. '[red] 01:23 ID 4: Phone-like object near hands for 1.2s'."""
    minutes, seconds = divmod(int(alert.timestamp), 60)
    return f"[{alert.tier}] {minutes:02d}:{seconds:02d} ID {alert.track_id}: {alert.reason}"


# ---------------------------------------------------------------------------
# Alerts CSV
# ---------------------------------------------------------------------------

def save_alerts_csv(alerts: list[Alert], path: str | Path) -> None:
    """Write alerts to CSV (alert_id first), with empty review_decision and
    reviewed_at columns for the reviewer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["alert_id"] + [f.name for f in fields(Alert) if f.name != "alert_id"] + CSV_EXTRA_COLUMNS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for alert in alerts:
            writer.writerow({**asdict(alert), **{c: "" for c in CSV_EXTRA_COLUMNS}})


def read_alerts_csv(path: str | Path) -> list[dict[str, str]]:
    """All rows of an alerts CSV as dicts (missing review columns added empty)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for column in CSV_EXTRA_COLUMNS:
            row.setdefault(column, "")
    return rows


def update_review_decision(path: str | Path, alert_id: str, decision: str) -> None:
    """Store a reviewer's decision ("confirmed", "dismissed" or "" to clear)
    for one alert in the CSV, with the time it was made (reviewed_at)."""
    if decision not in ("confirmed", "dismissed", ""):
        raise ValueError(f"decision must be confirmed / dismissed / empty, got {decision!r}")
    rows = read_alerts_csv(path)
    for row in rows:
        if row["alert_id"] == alert_id:
            row["review_decision"] = decision
            row["reviewed_at"] = datetime.now().isoformat(timespec="seconds") if decision else ""
    columns = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a video and list review alerts.")
    parser.add_argument("--video", help="Input video (default: paths.input_video in config.yaml).")
    parser.add_argument("--config", help="Config file (default: config.yaml at the repo root).")
    args = parser.parse_args()

    from src.tracking import print_summary, run_on_video  # here to avoid a circular import

    cfg = load_config(args.config)
    video = resolve_input_video(args.video, cfg)
    _, summary = run_on_video(video, cfg)
    print_summary(summary)


if __name__ == "__main__":
    main()
