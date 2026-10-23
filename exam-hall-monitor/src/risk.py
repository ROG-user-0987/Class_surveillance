"""Risk scoring: turns FrameRecords into Alerts for HUMAN review.

For every person (track_id) and every behaviour below, the scorer watches
how long the behaviour lasts:

    behaviour    present in a frame when ...               must last for
    phone        a phone is assigned to the person          phone.min_duration_s
    head_turn    head turned past pose.head_turn_threshold  pose.min_duration_s
    lean         torso past pose.lean_threshold_deg         pose.min_duration_s
    out_of_seat  box centre in no seat zone (zones set)     seats.max_outside_s

Detections flicker, so a behaviour may disappear for up to
alerts.gap_tolerance_s without restarting the timer. Once it has lasted long
enough, ONE Alert fires, with the tier from alerts.tiers. The same person +
behaviour cannot alert again until alerts.cooldown_s has passed.

An Alert is a request for an invigilator to look at a clip, never a verdict.
Reason texts describe what was observed ("Phone-like object near hands for
1.2s"), not intent. The alerts CSV has an empty review_decision column for
the human reviewer.

Run on its own (scans the video and lists the alerts):
    python -m src.risk --video data/sample.mp4
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.pose_rules import posture_flags
from src.seat_count import is_outside_seats, load_zones
from src.utils import load_config, resolve_input_video

if TYPE_CHECKING:  # type hints only; importing tracking here for real would be circular
    from src.tracking import FrameRecord

# "red" = review first, "orange" = review soon. Neither is an accusation.
Tier = Literal["red", "orange"]

# Neutral wording shown to reviewers (full) and on the video (short).
DESCRIPTIONS = {
    "phone": "Phone-like object near hands",
    "head_turn": "Head turned to the side",
    "lean": "Leaning",
    "out_of_seat": "Outside all seat zones",
}
SHORT_LABELS = {"phone": "phone", "head_turn": "head turn", "lean": "lean", "out_of_seat": "out of seat"}


@dataclass(frozen=True)
class Alert:
    """One item in the human review queue.

    Fields:
        timestamp: seconds from the start of the video when the observed
            behaviour began.
        track_id: the FrameRecord.track_id it refers to.
        tier: "red" (review first) or "orange" (review soon).
        reason: neutral description of what was observed, for the reviewer.
        clip_path: path to the saved review clip, or None if not saved.
        behaviour: "phone", "head_turn", "lean" or "out_of_seat".
    """

    timestamp: float
    track_id: int
    tier: Tier
    reason: str
    clip_path: str | None = None
    behaviour: str = ""


@dataclass
class _Streak:
    """How long one behaviour has been going on for one person."""

    start: float              # timestamp when it began
    last_seen: float          # latest timestamp it was present
    peak_conf: float = 0.0    # highest phone confidence seen (phone only)
    alerted: bool = False     # an alert was already raised for this streak


class RiskScorer:
    """Consumes FrameRecords frame by frame and emits Alerts.

    Use one instance per video (or call reset() between videos).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
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
        self.zones = load_zones(cfg)
        self.reset()

    def reset(self) -> None:
        """Forget all state (call before starting a new video)."""
        self.streaks: dict[tuple[int, str], _Streak] = {}
        self.last_alert: dict[tuple[int, str], float] = {}

    def behaviours_present(self, record: FrameRecord) -> dict[str, bool]:
        """Which behaviours this person shows in this one frame."""
        posture = posture_flags(record, self.cfg)
        return {
            "phone": record.phone_conf > 0,
            "head_turn": posture["head_turn"],
            "lean": posture["lean"],
            "out_of_seat": is_outside_seats(record, self.zones) is True,
        }

    def update(self, timestamp: float, records: list[FrameRecord]) -> list[Alert]:
        """Feed all FrameRecords of ONE processed frame; return any new Alerts.

        Call it for every processed frame, even when `records` is empty, so
        behaviours of people who left the view can time out.
        """
        new_alerts = []
        for record in records:
            for behaviour, present in self.behaviours_present(record).items():
                if not present:
                    continue
                key = (record.track_id, behaviour)
                streak = self.streaks.get(key)
                if streak is None or timestamp - streak.last_seen > self.gap_tolerance:
                    streak = self.streaks[key] = _Streak(start=timestamp, last_seen=timestamp)
                streak.last_seen = timestamp
                if behaviour == "phone":
                    streak.peak_conf = max(streak.peak_conf, record.phone_conf)

                alert = self._maybe_alert(key, streak)
                if alert is not None:
                    new_alerts.append(alert)

        # Forget behaviours that stopped (or whose person disappeared).
        self.streaks = {
            key: s for key, s in self.streaks.items() if timestamp - s.last_seen <= self.gap_tolerance
        }
        return new_alerts

    def _maybe_alert(self, key: tuple[int, str], streak: _Streak) -> Alert | None:
        track_id, behaviour = key
        duration = streak.last_seen - streak.start
        if streak.alerted or duration < self.min_duration[behaviour]:
            return None
        if streak.last_seen - self.last_alert.get(key, -math.inf) < self.cooldown:
            return None  # same person + behaviour alerted too recently

        streak.alerted = True
        self.last_alert[key] = streak.last_seen
        reason = f"{DESCRIPTIONS[behaviour]} for {duration:.1f}s"
        if behaviour == "phone":
            reason += f" (max confidence {streak.peak_conf:.2f})"
        return Alert(timestamp=streak.start, track_id=track_id, tier=self.tiers[behaviour],
                     reason=reason, behaviour=behaviour)

    def active_flags(self) -> dict[int, list[tuple[Tier, str]]]:
        """People whose behaviour has lasted long enough to be flagged right
        now, as {track_id: [(tier, short label), ...]}, for drawing."""
        flags: dict[int, list[tuple[Tier, str]]] = {}
        for (track_id, behaviour), s in self.streaks.items():
            if s.last_seen - s.start >= self.min_duration[behaviour]:
                flags.setdefault(track_id, []).append((self.tiers[behaviour], SHORT_LABELS[behaviour]))
        return flags


def format_alert(alert: Alert) -> str:
    """One readable line, e.g. '[red] 01:23 ID 4: Phone-like object near hands for 1.2s'."""
    minutes, seconds = divmod(int(alert.timestamp), 60)
    return f"[{alert.tier}] {minutes:02d}:{seconds:02d} ID {alert.track_id}: {alert.reason}"


def save_alerts_csv(alerts: list[Alert], path: str | Path) -> None:
    """Write alerts to CSV, with an empty review_decision column for the reviewer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [f.name for f in fields(Alert)] + ["review_decision"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for alert in alerts:
            writer.writerow({**asdict(alert), "review_decision": ""})


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
