"""Risk scoring: turns FrameRecords into Alerts for HUMAN review.

STATUS: interface only. The types below are fixed; RiskScorer has no logic yet.

An Alert is a request for an invigilator to look at a clip, never a verdict.
Reason texts must describe what was observed ("phone-like object near hands
for 3.2 s"), not intent ("cheating").

Planned logic (not implemented yet; all numbers come from config.yaml):
    - red:    a phone is assigned to the same track_id for >= phone.min_duration_s
    - orange: a pose rule (head turn / lean) holds for >= pose.min_duration_s,
              or a person is outside their seat zone for > seats.max_outside_s
    - at most one alert per (track_id, reason) every alerts.cooldown_s seconds
    - each alert gets a short clip (alerts.clip_before_s / clip_after_s)

Run on its own (prints the interface for now):
    python -m src.risk --video data/sample.mp4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Literal

from src.tracking import FrameRecord
from src.utils import load_config, resolve_input_video

# "red" = review first, "orange" = review when possible. Neither is an accusation.
Tier = Literal["red", "orange"]


@dataclass(frozen=True)
class Alert:
    """One item in the human review queue.

    Fields:
        timestamp: seconds from the start of the video when the observed
            behaviour began.
        track_id: the FrameRecord.track_id it refers to.
        tier: "red" (higher review priority) or "orange" (lower).
        reason: neutral description of what was observed, for the reviewer.
        clip_path: path to the saved review clip, or None if not saved yet.
    """

    timestamp: float
    track_id: int
    tier: Tier
    reason: str
    clip_path: str | None = None


class RiskScorer:
    """Consumes FrameRecords frame by frame and emits Alerts. (stub)"""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Read the phone / pose / seats / alerts sections of the config."""
        self.cfg = cfg

    def update(self, records: list[FrameRecord]) -> list[Alert]:
        """Feed all FrameRecords of ONE processed frame; return any new Alerts."""
        raise NotImplementedError("RiskScorer.update is not implemented yet")

    def finish(self) -> list[Alert]:
        """Call at the end of the video to flush events still in progress."""
        raise NotImplementedError("RiskScorer.finish is not implemented yet")

    def reset(self) -> None:
        """Forget all state (call before starting a new video)."""
        raise NotImplementedError("RiskScorer.reset is not implemented yet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score FrameRecords into review alerts (stub).")
    parser.add_argument("--video", help="Input video (default: paths.input_video in config.yaml).")
    parser.add_argument("--config", help="Config file (default: config.yaml at the repo root).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    video = resolve_input_video(args.video, cfg)
    print(f"risk.py: interface only, no scoring yet for {video}.")
    print("  Input : FrameRecord(frame_index, timestamp, track_id, person_box, keypoints, phone_conf, phone_box)")
    print("  Output: Alert(timestamp, track_id, tier 'red'|'orange', reason, clip_path)")
    print(f"  Alert settings from config: {cfg['alerts']}")


if __name__ == "__main__":
    main()
