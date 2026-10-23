"""Posture rules from keypoints. STATUS: stub.

Planned: read FrameRecords (src/tracking.py) and measure, per track_id,
    - head turn: nose offset from the shoulder midpoint / shoulder width
      (pose.head_turn_threshold)
    - lean: torso angle away from vertical (pose.lean_threshold_deg)
    - only report a posture that lasts >= pose.min_duration_s

These are measurements for risk.py, not judgements.

Run on its own (once implemented):
    python -m src.pose_rules --video data/sample.mp4
"""

from __future__ import annotations

import argparse

from src.utils import load_config, resolve_input_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-turn / lean rules (stub).")
    parser.add_argument("--video", help="Input video (default: paths.input_video in config.yaml).")
    parser.add_argument("--config", help="Config file (default: config.yaml at the repo root).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    video = resolve_input_video(args.video, cfg)
    print(f"pose_rules.py is a stub; nothing to do for {video} yet.")
    print(f"Pose settings from config: {cfg['pose']}")


if __name__ == "__main__":
    main()
