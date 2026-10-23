"""Seat zones and head count. STATUS: stub.

Planned:
    - count tracked people vs seats.expected_count
    - map each track_id to a seat polygon from seats.zones
    - report time spent outside one's seat beyond seats.max_outside_s

Run on its own (once implemented):
    python -m src.seat_count --video data/sample.mp4
"""

from __future__ import annotations

import argparse

from src.utils import load_config, resolve_input_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Seat zones and people count (stub).")
    parser.add_argument("--video", help="Input video (default: paths.input_video in config.yaml).")
    parser.add_argument("--config", help="Config file (default: config.yaml at the repo root).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    video = resolve_input_video(args.video, cfg)
    print(f"seat_count.py is a stub; nothing to do for {video} yet.")
    print(f"Seat settings from config: {cfg['seats']}")


if __name__ == "__main__":
    main()
