"""Phone detection upgrades. STATUS: stub.

Planned:
    - SAHI sliced inference for small / far-away phones
      (phone.sahi, phone.slice_size, phone.slice_overlap in config.yaml)
    - loading our fine-tuned 1-class model (phone.class_id = 0)
    - a duration filter: a phone must persist for phone.min_duration_s

For now, plain full-frame phone detection lives in src/tracking.py.

Run on its own (once implemented):
    python -m src.detect_phone --video data/sample.mp4
"""

from __future__ import annotations

import argparse

from src.utils import load_config, resolve_input_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Phone detection with SAHI (stub).")
    parser.add_argument("--video", help="Input video (default: paths.input_video in config.yaml).")
    parser.add_argument("--config", help="Config file (default: config.yaml at the repo root).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    video = resolve_input_video(args.video, cfg)
    print(f"detect_phone.py is a stub; nothing to do for {video} yet.")
    print(f"For now run: python -m src.tracking --video {video}")


if __name__ == "__main__":
    main()
