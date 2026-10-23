"""Evaluation against labelled clips. STATUS: stub.

Planned: compare Alerts with hand-labelled events (precision / recall per
tier, false alerts per minute) and plot results with pandas + matplotlib.

Run on its own (once implemented):
    python evaluate.py --video data/sample.mp4 --labels data/sample_labels.csv
"""

from __future__ import annotations

import argparse

from src.utils import load_config, resolve_input_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate alerts against labels (stub).")
    parser.add_argument("--video", help="Input video (default: paths.input_video in config.yaml).")
    parser.add_argument("--labels", help="CSV of hand-labelled events (format TBD).")
    parser.add_argument("--config", help="Config file (default: config.yaml at the repo root).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    video = resolve_input_video(args.video, cfg)
    print(f"evaluate.py is a stub; nothing to evaluate for {video} yet.")


if __name__ == "__main__":
    main()
