"""Seat zones: is a person inside any seat zone?

Seat zones are polygons in pixel coordinates of the video, listed under
seats.zones in config.yaml. With no zones the out-of-seat rule is simply off.
A person counts as "in a seat" when the centre of their box is inside any zone.

How to make zones: run src.tracking once, open a frame of
results/<video>_tracked.mp4 in any image viewer that shows pixel coordinates
(e.g. Paint), and note the corners of each seat area.

Run on its own (prints time spent outside seat zones per person):
    python -m src.seat_count --video data/sample.mp4
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.utils import load_config, resolve_input_video

if TYPE_CHECKING:  # type hints only; importing tracking here for real would be circular
    from src.tracking import FrameRecord


def load_zones(cfg: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    """Seat zones from config as (zone_id, polygon) pairs; [] if none are set."""
    return [
        (str(zone["id"]), np.array(zone["polygon"], dtype=np.float32))
        for zone in (cfg["seats"]["zones"] or [])
    ]


def find_zone(point: tuple[float, float], zones: list[tuple[str, np.ndarray]]) -> str | None:
    """The id of the zone that contains `point`, or None."""
    for zone_id, polygon in zones:
        if cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0:
            return zone_id
    return None


def is_outside_seats(record: FrameRecord, zones: list[tuple[str, np.ndarray]]) -> bool | None:
    """True if the person's box centre is in no seat zone; None if no zones are set."""
    if not zones:
        return None
    x1, y1, x2, y2 = record.person_box
    return find_zone(((x1 + x2) / 2, (y1 + y2) / 2), zones) is None


def main() -> None:
    parser = argparse.ArgumentParser(description="Time spent outside seat zones per person.")
    parser.add_argument("--video", help="Input video (default: paths.input_video in config.yaml).")
    parser.add_argument("--config", help="Config file (default: config.yaml at the repo root).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    video = resolve_input_video(args.video, cfg)
    zones = load_zones(cfg)
    if not zones:
        print("No seat zones in config.yaml (seats.zones is empty), so the out-of-seat rule is off.")
        print("See the docstring at the top of src/seat_count.py for how to add zones.")
        return

    from src.tracking import run_on_video  # imported here to avoid a circular import

    records, summary = run_on_video(video, cfg)
    seconds_per_frame = summary["every_nth"] / summary["video_fps"]

    outside: dict[int, float] = {}
    occupancy: dict[str, set[int]] = {zone_id: set() for zone_id, _ in zones}
    for record in records:
        x1, y1, x2, y2 = record.person_box
        zone_id = find_zone(((x1 + x2) / 2, (y1 + y2) / 2), zones)
        if zone_id is None:
            outside[record.track_id] = outside.get(record.track_id, 0.0) + seconds_per_frame
        else:
            occupancy[zone_id].add(record.track_id)

    print("\nTrack IDs seen in each seat zone:")
    for zone_id, ids in occupancy.items():
        print(f"  {zone_id}: {sorted(ids) or '-'}")
    print(f"\nSeconds outside all seat zones (flag after {cfg['seats']['max_outside_s']}s in a row):")
    for track_id, seconds in sorted(outside.items()):
        print(f"  ID {track_id}: {seconds:.1f}s")


if __name__ == "__main__":
    main()
