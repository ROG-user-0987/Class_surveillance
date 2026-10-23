"""Shared helpers used by every module.

Contents:
    - load_config / resolve_path / resolve_input_video: read config.yaml and
      turn the paths in it into absolute paths
    - get_video_info / read_video / make_video_writer: simple OpenCV video I/O
    - drawing helpers: boxes, labels, skeletons, lines
    - blur_head: privacy blur built from YOLOv8-Pose head keypoints
      (there is NO face detector and NO face recognition anywhere)

Run on its own to check that a video opens and the config loads:
    python -m src.utils --video data/sample.mp4
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import yaml

# The repo root is the folder that holds config.yaml (one level above src/).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

# COCO keypoint indices produced by YOLOv8-Pose (17 points per person).
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
HEAD_KEYPOINTS = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]
WRIST_KEYPOINTS = [LEFT_WRIST, RIGHT_WRIST]

# Pairs of keypoint indices joined by a line when drawing a skeleton.
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),        # legs and hips
    (5, 11), (6, 12), (5, 6),                                # torso
    (5, 7), (7, 9), (6, 8), (8, 10),                         # arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),  # head
]

# Colours are BGR (OpenCV order).
PHONE_COLOR = (255, 255, 0)        # cyan
UNASSIGNED_COLOR = (160, 160, 160)  # grey
TEXT_COLOR = (255, 255, 255)        # white
ZONE_COLOR = (200, 200, 200)        # light grey seat outlines
TIER_COLORS = {"red": (0, 0, 255), "orange": (0, 140, 255)}  # review priority colours


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml (or another YAML file) into a plain dict."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    """Turn a path from config.yaml into an absolute path.

    Relative paths are taken relative to the repo root, so the code works no
    matter which folder you run it from.
    """
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_input_video(cli_video: str | None, cfg: dict[str, Any]) -> Path:
    """Pick the input video: --video from the command line wins, else config.

    Exits with a clear message if no video was given or the file is missing.
    """
    if cli_video:
        video = Path(cli_video)  # relative to where you ran the command
    elif cfg["paths"].get("input_video"):
        video = resolve_path(cfg["paths"]["input_video"])
    else:
        raise SystemExit(
            "No input video. Pass --video path/to/clip.mp4 "
            "or set paths.input_video in config.yaml."
        )
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    return video


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def get_video_info(path: str | Path, fallback_fps: float) -> dict[str, Any]:
    """Return fps, width, height and frame_count of a video file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or fallback_fps  # some files report 0
    info = {
        "fps": float(fps),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def read_video(
    path: str | Path, every_nth: int, fallback_fps: float
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_seconds, frame) for every Nth frame.

    frame_index always counts ALL frames of the original video, so with
    every_nth=2 you get indices 0, 2, 4, ... and timestamps stay correct.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or fallback_fps
    frame_index = 0
    try:
        while True:
            if frame_index % every_nth != 0:
                # grab() skips a frame without fully decoding it (faster).
                if not cap.grab():
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame_index, frame_index / fps, frame
            frame_index += 1
    finally:
        cap.release()


def make_video_writer(
    path: str | Path, fps: float, width: int, height: int
) -> cv2.VideoWriter:
    """Create an .mp4 writer, making the output folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {path}")
    return writer


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    """Shrink a frame to `width` pixels wide, keeping its shape (never enlarges)."""
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


@dataclass
class _OpenClip:
    writer: cv2.VideoWriter
    frames_left: int


class ClipRecorder:
    """Saves a short clip around each alert for the human reviewer.

    It keeps the last `before_s` seconds of frames in memory (as small JPEGs,
    so memory stays low). When start_clip() is called, those frames are
    written first, so the clip begins BEFORE the alert, and then the next
    `after_s` seconds are added as they arrive.

    Only pass frames that are already annotated with heads blurred.
    """

    def __init__(
        self, folder: str | Path, fps: float, before_s: float, after_s: float,
        width: int, jpeg_quality: int,
    ) -> None:
        self.folder = Path(folder)
        self.fps = fps
        self.width = width
        self.jpeg_quality = jpeg_quality
        self.recent: deque[np.ndarray] = deque(maxlen=max(1, round(before_s * fps)))
        self.after_frames = max(1, round(after_s * fps))
        self.open_clips: list[_OpenClip] = []

    def add_frame(self, frame: np.ndarray) -> None:
        """Call once per processed frame."""
        small = resize_to_width(frame, self.width)

        # Feed clips that are still recording their "after" part.
        for clip in self.open_clips:
            clip.writer.write(small)
            clip.frames_left -= 1
        for clip in [c for c in self.open_clips if c.frames_left <= 0]:
            clip.writer.release()
            self.open_clips.remove(clip)

        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if ok:
            self.recent.append(jpg)

    def start_clip(self, name: str) -> Path:
        """Start a clip called <name>.mp4 (call right after add_frame)."""
        frames = [cv2.imdecode(jpg, cv2.IMREAD_COLOR) for jpg in self.recent]
        h, w = frames[-1].shape[:2]
        path = self.folder / f"{name}.mp4"
        writer = make_video_writer(path, self.fps, w, h)
        for frame in frames:
            writer.write(frame)
        self.open_clips.append(_OpenClip(writer, self.after_frames))
        return path

    def close(self) -> None:
        """Finish all clips (the last ones may be shorter if the video ended)."""
        for clip in self.open_clips:
            clip.writer.release()
        self.open_clips.clear()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def color_for_id(track_id: int | None) -> tuple[int, int, int]:
    """A stable, distinct-ish colour per track ID (grey if no ID yet)."""
    if track_id is None:
        return UNASSIGNED_COLOR
    rng = np.random.default_rng(track_id)  # same id -> same colour every frame
    b, g, r = rng.integers(64, 256, size=3)
    return int(b), int(g), int(r)


def to_point(xy: np.ndarray | tuple[float, float]) -> tuple[int, int]:
    """Convert an (x, y) float pair into integer pixel coords for OpenCV."""
    return int(round(xy[0])), int(round(xy[1]))


def draw_box(
    frame: np.ndarray,
    box: np.ndarray,
    color: tuple[int, int, int],
    label: str | None = None,
    thickness: int = 2,
) -> None:
    """Draw an [x1, y1, x2, y2] box with an optional label above it."""
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y1 - th - 6, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 4, top + th + 6), color, -1)
        cv2.putText(frame, label, (x1 + 2, top + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    color: tuple[int, int, int],
    min_conf: float,
) -> None:
    """Draw the 17-point COCO skeleton, skipping low-confidence keypoints."""
    for a, b in SKELETON:
        if keypoints[a, 2] >= min_conf and keypoints[b, 2] >= min_conf:
            cv2.line(frame, to_point(keypoints[a]), to_point(keypoints[b]),
                     color, 2, cv2.LINE_AA)
    for x, y, conf in keypoints:
        if conf >= min_conf:
            cv2.circle(frame, to_point((x, y)), 3, color, -1, cv2.LINE_AA)


def draw_line(
    frame: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    """Draw a straight line between two (x, y) points."""
    cv2.line(frame, to_point(start), to_point(end), color, 2, cv2.LINE_AA)


def draw_polygon(
    frame: np.ndarray, polygon: np.ndarray, color: tuple[int, int, int], label: str
) -> None:
    """Draw a closed polygon outline (e.g. a seat zone) with a small label."""
    points = polygon.astype(np.int32)
    cv2.polylines(frame, [points], True, color, 1, cv2.LINE_AA)
    cv2.putText(frame, label, (int(points[0][0]) + 3, int(points[0][1]) + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _outlined_text(frame: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    """White text with a dark outline, readable on any background."""
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                TEXT_COLOR, 1, cv2.LINE_AA)


def draw_header(frame: np.ndarray, text: str) -> None:
    """Write a line of text in the top-left corner of the frame."""
    _outlined_text(frame, text, (10, 24))


def draw_footer(frame: np.ndarray, text: str) -> None:
    """Write a line of text in the bottom-left corner of the frame."""
    _outlined_text(frame, text, (10, frame.shape[0] - 12))


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

def blur_head(
    frame: np.ndarray,
    keypoints: np.ndarray,
    *,
    min_conf: float,
    box_scale: float,
    min_width_fraction: float,
    fallback_fraction: float,
    kernel_size: int,
    person_box: np.ndarray | None = None,
) -> np.ndarray:
    """Gaussian-blur one person's head region, in place, and return the frame.

    The head box is built ONLY from the pose keypoints (nose, eyes, ears).
    No face detector or face recognition is used.

    Args:
        frame: BGR image, modified in place.
        keypoints: (17, 3) array of [x, y, conf] from YOLOv8-Pose.
        min_conf: head keypoints below this confidence are ignored.
        box_scale: box side = spread of the visible head points x box_scale.
        min_width_fraction: box side is at least this x person box width
            (only used when person_box is given).
        fallback_fraction: if fewer than 2 head points are visible, blur this
            top fraction of person_box instead, so nobody is left unblurred.
        kernel_size: Gaussian kernel size in pixels (forced to be odd).
        person_box: optional [x1, y1, x2, y2] of the person.
    """
    head = keypoints[HEAD_KEYPOINTS]
    visible = head[head[:, 2] >= min_conf][:, :2]  # keep only x, y

    if len(visible) >= 2:
        # Square box centred on the visible head points.
        cx, cy = visible.mean(axis=0)
        spread = max(np.ptp(visible[:, 0]), np.ptp(visible[:, 1]))
        side = spread * box_scale
        if person_box is not None:
            side = max(side, (person_box[2] - person_box[0]) * min_width_fraction)
        x1, y1 = cx - side / 2, cy - side / 2
        x2, y2 = cx + side / 2, cy + side / 2
    elif person_box is not None:
        # Head turned away or hidden: blur the top of the person box instead.
        x1, y1, x2, _ = person_box
        y2 = y1 + (person_box[3] - person_box[1]) * fallback_fraction
    else:
        return frame

    # Clip the box to the image and blur that region.
    h, w = frame.shape[:2]
    x1, x2 = int(max(0, x1)), int(min(w, x2))
    y1, y2 = int(max(0, y1)), int(min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return frame

    # Big heads (close to the camera) get a proportionally bigger kernel.
    k = max(kernel_size, min(x2 - x1, y2 - y1) // 2) | 1  # "| 1" makes it odd
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (k, k), 0)
    return frame


# ---------------------------------------------------------------------------
# Command line: quick sanity check of a video + config
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Check that a video opens and config.yaml loads.")
    parser.add_argument("--video", help="Input video (default: paths.input_video in config.yaml).")
    parser.add_argument("--config", help="Config file (default: config.yaml at the repo root).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    video = resolve_input_video(args.video, cfg)
    runtime = cfg["runtime"]

    info = get_video_info(video, runtime["fallback_fps"])
    print(f"Config sections: {', '.join(cfg)}")
    print(f"Video: {video}")
    print(f"  {info['width']}x{info['height']} @ {info['fps']:.1f} fps, "
          f"{info['frame_count']} frames")

    count, last_t = 0, 0.0
    for _, timestamp, _ in read_video(video, runtime["every_nth_frame"], runtime["fallback_fps"]):
        count += 1
        last_t = timestamp
    print(f"  read {count} frames (every {runtime['every_nth_frame']}), last timestamp {last_t:.1f}s")


if __name__ == "__main__":
    main()
