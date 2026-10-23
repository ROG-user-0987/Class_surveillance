"""Live scanning: watch a video being scanned while it plays.

All viewers use VideoScan from src/tracking.py, so the detections and the
saved annotated video are identical whichever one you use:
    - Colab / Jupyter notebook:  play_in_notebook(video, cfg)
          (see notebooks/colab_live_demo.ipynb)
    - Streamlit app with an upload button:  streamlit run app.py
    - OpenCV window (local machine with a screen; press q to stop):
          python -m src.live --video data/sample.mp4

Heads are blurred before any frame is shown or saved.
"""

from __future__ import annotations

import argparse
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.tracking import LiveFrame, VideoScan, print_summary
from src.utils import load_config, resolve_input_video


class RealtimePacer:
    """Keeps playback from running faster than the real video.

    If scanning is faster than real time, wait() sleeps so the video plays at
    normal speed. If scanning is slower, it never sleeps and the returned
    speed drops below 1.0x.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.wall_start: float | None = None
        self.video_start = 0.0

    def wait(self, timestamp: float) -> float:
        """Call once per frame. Returns the playback speed (1.0 = real time)."""
        now = time.time()
        if self.wall_start is None:  # first frame: start the clock
            self.wall_start, self.video_start = now, timestamp
            return 1.0

        video_elapsed = timestamp - self.video_start
        if self.enabled:
            ahead = video_elapsed - (now - self.wall_start)
            if ahead > 0:
                time.sleep(ahead)
                now = time.time()
        wall_elapsed = now - self.wall_start
        return video_elapsed / wall_elapsed if wall_elapsed > 0 else 1.0


def to_display(frame: np.ndarray, width: int) -> np.ndarray:
    """Shrink a frame to `width` pixels wide (keeping its shape) for display."""
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def status_line(live: LiveFrame, speed: float) -> str:
    """One line of live stats shown under the video."""
    minutes, seconds = divmod(int(live.timestamp), 60)
    out = live.output
    near = sum(owner is not None for owner in out.assignments)
    return (f"{minutes:02d}:{seconds:02d} | {live.progress:4.0%} | people now: {len(out.people)} | "
            f"phones now: {len(out.phones)} ({near} near a person) | {speed:.1f}x real time")


def play_in_notebook(
    video: str | Path, cfg: dict[str, Any], output_path: str | Path | None = None
) -> dict[str, Any]:
    """Scan a video and show it playing live in a Jupyter / Colab cell.

    Stop early with the cell's stop button; the video saved so far is kept.
    Returns the same summary dict as tracking.run_on_video.
    """
    from IPython.display import Image, Markdown, display  # only exists inside notebooks

    live_cfg = cfg["live"]
    scan = VideoScan(video, cfg, output_path)
    pacer = RealtimePacer(live_cfg["realtime"])
    frame_handle = status_handle = None

    try:
        with closing(scan.frames()) as frames:
            for live in frames:
                speed = pacer.wait(live.timestamp)
                small = to_display(live.annotated, live_cfg["display_width"])
                ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, live_cfg["jpeg_quality"]])
                image = Image(data=jpg.tobytes(), format="jpeg")
                status = Markdown(f"`{status_line(live, speed)}`")

                # The first frame creates the output area; later frames replace it in place.
                if frame_handle is None:
                    frame_handle = display(image, display_id=True)
                    status_handle = display(status, display_id=True)
                else:
                    frame_handle.update(image)
                    status_handle.update(status)
    except KeyboardInterrupt:
        print("Stopped. The annotated video up to this point was saved.")

    summary = scan.summary()
    print_summary(summary)
    return summary


def play_in_window(
    video: str | Path, cfg: dict[str, Any], output_path: str | Path | None = None
) -> dict[str, Any]:
    """Scan a video and show it in an OpenCV window (needs a screen; q stops)."""
    live_cfg = cfg["live"]
    scan = VideoScan(video, cfg, output_path)
    pacer = RealtimePacer(live_cfg["realtime"])
    window = "Exam Hall Monitor (q to stop)"

    try:
        with closing(scan.frames()) as frames:
            for live in frames:
                speed = pacer.wait(live.timestamp)
                cv2.imshow(window, to_display(live.annotated, live_cfg["display_width"]))
                print(f"\r{status_line(live, speed)}", end="", flush=True)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nStopped. The annotated video up to this point was saved.")
                    break
    except KeyboardInterrupt:
        print("\nStopped. The annotated video up to this point was saved.")
    finally:
        cv2.destroyAllWindows()

    summary = scan.summary()
    print_summary(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a video being scanned live (OpenCV window).")
    parser.add_argument("--video", help="Input video (default: paths.input_video in config.yaml).")
    parser.add_argument("--config", help="Config file (default: config.yaml at the repo root).")
    parser.add_argument("--output", help="Annotated video path "
                                         "(default: <output_dir>/<video name>_tracked.mp4).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    video = resolve_input_video(args.video, cfg)
    play_in_window(video, cfg, args.output)


if __name__ == "__main__":
    main()
