"""Person tracking + phone-to-person association (the core of the pipeline).

What happens to every processed frame:
    1. YOLOv8-Pose + ByteTrack find each person, give them a stable track_id
       and 17 body keypoints.
    2. A YOLO detector finds phones (COCO class 67 for now; our fine-tuned
       1-class model later; both are set in config.yaml).
    3. associate_phones() gives each phone to at most one person:
         a) the person with the nearest confident wrist, if close enough, else
         b) the person whose box contains most of the phone box, else
         c) nobody (the phone stays "unassigned").
    4. One FrameRecord per tracked person is produced. These records are the
       ONLY thing later modules (pose_rules.py, risk.py) consume.

VideoScan runs this over a whole video one frame at a time, so a viewer
(src/live.py, app.py) can show each annotated frame as soon as it is ready.

Nothing here decides whether anyone is cheating. It only measures.

Run on its own (no live view, just saves the annotated video):
    python -m src.tracking --video data/sample.mp4
"""

from __future__ import annotations

import argparse
import math
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from src.utils import (
    PHONE_COLOR,
    UNASSIGNED_COLOR,
    WRIST_KEYPOINTS,
    blur_head,
    color_for_id,
    draw_box,
    draw_footer,
    draw_header,
    draw_line,
    draw_skeleton,
    get_video_info,
    load_config,
    make_video_writer,
    read_video,
    resolve_input_video,
    resolve_path,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FrameRecord:
    """One tracked person in one processed frame.

    FROZEN INTERFACE: pose_rules.py and risk.py consume exactly these fields.
    Do not rename or remove fields without updating every consumer and
    CLAUDE.md.

    Fields:
        frame_index: 0-based index of the frame in the ORIGINAL video (not a
            count of processed frames). With every_nth_frame=2 you see
            0, 2, 4, ...
        timestamp: seconds from the start of the video (frame_index / fps).
            Use this, not frame_index, for any duration logic.
        track_id: ByteTrack ID. It stays the same while the person stays
            visible. If tracking is lost (occlusion, leaving the frame) the
            same person may come back with a NEW id, so downstream code must
            tolerate that. People ByteTrack has not confirmed yet are not
            emitted at all.
        person_box: float32 array, shape (4,): [x1, y1, x2, y2] in pixels of
            the original frame.
        keypoints: float32 array, shape (17, 3): [x, y, conf] for each COCO
            keypoint (0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows,
            9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles). All 17 are
            always present; low-confidence ones are NOT removed, so filter by
            conf yourself (pose.keypoint_conf in config.yaml).
        phone_conf: confidence of the phone assigned to this person in this
            frame, or 0.0 if none. If several phones were assigned, the
            highest-confidence one is kept.
        phone_box: float32 array, shape (4,): [x1, y1, x2, y2] of that phone,
            or None if no phone was assigned.
    """

    frame_index: int
    timestamp: float
    track_id: int
    person_box: np.ndarray
    keypoints: np.ndarray
    phone_conf: float = 0.0
    phone_box: np.ndarray | None = None


@dataclass
class Person:
    """A person found by the pose model in one frame (internal to this module)."""

    track_id: int | None  # None = detected, but ByteTrack has not confirmed a track yet
    box: np.ndarray       # (4,) [x1, y1, x2, y2]
    keypoints: np.ndarray  # (17, 3) [x, y, conf]


@dataclass
class PhoneDetection:
    """A phone box from the phone detector in one frame."""

    box: np.ndarray  # (4,) [x1, y1, x2, y2]
    conf: float


@dataclass
class FrameOutput:
    """Everything produced for one frame: records for later modules, plus the
    raw detections that are needed to draw the debug video."""

    timestamp: float
    records: list[FrameRecord]
    people: list[Person]            # includes people with no track_id yet (they still get blurred)
    phones: list[PhoneDetection]
    assignments: list[int | None]   # assignments[i] = track_id that phones[i] belongs to, or None


@dataclass
class LiveFrame:
    """One processed frame handed to a live viewer by VideoScan.frames()."""

    frame_index: int
    timestamp: float
    annotated: np.ndarray  # BGR image, heads already blurred, safe to show
    output: FrameOutput
    progress: float        # 0.0 .. 1.0 through the video


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def box_centre(box: np.ndarray) -> tuple[float, float]:
    """Centre (x, y) of an [x1, y1, x2, y2] box."""
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def overlap_fraction(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of the `inner` box's area that lies inside the `outer` box.

    We use this instead of plain IoU for the fallback because a phone is tiny
    compared with a person: even a phone fully inside a person box has an IoU
    of only ~0.02, which makes IoU thresholds hard to tune.
    """
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    inner_area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return float(intersection / inner_area) if inner_area > 0 else 0.0


# ---------------------------------------------------------------------------
# Phone -> person association
# ---------------------------------------------------------------------------

def associate_phones(
    phones: list[PhoneDetection],
    people: list[Person],
    *,
    keypoint_conf: float,
    max_wrist_distance: float,
    min_box_overlap: float,
) -> list[int | None]:
    """Assign each phone to at most one tracked person.

    For each phone:
      1. Wrist rule: find the nearest wrist (left or right) with confidence
         >= keypoint_conf. The distance is divided by that person's box
         height, so near and far students are treated the same. If it is
         <= max_wrist_distance, the phone goes to that person.
      2. Box rule (fallback, e.g. wrists hidden under a desk): give the phone
         to the person whose box contains the largest fraction of the phone
         box, if that fraction is >= min_box_overlap.
      3. Otherwise the phone stays unassigned (None).

    Only people with a track_id can receive phones.

    Returns:
        A list the same length as `phones`: the track_id for each phone, or None.
    """
    tracked = [p for p in people if p.track_id is not None]
    assignments: list[int | None] = []

    for phone in phones:
        cx, cy = box_centre(phone.box)

        # --- 1. Nearest confident wrist ---------------------------------
        best_id, best_distance = None, math.inf
        for person in tracked:
            height = max(float(person.box[3] - person.box[1]), 1.0)
            for k in WRIST_KEYPOINTS:
                x, y, conf = person.keypoints[k]
                if conf < keypoint_conf:
                    continue
                distance = math.hypot(x - cx, y - cy) / height
                if distance < best_distance:
                    best_id, best_distance = person.track_id, distance

        if best_id is not None and best_distance <= max_wrist_distance:
            assignments.append(best_id)
            continue

        # --- 2. Fallback: phone box inside a person box -------------------
        best_id, best_overlap = None, 0.0
        for person in tracked:
            overlap = overlap_fraction(phone.box, person.box)
            if overlap > best_overlap:
                best_id, best_overlap = person.track_id, overlap

        # --- 3. Assign, or leave unassigned --------------------------------
        assignments.append(best_id if best_overlap >= min_box_overlap else None)

    return assignments


def build_records(
    frame_index: int,
    timestamp: float,
    people: list[Person],
    phones: list[PhoneDetection],
    assignments: list[int | None],
) -> list[FrameRecord]:
    """Turn one frame's people + assigned phones into FrameRecords."""
    # For each track_id, keep only its highest-confidence phone.
    best_phone: dict[int, PhoneDetection] = {}
    for phone, owner in zip(phones, assignments):
        if owner is not None and (owner not in best_phone or phone.conf > best_phone[owner].conf):
            best_phone[owner] = phone

    records = []
    for person in people:
        if person.track_id is None:
            continue  # not confirmed by ByteTrack yet
        phone = best_phone.get(person.track_id)
        records.append(FrameRecord(
            frame_index=frame_index,
            timestamp=timestamp,
            track_id=person.track_id,
            person_box=person.box,
            keypoints=person.keypoints,
            phone_conf=phone.conf if phone else 0.0,
            phone_box=phone.box if phone else None,
        ))
    return records


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def pick_device(setting: Any) -> str:
    """Turn runtime.device into something Ultralytics accepts.

    "auto" means the first GPU if PyTorch can see one (e.g. on Colab),
    otherwise the CPU. Anything else ("cpu", 0, "0") is passed through.
    """
    if str(setting).lower() != "auto":
        return str(setting)
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


class ExamHallTracker:
    """Loads both models once, then processes frames one at a time.

    Use ONE instance per video: the ByteTrack state (persist=True) lives
    inside the pose model, so a new video needs a new ExamHallTracker.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        # Imported here, not at the top of the file, so other modules can
        # import FrameRecord without loading PyTorch.
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = pick_device(cfg["runtime"]["device"])
        print(f"Device: {'GPU ' + self.device if self.device != 'cpu' else 'CPU'}")

        paths = cfg["paths"]
        self.pose_model = YOLO(str(resolve_path(paths["pose_weights"])))
        self.phone_model = YOLO(str(resolve_path(paths["phone_weights"])))
        self.tracker_config = paths["tracker"]

        # Check the phone class id exists in this model, so swapping in the
        # fine-tuned model with the wrong class_id fails loudly.
        self.phone_class_id = int(cfg["phone"]["class_id"])
        names = self.phone_model.names
        if self.phone_class_id not in names:
            raise ValueError(
                f"phone.class_id={self.phone_class_id} is not in {paths['phone_weights']} "
                f"(classes: {names}). Use 67 for COCO yolov8n, 0 for the fine-tuned model."
            )
        print(f"Phone class: {self.phone_class_id} = '{names[self.phone_class_id]}'")

        if cfg["phone"]["sahi"]:
            print("[warning] phone.sahi is true, but SAHI is not wired in yet "
                  "(planned for detect_phone.py). Using full-frame detection.")

    def process_frame(self, frame_index: int, timestamp: float, frame: np.ndarray) -> FrameOutput:
        """Track people, detect phones, associate them, build FrameRecords."""
        people = self._track_people(frame)
        phones = self._detect_phones(frame)
        assoc = self.cfg["association"]
        assignments = associate_phones(
            phones,
            people,
            keypoint_conf=assoc["keypoint_conf"],
            max_wrist_distance=assoc["max_wrist_distance"],
            min_box_overlap=assoc["min_box_overlap"],
        )
        records = build_records(frame_index, timestamp, people, phones, assignments)
        return FrameOutput(
            timestamp=timestamp,
            records=records,
            people=people,
            phones=phones,
            assignments=assignments,
        )

    def _track_people(self, frame: np.ndarray) -> list[Person]:
        result = self.pose_model.track(
            frame,
            persist=True,                  # keep ByteTrack state between calls
            tracker=self.tracker_config,
            conf=self.cfg["pose"]["person_conf"],
            imgsz=self.cfg["runtime"]["imgsz"],
            device=self.device,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        # boxes.id is None when ByteTrack has not confirmed any track yet.
        if result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
        else:
            ids = [None] * len(boxes)
        if result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy().astype(np.float32)
        else:
            keypoints = np.zeros((len(boxes), 17, 3), dtype=np.float32)

        return [Person(track_id=t, box=b, keypoints=k) for t, b, k in zip(ids, boxes, keypoints)]

    def _detect_phones(self, frame: np.ndarray) -> list[PhoneDetection]:
        result = self.phone_model.predict(
            frame,
            classes=[self.phone_class_id],  # keep ONLY the phone class
            conf=self.cfg["phone"]["conf_threshold"],
            imgsz=self.cfg["runtime"]["imgsz"],
            device=self.device,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = result.boxes.conf.cpu().numpy()
        return [PhoneDetection(box=b, conf=float(c)) for b, c in zip(boxes, confs)]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_frame(frame: np.ndarray, output: FrameOutput, cfg: dict[str, Any]) -> np.ndarray:
    """Return an annotated copy of `frame` (heads blurred FIRST)."""
    canvas = frame.copy()
    privacy = cfg["privacy"]

    # 1. Blur every detected person's head, including people without a
    #    track_id yet, before drawing anything else.
    if privacy["blur_heads"]:
        for person in output.people:
            blur_head(
                canvas,
                person.keypoints,
                min_conf=privacy["head_keypoint_conf"],
                box_scale=privacy["head_box_scale"],
                min_width_fraction=privacy["min_head_width_fraction"],
                fallback_fraction=privacy["fallback_head_fraction"],
                kernel_size=privacy["blur_kernel"],
                person_box=person.box,
            )

    # 2. People: skeleton, box and track ID.
    person_centres: dict[int, tuple[float, float]] = {}
    for person in output.people:
        color = color_for_id(person.track_id)
        draw_skeleton(canvas, person.keypoints, color, cfg["pose"]["keypoint_conf"])
        label = f"ID {person.track_id}" if person.track_id is not None else "ID ?"
        draw_box(canvas, person.box, color, label)
        if person.track_id is not None:
            person_centres[person.track_id] = box_centre(person.box)

    # 3. Phones, with a line from each assigned phone to its person.
    for phone, owner in zip(output.phones, output.assignments):
        if owner is None:
            draw_box(canvas, phone.box, UNASSIGNED_COLOR, f"phone {phone.conf:.2f} unassigned")
        else:
            draw_box(canvas, phone.box, PHONE_COLOR, f"phone {phone.conf:.2f} -> ID {owner}")
            draw_line(canvas, box_centre(phone.box), person_centres[owner], PHONE_COLOR)

    # 4. Status line at the top: video time and what is visible right now.
    minutes, seconds = divmod(int(output.timestamp), 60)
    header = f"{minutes:02d}:{seconds:02d}  people: {len(output.people)}  phones: {len(output.phones)}"
    near = sorted({owner for owner in output.assignments if owner is not None})
    if near:
        header += "  (phone near ID " + ", ".join(str(i) for i in near) + ")"
    draw_header(canvas, header)

    draw_footer(canvas, "Detections are for human review only")
    return canvas


# ---------------------------------------------------------------------------
# Whole-video scan, one frame at a time
# ---------------------------------------------------------------------------

class VideoScan:
    """Scan one video frame by frame, saving the annotated copy as it goes.

    Usage:
        scan = VideoScan(video_path, cfg)
        with closing(scan.frames()) as frames:   # closing() makes sure the
            for live in frames:                   # output file is finished
                show(live.annotated)              # even if you stop early
        print(scan.summary())
    """

    def __init__(
        self, video_path: str | Path, cfg: dict[str, Any], output_path: str | Path | None = None
    ) -> None:
        self.video_path = Path(video_path)
        self.cfg = cfg
        self.every_nth = int(cfg["runtime"]["every_nth_frame"])
        self.info = get_video_info(self.video_path, cfg["runtime"]["fallback_fps"])
        if output_path is None:
            output_path = resolve_path(cfg["paths"]["output_dir"]) / f"{self.video_path.stem}_tracked.mp4"
        self.output_path = Path(output_path)
        self.tracker = ExamHallTracker(cfg)

        # Running totals, used for the summary.
        self.frames_processed = 0
        self.track_ids: set[int] = set()
        self.phone_detections = 0
        self.phones_assigned = 0
        self.seconds = 0.0

    def frames(self) -> Iterator[LiveFrame]:
        """Yield each processed frame as soon as it is ready."""
        fps_out = self.info["fps"] / self.every_nth  # saved video plays at real-time speed
        writer = make_video_writer(self.output_path, fps_out, self.info["width"], self.info["height"])
        expected = max(self.info["frame_count"] // self.every_nth, 1)
        start = time.time()

        try:
            for frame_index, timestamp, frame in read_video(
                self.video_path, self.every_nth, self.cfg["runtime"]["fallback_fps"]
            ):
                output = self.tracker.process_frame(frame_index, timestamp, frame)
                annotated = draw_frame(frame, output, self.cfg)
                writer.write(annotated)

                self.frames_processed += 1
                self.track_ids.update(r.track_id for r in output.records)
                self.phone_detections += len(output.phones)
                self.phones_assigned += sum(owner is not None for owner in output.assignments)
                self.seconds = time.time() - start

                yield LiveFrame(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    annotated=annotated,
                    output=output,
                    progress=min(self.frames_processed / expected, 1.0),
                )
        finally:
            writer.release()  # always finish the file, even if the viewer stops early

    def summary(self) -> dict[str, Any]:
        """Totals so far (call after frames() has finished or been stopped)."""
        return {
            "video": str(self.video_path),
            "output": str(self.output_path),
            "frames_total": self.info["frame_count"],
            "every_nth": self.every_nth,
            "frames_processed": self.frames_processed,
            "people_seen": len(self.track_ids),
            "phone_detections": self.phone_detections,
            "phones_assigned": self.phones_assigned,
            "phones_unassigned": self.phone_detections - self.phones_assigned,
            "seconds": self.seconds,
            "fps": self.frames_processed / self.seconds if self.seconds > 0 else 0.0,
        }


def run_on_video(
    video_path: str | Path, cfg: dict[str, Any], output_path: str | Path | None = None
) -> tuple[list[FrameRecord], dict[str, Any]]:
    """Process a whole video without a live view; return (records, summary)."""
    scan = VideoScan(video_path, cfg, output_path)
    expected = max(scan.info["frame_count"] // scan.every_nth, 1)
    all_records: list[FrameRecord] = []

    with closing(scan.frames()) as frames:
        for live in frames:
            all_records.extend(live.output.records)
            if scan.frames_processed % cfg["runtime"]["log_every"] == 0:
                print(f"  processed {scan.frames_processed}/~{expected} frames (t = {live.timestamp:.1f}s)")

    return all_records, scan.summary()


def print_summary(summary: dict[str, Any]) -> None:
    """Print the run summary in a readable block."""
    print("\n=== Tracking summary ===")
    print(f"Video:             {summary['video']}")
    print(f"Frames processed:  {summary['frames_processed']} "
          f"(every {summary['every_nth']} of {summary['frames_total']})")
    print(f"People seen:       {summary['people_seen']} unique track IDs "
          f"(can exceed real people if tracks are lost and re-found)")
    print(f"Phone detections:  {summary['phone_detections']}")
    print(f"  assigned:        {summary['phones_assigned']}")
    print(f"  unassigned:      {summary['phones_unassigned']}")
    print(f"Wall time:         {summary['seconds']:.0f}s "
          f"({summary['fps']:.1f} processed frames/s, including any live display)")
    print(f"Annotated video:   {summary['output']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track people, detect phones and link each phone to a person."
    )
    parser.add_argument("--video", help="Input video (default: paths.input_video in config.yaml).")
    parser.add_argument("--config", help="Config file (default: config.yaml at the repo root).")
    parser.add_argument("--output", help="Annotated video path "
                                         "(default: <output_dir>/<video name>_tracked.mp4).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    video = resolve_input_video(args.video, cfg)

    print(f"Processing {video} ...")
    _, summary = run_on_video(video, cfg, args.output)
    print_summary(summary)


if __name__ == "__main__":
    main()
