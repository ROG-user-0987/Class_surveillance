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

Nothing here decides whether anyone is cheating. It only measures.

Run on its own:
    python -m src.tracking --video data/sample.mp4
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils import (
    PHONE_COLOR,
    UNASSIGNED_COLOR,
    WRIST_KEYPOINTS,
    blur_head,
    color_for_id,
    draw_box,
    draw_footer,
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

    records: list[FrameRecord]
    people: list[Person]            # includes people with no track_id yet (they still get blurred)
    phones: list[PhoneDetection]
    assignments: list[int | None]   # assignments[i] = track_id that phones[i] belongs to, or None


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
        return FrameOutput(records, people, phones, assignments)

    def _track_people(self, frame: np.ndarray) -> list[Person]:
        runtime = self.cfg["runtime"]
        result = self.pose_model.track(
            frame,
            persist=True,                  # keep ByteTrack state between calls
            tracker=self.tracker_config,
            conf=self.cfg["pose"]["person_conf"],
            imgsz=runtime["imgsz"],
            device=runtime["device"],
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
        runtime = self.cfg["runtime"]
        result = self.phone_model.predict(
            frame,
            classes=[self.phone_class_id],  # keep ONLY the phone class
            conf=self.cfg["phone"]["conf_threshold"],
            imgsz=runtime["imgsz"],
            device=runtime["device"],
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

    draw_footer(canvas, "Detections are for human review only")
    return canvas


# ---------------------------------------------------------------------------
# Whole-video run
# ---------------------------------------------------------------------------

def run_on_video(
    video_path: Path, cfg: dict[str, Any], output_path: Path | None = None
) -> tuple[list[FrameRecord], dict[str, Any]]:
    """Process a whole video, save the annotated copy, return (records, summary)."""
    runtime = cfg["runtime"]
    every_nth = int(runtime["every_nth_frame"])
    info = get_video_info(video_path, runtime["fallback_fps"])

    if output_path is None:
        output_path = resolve_path(cfg["paths"]["output_dir"]) / f"{video_path.stem}_tracked.mp4"
    # Output FPS = input FPS / N, so the saved video plays at real-time speed.
    writer = make_video_writer(output_path, info["fps"] / every_nth, info["width"], info["height"])
    tracker = ExamHallTracker(cfg)

    all_records: list[FrameRecord] = []
    track_ids: set[int] = set()
    frames_processed = phones_total = phones_assigned = 0
    expected = max(info["frame_count"] // every_nth, 1)
    start = time.time()

    try:
        for frame_index, timestamp, frame in read_video(video_path, every_nth, runtime["fallback_fps"]):
            output = tracker.process_frame(frame_index, timestamp, frame)
            writer.write(draw_frame(frame, output, cfg))

            all_records.extend(output.records)
            track_ids.update(r.track_id for r in output.records)
            phones_total += len(output.phones)
            phones_assigned += sum(owner is not None for owner in output.assignments)
            frames_processed += 1

            if frames_processed % runtime["log_every"] == 0:
                print(f"  processed {frames_processed}/~{expected} frames (t = {timestamp:.1f}s)")
    finally:
        writer.release()  # always close the file, even on Ctrl+C

    elapsed = time.time() - start
    summary = {
        "video": str(video_path),
        "output": str(output_path),
        "frames_total": info["frame_count"],
        "every_nth": every_nth,
        "frames_processed": frames_processed,
        "people_seen": len(track_ids),
        "phone_detections": phones_total,
        "phones_assigned": phones_assigned,
        "phones_unassigned": phones_total - phones_assigned,
        "seconds": elapsed,
        "fps": frames_processed / elapsed if elapsed > 0 else 0.0,
    }
    return all_records, summary


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
    print(f"Speed:             {summary['fps']:.1f} processed frames/s "
          f"({summary['seconds']:.0f}s total)")
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
    output = Path(args.output) if args.output else None

    print(f"Processing {video} on device '{cfg['runtime']['device']}' ...")
    _, summary = run_on_video(video, cfg, output)
    print_summary(summary)


if __name__ == "__main__":
    main()
