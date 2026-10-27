"""Phone detection in three modes, plus a fair comparison of the modes.

Modes (phone.mode in config.yaml). All three use the same weights
(paths.phone_weights) and the same class id (phone.class_id):

    full  One pass over the whole frame. This is the original behaviour.
    sahi  SAHI sliced inference: the frame is cut into overlapping tiles
          (phone.slice_size, phone.slice_overlap), every tile is detected,
          and overlapping boxes are merged with NMS (phone.nms_iou). Helps
          with small, far-away phones.
    crop  One crop per detected person, covering their hands and desk: the
          person box grown by phone.crop_pad on every side, with more of the
          vertical growth below the box than above (phone.crop_down_bias),
          and stretched to include any visible wrist. Each crop is resized to
          phone.crop_size and detected; boxes are mapped back to full-frame
          coordinates and duplicates from overlapping crops are merged with
          NMS (phone.nms_iou). A phone found in a tracked person's crop
          remembers that person (crop_owner); tracking.associate_phones()
          tries that person first but still applies the normal checks.

Comparison (python evaluate.py --compare-modes --gt ...): every mode exports
its boxes to COCO JSON and all are scored by the SAME pycocotools evaluator
(mAP50, plus precision/recall at phone.conf_threshold from the same box
matching). FPS is measured end to end per image on the same device: person
detection (the pipeline always runs it) + the mode's cropping/slicing +
detection + merging.

Run on its own (detect phones in a video with one mode, print counts + speed):
    python -m src.detect_phone --video data/sample.mp4 --mode crop
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.utils import WRIST_KEYPOINTS, load_config, read_video, resolve_input_video, resolve_path

if TYPE_CHECKING:  # type hints only; importing tracking here for real would be circular
    from src.tracking import Person

MODES = ("full", "sahi", "crop")


@dataclass
class PhoneDetection:
    """A phone box from the phone detector in one frame."""

    box: np.ndarray  # (4,) [x1, y1, x2, y2]
    conf: float
    crop_owner: int | None = None  # crop mode: track_id of the person whose crop found it


def detection_mode(cfg: dict[str, Any]) -> str:
    """phone.mode, also accepting the old `phone.sahi: true` setting."""
    phone = cfg["phone"]
    mode = phone.get("mode") or ("sahi" if phone.get("sahi") else "full")
    if mode not in MODES:
        raise ValueError(f"phone.mode must be one of {MODES}, got {mode!r}")
    return mode


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Non-maximum suppression: indices of the boxes to keep, best first.

    A box is dropped when it overlaps an already-kept, higher-scoring box by
    more than iou_threshold (intersection over union).
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    areas = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    order = np.argsort(-np.asarray(scores))
    keep: list[int] = []
    while order.size:
        best, rest = int(order[0]), order[1:]
        keep.append(best)
        ix1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        iy1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        ix2 = np.minimum(boxes[best, 2], boxes[rest, 2])
        iy2 = np.minimum(boxes[best, 3], boxes[rest, 3])
        inter = (ix2 - ix1).clip(0) * (iy2 - iy1).clip(0)
        iou = inter / (areas[best] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def crop_region(
    person_box: np.ndarray,
    keypoints: np.ndarray,
    frame_shape: tuple[int, ...],
    *,
    pad: float,
    down_bias: float,
    kp_conf: float,
) -> tuple[int, int, int, int] | None:
    """Pixel region (x1, y1, x2, y2) covering a person's hands and desk.

    The person box grows by `pad` x its width on the left and on the right.
    Vertically it grows by `pad` x its height on average above and below,
    with `down_bias` shifting that growth downward: above = pad*h*(1-bias),
    below = pad*h*(1+bias) (bias 0.5 -> 25% of the growth above, 75% below).
    The region is then stretched to include every visible wrist (plus a
    margin of pad x width), and clipped to the frame. None if nothing is left.
    """
    x1, y1, x2, y2 = (float(v) for v in person_box)
    w, h = x2 - x1, y2 - y1
    rx1, rx2 = x1 - pad * w, x2 + pad * w
    ry1 = y1 - pad * h * (1 - down_bias)
    ry2 = y2 + pad * h * (1 + down_bias)

    margin = pad * w
    for k in WRIST_KEYPOINTS:
        wx, wy, conf = keypoints[k]
        if conf >= kp_conf:
            rx1, rx2 = min(rx1, wx - margin), max(rx2, wx + margin)
            ry1, ry2 = min(ry1, wy - margin), max(ry2, wy + margin)

    frame_h, frame_w = frame_shape[:2]
    rx1, ry1 = int(max(0, rx1)), int(max(0, ry1))
    rx2, ry2 = int(min(frame_w, rx2)), int(min(frame_h, ry2))
    if rx2 - rx1 < 2 or ry2 - ry1 < 2:
        return None
    return rx1, ry1, rx2, ry2


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------

class PhoneDetector:
    """Detects phones in a frame with the mode set in phone.mode."""

    def __init__(self, model: Any, cfg: dict[str, Any], device: str) -> None:
        """`model` is a loaded Ultralytics YOLO model with the phone weights."""
        self.model = model
        self.cfg = cfg
        self.device = device
        self.mode = detection_mode(cfg)
        self.class_id = int(cfg["phone"]["class_id"])
        self._sahi_model: Any = None  # loaded the first time sahi mode is used

    def detect(
        self, frame: np.ndarray, people: list[Person] | tuple = (), conf: float | None = None
    ) -> list[PhoneDetection]:
        """Phones in `frame`. `people` is only needed for crop mode. `conf`
        overrides phone.conf_threshold (evaluation exports low-confidence boxes)."""
        conf = self.cfg["phone"]["conf_threshold"] if conf is None else conf
        if self.mode == "full":
            return self._detect_full(frame, conf)
        if self.mode == "sahi":
            return self._detect_sahi(frame, conf)
        return self._detect_crops(frame, people, conf)

    def _detect_full(self, frame: np.ndarray, conf: float) -> list[PhoneDetection]:
        result = self.model.predict(
            frame,
            classes=[self.class_id],  # keep ONLY the phone class
            conf=conf,
            imgsz=self.cfg["runtime"]["imgsz"],
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = result.boxes.conf.cpu().numpy()
        return [PhoneDetection(box=b, conf=float(c)) for b, c in zip(boxes, confs)]

    def _detect_sahi(self, frame: np.ndarray, conf: float) -> list[PhoneDetection]:
        from sahi.predict import get_sliced_prediction

        model = self._load_sahi_model()
        model.confidence_threshold = conf
        if hasattr(model, "model") and hasattr(model.model, "overrides"):
            model.model.overrides["conf"] = conf  # Ultralytics' own default (0.25) would cut lower boxes
        phone = self.cfg["phone"]
        result = get_sliced_prediction(
            np.ascontiguousarray(frame[:, :, ::-1]),  # SAHI expects RGB; OpenCV frames are BGR
            model,
            slice_height=phone["slice_size"],
            slice_width=phone["slice_size"],
            overlap_height_ratio=phone["slice_overlap"],
            overlap_width_ratio=phone["slice_overlap"],
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=phone["nms_iou"],
            verbose=0,
        )
        detections = []
        for prediction in result.object_prediction_list:
            if prediction.category.id != self.class_id:
                continue
            box = np.array(prediction.bbox.to_xyxy(), dtype=np.float32)
            detections.append(PhoneDetection(box=box, conf=float(prediction.score.value)))
        return detections

    def _load_sahi_model(self) -> Any:
        if self._sahi_model is not None:
            return self._sahi_model
        from sahi import AutoDetectionModel

        weights = str(resolve_path(self.cfg["paths"]["phone_weights"]))
        device = "cpu" if self.device == "cpu" else f"cuda:{self.device}"
        errors = []
        # SAHI renamed its Ultralytics model type between versions; try both names.
        for model_type in ("ultralytics", "yolov8"):
            try:
                self._sahi_model = AutoDetectionModel.from_pretrained(
                    model_type=model_type,
                    model_path=weights,
                    confidence_threshold=self.cfg["phone"]["conf_threshold"],
                    device=device,
                    image_size=self.cfg["runtime"]["imgsz"],
                )
                return self._sahi_model
            except Exception as error:  # noqa: BLE001 - SAHI raises different errors per version
                errors.append(f"{model_type}: {error}")
        raise RuntimeError("Could not load the phone model in SAHI:\n" + "\n".join(errors))

    def _detect_crops(self, frame: np.ndarray, people: list[Person] | tuple, conf: float) -> list[PhoneDetection]:
        phone = self.cfg["phone"]
        regions, owners = [], []
        for person in people:
            region = crop_region(person.box, person.keypoints, frame.shape,
                                 pad=phone["crop_pad"], down_bias=phone["crop_down_bias"],
                                 kp_conf=self.cfg["association"]["keypoint_conf"])
            if region is not None:
                regions.append(region)
                owners.append(person.track_id)
        if not regions:
            return []

        crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in regions]
        results = self.model.predict(crops, classes=[self.class_id], conf=conf,
                                     imgsz=phone["crop_size"], device=self.device, verbose=False)

        # Map every crop's boxes back to full-frame coordinates.
        boxes, scores, box_owners = [], [], []
        for (x1, y1, _, _), owner, result in zip(regions, owners, results):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            offset = np.array([x1, y1, x1, y1], dtype=np.float32)
            for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                boxes.append(box.astype(np.float32) + offset)
                scores.append(float(score))
                box_owners.append(owner)
        if not boxes:
            return []

        # The same phone can appear in two neighbours' crops: keep the best box.
        keep = nms(np.array(boxes), np.array(scores), phone["nms_iou"])
        return [PhoneDetection(box=boxes[i], conf=scores[i], crop_owner=box_owners[i]) for i in keep]


# ---------------------------------------------------------------------------
# Comparing the modes on labelled images (pycocotools)
# ---------------------------------------------------------------------------

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def load_ground_truth(
    gt_path: str | Path, images_dir: str | Path | None, gt_class_id: int
) -> tuple[dict[str, Any], list[tuple[int, Path]]]:
    """Ground-truth phone boxes as a COCO dict (one category, id 1), plus
    (image_id, image_path) pairs.

    Accepts either:
      - a COCO JSON file (images_dir = folder with the images; every
        annotation is treated as a phone), or
      - a YOLO-format folder: images under .../images/, labels in the matching
        .../labels/*.txt ("class cx cy w h", normalised); only lines with
        class == gt_class_id are used.
    """
    gt_path = Path(gt_path)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    pairs: list[tuple[int, Path]] = []

    if gt_path.suffix.lower() == ".json":
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        folder = Path(images_dir) if images_dir else gt_path.parent
        for image in data["images"]:
            images.append({"id": image["id"], "file_name": image["file_name"],
                           "width": image.get("width", 0), "height": image.get("height", 0)})
            pairs.append((image["id"], folder / image["file_name"]))
        for i, ann in enumerate(data.get("annotations", []), start=1):
            x, y, w, h = ann["bbox"]
            annotations.append({"id": i, "image_id": ann["image_id"], "category_id": 1,
                                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
    else:
        image_paths = sorted(p for p in gt_path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        ann_id = 0
        for image_id, path in enumerate(image_paths, start=1):
            height, width = cv2.imread(str(path)).shape[:2]
            images.append({"id": image_id, "file_name": path.name, "width": width, "height": height})
            pairs.append((image_id, path))
            parts = list(path.with_suffix(".txt").parts)
            if "images" in parts:  # YOLO layout: .../images/x.jpg -> .../labels/x.txt
                parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
            label_path = Path(*parts)
            if not label_path.exists():
                continue
            for line in label_path.read_text().splitlines():
                values = line.split()
                if len(values) < 5 or int(float(values[0])) != gt_class_id:
                    continue
                cx, cy, w, h = (float(v) for v in values[1:5])
                ann_id += 1
                annotations.append({"id": ann_id, "image_id": image_id, "category_id": 1,
                                    "bbox": [(cx - w / 2) * width, (cy - h / 2) * height,
                                             w * width, h * height],
                                    "area": w * width * h * height, "iscrowd": 0})

    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "phone"}]}
    return coco, pairs


def score_coco(
    coco_gt: Any, predictions: list[dict[str, Any]], *, conf_threshold: float, iou_threshold: float
) -> dict[str, float]:
    """Score predictions with pycocotools: mAP50 from COCOeval, and
    precision/recall at `conf_threshold` from the SAME per-image box matching
    COCOeval computed (at `iou_threshold`, all object sizes)."""
    from pycocotools.cocoeval import COCOeval

    if not predictions:
        return {"precision": 0.0, "recall": 0.0, "mAP50": 0.0}
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.catIds = [1]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    map50 = float(evaluator.stats[1])

    iou_index = int(np.argmin(np.abs(evaluator.params.iouThrs - iou_threshold)))
    all_sizes = evaluator.params.areaRng[0]
    tp = fp = n_gt = 0
    for result in evaluator.evalImgs:
        if result is None or list(result["aRng"]) != list(all_sizes):
            continue
        n_gt += int(np.sum(np.logical_not(np.asarray(result["gtIgnore"], dtype=bool))))
        scores = np.asarray(result["dtScores"])
        if scores.size == 0:
            continue
        matched = np.asarray(result["dtMatches"])[iou_index] > 0
        ignored = np.asarray(result["dtIgnore"])[iou_index].astype(bool)
        counted = (scores >= conf_threshold) & ~ignored
        tp += int(np.sum(matched & counted))
        fp += int(np.sum(~matched & counted))
    return {
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "recall": round(tp / n_gt, 4) if n_gt else 0.0,
        "mAP50": round(map50, 4),
    }


def detect_people(pose_model: Any, frame: np.ndarray, cfg: dict[str, Any], device: str) -> list[Person]:
    """People (boxes + keypoints, no track ids) in one still image."""
    from src.tracking import Person  # here to avoid a circular import

    result = pose_model.predict(frame, conf=cfg["pose"]["person_conf"], imgsz=cfg["runtime"]["imgsz"],
                                device=device, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    if result.keypoints is not None:
        keypoints = result.keypoints.data.cpu().numpy().astype(np.float32)
    else:
        keypoints = np.zeros((len(boxes), 17, 3), dtype=np.float32)
    return [Person(track_id=None, box=b, keypoints=k) for b, k in zip(boxes, keypoints)]


def compare_modes(
    cfg: dict[str, Any],
    gt_path: str | Path,
    images_dir: str | Path | None = None,
    modes: tuple[str, ...] = MODES,
) -> list[dict[str, Any]]:
    """Score every detection mode on the same labelled images with the same
    evaluator; write results/mode_comparison.csv; return the rows."""
    from pycocotools.coco import COCO
    from ultralytics import YOLO

    from src.tracking import prepare_runtime  # here to avoid a circular import

    cfg = prepare_runtime(cfg)
    device = cfg["runtime"]["device"]
    evaluation = cfg["evaluation"]
    gt, images = load_ground_truth(gt_path, images_dir, evaluation["gt_class_id"])
    print(f"{len(images)} images, {len(gt['annotations'])} labelled phones")

    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()

    weights = str(resolve_path(cfg["paths"]["phone_weights"]))
    pose_model = YOLO(str(resolve_path(cfg["paths"]["pose_weights"])))
    phone_model = YOLO(weights)
    out_dir = resolve_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode in modes:
        mode_cfg = copy.deepcopy(cfg)
        mode_cfg["phone"]["mode"] = mode
        detector = PhoneDetector(phone_model, mode_cfg, device)
        predictions, timed_images, elapsed = [], 0, 0.0
        for index, (image_id, path) in enumerate(images):
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"  [skip] cannot read {path}")
                continue
            start = time.perf_counter()
            people = detect_people(pose_model, frame, cfg, device)  # same work for every mode
            detections = detector.detect(frame, people, conf=evaluation["map_conf"])
            if index >= evaluation["fps_warmup"]:
                elapsed += time.perf_counter() - start
                timed_images += 1
            for det in detections:
                x1, y1, x2, y2 = (float(v) for v in det.box)
                predictions.append({"image_id": image_id, "category_id": 1,
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": det.conf})

        (out_dir / f"mode_predictions_{mode}.json").write_text(json.dumps(predictions), encoding="utf-8")
        metrics = score_coco(coco_gt, predictions, conf_threshold=cfg["phone"]["conf_threshold"],
                             iou_threshold=evaluation["map_iou"])
        rows.append({"model": Path(weights).stem, "weights": weights, "mode": mode, **metrics,
                     "fps": round(timed_images / elapsed, 2) if elapsed > 0 else 0.0,
                     "device": device})

    out_csv = out_dir / "mode_comparison.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "weights", "mode", "precision", "recall",
                                               "mAP50", "fps", "device"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv}")
    return rows


# ---------------------------------------------------------------------------
# Command line: run one mode over a video
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Detect phones in a video with one detection mode.")
    parser.add_argument("--video", help="Input video (default: paths.input_video in config.yaml).")
    parser.add_argument("--config", help="Config file (default: config.yaml at the repo root).")
    parser.add_argument("--mode", choices=MODES, help="Override phone.mode from config.yaml.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many processed frames.")
    args = parser.parse_args()

    from ultralytics import YOLO

    from src.tracking import prepare_runtime  # here to avoid a circular import

    cfg = prepare_runtime(load_config(args.config))
    if args.mode:
        cfg["phone"]["mode"] = args.mode
    video = resolve_input_video(args.video, cfg)
    device = cfg["runtime"]["device"]
    detector = PhoneDetector(YOLO(str(resolve_path(cfg["paths"]["phone_weights"]))), cfg, device)
    pose_model = YOLO(str(resolve_path(cfg["paths"]["pose_weights"]))) if detector.mode == "crop" else None

    frames = phones = 0
    start = time.perf_counter()
    for _, _, frame in read_video(video, cfg["runtime"]["every_nth_frame"], cfg["runtime"]["fallback_fps"]):
        people = detect_people(pose_model, frame, cfg, device) if pose_model is not None else []
        phones += len(detector.detect(frame, people))
        frames += 1
        if args.max_frames and frames >= args.max_frames:
            break
    elapsed = time.perf_counter() - start
    print(f"Mode {detector.mode} on {device}: {phones} phone boxes in {frames} frames "
          f"({frames / elapsed:.1f} frames/s)" if elapsed > 0 else "No frames read.")


if __name__ == "__main__":
    main()
