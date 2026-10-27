"""Shared test helpers: a config dict (same values as config.yaml) and
synthetic students, so tests need no video, models or YAML file."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make "import src..." work

import numpy as np  # noqa: E402

from src.tracking import FrameRecord  # noqa: E402

BASE_CFG = {
    "features": {"temporal_hysteresis": False, "baseline_calibration": False, "evidence_alerts": False,
                 "neighbour_peek": False, "object_passing": False, "hands_hidden": False,
                 "low_visibility": False, "cue_fusion": False},
    "phone": {"class_id": 0, "conf_threshold": 0.25, "min_duration_s": 1.0, "mode": "full",
              "slice_size": 640, "slice_overlap": 0.2, "crop_pad": 0.25, "crop_down_bias": 0.5,
              "crop_size": 640, "nms_iou": 0.5},
    "association": {"keypoint_conf": 0.5, "max_wrist_distance": 0.35, "min_box_overlap": 0.5},
    "pose": {"person_conf": 0.4, "keypoint_conf": 0.5, "head_turn_threshold": 0.2,
             "min_shoulder_box_ratio": 0.3, "lean_threshold_deg": 25, "min_duration_s": 2.0},
    "temporal": {"ema_alpha": 0.3, "phone_on": 0.3, "phone_off": 0.15, "posture_on": 1.0,
                 "posture_off": 0.7, "cue_on": 0.6, "cue_off": 0.3, "max_gap_frames": 3},
    "baseline": {"baseline_seconds": 20, "min_samples": 10},
    "cues": {"kp_conf": 0.5, "peek_min_turn": 0.4, "peek_max_distance": 2.0, "peek_min_s": 2.0,
             "pass_distance": 0.5, "pass_min_s": 1.5, "shoulder_fallback_box_fraction": 0.5,
             "head_down_pitch": -0.2, "head_down_delta": 0.25, "hands_hidden_min_s": 5.0,
             "min_visible_keypoints": 6, "low_vis_min_s": 2.0, "fusion_window_s": 3.0},
    "seats": {"expected_count": 0, "zones": [], "max_outside_s": 10},
    "alerts": {"cooldown_s": 30, "gap_tolerance_s": 0.5, "save_clips": False, "clip_before_s": 5,
               "clip_after_s": 5,
               "tiers": {"phone": "red", "head_turn": "orange", "lean": "orange", "out_of_seat": "orange",
                         "neighbour_peek": "orange", "object_passing": "orange", "hands_hidden": "orange",
                         "combined": "orange", "low_visibility": "grey"}},
    "evaluation": {"match_tolerance_s": 1.0, "sweep_conf": [0.2, 0.3], "sweep_min_duration": [0.5, 1.0],
                   "gallery_max": 6, "map_conf": 0.01, "map_iou": 0.5, "gt_class_id": 0, "fps_warmup": 0},
    "metrics": {"near_threshold_factor": 1.5},
    "runtime": {"every_nth_frame": 1, "device": "cpu", "imgsz": 640, "fallback_fps": 30, "log_every": 1000},
    "live": {"realtime": False, "display_width": 640, "jpeg_quality": 80},
    "privacy": {"blur_heads": True, "head_keypoint_conf": 0.3, "head_box_scale": 2.5,
                "min_head_width_fraction": 0.3, "fallback_head_fraction": 0.25, "blur_kernel": 51},
}


def make_cfg(**features: bool) -> dict:
    """A fresh config with the given features switched on (all others off)."""
    cfg = copy.deepcopy(BASE_CFG)
    cfg["features"].update(features)
    return cfg


def make_record(
    track_id: int = 1, x: float = 100, t: float = 0.0, *, turn: float = 0.0, ear_turn: float | None = None,
    lean_dx: float = 0.0, pitch: float = -0.5, wrists: bool = True, conf: float = 0.9,
    visible: set[int] | None = None, phone_conf: float = 0.0,
) -> FrameRecord:
    """A seated student facing the camera, box from x to x+150 (y 50..400),
    shoulders 100 px apart.

    turn      nose offset from the shoulder midpoint, in shoulder widths
    ear_turn  nose offset from the ear midpoint, in half ear-distances (default: same as turn x 2)
    lean_dx   hips shifted sideways by this many pixels (torso lean)
    pitch     nose height relative to the shoulder line, in shoulder widths
    wrists    whether the wrists are visible
    visible   if given, only these keypoint indices are visible
    """
    k = np.zeros((17, 3), np.float32)
    mid = x + 75
    ear_offset = (turn * 2 if ear_turn is None else ear_turn) * 20  # ears 40 px apart
    k[0] = [mid + turn * 100, 150 + pitch * 100, conf]         # nose
    k[1], k[2] = [mid - 8, 140, conf], [mid + 8, 140, conf]    # eyes
    k[3], k[4] = [mid - 20 - ear_offset, 145, conf], [mid + 20 - ear_offset, 145, conf]  # ears
    k[5], k[6] = [mid - 50, 150, conf], [mid + 50, 150, conf]  # shoulders
    k[7], k[8] = [mid - 55, 220, conf], [mid + 55, 220, conf]  # elbows
    if wrists:
        k[9], k[10] = [mid - 40, 280, conf], [mid + 40, 280, conf]
    k[11], k[12] = [mid - 40 + lean_dx, 330, conf], [mid + 40 + lean_dx, 330, conf]  # hips
    if visible is not None:
        for i in range(17):
            if i not in visible:
                k[i, 2] = 0.0
    phone_box = np.array([mid + 30, 270, mid + 50, 290], np.float32) if phone_conf else None
    return FrameRecord(frame_index=int(round(t * 10)), timestamp=t, track_id=track_id,
                       person_box=np.array([x, 50, x + 150, 400], np.float32), keypoints=k,
                       phone_conf=phone_conf, phone_box=phone_box)


def run_scorer(cfg: dict, frames: list[list[FrameRecord]], fps: int = 10) -> list:
    """Feed frames (one list of records per frame, fps per second) to a
    RiskScorer and return every alert, including those released by finish()."""
    from src.risk import RiskScorer

    scorer = RiskScorer(cfg)
    alerts = []
    for i, records in enumerate(frames):
        alerts += scorer.update(i / fps, records)
    alerts += scorer.finish((len(frames) - 1) / fps)
    return alerts


def run_all(namespace: dict) -> None:
    """Run every test_* function in a module (for running without pytest)."""
    tests = [obj for name, obj in sorted(namespace.items()) if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"All {len(tests)} tests passed.")
