"""Tests for the model-free parts of src/detect_phone.py and the crop-owner
step of tracking.associate_phones. Run: python tests/test_detect_phone.py"""

from __future__ import annotations

import numpy as np
from helpers import make_cfg, run_all

from src.detect_phone import PhoneDetection, crop_region, detection_mode, nms
from src.tracking import Person, associate_phones


def person(track_id: int, x: float, wrist_x: float | None = None) -> Person:
    k = np.zeros((17, 3), np.float32)
    if wrist_x is not None:
        k[10] = [wrist_x, 300, 0.9]
    return Person(track_id, np.array([x, 50, x + 150, 400], np.float32), k)


def test_nms_keeps_best_of_overlapping_boxes() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    assert nms(boxes, np.array([0.6, 0.9, 0.5]), 0.5) == [1, 2]
    assert nms(boxes, np.array([0.6, 0.9, 0.5]), 0.95) == [1, 0, 2]  # barely-overlapping kept


def test_crop_region_grows_more_downward_and_includes_wrists() -> None:
    k = np.zeros((17, 3), np.float32)
    x1, y1, x2, y2 = crop_region(np.array([100, 100, 200, 300]), k, (1000, 1000, 3),
                                 pad=0.25, down_bias=0.5, kp_conf=0.5)
    assert (x1, x2) == (75, 225)
    assert 100 - y1 == 25 and y2 - 300 == 75  # 25% of the growth above, 75% below
    k[9] = [400, 150, 0.9]                     # a wrist stretched far to the right
    assert crop_region(np.array([100, 100, 200, 300]), k, (1000, 1000, 3),
                       pad=0.25, down_bias=0.5, kp_conf=0.5)[2] >= 400
    assert crop_region(np.array([100, 100, 200, 300]), k, (500, 300, 3),   # clipped to the frame
                       pad=0.25, down_bias=0.5, kp_conf=0.5)[2] == 300


def test_detection_mode_accepts_old_sahi_flag() -> None:
    cfg = make_cfg()
    assert detection_mode(cfg) == "full"
    del cfg["phone"]["mode"]
    cfg["phone"]["sahi"] = True
    assert detection_mode(cfg) == "sahi"


def test_crop_owner_is_tried_first_but_must_pass_checks() -> None:
    args = {"keypoint_conf": 0.5, "max_wrist_distance": 0.35, "min_box_overlap": 0.5}
    owner = person(1, x=100, wrist_x=230)        # owner's wrist at (230, 300)
    other = person(2, x=230, wrist_x=232)        # other wrist even closer to the phone
    phone = np.array([225, 290, 245, 310], np.float32)  # centre (235, 300)
    # Without crop info the nearest wrist (person 2) wins ...
    assert associate_phones([PhoneDetection(phone, 0.8)], [owner, other], **args) == [2]
    # ... but a phone found in person 1's own crop goes to person 1 (their wrist is close enough).
    assert associate_phones([PhoneDetection(phone, 0.8, crop_owner=1)], [owner, other], **args) == [1]
    # The crop owner still has to pass the normal checks: here they are far away.
    far_owner = person(1, x=600, wrist_x=650)
    assert associate_phones([PhoneDetection(phone, 0.8, crop_owner=1)], [far_owner, other], **args) == [2]


if __name__ == "__main__":
    run_all(globals())
