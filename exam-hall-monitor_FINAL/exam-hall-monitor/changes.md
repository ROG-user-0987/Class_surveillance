# Changes: the 5-phase upgrade

This file explains every change made to the code for the 5-phase upgrade, why
it was made, the decisions taken on your behalf, and what still needs testing.
All work is on the git branch `improvements` (nothing is committed yet).

**The one rule behind everything:** every new behaviour has an on/off switch
under `features:` in `config.yaml`. With **all switches `false` and
`phone.mode: full`, the pipeline behaves exactly like the original**. This
was checked by running the same synthetic video through the old code
(`main`) and the new code: identical alerts and identical annotated frames.
The only difference is extra columns in the alerts CSV. This lets you run
fair before/after comparisons (ablations) for your report.

---

## Phase 1: Temporal logic, per-student baseline, better alert text

### New file `src/temporal.py`: `EventTracker`
- **What:** follows one signal per student (e.g. phone confidence) and turns
  it into events:
  - **Smoothing (EMA):** `ema = 0.3 × new value + 0.7 × previous`, starting at 0.
  - **Hysteresis:** an event *starts* above `on` and only *ends* below `off`
    (off < on).
  - **Gap tolerance:** up to `max_gap_frames` bad or missing frames don't end
    an event.
  - **Minimum duration:** it fires once after `min_duration_s`.
  - **Statistics:** each event keeps its start, duration, mean/max signal and
    peak time.
- **Why:** the original timer reacted to raw, flickering detections. Smoothing
  plus two thresholds means one bright false detection can't start an event,
  and one missed frame can't end a real one.
- **Switch:** `features.temporal_hysteresis`; settings under `temporal:`.
- **Later additions (Phase 3):** `pop_ended()` and `end_all()`, so grey periods
  can be logged when they finish, and keys that can be a *pair* of students
  (used by object passing).

### `src/pose_rules.py`: signed measurements, head pitch, per-student baseline
- **What:**
  - New `head_turn_signed`, `lean_angle_signed` and `head_pitch_ratio` (nose
    height relative to the shoulders).
  - The old `head_turn_ratio` and `lean_angle_deg` now return the absolute
    value of the signed versions. A test checks they give exactly the same
    answers as before on 3,000 random poses.
  - New `PostureBaseline`: for each student, the median of their first
    `baseline_seconds` (20 s) of head turn, lean and pitch becomes "normal".
    Posture is then measured as distance from *their* normal.
- **Why:** a camera placed at an angle makes everyone look slightly "turned".
  Measuring each student against their own normal removes that bias.
- **Limitations (documented in the code):**
  - A student who gets a new track ID (ByteTrack lost them) starts
    calibrating again.
  - Whatever a student does during calibration becomes their "normal".
  - Absolute thresholds still apply until a baseline exists.
- **Switch:** `features.baseline_calibration`; settings under `baseline:`.

### `src/risk.py`: evidence-rich alerts
- **What:**
  - Reason text now says which cue, which side, how long and how confident,
    e.g. "Phone near right wrist, 4.2 s, mean conf 0.71, max 0.88" or
    "Head turned toward image left, 5.1 s … (baseline-adjusted)".
    "Left/right wrist" means the *student's own* wrist (the pose model's
    convention).
  - A head-blurred **thumbnail** of the peak frame is saved next to each clip.
  - New `Alert` fields:
    - `thumbnail_path`;
    - `alert_id` (unique, e.g. `exam1-A0003`);
    - `signals` (JSON with the numbers behind the alert: duration, mean, max,
      peak time, baseline, side, method).
- **Why:** a reviewer needs to see *why* something was flagged before
  watching the clip.
- **Switch:** `features.evidence_alerts`. `alert_id` and `signals` are always
  filled in; they are bookkeeping and don't change what gets flagged.

### `src/utils.py`, `src/tracking.py`
- `ClipRecorder` remembers the time of each buffered frame (`jpeg_near()`),
  so the thumbnail can be taken at the peak moment.
- New `encode_jpeg()` helper.
- `VideoScan` gives each alert its id, clip and thumbnail (`_finish_alert`).

### Tests
- `tests/test_temporal.py` (8 tests):
  - flicker does not trigger;
  - a single-frame dropout does not end an event;
  - a sustained signal fires after `min_duration_s`;
  - a long gap ends an event;
  - baseline subtraction works;
  - the new posture readings match the original rule exactly.

---

## Phase 2: Phone detection modes (full / sahi / crop) and a fair comparison

### `src/detect_phone.py` (was a stub)
- **`phone.mode` in config:**
  - `full`: one pass over the whole frame (the original behaviour).
  - `sahi`: the frame is cut into overlapping tiles (`slice_size`,
    `slice_overlap`), each tile is detected, and boxes are merged with NMS.
  - `crop`: one crop per person covering hands and desk:
    - the person box grows by `crop_pad`, with most of the growth downward
      (`crop_down_bias`);
    - it is stretched to include any visible wrist;
    - it is resized to `crop_size` and detected;
    - boxes are mapped back to the full frame, and duplicates from
      neighbouring crops are merged with NMS (`nms_iou`).
- All three modes use the **same weights** (`paths.phone_weights`).
- **Why:** phones are small. Slicing or cropping lets the detector see them at
  a larger scale. Crop mode only looks where hands are.
- **Pre-association:** a phone found in a student's own crop remembers that
  student (`crop_owner`). `associate_phones()` tries them first, but they
  still have to pass the normal wrist-distance or box-overlap check.
- **Comparison:** `compare_modes()` runs every mode on the same labelled
  images:
  - Every mode's boxes are exported to COCO JSON and scored by the **same
    pycocotools evaluator** (mAP50, plus precision/recall at your confidence
    threshold, taken from the same box matching).
  - FPS is measured end to end, on the same device.
  - Results go to `results/mode_comparison.csv` with columns `model, weights,
    mode, precision, recall, mAP50, fps, device`.
- **Ground truth:** you hadn't said which label format your dataset uses, so it
  accepts **both** a YOLO folder (`images/` + `labels/*.txt`) and a COCO JSON
  file.
- Command-line use: `python -m src.detect_phone --video x.mp4 --mode crop`
  prints box counts and speed for one mode.

### `src/tracking.py`
- `PhoneDetection` moved to `detect_phone.py` (still importable from
  `tracking`, so old imports keep working) and gained `crop_owner`.
- `ExamHallTracker` uses `PhoneDetector` (the old "SAHI not wired in yet"
  warning is gone).
- New public `detect()` method (people + phones, before association) for the
  evaluation cache.
- `associate_phones()` got the crop-owner step (step 0). A new
  `wrist_distance()` helper is shared by both steps; same result as before
  when there's no crop owner.

### Tests
`tests/test_detect_phone.py` (4 tests):
- NMS;
- the crop region shape;
- the old `phone.sahi: true` setting still works;
- the crop owner is preferred but must still pass the normal checks.

---

## Phase 3: Extra cues and a "cannot assess" state

### New file `src/cues.py` (kept separate because `pose_rules.py` was getting long)
All of these are orange tier, and each has its own switch:
- **`neighbour_peek`:** head direction is taken from the nose position relative
  to the ear midpoint (eyes if an ear is hidden). A horizontal ray goes from
  the nose in that direction. If it reaches another student's box within
  `peek_max_distance` shoulder widths for `peek_min_s`, an alert fires naming
  that student: "Head turned toward student #7, 5.1 s".
- **`object_passing`:** a wrist of one student within `pass_distance` shoulder
  widths of a wrist of another student, for `pass_min_s`. The cooldown applies
  to **both** students.
- **`hands_hidden`:** the head is pitched down (relative to the student's own
  baseline if one exists, else an absolute threshold) **and** both wrists are
  hidden, for `hands_hidden_min_s`. Reason: "Head down with hands not visible".
- **`low_visibility`:** fewer than `min_visible_keypoints` keypoints visible
  puts the student in the **grey "cannot assess"** state.
  - Posture alerts for that student are paused while the state lasts. Phone
    alerts are **not** paused.
  - The grey period is logged in the alerts CSV with tier `grey` and its full
    length.

### `src/risk.py`
- Runs the cues through `EventTracker`s.
- Handles the grey state, logging grey periods when they end and at the end of
  the video (`finish()`).
- **Combined alerts (`features.cue_fusion`):** if two *different* orange cues
  fire for the same student within `fusion_window_s`, you get **one** combined
  orange alert ("Combined: Head turned …; Leaning …") instead of two.

### `src/tracking.py`, `src/utils.py`
- A grey colour for the new tier.
- `VideoScan` collects held and grey alerts at the end of the video.
- The summary counts grey alerts separately.

### Tests
`tests/test_cues.py` (9 tests): each cue, the grey state pausing posture but
not phone alerts, and combined alerts.

---

## Phase 4: Dashboard reviewer workflow (`app.py`)

- **Banner at the top:** "Alerts are for human review only. They are not
  evidence of misconduct."
- **Sidebar:**
  - a switch for every flag in `features:`;
  - a selector for the phone detection mode (full / sahi / crop);
  - both apply to the **next scan**, so you can demo before/after live;
  - a picker for which alerts CSV to review.
- **Live scan tab:** as before.
- **Review alerts tab:**
  - **Alert cards:** tier colour (red/orange/grey), reason, time, student ID,
    thumbnail, a **Play clip** button, and **Confirm** / **Dismiss** buttons.
  - **Decisions** are stored in `st.session_state` and written to the CSV
    (`review_decision` = confirmed / dismissed, plus a `reviewed_at` time).
    New helpers in `risk.py`: `read_alerts_csv`, `update_review_decision`.
  - **Summary panel:** alerts by tier, how many are reviewed, and reviewer
    precision = confirmed / (confirmed + dismissed). Grey alerts are left out
    of precision because they aren't detections.
- **Why "Play clip" shows frames rather than a video player:** OpenCV saves
  clips in the mp4v format, which browsers can't play. The app reads the clip
  with OpenCV and shows it frame by frame, which works everywhere.
- If a CSV was made on Colab and moved, the app also looks for clips and
  thumbnails in `clips/` next to the CSV.

---

## Phase 5: Evaluation for the report (`evaluate.py`, was a stub)

- **Labels (`labels.csv`):** one row per event: `clip,event_type,start_s,end_s`
  plus optional `angle,lighting`. A row with `event_type` `none` marks an
  honest clip. The full format is at the top of `evaluate.py` and in the README.
- **5a, matching rule (written in the code):**
  - An alert is a **true positive** if it overlaps a labelled interval of the
    same type, allowing `match_tolerance_s` (1 s) on each side.
  - Each labelled event can be matched once. A second alert on the same event
    counts as a false positive.
  - A combined alert matches any of its cues. Grey alerts are ignored.
  - False alerts per hour are counted on **honest clips**, or on all clips if
    there are no honest ones.
- **5b, `--sweep`:**
  - Tries every `evaluation.sweep_conf` (10 values) and
    `evaluation.sweep_min_duration` (8 values).
  - Writes `results/sweep.csv` and `results/sweep.png`: recall vs false
    alerts/hour, with the current config setting ringed in red.
  - **Detections are cached** in `results/cache/`, so the models run once per
    clip. Every later run only re-applies the rules (seconds instead of minutes).
- **5c, `--ablation`:** all features off / each feature alone / all on →
  `results/ablation.csv` (precision, recall, F1, false alerts/hour).
- **5d, per-condition:** if `labels.csv` has `angle`/`lighting`, recall and
  false alerts/hour per value go to `results/per_condition.csv`. Otherwise a
  note is printed (no crash).
- **5e, `--gallery`:** up to 6 false positives and 6 false negatives as
  head-blurred images in `results/gallery/`, plus `gallery.csv` with an empty
  `cause` column for you to fill in.
- **Console output:** a summary table is printed after each run.
- **Undefined values:** precision with no alerts, and recall with no labelled
  events, are shown as "-" (blank in the CSV), not 0. A 0 would be misleading
  in a report.
- **Mode comparison** from the command line:
  `python evaluate.py --compare-modes --gt <dataset>`.

### Tests
`tests/test_evaluate.py` (5 tests):
- the tolerance window;
- types must match;
- each event is matched only once;
- combined and grey alerts are handled correctly;
- false alerts/hour are counted on honest clips.

---

## Other files

- **`config.yaml`:**
  - new sections `features:` (8 switches), `temporal:`, `baseline:`,
    `cues:` and `evaluation:`;
  - new `phone:` crop/NMS settings;
  - new tiers for the new behaviours.
  - Every new number has a comment.
  - `phone.sahi: false` was replaced by `phone.mode: full`; the old key is
    still understood.
- **`requirements.txt`:** added `pycocotools` (see decisions below).
- **`.gitignore`:** also ignores result images, JSON, the detection cache and
  the pytest cache. They come from footage of real people.
- **`README.md`, `CLAUDE.md`:** updated for all of the above.
- **`notebooks/colab_live_demo.ipynb`:** new cells to run the tests and the
  evaluation.
- **`tests/helpers.py`:** builds a test config and synthetic students, so tests
  need no video, models or YAML file.

---

## Decisions made on your behalf

1. **Added `pycocotools`:** your Phase 2 spec named it for scoring, and you
   asked for all phases to be done.
2. **Both label formats** (YOLO txt and COCO JSON) are accepted for the mode
   comparison, because the dataset's format wasn't known.
3. **Mode-comparison FPS includes person detection:** the real pipeline always
   runs the pose model, and crop mode needs it. Including it in every mode
   keeps the comparison fair.
4. **Out-of-seat keeps the simple timer:** it's a yes/no zone check, not a
   posture signal, so smoothing adds nothing.
5. **The new cues always use `EventTracker`,** whatever
   `temporal_hysteresis` says. They are new behaviours, so there is no
   "original" timing to preserve.
6. **The grey pause starts as soon as low visibility is detected,** not after
   `low_vis_min_s`. Otherwise a 2-second posture cue could fire just before a
   2-second grey state began. `low_vis_min_s` only decides whether the period
   is logged.
7. **Combined alerts are delayed:** orange alerts wait up to
   `fusion_window_s` (3 s) so a second cue can be combined with them. Red
   (phone) alerts are never delayed.
8. **No phone + posture fusion was added.** Your spec said "keep the existing
   phone+posture fusion", but the code never had one. Adding it would be a new
   design decision, so it's left for you to decide.

## Trade-offs to mention in your report
- **Hysteresis vs flicker:** a phone detected weakly on every other frame
  (e.g. confidence 0.4) may never start an event. The smoothed value stays
  below `phone_on`. This is the price of fewer false alerts.
- **False alerts during calibration:** in the first `baseline_seconds`,
  absolute thresholds apply. A student who looks turned only because of the
  camera angle can be flagged then, and that alert's cooldown can hide a real
  event shortly after.
- **Crop mode on CPU is slow:** one detection per student (e.g. 30 crops per
  frame).

## Not tested yet (PyTorch is blocked on the development laptop)
Everything was tested with synthetic data and a fake detector: 26 unit tests,
the main-vs-new comparison, and the full evaluation run including the sweep
chart, cache and gallery. **Not yet run with the real models:**
- real YOLO pose + phone detection in all three modes;
- SAHI: its API changed between versions, and the code tries both the
  `ultralytics` and `yolov8` model-type names;
- pycocotools scoring (`--compare-modes`);
- the Streamlit dashboard in a browser.

## How to test (on Colab, after unzipping into Drive)
```bash
python -m pytest -q tests                                  # 26 tests, no models needed
python -m src.risk --video data/<clip>.mp4                 # alerts with the new text + thumbnails
python evaluate.py --clips-dir data/eval --labels data/labels.csv --sweep --ablation --gallery
python evaluate.py --compare-modes --gt data/<phone dataset>
streamlit run app.py                                       # review dashboard
```
To reproduce the original behaviour for comparison, set every switch under
`features:` to `false` and `phone.mode: full`.

## Fixes after the upgrade

- **`notebooks/colab_live_demo.ipynb`, cell 2**: it used to `%cd` into a fixed
  path (`My Drive/exam-hall-monitor`). If the folder wasn't exactly there, the
  next cell failed with `Could not open requirements file: 'requirements.txt'`.
  Google Drive does not unzip uploaded zips, so uploading `exam-hall-monitor.zip`
  alone caused exactly this. The cell now looks for the project folder in My
  Drive (up to two folders deep, any name starting with `exam-hall-monitor`)
  and, if it only finds the zip, unzips it next to the zip first.

---

## Metrics (added after the first demo run)

### New file `src/metrics.py`
- **What:** `python -m src.metrics --alerts results/<video>_alerts.csv [--plot]`
  measures a scan without labelled clips:
  - **Run:** processed frames/s, milliseconds per frame for each stage
    (models, rules, drawing), people per frame, and track IDs per visible
    person (how often ByteTrack lost and re-issued IDs).
  - **Alerts:** alerts per minute and per hour by tier and cue, students
    alerted, alerts per student, how many head-turn detections used a personal
    baseline, how far above `pose.head_turn_threshold` the turns were, share of
    events that fired exactly at the minimum duration, smallest gap between
    two alerts of the same cue for one student (cooldown check).
  - **Review:** precision = confirmed / (confirmed + dismissed) for red +
    orange alerts, overall, per tier and per cue, with 95% Wilson confidence
    intervals. Grey alerts are counted separately (correct abstentions).
    Uses `review_decision` from the dashboard, or a `draft_verdict` /
    `draft_cause` column when no one has reviewed yet.
- **Why:** our recording has no labelled events yet, so `evaluate.py` could not
  produce numbers. These metrics come straight from the alert log, and review
  precision needs only the Confirm / Dismiss buttons. Confidence intervals
  matter because we have dozens of alerts, not thousands.
- **Settings:** `metrics.near_threshold_factor` in `config.yaml`.

### `src/tracking.py`: stage timing and a summary file
- `VideoScan` times three stages per frame with `StageTimer` (models, rules,
  drawing), counts people per frame, and writes
  `results/<video>_summary.json` when a scan ends (also when it is stopped
  early). `print_summary` shows ms per stage and people per frame.
- Outputs (video, alerts, clips) are unchanged, so the "all switches off =
  original" rule still holds.

### `evaluate.py` and `app.py`
- `score()` also returns `precision_ci95` and `recall_ci95` (Wilson); the
  console prints them under the event table.
- The dashboard shows the 95% interval under "Reviewer precision".

### Tests
- `tests/test_metrics.py` (6 tests, synthetic alerts, no video or models):
  Wilson interval, alert metrics, review precision ignoring grey and
  unclear, dashboard decisions taking priority over drafts, summary
  metrics, stage timer.
