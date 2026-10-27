# Exam Hall Monitor

A CCTV-based exam hall monitor that detects **likely** phone use and
suspicious posture and **flags them for human review**. It never accuses
anyone: every output is a prompt for an invigilator to look at a clip.

Heads are blurred in every saved video, using pose keypoints only
(no face detection, no face recognition).

## Quick start on Colab (live scan)

1. Copy the `exam-hall-monitor` folder (or unzip `exam-hall-monitor.zip`) into
   **Google Drive → My Drive**.
2. Open [`notebooks/colab_live_demo.ipynb`](notebooks/colab_live_demo.ipynb)
   from Drive with Google Colab, and set *Runtime → Change runtime type → T4 GPU*.
3. Run the cells top to bottom. The upload cell asks for a video, and the scan
   cell plays the annotated video live inside the notebook.

## Setup (local)

```bash
cd exam-hall-monitor
python -m venv .venv
# Windows:    .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads `yolov8n-pose.pt` and `yolov8n.pt` (~6 MB each) into
the repo root. On its first tracking call Ultralytics may also auto-install
`lap`, which ByteTrack needs.

Put sample videos in `data/` (git-ignored). Footage of real people goes in
`data/our_clips/` and must never be committed or shared.

## Configuration

All thresholds and paths are in [`config.yaml`](config.yaml). Relative paths
are relative to the repo root. Useful knobs while testing:

| Setting | What it does |
|---|---|
| `runtime.every_nth_frame` | Process 1 frame in N. Raise it if CPU is too slow. |
| `runtime.device` | `auto` uses the GPU when PyTorch sees one, otherwise the CPU. |
| `live.realtime` | `true` = live view never plays faster than the real video. |
| `phone.conf_threshold` | Lower = more phone boxes (and more false ones). |
| `paths.phone_weights` / `phone.class_id` | `yolov8n.pt` + `67` (COCO) now; fine-tuned `best.pt` + `0` later. |
| `association.max_wrist_distance` | How close (as a fraction of person height) a phone must be to a wrist. |

## Running each module

Run everything from the repo root.

| Module | Command | Status |
|---|---|---|
| Video/config check | `python -m src.utils --video data/sample.mp4` | done |
| Tracking + phones (saves video, no live view) | `python -m src.tracking --video data/sample.mp4` | done |
| Live scan in an OpenCV window (needs a screen, q stops) | `python -m src.live --video data/sample.mp4` | done |
| Dashboard: live scan + review alerts (Confirm / Dismiss) | `streamlit run app.py --server.maxUploadSize 1000` | done |
| Live scan in a notebook | `play_in_notebook(video, cfg)`, see the Colab notebook | done |
| Flags for review (scan + list alerts) | `python -m src.risk --video data/sample.mp4` | done |
| Pose statistics per person (for tuning thresholds) | `python -m src.pose_rules --video data/sample.mp4` | done |
| Time outside seat zones (needs `seats.zones`) | `python -m src.seat_count --video data/sample.mp4` | done |
| Phone detection with one mode (full / sahi / crop) | `python -m src.detect_phone --video data/sample.mp4 --mode crop` | done |
| Evaluation against labelled clips | `python evaluate.py --video data/clip.mp4 --labels data/labels.csv` | done |
| Compare the 3 phone detection modes | `python evaluate.py --compare-modes --gt data/phone_set` | done |
| Metrics for a scanned video (no labels needed) | `python -m src.metrics --alerts results/sample_alerts.csv --plot` | done |
| Tests (no video or models needed) | `python -m pytest tests` (or `python tests/test_cues.py` etc.) | done |

`--video` can be left out if `paths.input_video` is set in `config.yaml`.

### What the scan produces (tracking, live window, app and notebook alike)

- `results/<video name>_tracked.mp4` showing blurred heads, a coloured box +
  `ID n` + skeleton per person, cyan phone boxes with a line to the person
  they were assigned to, grey "unassigned" phone boxes, and a status line at
  the top (video time, people, phones). The live viewers show the same frames
  while they are being processed, plus a stats line (progress, speed vs real
  time).
- A summary printed to the terminal: frames processed, unique track IDs,
  phone detections, assigned vs unassigned, and the list of flags.
- `results/<video name>_alerts.csv`: one row per flag: `alert_id`, time,
  student ID, tier, reason, clip/thumbnail paths, `signals` (JSON with the
  numbers behind the flag), and `review_decision` / `reviewed_at`, which the
  dashboard fills in when you press Confirm or Dismiss.
- `results/clips/<video name>/`: a short clip around every flag (heads
  blurred), e.g. `01m23s_ID4_phone.mp4`, plus a `.jpg` of the peak frame when
  `evidence_alerts` is on.
- `results/<video name>_summary.json`: the run summary for `src/metrics.py`
  (speed, milliseconds per frame for each stage, people per frame, track IDs).

### Flags for human review

The system flags behaviours that are worth a look. It never decides that
anyone cheated; a person reviews every flag.

| Behaviour | Flagged when it lasts at least | Default tier |
|---|---|---|
| Phone near a person's hands | `phone.min_duration_s` (1 s) | red |
| Head turned to the side | `pose.min_duration_s` (2 s) | orange |
| Leaning (needs hips visible) | `pose.min_duration_s` (2 s) | orange |
| Outside all seat zones (needs `seats.zones`) | `seats.max_outside_s` (10 s) | orange |
| Head turned toward a neighbour's box | `cues.peek_min_s` (2 s) | orange |
| Wrists of two students close together (possible object passing) | `cues.pass_min_s` (1.5 s) | orange |
| Head down with both hands not visible | `cues.hands_hidden_min_s` (5 s) | orange |
| Two different orange cues within `cues.fusion_window_s` | one combined alert instead of two | orange |
| Too few keypoints visible: "cannot assess" (posture alerts paused, phone alerts not) | `cues.low_vis_min_s` (2 s) | grey |

Flagged people get a thick red/orange/grey box with `REVIEW: ...` in the video.
The same person + behaviour is not flagged again within `alerts.cooldown_s`.
Before trusting the head-turn rule on a new camera, run
`python -m src.pose_rules` on a normal clip and set
`pose.head_turn_threshold` above what most people show when facing forward.

### Feature switches (for before/after comparisons)

Each improvement can be turned on or off under `features:` in `config.yaml`
(or with the sidebar switches in the dashboard). With all of them `false` and
`phone.mode: full`, the pipeline behaves exactly like the original.

| Switch | What it changes | Settings |
|---|---|---|
| `temporal_hysteresis` | Phone, head-turn and lean events use a smoothed signal with separate start/end thresholds and gap tolerance (`src/temporal.py`), instead of the simple timer | `temporal:` |
| `baseline_calibration` | Head turn, lean and head pitch are measured relative to each student's own first `baseline.baseline_seconds` on camera | `baseline:` |
| `evidence_alerts` | Detailed reason text (e.g. "Phone near right wrist, 4.2 s, mean conf 0.71, max 0.88") and a head-blurred thumbnail of the peak frame next to each clip | none |
| `neighbour_peek` | Head turned toward another student's box | `cues:` |
| `object_passing` | Wrists of two students close together | `cues:` |
| `hands_hidden` | Head down and both wrists not visible | `cues:` |
| `low_visibility` | Grey "cannot assess" state; pauses posture alerts for that student | `cues:` |
| `cue_fusion` | Two different orange cues close together become one combined alert (orange alerts wait up to `cues.fusion_window_s` before appearing) | `cues:` |

### Phone detection modes (`phone.mode`)

| Mode | How it works | Good for |
|---|---|---|
| `full` | One detection pass over the whole frame (original) | speed |
| `sahi` | The frame is cut into overlapping tiles (SAHI), each tile is detected, boxes merged | small, far-away phones |
| `crop` | One crop per person around hands and desk, resized to `phone.crop_size`; boxes merged with NMS | phones in laps / on desks |

All three use the same weights. To compare them fairly on labelled images:
`python evaluate.py --compare-modes --gt <YOLO dataset folder or COCO JSON>`
writes `results/mode_comparison.csv` (precision, recall, mAP50, fps, device).

## Evaluation for the report (`evaluate.py`)

Label your test clips in a `labels.csv`, one row per event:

```
clip,event_type,start_s,end_s,angle,lighting
exam1.mp4,phone,12.0,18.5,front,bright
exam2.mp4,none,,,side,dim
```

`event_type` is one of `phone, head_turn, lean, out_of_seat, neighbour_peek,
object_passing, hands_hidden`; `none` marks an honest clip (no events).
`angle` and `lighting` are optional.

```bash
python evaluate.py --clips-dir data/eval --labels data/labels.csv --sweep --ablation --gallery
```

| Output | What it is |
|---|---|
| console tables | per-clip and overall precision / recall / F1, false alerts per hour |
| `results/sweep.csv`, `results/sweep.png` | phone recall vs false alerts/hour for 8-10 confidence thresholds and minimum durations (current setting ringed) |
| `results/ablation.csv` | all features off / each feature alone / all on |
| `results/per_condition.csv` | recall and false alerts/hour per angle and lighting (only if those columns exist) |
| `results/gallery/` | up to 6 false positives and 6 false negatives (head-blurred) + `gallery.csv` with an empty `cause` column for you to fill in |

An alert counts as correct if it overlaps a labelled event of the same type
(± `evaluation.match_tolerance_s`); each labelled event can be matched once.
Detections are cached in `results/cache/`, so sweeps and ablations only re-run
the cheap rule logic, not the models.

## Metrics without labels (`src/metrics.py`)

`evaluate.py` needs labelled clips. `src/metrics.py` measures a scan without them,
and turns review decisions into precision:

```bash
python -m src.metrics --alerts results/sample_alerts.csv --plot
```

| Group | Metrics |
|---|---|
| Run (from `<video>_summary.json`) | processed frames/s, ms per frame for models / rules / drawing, max and mean people per frame, track IDs per visible person (1.0 = nobody lost their ID) |
| Alerts | alerts per minute / hour by tier and cue, students alerted, alerts per student, share of head-turn detections that used a personal baseline, median head turn vs `pose.head_turn_threshold`, share of events that fired at the minimum duration, smallest same-cue gap (cooldown check) |
| Review | precision = confirmed / (confirmed + dismissed) for red + orange, per tier and per cue, each with a 95% Wilson confidence interval; grey alerts are reported separately as correct abstentions |

Review decisions come from the `review_decision` column (dashboard). If nobody
has reviewed yet, a `draft_verdict` column (`supported` / `not_supported` /
`unclear`) with an optional `draft_cause` is used instead. Output:
`results/<video>_metrics.json` (+ `_metrics.png` with `--plot`).
`evaluate.py` also prints 95% confidence intervals for precision and recall now.

## Reviewing alerts (dashboard)

`streamlit run app.py` → **Review alerts** tab: one card per alert with its
tier colour, reason, thumbnail, a **Play clip** button and **Confirm** /
**Dismiss** buttons. Decisions are saved to the alerts CSV. The summary panel
shows alerts by tier, how many are reviewed, and reviewer precision
(confirmed / (confirmed + dismissed)).

### Running on CPU

`runtime.device: auto` uses the GPU when PyTorch can see one, otherwise the
CPU. On CPU, `runtime.cpu_overrides` applies (every 3rd frame, image size 480)
to keep it usable. The live view then plays slower than real time; the
stats line shows the speed (e.g. `0.4x real time`).

## Training (Colab)

Training happens on Google Colab, not locally. Put notebooks in
`notebooks/`, then copy the resulting `best.pt` into the repo root and point
`paths.phone_weights` at it (with `phone.class_id: 0`).

## Troubleshooting

- **`OSError: [WinError 4551] An Application Control policy has blocked this
  file ... c10.dll`**: Windows (Smart App Control / WDAC) is blocking
  PyTorch's DLLs. Either run on Colab, or turn Smart App Control off
  (Windows Security → App & browser control → Smart App Control).
  **Warning:** turning it off is one-way; Windows only lets you turn it back
  on by resetting or reinstalling Windows.
- **Very slow on CPU**: raise `runtime.every_nth_frame` (e.g. 3–5) or lower
  `runtime.imgsz` to 480.
