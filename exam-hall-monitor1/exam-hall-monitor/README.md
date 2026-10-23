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
| Live scan web app with upload button | `streamlit run app.py --server.maxUploadSize 1000` | done |
| Live scan in a notebook | `play_in_notebook(video, cfg)`, see the Colab notebook | done |
| Flags for review (scan + list alerts) | `python -m src.risk --video data/sample.mp4` | done |
| Pose statistics per person (for tuning thresholds) | `python -m src.pose_rules --video data/sample.mp4` | done |
| Time outside seat zones (needs `seats.zones`) | `python -m src.seat_count --video data/sample.mp4` | done |
| Phone detector (SAHI) | `python -m src.detect_phone --video data/sample.mp4` | stub |
| Evaluation | `python evaluate.py --video data/sample.mp4` | stub |

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
- `results/<video name>_alerts.csv`: one row per flag, with an empty
  `review_decision` column for the reviewer.
- `results/clips/<video name>/`: a short clip around every flag (heads
  blurred), e.g. `01m23s_ID4_phone.mp4`.

### Flags for human review

The system flags behaviours that are worth a look. It never decides that
anyone cheated; a person reviews every flag.

| Behaviour | Flagged when it lasts at least | Default tier |
|---|---|---|
| Phone near a person's hands | `phone.min_duration_s` (1 s) | red |
| Head turned to the side | `pose.min_duration_s` (2 s) | orange |
| Leaning (needs hips visible) | `pose.min_duration_s` (2 s) | orange |
| Outside all seat zones (needs `seats.zones`) | `seats.max_outside_s` (10 s) | orange |

Flagged people get a thick red/orange box with `REVIEW: ...` in the video.
The same person + behaviour is not flagged again within `alerts.cooldown_s`.
Before trusting the head-turn rule on a new camera, run
`python -m src.pose_rules` on a normal clip and set
`pose.head_turn_threshold` above what most people show when facing forward.

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
