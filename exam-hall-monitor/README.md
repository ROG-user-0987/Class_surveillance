# Exam Hall Monitor

A CCTV-based exam hall monitor that detects **likely** phone use and
suspicious posture and **flags them for human review**. It never accuses
anyone: every output is a prompt for an invigilator to look at a clip.

Heads are blurred in every saved video, using pose keypoints only
(no face detection, no face recognition).

## Setup (local, CPU)

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
| `runtime.device` | `cpu` locally, `0` for GPU on Colab. |
| `phone.conf_threshold` | Lower = more phone boxes (and more false ones). |
| `paths.phone_weights` / `phone.class_id` | `yolov8n.pt` + `67` (COCO) now; fine-tuned `best.pt` + `0` later. |
| `association.max_wrist_distance` | How close (as a fraction of person height) a phone must be to a wrist. |

## Running each module

Run everything from the repo root.

| Module | Command | Status |
|---|---|---|
| Video/config check | `python -m src.utils --video data/sample.mp4` | done |
| Tracking + phones | `python -m src.tracking --video data/sample.mp4` | done |
| Phone detector (SAHI) | `python -m src.detect_phone --video data/sample.mp4` | stub |
| Pose rules | `python -m src.pose_rules --video data/sample.mp4` | stub |
| Seat count | `python -m src.seat_count --video data/sample.mp4` | stub |
| Risk scoring | `python -m src.risk --video data/sample.mp4` | interface only |
| Review app | `streamlit run app.py` | stub |
| Evaluation | `python evaluate.py --video data/sample.mp4` | stub |

`--video` can be left out if `paths.input_video` is set in `config.yaml`.

### What `src.tracking` produces

- `results/<video name>_tracked.mp4` showing blurred heads, a coloured box +
  `ID n` + skeleton per person, cyan phone boxes with a line to the person
  they were assigned to, and grey "unassigned" phone boxes.
- A summary printed to the terminal: frames processed, unique track IDs,
  phone detections, assigned vs unassigned.

## Training (Colab)

Training happens on Google Colab, not locally. Put notebooks in
`notebooks/`, then copy the resulting `best.pt` into the repo root and point
`paths.phone_weights` at it (with `phone.class_id: 0`).

## Troubleshooting

- **`OSError: [WinError 4551] An Application Control policy has blocked this
  file ... c10.dll`**: Windows (Smart App Control / WDAC) is blocking
  PyTorch's DLLs. Turn Smart App Control off (Windows Security → App & browser
  control → Smart App Control) or ask your admin to allow it. Alternatively,
  run the demo on Colab.
- **Very slow on CPU**: raise `runtime.every_nth_frame` (e.g. 3–5) or lower
  `runtime.imgsz` to 480.
