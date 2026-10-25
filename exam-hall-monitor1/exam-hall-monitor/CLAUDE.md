# Exam Hall Monitor

## Project context
Deep learning case study: a CCTV-based exam hall monitor that detects *likely*
phone use and suspicious posture and **flags them for HUMAN review**.

- The system must **never accuse anyone automatically**. Every output is a
  request for an invigilator to look, worded neutrally ("phone-like object near
  hands for 3 s"), never "cheating" or "violation".
- Deadline is Sunday: favour simple, working code over clever code.
- Training happens on Google Colab (GPU). This repo must also run locally on
  CPU for the demo.
- The user currently runs everything on Colab: on their Windows laptop,
  Smart App Control blocks PyTorch's unsigned DLLs. `runtime.device: auto`
  picks the GPU when one is visible; on CPU `runtime.cpu_overrides` applies.
  The code must keep working on both.

## Stack
Python 3.10+, PyTorch, Ultralytics (YOLOv8, YOLO11, YOLOv8-Pose, ByteTrack),
OpenCV, SAHI, Streamlit, pandas, matplotlib, pycocotools (added for the
detection-mode comparison). (PyYAML and NumPy are used directly too; both
come with Ultralytics.)

## Repo layout
```
exam-hall-monitor/
  CLAUDE.md          this file
  README.md          setup + how to run each module
  changes.md         every change of the 5-phase upgrade and why
  requirements.txt
  .gitignore         ignores data/, results/ outputs, *.pt, runs/, venvs, caches
  config.yaml        ALL thresholds and paths live here
  data/              videos/datasets (git-ignored; data/our_clips/ = real people)
  notebooks/         Colab notebooks (colab_live_demo.ipynb = live scan demo)
  src/
    __init__.py
    utils.py         config loading, video I/O, drawing, head-region blur, ClipRecorder
    tracking.py      person tracking + phone-to-person association,
                     VideoScan (frame-by-frame generator)
    live.py          live viewers: notebook, OpenCV window
    detect_phone.py  phone detection modes full / sahi / crop, compare_modes (pycocotools)
    pose_rules.py    head-turn / lean / pitch, per-student baseline
    cues.py          neighbour peek, object passing, hands hidden, low visibility
    seat_count.py    seat zones, time outside seat
    temporal.py      EventTracker: EMA + hysteresis + gap + min duration
    risk.py          FrameRecord -> Alert (timer or EventTracker, cues, grey
                     state, fusion, cooldown, evidence text, CSV + review decisions)
  tests/             plain-assert tests; tests/helpers.py builds cfg + synthetic students
  app.py             Streamlit: live scan + review dashboard (Confirm / Dismiss)
  evaluate.py        event matching, sweep, ablation, per-condition, gallery, mode comparison
  results/           outputs (all git-ignored: videos, CSVs, images, cache)
```

## Frozen interfaces (do not change without updating every consumer)
- `src.tracking.FrameRecord`: one record per tracked person per processed
  frame: `frame_index, timestamp, track_id, person_box (4,), keypoints (17,3),
  phone_conf (0.0 if none), phone_box or None`. Consumed by `pose_rules.py`
  and `risk.py`. The full contract is in its docstring.
- `src.risk.Alert`: `timestamp, track_id, tier ("red"|"orange"|"grey"), reason,
  clip_path, behaviour, thumbnail_path, alert_id, signals (JSON str)`. Alerts
  are flags for human review; reason texts describe what was observed, never
  intent ("cheating"). Grey = "cannot assess", not a detection.
- `src.detect_phone.PhoneDetection(box, conf, crop_owner=None)` (re-exported
  from `src.tracking`).

## Feature flags (ablations)
Every improvement has a switch under `features:` in config.yaml. With ALL
switches false and `phone.mode: full` the pipeline must behave exactly as the
original (same alerts, same annotated frames; only extra CSV columns). Any new
behaviour must get its own switch and keep this property.
Design notes: out_of_seat always uses the simple timer; the extra cues and
the grey state always use EventTrackers; cue_fusion holds ORANGE alerts for
cues.fusion_window_s (red is never held). The original code had no
phone + posture fusion.

## Privacy
- `data/our_clips/` contains videos of real people. Never commit, upload,
  or paste frames from it.
- Saved/annotated video always has heads blurred via `utils.blur_head`, which
  builds the box from YOLOv8-Pose nose/eye/ear keypoints. **No face detector
  and no face recognition, ever.**

## Rules
- Ask the user before adding any dependency not listed in the stack above.
- Every module must run on its own from the command line with a sample video
  (`python -m src.<module> --video path.mp4`).
- Write clean, commented, beginner-readable code. Type hints on public
  functions.
- No hardcoded thresholds or paths in .py files; read them from `config.yaml`.
- Do not train models in a local session (training is done on Colab).
- If no sample video exists in `data/`, ask the user for a path rather than
  inventing one.
- When a task is done, tell the user the exact command to test it and what
  they should see.
