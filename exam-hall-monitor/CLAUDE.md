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
  picks the GPU when one is visible.

## Stack
Python 3.10+, PyTorch, Ultralytics (YOLOv8, YOLO11, YOLOv8-Pose, ByteTrack),
OpenCV, SAHI, Streamlit, pandas, matplotlib. (PyYAML and NumPy are used
directly too; both come with Ultralytics.)

## Repo layout
```
exam-hall-monitor/
  CLAUDE.md          this file
  README.md          setup + how to run each module
  requirements.txt
  .gitignore         ignores data/, results/*.mp4, *.pt, runs/, venvs, caches
  config.yaml        ALL thresholds and paths live here
  data/              videos/datasets (git-ignored; data/our_clips/ = real people)
  notebooks/         Colab notebooks (colab_live_demo.ipynb = live scan demo)
  src/
    __init__.py
    utils.py         config loading, video I/O, drawing, head-region blur
    tracking.py      person tracking + phone-to-person association,
                     VideoScan (frame-by-frame generator)          [DONE]
    live.py          live viewers: notebook, OpenCV window          [DONE]
    detect_phone.py  SAHI / fine-tuned phone detector                [stub]
    pose_rules.py    head-turn / lean rules from keypoints           [stub]
    seat_count.py    seat zones, people count, time outside seat     [stub]
    risk.py          FrameRecord -> Alert (interface done, logic stub)
  app.py             Streamlit: upload video + live scan [DONE]; review queue [todo]
  evaluate.py        metrics against labelled clips                  [stub]
  results/           annotated videos, alert clips (videos git-ignored)
```

## Frozen interfaces (do not change without updating every consumer)
- `src.tracking.FrameRecord`: one record per tracked person per processed
  frame: `frame_index, timestamp, track_id, person_box (4,), keypoints (17,3),
  phone_conf (0.0 if none), phone_box or None`. Consumed by `pose_rules.py`
  and `risk.py`. The full contract is in its docstring.
- `src.risk.Alert`: `timestamp, track_id, tier ("red"|"orange"), reason,
  clip_path`.

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
