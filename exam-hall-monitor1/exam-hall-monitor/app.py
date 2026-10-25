"""Streamlit app: scan a video live, then review every flag.

Run locally:
    streamlit run app.py --server.maxUploadSize 1000
On Colab: see the optional Streamlit cell in notebooks/colab_live_demo.ipynb.

Layout:
    sidebar   switches for every features: flag and the phone detection mode
              (used by the NEXT scan, for live before/after demos), and a
              picker for which alerts CSV to review
    Live scan tab      upload a video, watch it being scanned
    Review alerts tab  summary panel + one card per alert: tier colour, reason,
                       thumbnail, Play clip, Confirm / Dismiss. Decisions are
                       kept in st.session_state and written to the CSV
                       (review_decision, reviewed_at).

Every flag is a request for a human to look, never an accusation.
"""

import copy
import tempfile
import time
from contextlib import closing
from pathlib import Path

import cv2
import streamlit as st

from src.detect_phone import MODES, detection_mode
from src.live import RealtimePacer, status_line
from src.risk import format_alert, read_alerts_csv, update_review_decision
from src.tracking import VideoScan
from src.utils import load_config, resize_to_width, resolve_path

BANNER = "Alerts are for human review only. They are not evidence of misconduct."
TIER_STYLE = {"red": ":red", "orange": ":orange", "grey": ":gray"}  # Streamlit coloured text

st.set_page_config(page_title="Exam Hall Monitor", layout="wide")
base_cfg = load_config()
live_cfg = base_cfg["live"]
results_dir = resolve_path(base_cfg["paths"]["output_dir"])
st.session_state.setdefault("decisions", {})   # alert_id -> "confirmed" / "dismissed"
st.session_state.setdefault("playing", None)   # alert_id whose clip is playing

# ---------------------------------------------------------------------------
# Sidebar: settings for the next scan + which results to review
# ---------------------------------------------------------------------------

st.sidebar.header("Settings for the next scan")
cfg = copy.deepcopy(base_cfg)
st.sidebar.caption("Feature switches (all off + mode 'full' = original behaviour)")
for name, default in (base_cfg.get("features") or {}).items():
    cfg["features"][name] = st.sidebar.toggle(name, value=bool(default), key=f"feature_{name}")
cfg["phone"]["mode"] = st.sidebar.selectbox(
    "Phone detection mode", MODES, index=MODES.index(detection_mode(base_cfg)))

st.sidebar.header("Review")
csv_files = sorted(results_dir.glob("*_alerts.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
default_csv = st.session_state.get("review_csv")
names = [p.name for p in csv_files]
review_csv = None
if names:
    index = names.index(Path(default_csv).name) if default_csv and Path(default_csv).name in names else 0
    review_csv = results_dir / st.sidebar.selectbox("Alerts file", names, index=index)
else:
    st.sidebar.caption("No alerts files in results/ yet.")

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.error(BANNER)
st.title("Exam Hall Monitor")
scan_tab, review_tab = st.tabs(["Live scan", "Review alerts"])


def find_file(path_text: str, csv_path: Path) -> Path | None:
    """A clip/thumbnail path from the CSV; if the CSV was moved (e.g. from
    Colab), also look in clips/ next to it."""
    if not path_text:
        return None
    path = Path(path_text)
    if path.exists():
        return path
    name = csv_path.stem.removesuffix("_alerts")
    nearby = csv_path.parent / "clips" / name / path.name
    return nearby if nearby.exists() else None


def play_clip(clip: Path, slot) -> None:
    """Play a review clip frame by frame (works for any codec OpenCV reads;
    browsers can't play the mp4v files OpenCV writes)."""
    cap = cv2.VideoCapture(str(clip))
    delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or base_cfg["runtime"]["fallback_fps"])
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        slot.image(frame, channels="BGR")
        time.sleep(delay)
    cap.release()


with scan_tab:
    uploaded = st.file_uploader("Upload an exam hall video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is not None and st.button("Start scan", type="primary"):
        # OpenCV needs a real file, so save the upload to a temporary file first.
        with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
            tmp.write(uploaded.getbuffer())
            video_path = Path(tmp.name)
        output_path = results_dir / f"{Path(uploaded.name).stem}_tracked.mp4"

        video_col, flags_col = st.columns([3, 1])
        with video_col:
            frame_slot = st.empty()     # the live video frame is redrawn here
            status_slot = st.empty()    # one line of live stats
            progress_bar = st.progress(0.0)
        with flags_col:
            st.subheader("Flags for review")

        try:
            with st.spinner("Loading models..."):
                scan = VideoScan(video_path, cfg, output_path)
            pacer = RealtimePacer(live_cfg["realtime"])
            # closing() finishes the saved files even if you press Stop in Streamlit.
            with closing(scan.frames()) as frames:
                for live in frames:
                    speed = pacer.wait(live.timestamp)
                    frame_slot.image(resize_to_width(live.annotated, live_cfg["display_width"]), channels="BGR")
                    status_slot.code(status_line(live, speed, len(scan.alerts)))
                    progress_bar.progress(live.progress)
                    for alert in live.new_alerts:
                        with flags_col:
                            st.markdown(f"{TIER_STYLE.get(alert.tier, '')}[**{alert.tier.upper()}**] "
                                        f"{format_alert(alert)}")
        finally:
            # Don't keep an extra copy of the uploaded footage (real people).
            video_path.unlink(missing_ok=True)

        summary = scan.summary()
        st.session_state["review_csv"] = summary["alerts_csv"]
        st.success(f"Scan finished: {len(summary['alerts'])} flags. Open the 'Review alerts' tab.")
        output = Path(summary["output"])
        if output.exists():
            with open(output, "rb") as f:
                st.download_button("Download annotated video", f, file_name=output.name, mime="video/mp4")


with review_tab:
    if review_csv is None or not review_csv.exists():
        st.info("Run a scan first (or put an *_alerts.csv in results/).")
    else:
        rows = read_alerts_csv(review_csv)
        decisions = st.session_state["decisions"]
        for row in rows:  # decisions saved earlier (in the CSV) count too
            if row["review_decision"] and row["alert_id"] not in decisions:
                decisions[row["alert_id"]] = row["review_decision"]

        # --- Summary panel ---------------------------------------------------
        by_tier = {tier: sum(r["tier"] == tier for r in rows) for tier in ("red", "orange", "grey")}
        flagged = [r for r in rows if r["tier"] != "grey"]  # grey = not a detection
        confirmed = sum(decisions.get(r["alert_id"]) == "confirmed" for r in flagged)
        dismissed = sum(decisions.get(r["alert_id"]) == "dismissed" for r in flagged)
        cols = st.columns(6)
        cols[0].metric("Red", by_tier["red"])
        cols[1].metric("Orange", by_tier["orange"])
        cols[2].metric("Grey (cannot assess)", by_tier["grey"])
        cols[3].metric("Reviewed", f"{confirmed + dismissed} / {len(flagged)}")
        cols[4].metric("Confirmed / dismissed", f"{confirmed} / {dismissed}")
        precision = f"{confirmed / (confirmed + dismissed):.0%}" if confirmed + dismissed else "-"
        cols[5].metric("Reviewer precision", precision,
                       help="confirmed / (confirmed + dismissed), red and orange only")

        show = st.multiselect("Show tiers", ["red", "orange", "grey"], default=["red", "orange", "grey"])
        st.caption(f"File: {review_csv}")

        # --- One card per alert ----------------------------------------------
        for row in rows:
            if row["tier"] not in show:
                continue
            alert_id = row["alert_id"]
            with st.container(border=True):
                left, right = st.columns([1, 2])
                thumbnail = find_file(row.get("thumbnail_path", ""), review_csv)
                if thumbnail:
                    left.image(str(thumbnail))
                else:
                    left.caption("no thumbnail")

                minutes, seconds = divmod(int(float(row["timestamp"])), 60)
                tier = row["tier"]
                right.markdown(f"{TIER_STYLE.get(tier, '')}[**{tier.upper()}**] · {minutes:02d}:{seconds:02d}"
                               f" · student ID {row['track_id']} · `{alert_id}`")
                right.write(row["reason"])
                decision = decisions.get(alert_id, "")
                right.caption(f"Decision: {decision or 'not reviewed yet'}")

                clip = find_file(row.get("clip_path", ""), review_csv)
                play_col, confirm_col, dismiss_col = right.columns(3)
                if play_col.button("Play clip", key=f"play_{alert_id}", disabled=clip is None):
                    st.session_state["playing"] = alert_id
                if confirm_col.button("Confirm", key=f"confirm_{alert_id}"):
                    decisions[alert_id] = "confirmed"
                    update_review_decision(review_csv, alert_id, "confirmed")
                    st.rerun()
                if dismiss_col.button("Dismiss", key=f"dismiss_{alert_id}"):
                    decisions[alert_id] = "dismissed"
                    update_review_decision(review_csv, alert_id, "dismissed")
                    st.rerun()
                if st.session_state["playing"] == alert_id and clip is not None:
                    play_clip(clip, right.empty())
                    st.session_state["playing"] = None
