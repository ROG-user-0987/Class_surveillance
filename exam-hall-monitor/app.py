"""Streamlit app: upload a video, watch it being scanned, review the flags.

Run locally:
    streamlit run app.py --server.maxUploadSize 1000
On Colab: see the optional Streamlit cell in notebooks/colab_live_demo.ipynb.

Every flag is a request for a human to look, never an accusation.
"""

import tempfile
from contextlib import closing
from pathlib import Path

import streamlit as st

from src.live import RealtimePacer, status_line
from src.risk import format_alert
from src.tracking import VideoScan
from src.utils import load_config, resize_to_width, resolve_path

st.set_page_config(page_title="Exam Hall Monitor", layout="wide")
cfg = load_config()
live_cfg = cfg["live"]


def show_results(summary: dict) -> None:
    """Totals, the flag list and download buttons for the finished scan."""
    st.success("Scan finished. Review every flag before deciding anything.")
    cols = st.columns(4)
    cols[0].metric("Frames processed", summary["frames_processed"])
    cols[1].metric("Unique track IDs", summary["people_seen"])
    cols[2].metric("Red flags", summary["alerts_red"])
    cols[3].metric("Orange flags", summary["alerts_orange"])

    for alert in summary["alerts"]:
        st.write(format_alert(alert))

    output = Path(summary["output"])
    if output.exists():
        with open(output, "rb") as f:
            st.download_button("Download annotated video", f, file_name=output.name, mime="video/mp4")
    alerts_csv = Path(summary["alerts_csv"])
    if alerts_csv.exists():
        with open(alerts_csv, "rb") as f:
            st.download_button("Download flags (CSV)", f, file_name=alerts_csv.name, mime="text/csv")
    st.caption(f"Review clips are in {summary['clips_dir']}. "
               "Videos play in VLC or the Windows video player.")


st.title("Exam Hall Monitor: live scan")
st.warning("Every flag here is a request for a human to review, not an accusation.")

uploaded = st.file_uploader("Upload an exam hall video", type=["mp4", "avi", "mov", "mkv"])
start = uploaded is not None and st.button("Start scan", type="primary")
if not start:
    # Streamlit reruns the script on every click (e.g. Download), so show the
    # last finished scan again instead of an empty page.
    if "last_summary" in st.session_state:
        show_results(st.session_state["last_summary"])
    st.stop()

# OpenCV needs a real file, so save the upload to a temporary file first.
with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
    tmp.write(uploaded.getbuffer())
    video_path = Path(tmp.name)

output_path = resolve_path(cfg["paths"]["output_dir"]) / f"{Path(uploaded.name).stem}_tracked.mp4"

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
                    st.markdown(f"**{alert.tier.upper()}**: {format_alert(alert)}")
                    st.image(resize_to_width(live.annotated, 320), channels="BGR")
finally:
    # Don't keep an extra copy of the uploaded footage (real people).
    video_path.unlink(missing_ok=True)

st.session_state["last_summary"] = scan.summary()
show_results(st.session_state["last_summary"])
