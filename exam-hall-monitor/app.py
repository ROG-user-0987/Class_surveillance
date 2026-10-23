"""Streamlit app: upload a video and watch it being scanned while it plays.

Run locally:
    streamlit run app.py --server.maxUploadSize 1000
On Colab: see the optional Streamlit cell in notebooks/colab_live_demo.ipynb.

Everything shown is a flag for a human to review, never an accusation.
Planned next: a review queue of Alerts (red first) once src/risk.py has logic.
"""

import tempfile
from contextlib import closing
from pathlib import Path

import streamlit as st

from src.live import RealtimePacer, status_line, to_display
from src.tracking import VideoScan
from src.utils import load_config, resolve_path

st.set_page_config(page_title="Exam Hall Monitor", layout="wide")
cfg = load_config()
live_cfg = cfg["live"]


def show_results(summary: dict) -> None:
    """Totals and a download button for the finished scan."""
    st.success("Scan finished.")
    cols = st.columns(4)
    cols[0].metric("Frames processed", summary["frames_processed"])
    cols[1].metric("Unique track IDs", summary["people_seen"])
    cols[2].metric("Phones near a person", summary["phones_assigned"])
    cols[3].metric("Phones unassigned", summary["phones_unassigned"])

    output = Path(summary["output"])
    if output.exists():
        with open(output, "rb") as f:
            st.download_button("Download annotated video", f, file_name=output.name, mime="video/mp4")
        st.caption(f"Also saved to {output}. It plays in VLC or the Windows video player.")


st.title("Exam Hall Monitor: live scan")
st.warning("Every detection here is a flag for a human to review, not an accusation.")

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

frame_slot = st.empty()     # the live video frame is redrawn here
status_slot = st.empty()    # one line of live stats
progress_bar = st.progress(0.0)

try:
    with st.spinner("Loading models..."):
        scan = VideoScan(video_path, cfg, output_path)
    pacer = RealtimePacer(live_cfg["realtime"])

    # closing() finishes the saved video even if you press Stop in Streamlit.
    with closing(scan.frames()) as frames:
        for live in frames:
            speed = pacer.wait(live.timestamp)
            frame_slot.image(to_display(live.annotated, live_cfg["display_width"]), channels="BGR")
            status_slot.code(status_line(live, speed))
            progress_bar.progress(live.progress)
finally:
    # Don't keep an extra copy of the uploaded footage (real people).
    video_path.unlink(missing_ok=True)

st.session_state["last_summary"] = scan.summary()
show_results(st.session_state["last_summary"])
