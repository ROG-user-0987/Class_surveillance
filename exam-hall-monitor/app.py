"""Streamlit review UI. STATUS: stub.

Planned: upload/pick a video, run the pipeline, then show a review queue of
Alerts (red first) with their clips, so an invigilator decides what happened.

Run:
    streamlit run app.py
"""

import streamlit as st

from src.utils import load_config

cfg = load_config()

st.title("Exam Hall Monitor: review queue")
st.warning("Every item here is a flag for a human to review, not an accusation.")
st.info("Not implemented yet. For now run: `python -m src.tracking --video path/to/clip.mp4`")
st.caption(f"Output folder from config: {cfg['paths']['output_dir']}")
