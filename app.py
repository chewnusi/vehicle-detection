from pathlib import Path
import PIL
import streamlit as st
import config
import worker
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Military Equipment Detection",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Military Equipment Detection 🔍")

st.sidebar.header("⚙️ Model Settings")

confidence = st.sidebar.slider("Select confidence level", 0.05, 1.0, 0.15)

iou_threshold = st.sidebar.slider("IOU Threshold for NMS", 0.05, 1.0, 0.36,
                                 help="Intersection Over Union - lower values remove more overlapping boxes")

with st.sidebar.expander("🛠️ Additional Settings"):
    img_size = st.number_input(
        "Image Size",
        min_value=128,
        max_value=1920,
        value=512,
        step=32,
        help="Larger size - higher accuracy, lower speed"
    )
    st.caption(f"The image will be processed as {img_size}x{img_size} px")

st.sidebar.header("Source Type")
source_radio = st.sidebar.radio("Select source", config.SOURCES_LIST)

tracker_option = "bytetrack" 
if source_radio in [config.VIDEO, config.RTSP, config.YOUTUBE]:
    st.sidebar.subheader("🎯 Tracker Selection")
    tracker_option = st.sidebar.radio(
        "Select tracker:",
        list(config.TRACKERS.keys()),
        format_func=lambda x: config.TRACKERS[x]["name"]
    )
    st.sidebar.info(config.TRACKERS[tracker_option]["description"])

model_path = Path(config.DETECTION_MODEL)
try:
    model = worker.load_model(model_path)
except Exception as ex:
    st.error(f"❌ Could not load model: {model_path}")
    st.error(ex)

if source_radio == config.IMAGE:
    worker.detect_on_image(confidence, model, iou_threshold, img_size)

elif source_radio == config.VIDEO:
    worker.play_stored_video(confidence, model, config.TRACKERS[tracker_option]["config"], iou_threshold, img_size)

elif source_radio == config.RTSP:
    worker.play_rtsp_stream(confidence, model, config.TRACKERS[tracker_option]["config"], iou_threshold, img_size)

elif source_radio == config.YOUTUBE:
    worker.play_youtube_video(confidence, model, config.TRACKERS[tracker_option]["config"], iou_threshold, img_size)
else:
    st.error("❗ Please select a correct source type!")
