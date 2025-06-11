from pathlib import Path
import PIL
import streamlit as st
import config
import worker
import cv2
import numpy as np
from PIL import Image

# Налаштування сторінки
st.set_page_config(
    page_title="Military Equipment Detection",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Military Equipment Detection 🔍")

# Бічна панель
st.sidebar.header("⚙️ Налаштування моделі")

# Confidence slider
confidence = st.sidebar.slider("Виберіть рівень впевненості", 0.05, 1.0, 0.4)

# IOU slider
iou_threshold = st.sidebar.slider("Поріг IOU для NMS", 0.1, 0.9, 0.5, 0.1, 
                                 help="Intersection Over Union - нижчі значення видаляють більше перетинаючих боксів")

# Advanced options
with st.sidebar.expander("🛠️ Додаткові налаштування"):
    # Image size input
    img_size = st.number_input(
        "Розмір зображення",
        min_value=128,
        max_value=1920,
        value=512,
        step=32,
        help="Більший розмір - вища точність, нижча швидкість"
    )
    st.caption(f"Зображення буде оброблятися як {img_size}x{img_size} px")

st.sidebar.header("Тип джерела")
source_radio = st.sidebar.radio("Виберіть джерело", config.SOURCES_LIST)

tracker_option = "bytetrack" 
if source_radio in [config.VIDEO, config.RTSP, config.YOUTUBE]:
    st.sidebar.subheader("🎯 Вибір трекера")
    tracker_option = st.sidebar.radio(
        "Оберіть трекер:",
        list(config.TRACKERS.keys()),
        format_func=lambda x: config.TRACKERS[x]["name"]
    )
    st.sidebar.info(config.TRACKERS[tracker_option]["description"])

model_path = Path(config.DETECTION_MODEL)
try:
    model = worker.load_model(model_path)
except Exception as ex:
    st.error(f"❌ Неможливо завантажити модель: {model_path}")
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
    st.error("❗ Будь ласка, оберіть коректний тип джерела!")
