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

model_path = Path(config.DETECTION_MODEL)
try:
    model = worker.load_model(model_path)
except Exception as ex:
    st.error(f"❌ Неможливо завантажити модель: {model_path}")
    st.error(ex)

st.sidebar.header("Тип джерела")
source_radio = st.sidebar.radio("Виберіть джерело", config.SOURCES_LIST)

if source_radio == config.IMAGE:
    worker.detect_on_image(confidence, model)

elif source_radio == config.VIDEO:
    worker.play_stored_video(confidence, model)

elif source_radio == config.RTSP:
    worker.play_rtsp_stream(confidence, model)

elif source_radio == config.YOUTUBE:
    worker.play_youtube_video(confidence, model)
else:
    st.error("❗ Будь ласка, оберіть коректний тип джерела!")
