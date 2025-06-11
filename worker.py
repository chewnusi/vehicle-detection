import streamlit as st
import cv2
from ultralytics import YOLO
import yt_dlp  
import config
from pathlib import Path
import PIL
import streamlit as st
import config
import cv2
import numpy as np
from PIL import Image
import os
import requests
import subprocess
import shutil
import traceback
import re
import time
import tempfile


def load_model(model_path):
    """Завантажує модель YOLO."""
    return YOLO(model_path)


def detect_on_image(conf, model):
    """
    Виконує детекцію на зображеннях.
    """
    st.title("🖼️ Обробка зображень")
    
    image_option = st.sidebar.radio(
        "Виберіть джерело зображення",
        ("Вибрати зі списку", "Завантажити зображення")
    )
    
    if image_option == "Вибрати зі списку":
        source_img = st.sidebar.selectbox(
            "Виберіть зображення...",
            list(config.IMAGES_DICT.keys())
        )
        image_path = str(config.IMAGES_DICT[source_img])
        
        col1, col2 = st.columns(2)
        with col1:
            original = Image.open(image_path)
            st.image(original, use_column_width=True)
        
        with col2:
            img_array = np.array(original)
            res = model.predict(img_array, conf=conf)
            annotated_img = res[0].plot()
            st.image(annotated_img, use_column_width=True)
        
        st.write("")
        with st.expander("Результати обробки"):
            for i, box in enumerate(res[0].boxes):
                data = box.data[0]
                st.write(f"Об'єкт #{i+1}:")
                st.write(f"- Клас: {data[5]}")
                st.write(f"- Впевненість: {data[4]*100:.2f}%")
                st.write(f"- Координати: x1={data[0]:.1f}, y1={data[1]:.1f}, x2={data[2]:.1f}, y2={data[3]:.1f}")
        st.write("")
    else:
        source_imgs = st.sidebar.file_uploader(
            "Завантаження зображень...",
            type=("jpg", "jpeg", "png"),
            accept_multiple_files=True
        )
        
        if source_imgs:
            for i, source_img in enumerate(source_imgs):
                col1, col2 = st.columns(2)
                
                with col1:
                    original = Image.open(source_img)
                    st.image(original, use_column_width=True)
                
                with col2:
                    img_array = np.array(original)
                    res = model.predict(img_array, conf=conf)
                    annotated_img = res[0].plot()
                    st.image(annotated_img, use_column_width=True)
                
                st.write("")
                with st.expander(f"Результати обробки для зображення {i+1}"):
                    for i, box in enumerate(res[0].boxes):
                        data = box.data[0]
                        st.write(f"Об'єкт #{i+1}:")
                        st.write(f"- Клас: {data[5]}")
                        st.write(f"- Впевненість: {data[4]*100:.2f}%")
                        st.write(f"- Координати: x1={data[0]:.1f}, y1={data[1]:.1f}, x2={data[2]:.1f}, y2={data[3]:.1f}")
                st.write("")


def get_frames_and_detect(conf, model, source, tracker="bytetrack.yaml"):
    """
    Допоміжна функція: зчитує кадри із source та виконує детекцію об'єктів.
    Відображає результати детекції в реальному часі.
    Зберігає оброблене відео у стандартний MP4 файл для відео.
    """
    try:
        vid_cap = cv2.VideoCapture(source)
        st_frame = st.empty()
        
        if not vid_cap.isOpened():
            st.error("❌ Не вдається відкрити потік/відео.")
            return None
        
        # Get original video properties
        fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:  # If fps is 0, set a default value
            fps = 30
        
        if source.startswith('rtsp://'):
            while vid_cap.isOpened():
                success, frame = vid_cap.read()
                if not success:
                    break
                
                if tracker:
                    res = model.track(frame, conf=conf, tracker=tracker)
                else:
                    res = model.predict(frame, conf=conf, stream=True)
                
                processed_frame = res[0].plot()
                
                st_frame.image(
                    processed_frame,
                    caption="Processing...",
                    use_column_width=True,
                    channels="BGR"
                )
            
            vid_cap.release()
            clean_temp_files()
            return None  
        
        temp_frames_dir = "temp_frames"
        os.makedirs(temp_frames_dir, exist_ok=True)
        
        frame_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        frame_idx = 0
        
        while vid_cap.isOpened():
            success, frame = vid_cap.read()
            if not success:
                break
            
            if frame_count > 0:
                progress_bar.progress(min(frame_idx / frame_count, 1.0))
            
            if tracker:
                res = model.track(frame, conf=conf, tracker=tracker)
            else:
                res = model.predict(frame, conf=conf, stream=True)
            
            processed_frame = res[0].plot()
            
            st_frame.image(
                processed_frame,
                caption="Обробка...",
                use_column_width=True,
                channels="BGR"
            )
            
            frame_path = os.path.join(temp_frames_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, processed_frame)
            
            frame_idx += 1
        
        vid_cap.release()
        
        videos_dir = "videos"
        os.makedirs(videos_dir, exist_ok=True)
        output_path = os.path.join(videos_dir, f"processed_{Path(source).stem}.mp4")
        
        try:       
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(temp_frames_dir, "frame_%06d.jpg"),  
                "-c:v", "libx264", 
                "-profile:v", "high",  
                "-preset", "medium", 
                "-pix_fmt", "yuv420p",
                "-r", str(fps), 
                "-movflags", "+faststart",
                output_path
            ]

            
            # cmd = [
            #     "ffmpeg", "-y",  
            #     "-framerate", str(fps),  
            #     "-i", os.path.join(temp_frames_dir, "frame_%06d.jpg"),  
            #     "-c:v", "libx264",  
            #     "-profile:v", "main", 
            #     "-preset", "medium", 
            #     "-r", str(fps), 
            #     "-tune", "zerolatency", 
            #     "-crf", "23", 
            #     "-pix_fmt", "yuv420p", 
            #     "-movflags", "+faststart", 
            #     output_path
            # ]
            
            save_msg = st.empty()
            save_msg.info("⏳ Збереження відео...")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                st.error(f"❌ Помилка при створенні відео: {stderr.decode()}")
                return None
        except Exception as e:
            st.error(f"❌ Помилка при створенні відео: {str(e)}")
            return None
        finally:
            for file in os.listdir(temp_frames_dir):
                file_path = os.path.join(temp_frames_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    st.warning(f"Не вдалося видалити тимчасовий файл {file_path}: {str(e)}")
            
            if os.path.exists(temp_frames_dir) and not os.listdir(temp_frames_dir):
                os.rmdir(temp_frames_dir)
        
        progress_bar.empty()
        st_frame.empty()
        save_msg.empty()  
        
        clean_temp_files()
        
        return output_path
        
    except Exception as e:
        st.error(f"Помилка обробки відео: {str(e)}")
        clean_temp_files()
        return None


def play_stored_video(conf, model, tracker="bytetrack.yaml"):
    """
    Функція для обробки та відтворення відео:
    1. Користувач вибирає відео зі списку або завантажує своє
    2. Після натискання кнопки "Запуск детекції" відео обробляється та показується в реальному часі з детекцією
    3. Після закінчення обробки, показується оброблене відео у звичайному форматі та надається можливість завантажити його
    
    Args:
        conf: Рівень впевненості
        model: Завантажена модель YOLO
        tracker: Конфігурація трекера (default: "bytetrack.yaml")
    """
    st.title("🎥 Обробка відео")
    
    # Створюємо тимчасову директорію для збереження кадрів
    temp_dir = "temp_frames"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    video_option = st.sidebar.radio(
        "Виберіть джерело відео",
        ("Вибрати зі списку", "Завантажити відео")
    )
    
    if video_option == "Вибрати зі списку":
        source_vid = st.sidebar.selectbox(
            "Виберіть відео...",
            list(config.VIDEOS_DICT.keys())
        )
        video_path = str(config.VIDEOS_DICT[source_vid])
    else:
        uploaded_file = st.sidebar.file_uploader("Оберіть відео файл", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            video_path = os.path.join(temp_dir, uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        else:
            st.warning("Будь ласка, завантажте відео файл")
            return
    
    video_container = st.empty()
    
    if st.sidebar.button("Запуск детекції 🎯"):
        try:
            processed_video_path = get_frames_and_detect(conf, model, video_path, tracker)
            
            if processed_video_path and os.path.exists(processed_video_path):
                file_size = os.path.getsize(processed_video_path) / (1024 * 1024)
                st.success(f"✅ Файл сформовано. Розмір файлу: {file_size:.2f} MB")
                
                video_container.video(processed_video_path)
                
                with open(processed_video_path, "rb") as file:
                    video_bytes = file.read()
                    st.download_button(
                        label="📥 Завантажити оброблене відео",
                        data=video_bytes,
                        file_name=Path(processed_video_path).name,
                        mime="video/mp4"
                    )
                
                if st.button("Очистити відео"):
                    video_container.empty()
                    clean_temp_files()
        except Exception as e:
            st.error(f"Помилка: {str(e)}")
            clean_temp_files()


def play_youtube_video(conf, model, tracker="bytetrack.yaml"):
    """
    Відтворення YouTube-відео за посиланням у реальному часі.
    Детекція + трекінг на кожному кадрі.
    
    Args:
        conf: Рівень впевненості
        model: Завантажена модель YOLO
        tracker: Конфігурація трекера (default: "bytetrack.yaml")
    """
    youtube_url = st.sidebar.text_input("YouTube Video URL", "https://www.youtube.com/watch?v=FQijTjkH7-0")
    
    with st.sidebar.expander("Додаткові налаштування"):
        debug_mode = st.checkbox("Режим діагностики", value=False)
    
    video_container = st.empty()
    
    if st.sidebar.button("Обробити відео 🎬"):
        if not youtube_url:
            st.error("❌ Введіть посилання на YouTube-відео.")
            return

        try:
            videos_dir = "videos"
            os.makedirs(videos_dir, exist_ok=True)
            video_id = extract_youtube_id(youtube_url)
            output_path = os.path.join(videos_dir, f"youtube_{video_id}.mp4")
            
            connection_msg = st.empty()
            connection_msg.info("Отримання відео з YouTube...")
            
            stream_url = get_youtube_stream_url(youtube_url)
            
            connection_msg.success("✅ Відео отримано. Обробка...")
            
            temp_file = download_youtube_to_temp(stream_url)
            
            processed_video_path = get_frames_and_detect(conf, model, temp_file, tracker)
            
            if processed_video_path and os.path.exists(processed_video_path):
                shutil.copy(processed_video_path, output_path)
                processed_video_path = output_path
                
            connection_msg.empty()
            
            if processed_video_path and os.path.exists(processed_video_path):
                file_size = os.path.getsize(processed_video_path) / (1024 * 1024)
                st.success(f"✅ Файл сформовано. Розмір файлу: {file_size:.2f} MB")
                
                video_container.video(processed_video_path)
                
                with open(processed_video_path, "rb") as file:
                    video_bytes = file.read()
                    st.download_button(
                        label="📥 Завантажити оброблене відео",
                        data=video_bytes,
                        file_name=f"youtube_video_{video_id}.mp4",
                        mime="video/mp4"
                    )
                
                clean_temp_files()
                
                if st.button("Очистити відео"):
                    video_container.empty()
                    clean_temp_files()
            
        except Exception as e:
            st.error(f"Помилка: {str(e)}")
            if debug_mode:
                st.code(traceback.format_exc())
            clean_temp_files()


def extract_youtube_id(youtube_url):
    """
    Витягує ID відео з YouTube URL.
    """
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  
        r'(?:embed\/|v\/|youtu\.be\/)([0-9A-Za-z_-]{11})', 
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})' 
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    
    return f"video_{int(time.time())}"


def download_youtube_to_temp(stream_url):
    """
    Завантажує відео з потоку у тимчасовий файл і повертає шлях до нього.
    """
    
    temp_dir = "temp_youtube"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, f"temp_video_{int(time.time())}.mp4")
    
    try:
        response = requests.get(stream_url, stream=True)
        response.raise_for_status()
        
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return temp_file
    except Exception as e:
        try:
            ydl_opts = {
                'format': 'bestvideo[height<=720][vcodec!*=av01]+bestaudio/best[height<=720]',
                'outtmpl': temp_file,
                'quiet': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                stream_info = {'url': stream_url}
                ydl.process_ie_result(stream_info, download=True)
            
            return temp_file
        except Exception:
            raise Exception(f"Не вдалося завантажити відео: {str(e)}")


def clean_temp_files():
    """
    Очищає всі тимчасові файли.
    """
    temp_dirs = ["temp_youtube", "temp_frames", "temp_uploads"]
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                
                if not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                print(f"Помилка при очистці {temp_dir}: {str(e)}")
                continue


def get_youtube_stream_url(youtube_url):
    """
    За допомогою yt_dlp витягує пряме посилання на відео-потік YouTube.
    Уникає формати AV1, які можуть викликати проблеми з декодуванням.
    """
    if not youtube_url:
        raise ValueError("URL відео не може бути порожнім")
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][vcodec!*=av01][vcodec^=avc]+bestaudio[ext=m4a]/best[ext=mp4][vcodec!*=av01]/best[vcodec!*=av01]/best',
        'quiet': True,
        'no_warnings': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            
            if 'url' in info:
                return info['url']
            
            if 'requested_formats' in info:
                for fmt in info['requested_formats']:
                    if fmt.get('vcodec', '').startswith('avc'):
                        return fmt['url']
            
            if 'formats' in info:
                for fmt in info['formats']:
                    vcodec = fmt.get('vcodec', 'none')
                    if (vcodec != 'none' and 'av01' not in vcodec and 
                        fmt.get('height', 0) >= 360):
                        return fmt['url']
            
            if 'formats' in info and info['formats']:
                for fmt in sorted(info['formats'], key=lambda x: x.get('height', 0), reverse=True):
                    if fmt.get('vcodec', 'none') != 'none' and fmt.get('height', 0) > 0:
                        return fmt['url']
            
            raise ValueError("Не вдалося отримати відповідний URL потоку")
            
    except Exception as e:
        raise Exception(f"Помилка при обробці відео: {str(e)}")


def play_rtsp_stream(conf, model, tracker="bytetrack.yaml"):
    """
    Відтворення RTSP стріму: користувач вводить URL, 
    далі кожен кадр обробляється та показується.
    
    Args:
        conf: Рівень впевненості
        model: Завантажена модель YOLO
        tracker: Конфігурація трекера (default: "bytetrack.yaml")
    """
    # rtsp://rtspstream:NuNGxzjfxj6QeLHwbJ9us@zephyr.rtsp.stream/people
    source_rtsp = st.sidebar.text_input("RTSP stream URL:", "rtsp://rtspstream:NuNGxzjfxj6QeLHwbJ9us@zephyr.rtsp.stream/traffic")
    st.sidebar.caption("Приклад: rtsp://rtspstream:NuNGxzjfxj6QeLHwbJ9us@zephyr.rtsp.stream/traffic")

    if st.sidebar.button("Start RTSP 🚀"):
        if not source_rtsp:
            st.error("❌ Будь ласка, введіть коректну RTSP-адресу.")
            return
        get_frames_and_detect(conf, model, source_rtsp, tracker)