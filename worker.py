import streamlit as st
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
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
import threading


class ThreadedCamera:
    """
    A class to read frames from a camera in a separate thread.
    This is used to prevent the main thread from blocking while waiting for a new frame,
    which is crucial for real-time applications like RTSP streaming to avoid lag.
    """
    def __init__(self, src):
        # Use FFMPEG backend for better RTSP handling
        self.capture = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        # Daemon thread to read frames from the camera
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
        self.status = False
        self.frame = None
        self.lock = threading.Lock()

    def update(self):
        """Continuously read frames from the camera."""
        while True:
            if self.capture.isOpened():
                (status, frame) = self.capture.read()
                if status:
                    with self.lock:
                        self.frame = frame
            else:
                break
            time.sleep(0.01)  # Reduce CPU load

    def read(self):
        """Read the latest frame."""
        with self.lock:
            if self.frame is not None:
                frame = self.frame.copy()
                return True, frame
            return False, None
    
    def release(self):
        """Release the camera capture."""
        self.capture.release()


def draw_boxes(frame, results, model_names):
    """Draw bounding boxes on a frame."""
    annotator = Annotator(frame, line_width=2, example=str(model_names))
    if results.boxes is not None:
        for box in results.boxes:
            if box.cls is not None and box.conf is not None:
                class_id = int(box.cls)
                confidence = float(box.conf)
                label = f'{model_names[class_id]} {confidence:.2f}'
                annotator.box_label(box.xyxy[0], label, color=colors(class_id, True))
    return annotator.result()


def load_model(model_path):
    """Loads a YOLO model."""
    return YOLO(model_path)


def detect_on_image(conf, model, iou=0.5, img_size=520):
    """
    Performs detection on images.
    
    Args:
        conf: Confidence level for detection
        model: Loaded YOLO model
        iou: IOU threshold for NMS (default: 0.5)
        img_size: Image size for inference (default: 520)
    """
    st.title("🖼️ Image Processing")
    
    def process_image(image):
        """Helper function to process a single image"""
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, use_column_width=True)
        
        with col2:
            img_array = np.array(image)
            res = model.predict(img_array, conf=conf, iou=iou, imgsz=img_size)
            annotated_img = res[0].plot()
            st.image(annotated_img, use_column_width=True)
        
        st.write("")
        with st.expander("Processing Results"):
            for i, box in enumerate(res[0].boxes):
                data = box.data[0]
                class_id = int(data[5])
                class_name = config.CLASSES[class_id] if class_id < len(config.CLASSES) else f"Class {class_id}"
                st.write(f"Object #{i+1}:")
                st.write(f"- Class: {class_name}")
                st.write(f"- Confidence: {data[4]*100:.2f}%")
                st.write(f"- Coordinates: x1={data[0]:.1f}, y1={data[1]:.1f}, x2={data[2]:.1f}, y2={data[3]:.1f}")
        st.write("")
    
    image_option = st.sidebar.radio(
        "Select image source",
        ("Select from list", "Upload image")
    )
    
    if image_option == "Select from list":
        source_img = st.sidebar.selectbox(
            "Select image...",
            list(config.IMAGES_DICT.keys())
        )
        image = Image.open(str(config.IMAGES_DICT[source_img]))
        process_image(image)
    else:
        source_imgs = st.sidebar.file_uploader(
            "Uploading images...",
            type=("jpg", "jpeg", "webp", "bmp", "dng", "mpo", "tif", "tiff", "pfm", "HEIC"),
            accept_multiple_files=True
        )
        if source_imgs:
            for source_img in source_imgs:
                image = Image.open(source_img)
                process_image(image)


def get_frames_and_detect(conf, model, source, tracker="bytetrack.yaml", iou=0.5, img_size=512):
    """
    Helper function: reads frames from source and performs object detection.
    Displays detection results in real-time and saves the processed video.
    
    Args:
        conf: Confidence level for detection
        model: Loaded YOLO model
        source: Path to video file or RTSP link
        tracker: Tracker configuration (bytetrack.yaml, botsort.yaml)
        iou: IOU threshold for NMS (default: 0.5)
        img_size: Image size for inference (default: 520)
    """
    try:
        if source.startswith('rtsp://'):
            threaded_camera = ThreadedCamera(source)
            st_frame = st.empty()

            track_config = {
                'conf': conf,
                'iou': iou,
                'imgsz': img_size,
                'max_det': 300,
                'tracker': tracker,
                'persist': True,
                'agnostic_nms': True,
                'verbose': False,  # Disabled for performance
                'half': False,
                'show_conf': True,
                'show_labels': True,
            }

            while True:
                success, frame = threaded_camera.read()
                if not success:
                    time.sleep(0.1)
                    continue

                if tracker:
                    res = model.track(frame, **track_config)
                else:
                    res = model.predict(frame, conf=conf)

                processed_frame = draw_boxes(frame, res[0], model.names)

                st_frame.image(
                    processed_frame,
                    caption="Processing RTSP stream...",
                    use_column_width=True,
                    channels="BGR"
                )

            threaded_camera.release()
            return None

        vid_cap = cv2.VideoCapture(source)
        st_frame = st.empty()

        if not vid_cap.isOpened():
            st.error("❌ Could not open stream/video.")
            return None

        fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30
        frame_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        track_config = {
            'conf': conf,
            'iou': iou,
            'imgsz': img_size,
            'max_det': 300,
            'tracker': tracker,
            'persist': True,
            'agnostic_nms': True,
            'verbose': True,
            'half': False,
            'show_conf': True,
            'show_labels': True,
        }

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
                res = model.track(frame, **track_config)
            else:
                res = model.predict(frame, conf=conf, stream=True)
            
            processed_frame = draw_boxes(frame, res[0], model.names)
            
            st_frame.image(
                processed_frame,
                caption="Processing...",
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
            fps_file = os.path.join(temp_frames_dir, "fps.txt")
            with open(fps_file, "w") as f:
                f.write(f"fps={fps}")
            
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(temp_frames_dir, "frame_%06d.jpg"),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-vsync", "vfr",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ]
            
            save_msg = st.empty()
            save_msg.info("⏳ Saving video...")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                st.error(f"❌ Error creating video: {stderr.decode()}")
                return None
        except Exception as e:
            st.error(f"❌ Error creating video: {str(e)}")
            return None
        finally:
            for file in os.listdir(temp_frames_dir):
                file_path = os.path.join(temp_frames_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    st.warning(f"Could not delete temporary file {file_path}: {str(e)}")
            
            if os.path.exists(temp_frames_dir) and not os.listdir(temp_frames_dir):
                os.rmdir(temp_frames_dir)
        
        progress_bar.empty()
        st_frame.empty()
        save_msg.empty()
        
        clean_temp_files()
        
        return output_path
        
    except Exception as e:
        st.error(f"Video processing error: {str(e)}")
        clean_temp_files()
        return None


def play_stored_video(conf, model, tracker="bytetrack.yaml", iou=0.5, img_size=520):
    """
    Function for processing and playing video:
    1. The user selects a video from the list or uploads their own
    2. After clicking the \"Start Detection\" button, the video is processed and displayed in real-time with detection
    3. After processing is complete, the processed video is shown in the standard format and an option to download it is provided
    
    Args:
        conf: Confidence level
        model: Loaded YOLO model
        tracker: Tracker configuration (default: "bytetrack.yaml")
        iou: IOU threshold for NMS (default: 0.5)
        img_size: Image size for inference (default: 520)
    """
    st.title("🎥 Video Processing")

    video_option = st.sidebar.radio(
        "Select video source",
        ("Select from list", "Upload video")
    )

    video_path = None
    if video_option == "Select from list":
        source_vid = st.sidebar.selectbox(
            "Select video...",
            list(config.VIDEOS_DICT.keys())
        )
        video_path = str(config.VIDEOS_DICT[source_vid])
    else:
        uploaded_file = st.sidebar.file_uploader("Select video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            temp_dir_uploads = "temp_uploads"
            os.makedirs(temp_dir_uploads, exist_ok=True)
            video_path = os.path.join(temp_dir_uploads, uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        else:
            st.warning("Please upload a video file")
            return

    if video_path:
        video_container = st.empty()

        if st.sidebar.button("Start Detection 🎯"):
            try:
                processed_video_path = get_frames_and_detect(conf, model, video_path, tracker, iou, img_size)

                if processed_video_path and os.path.exists(processed_video_path):
                    file_size = os.path.getsize(processed_video_path) / (1024 * 1024)
                    st.success(f"✅ File created. File size: {file_size:.2f} MB")

                    video_container.video(processed_video_path)

                    with open(processed_video_path, "rb") as file:
                        video_bytes = file.read()
                        st.download_button(
                            label="📥 Download processed video",
                            data=video_bytes,
                            file_name=Path(processed_video_path).name,
                            mime="video/mp4"
                        )

                    if st.button("Clear video"):
                        video_container.empty()
                        clean_temp_files()
            except Exception as e:
                st.error(f"Error: {str(e)}")
                clean_temp_files()


def play_youtube_video(conf, model, tracker="bytetrack.yaml", iou=0.5, img_size=520):
    """
    Plays a YouTube video from a link in real-time.
    1. The user pastes a link to a YouTube video
    2. After clicking the \"Start Detection\" button, the video is downloaded, processed, and displayed in real-time
    3. After processing is complete, the processed video is shown in the standard format and an option to download it is provided
    """
    youtube_url = st.sidebar.text_input("YouTube Video URL", "https://youtu.be/970Vdfu25yw")

    video_container = st.empty()

    if st.sidebar.button("Process Video 🎬"):
        if not youtube_url:
            st.error("❌ Please enter a YouTube video link.")
            return

        try:
            videos_dir = "videos"
            os.makedirs(videos_dir, exist_ok=True)
            video_id = extract_youtube_id(youtube_url)
            output_path = os.path.join(videos_dir, f"youtube_{video_id}.mp4")

            connection_msg = st.empty()
            connection_msg.info("Getting video from YouTube...")

            stream_url = get_youtube_stream_url(youtube_url)

            connection_msg.success("✅ Video received. Processing...")

            temp_file = download_youtube_to_temp(stream_url)

            processed_video_path = get_frames_and_detect(conf, model, temp_file, tracker, iou, img_size)

            if processed_video_path and os.path.exists(processed_video_path):
                shutil.copy(processed_video_path, output_path)
                processed_video_path = output_path

            connection_msg.empty()

            if processed_video_path and os.path.exists(processed_video_path):
                file_size = os.path.getsize(processed_video_path) / (1024 * 1024)
                st.success(f"✅ File created. File size: {file_size:.2f} MB")

                video_container.video(processed_video_path)

                with open(processed_video_path, "rb") as file:
                    video_bytes = file.read()
                    st.download_button(
                        label="📥 Download processed video",
                        data=video_bytes,
                        file_name=f"youtube_video_{video_id}.mp4",
                        mime="video/mp4"
                    )

                clean_temp_files()

                if st.button("Clear video"):
                    video_container.empty()
                    clean_temp_files()

        except Exception as e:
            st.error(f"Error: {str(e)}")
            clean_temp_files()


def extract_youtube_id(youtube_url):
    """
    Extracts the video ID from a YouTube URL.
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
    Downloads a YouTube video to a temporary file and returns the path.
    """

    temp_dir = "temp_youtube"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, f"temp_video_{int(time.time())}.mp4")

    try:
        ydl_opts = {
            'format': '(232+234)/(230+234)/best',
            'outtmpl': temp_file,
            'quiet': True,
            'no_warnings': True,
            'allow_unplayable_formats': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([stream_url])

        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
            return temp_file
        else:
            raise Exception("File was not downloaded")

    except Exception as e:
        try:
            ydl_opts['format'] = 'best'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([stream_url])

            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                return temp_file
            else:
                raise Exception("File was not downloaded")

        except Exception as e2:
            raise Exception(f"Could not download video: {str(e2)}")


def clean_temp_files():
    """
    Cleans all temporary files.
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
                print(f"Error while cleaning {temp_dir}: {str(e)}")
                continue


def get_youtube_stream_url(youtube_url):
    """
    Uses yt_dlp to extract the direct URL of a YouTube video stream.
    """
    if not youtube_url:
        raise ValueError("Video URL cannot be empty")

    ydl_opts = {
        'format': '(232+234)/(230+234)/best',
        'quiet': True,
        'no_warnings': True,
        'allow_unplayable_formats': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            
            if 'url' in info:
                return info['url']
            
            if 'requested_formats' in info:
                video_format = None
                audio_format = None
                
                for fmt in info['requested_formats']:
                    if fmt.get('vcodec', 'none') != 'none':
                        video_format = fmt
                    if fmt.get('acodec', 'none') != 'none':
                        audio_format = fmt
                
                if video_format:
                    return video_format['url']
            
            # Fallback to any available format with video
            if 'formats' in info:
                formats = sorted(info['formats'], 
                              key=lambda x: (x.get('height', 0) or 0),
                              reverse=True)
                
                for fmt in formats:
                    if fmt.get('vcodec', 'none') != 'none':
                        return fmt['url']
            
            raise ValueError("Could not find a suitable video format")

    except Exception as e:
        if 'Requested format is not available' in str(e):
            ydl_opts['format'] = 'best'
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                    if 'url' in info:
                        return info['url']
            except Exception as e2:
                raise Exception(f"Could not get video even with the simplest format: {str(e2)}")
        raise Exception(f"Error processing video: {str(e)}")


def play_rtsp_stream(conf, model, tracker="bytetrack.yaml", iou=0.5, img_size=520):
    """
    Plays an RTSP stream: the user enters a URL,
    then each frame is processed and displayed.

    Args:
        conf: Confidence level
        model: Loaded YOLO model
        tracker: Tracker configuration (default: "bytetrack.yaml")
        iou: IOU threshold for NMS (default: 0.5)
        img_size: Image size for inference (default: 520)
    """
    source_rtsp = st.sidebar.text_input("RTSP stream URL:", "rtsp://127.0.0.1:8554/live/vehicles_stream")
    st.sidebar.caption("Example: rtsp://127.0.0.1:8554/live/vehicles_stream")

    if st.sidebar.button("Start RTSP 🚀"):
        if not source_rtsp:
            st.error("❌ Please enter a valid RTSP address.")
            return
        get_frames_and_detect(conf, model, source_rtsp, tracker, iou, 416)