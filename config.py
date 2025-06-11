from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())

IMAGE = "Image"
VIDEO = "Video"
RTSP = "RTSP"
YOUTUBE = "YouTube"

CLASSES = ["AFV", "APC", "Artillery", "Air-Defense"]
SOURCES_LIST = [IMAGE, VIDEO, RTSP, YOUTUBE]

IMAGES_DIR = ROOT / "images"
IMAGES_DICT = {
    "image_1": IMAGES_DIR / "afv.jpg",
}

VIDEO_DIR = ROOT / "videos"
VIDEOS_DICT = {
    "video_1": VIDEO_DIR / "afv_apc.mp4",
    "video_short": VIDEO_DIR / "afv.mp4",
}

MODEL_DIR = ROOT / "weights"
DETECTION_MODEL = MODEL_DIR / "best_last_colab_0306.pt"
