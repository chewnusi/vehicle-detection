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

CLASSES = ['']
SOURCES_LIST = [IMAGE, VIDEO, RTSP, YOUTUBE]

IMAGES_DIR = ROOT / "images"
DEFAULT_IMAGE = IMAGES_DIR / "image.jpg"

VIDEO_DIR = ROOT / "videos"
VIDEOS_DICT = {
    "video_1": VIDEO_DIR / "afv_apc.mp4",
}

MODEL_DIR = ROOT / "weights"
DETECTION_MODEL = MODEL_DIR / "best1.pt"
