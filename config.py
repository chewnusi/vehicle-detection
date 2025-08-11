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
    "afv": IMAGES_DIR / "afv.jpg",
    "apc": IMAGES_DIR / "apc.webp",
    "artillery": IMAGES_DIR / "artillery.jpg",
    "air-defense": IMAGES_DIR / "air-defence.jpg",
}

VIDEO_DIR = ROOT / "videos"
VIDEOS_DICT = {
    "afv": VIDEO_DIR / "afv.mp4",
    "apc": VIDEO_DIR / "apc.mp4",
    "artillery": VIDEO_DIR / "artillery.mp4",
    "air-defense": VIDEO_DIR / "air-defense.mp4",
}

MODEL_DIR = ROOT / "weights"
DETECTION_MODEL = MODEL_DIR / "best_last_colab_0306.pt"

TRACKERS = {
    "bytetrack": {
        "name": "ByteTrack",
        "config": "trackers/bytetrack.yaml",
        "description": "Optimal for fast performance and high accuracy. Works best in occlusion conditions (when objects overlap each other)."
    },
    "botsort": {
        "name": "BotSORT",
        "config": "trackers/botsort.yaml",
        "description": "Better for complex scenes with many objects. Uses additional visual features for more stable tracking, but runs slower."
    }
}
