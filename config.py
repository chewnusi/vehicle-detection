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

# Tracker configurations
TRACKER_TYPES = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
    "strongsort": "strong_sort.yaml",
    "deepocsort": "deepocsort.yaml",
}

# Default tracker settings optimized for military equipment tracking
TRACKER_CONFIG = {
    # General tracking parameters
    'tracker_type': "bytetrack",  # Default tracker
    'conf_thres': 0.4,  # Detection confidence threshold
    'iou_thres': 0.5,  # IOU threshold for tracking
    'min_hits': 3,  # Minimum hits to confirm track
    'max_age': 30,  # Maximum frames to keep dead tracks
    
    # Motion prediction parameters
    'velocity_persist': True,  # Use velocity prediction
    'velocity_weight': 0.7,  # Weight of velocity in prediction
    
    # Track management
    'track_buffer': 60,  # How many frames to keep track history
    'match_thresh': 0.8,  # Threshold for track matching
    'track_high_thresh': 0.6,  # High confidence threshold
    'track_low_thresh': 0.1,  # Low confidence threshold
    'new_track_thresh': 0.6,  # Threshold for creating new tracks
    
    # Frame processing
    'frame_rate': 30,  # Default frame rate
    'min_box_area': 100,  # Minimum detection box area
    'aspect_ratio_thresh': 3.0,  # Maximum allowed aspect ratio
    
    # Advanced options
    'appearance_thresh': 0.25,  # Threshold for appearance feature matching
    'proximity_thresh': 0.5,  # Threshold for spatial proximity
    'use_byte': True,  # Use ByteTrack matching strategy
    
    # Military equipment specific
    'motion_weight': 0.2,  # Weight for motion prediction (lower for slow vehicles)
    'size_weight': 0.4,  # Weight for size consistency
    'class_weight': 0.4,  # Weight for class consistency
}
