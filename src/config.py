# src/config.py
import os
from pathlib import Path

# Update this path to where you extracted the v1.0-mini folder
# It should contain subfolders: maps, samples, sweeps, v1.0-mini
NUSCENES_DATAROOT = "nuscenes_data"  # MODIFY THIS PATH AS NEEDED "/path/to/your/nuscenes"
NUSCENES_VERSION = "v1.0-trainval"

# MODIFIED: Front-Hemisphere Only
# We dropped CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
CAM_ORDER = [
    "CAM_FRONT_LEFT", 
    "CAM_FRONT", 
    "CAM_FRONT_RIGHT"
]

# Visual settings
# 0.5 reduces 1600x900 -> 800x450 per image. 
# Total Montage: 2400x900. Fits easily in Qwen3-VL context window.
RESIZE_FACTOR = 1.0