# src/data/loader.py
import os
import sys
sys.path.append(os.path.abspath('..'))

from nuscenes.nuscenes import NuScenes
from src.config import NUSCENES_DATAROOT, NUSCENES_VERSION, CAM_ORDER

class NuScenesLoader:
    def __init__(self, dataroot=NUSCENES_DATAROOT, version=NUSCENES_VERSION):
        if not os.path.exists(dataroot):
            raise FileNotFoundError(f"Dataset not found at {dataroot}. Check your drive mount.")
            
        print(f"Loading NuScenes {version} database from {dataroot}...")
        # verbose=False to keep logs clean
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    
    def get_all_samples(self):
        """Returns a list of all sample tokens in the dataset."""
        return [s['token'] for s in self.nusc.sample]

    def get_camera_paths(self, sample_token):
        """
        Given a sample token, returns a dict mapping camera names to absolute file paths.
        dictionary keys match the CAM_ORDER in config.
        """
        sample_record = self.nusc.get('sample', sample_token)
        camera_paths = {}

        for cam_channel in CAM_ORDER:
            # Get the sample data token for this camera
            sd_token = sample_record['data'][cam_channel]
            # Get the absolute path
            path = self.nusc.get_sample_data_path(sd_token)
            camera_paths[cam_channel] = path
            
        return camera_paths

    def get_scene_description(self, sample_token):
        """Helper to get the human-readable description of the scene."""
        sample = self.nusc.get('sample', sample_token)
        scene = self.nusc.get('scene', sample['scene_token'])
        return scene['description']