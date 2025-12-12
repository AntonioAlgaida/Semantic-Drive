# src/data/loader.py
import os
import sys
import numpy as np

sys.path.append(os.path.abspath('..'))

from nuscenes.nuscenes import NuScenes
from src.config import NUSCENES_DATAROOT, NUSCENES_VERSION, CAM_ORDER

class NuScenesLoader:
    def __init__(self, dataroot=NUSCENES_DATAROOT, version=NUSCENES_VERSION):
        if not os.path.exists(dataroot):
            # print the whole path for easier debugging
            print(f"Debug: Checking existence of path: {dataroot}")
            print(f"Debug: Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"Dataset not found at {dataroot}. Check your drive mount.")
            
        print(f"Loading NuScenes {version} database from {dataroot}...")
        # verbose=False to keep logs clean
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    
    def get_all_samples(self):
        """Returns a list of all sample tokens in the dataset."""
        return [s['token'] for s in self.nusc.sample]

    def get_sparse_samples(self, frames_per_scene=3):
        """
        Smart Sampling: Returns 'frames_per_scene' tokens from EACH scene.
        Spaced out evenly (e.g., Start, Middle, End).
        """
        selected_tokens = []
        
        print(f"Selecting {frames_per_scene} frames from {len(self.nusc.scene)} scenes...")
        
        for scene in self.nusc.scene:
            # 1. Get all sample tokens for this scene in order
            scene_samples = []
            current_token = scene['first_sample_token']
            
            while current_token:
                scene_samples.append(current_token)
                # Traverse linked list
                current_record = self.nusc.get('sample', current_token)
                current_token = current_record['next']
            
            # 2. Select Indices (e.g., [0, 20, 39])
            total = len(scene_samples)
            if total < frames_per_scene:
                # Scene too short? Take all.
                indices = range(total)
            else:
                # Linspace gives us evenly spaced indices (float), cast to int
                indices = np.linspace(0, total - 1, num=frames_per_scene, dtype=int)
            
            for idx in indices:
                selected_tokens.append(scene_samples[idx])
                
        return selected_tokens
    
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