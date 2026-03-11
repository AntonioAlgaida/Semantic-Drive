# src/benchmark_grounding_dino.py

import sys
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# --- PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)              
sys.path.append(PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "grounding_dino_baseline.jsonl")

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

from src.data.loader import NuScenesLoader
from src.data.visuals import create_surround_montage

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "IDEA-Research/grounding-dino-base"

# --- THE EXACT SAME QUERIES AS CLIP ---
QUERIES = {
    "vru_on_road_hazard": "pedestrian standing in the middle of the driving lane",
    "vru_on_sidewalk_safe": "pedestrian standing safely on the sidewalk",
    "bicyclist_on_road_hazard": "bicyclist riding in the middle of the driving lane",
    "bicyclist_on_bike_lane_safe": "bicyclist riding safely in the bike lane",
    "construction_blocking": "orange construction barrels physically blocking the lane",
    "construction_roadside": "construction signs and cones on the side of the road not blocking traffic",
    "traffic_light_red": "red traffic light signal",
    "traffic_light_green": "green traffic light signal",
    "traffic_light_off": "traffic light that is turned off",
    "weather_rain_night": "wet road surface at night with streetlights reflecting",
    "weather_clear_day": "dry road surface with bright sunlight",
    "fog_hazard": "dense fog reducing visibility on the road",
    "clear_safe": "clear weather with good visibility on the road",
    "special_police": "police vehicle with flashing lights",
    "special_ambulance": "ambulance vehicle with flashing lights",
    "debris_hazard": "trash or large debris object lying on the road",
    "animal_crossing": "animal crossing the road ahead",
}

def get_all_gold_tokens():
    tokens = set()
    files = ["output/gold_annotations_master.json", "output/gold_annotations_unbiased.json"]
    for f in files:
        if os.path.exists(f):
            with open(f, 'r') as file:
                data = json.load(file)
                tokens.update(data.keys())
    return list(tokens)

def main():
    print(f"🚀 Loading Grounding DINO ({MODEL_ID})...")
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)

    loader = NuScenesLoader(dataroot="nuscenes_data", version="v1.0-trainval")
    target_samples = get_all_gold_tokens()
    
    if not target_samples:
        print("⚠️ No gold tokens found.")
        return

    # DINO expects a single string with classes separated by " . "
    text_prompt = " . ".join(QUERIES.values()).lower() + " ."

    results = []
    print(f"📉 Benchmarking Grounding DINO on {len(target_samples)} gold frames...")

    for token in tqdm(target_samples):
        paths = loader.get_camera_paths(token)
        front_paths = {k:v for k,v in paths.items() if "FRONT" in k}
        
        montage = create_surround_montage(front_paths, resize_factor=0.5)
        if not montage: continue
        
        # Inference
        inputs = processor(images=montage, text=text_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            
        # We use a very low threshold here (0.01) to capture raw scores
        # We will handle the actual thresholding in benchmark_final.py
        target_sizes = torch.tensor([montage.size[::-1]])
        processed_outputs = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.2, # Capture all potential detections
            text_threshold=0.2,
            target_sizes=target_sizes
        )[0]

        detected_labels = processed_outputs["labels"]
        detected_scores = processed_outputs["scores"]
        
        # --- NEW LOGIC: Map detections to the Score Dictionary ---
        # Initialize all scores to 0.0
        frame_scores = {key: 0.0 for key in QUERIES.keys()}
        
        for label, score in zip(detected_labels, detected_scores):
            label_clean = label.lower().strip()
            score_val = float(score.cpu().item())
            
            # Find which query key this detection belongs to
            for key, query_text in QUERIES.items():
                # DINO sometimes returns sub-phrases, so we check for overlap
                if label_clean in query_text.lower():
                    # We take the maximum score found for this query key
                    if score_val > frame_scores[key]:
                        frame_scores[key] = score_val

        results.append({
            "token": token, 
            "scores": frame_scores
        })

    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"✅ Grounding DINO Baseline generated at: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()