# src/benchmark_owlv2.py

import sys
import os
import json
import torch
from tqdm import tqdm
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# --- PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)              
sys.path.append(PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "grounding_OWLv2_baseline.jsonl")

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

from src.data.loader import NuScenesLoader
from src.data.visuals import create_surround_montage

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "google/owlv2-base-patch16-ensemble"

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
    print(f"🚀 Loading OWL-v2 ({MODEL_ID})...")
    
    try:
        processor = Owlv2Processor.from_pretrained(MODEL_ID)
        model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    loader = NuScenesLoader(dataroot="nuscenes_data", version="v1.0-trainval")
    target_samples = get_all_gold_tokens()
    
    if not target_samples:
        print("⚠️ No gold tokens found.")
        return

    # OWL-v2 expects a flat list of text queries
    query_keys = list(QUERIES.keys())
    query_texts = [QUERIES[k] for k in query_keys]
    # Processor expects nested list for batching: [[q1, q2, ...]]
    text_queries = [query_texts]

    results = []
    print(f"📉 Benchmarking OWL-v2 on {len(target_samples)} gold frames...")

    for token in tqdm(target_samples):
        paths = loader.get_camera_paths(token)
        front_paths = {k:v for k,v in paths.items() if "FRONT" in k}
        
        montage = create_surround_montage(front_paths, resize_factor=0.5)
        if not montage: continue
        
        # Inference
        inputs = processor(text=text_queries, images=montage, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Post-process with very low threshold to get raw max scores
        target_sizes = torch.tensor([(montage.height, montage.width)]).to(DEVICE)
        processed_outputs = processor.post_process_grounded_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=0.2
        )[0]

        scores = processed_outputs["scores"].cpu().numpy()
        labels = processed_outputs["labels"].cpu().numpy() # This gives the index in query_texts
        
        # Initialize frame scores to 0.0
        frame_scores = {key: 0.0 for key in query_keys}
        
        for score, label_idx in zip(scores, labels):
            key = query_keys[label_idx]
            if score > frame_scores[key]:
                frame_scores[key] = float(score)

        results.append({
            "token": token, 
            "scores": frame_scores
        })

    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"✅ OWL-v2 Baseline generated at: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()