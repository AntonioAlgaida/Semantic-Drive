import sys
import os
import json
import torch
import open_clip
from tqdm import tqdm
from PIL import Image

# --- 1. PATH FIXES ---
# Get the absolute path of the script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # e.g., .../Semantic-Drive/src
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)              # e.g., .../Semantic-Drive

# Add Project Root to sys.path so we can import 'src' modules reliably
sys.path.append(PROJECT_ROOT)

# Define Output Directory Absolute Path
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "clip_baseline.jsonl")

# Ensure output dir exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")

# Import local modules after path fix
from src.data.loader import NuScenesLoader
from src.data.visuals import create_surround_montage

# CONFIG
MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The queries we want to compare
# The queries we want to compare.
# We include "Negative Controls" (Safe versions) to test Spatial Reasoning.
QUERIES = {
    # --- 1. VRU SPATIAL TEST (The "Sidewalk vs. Road" Problem) ---
    # CLIP often fails here, flagging both as "Hazard" because it sees a person.
    "vru_on_road_hazard": "pedestrian standing in the middle of the driving lane",
    "vru_on_sidewalk_safe": "pedestrian standing safely on the sidewalk",
    "bicyclist_on_road_hazard": "bicyclist riding in the middle of the driving lane",
    "bicyclist_on_bike_lane_safe": "bicyclist riding safely in the bike lane",

    # --- 2. CONSTRUCTION TOPOLOGY (The "Blocking" Test) ---
    # Can it distinguish "presence" (roadside) from "obstruction" (blocking)?
    "construction_blocking": "orange construction barrels physically blocking the lane",
    "construction_roadside": "construction signs and cones on the side of the road not blocking traffic",

    # --- 3. ATTRIBUTE BINDING (The "State" Test) ---
    # Testing if it binds "Red" to the light or just sees red things.
    "traffic_light_red": "red traffic light signal",
    "traffic_light_green": "green traffic light signal",
    "traffic_light_off": "traffic light that is turned off",

    # --- 4. ENVIRONMENTAL ODD ---
    "weather_rain_night": "wet road surface at night with streetlights reflecting",
    "weather_clear_day": "dry road surface with bright sunlight",
    "fog_hazard": "dense fog reducing visibility on the road",
    "clear_safe": "clear weather with good visibility on the road",

    # --- 5. RARE CLASSES (Long-Tail) ---
    "special_police": "police vehicle with flashing lights",
    "special_ambulance": "ambulance vehicle with flashing lights",
    "debris_hazard": "trash or large debris object lying on the road",
    "animal_crossing": "animal crossing the road ahead",
}

def main():
    print(f"🚀 Loading CLIP ({MODEL_NAME})...")
    print(f"📂 Saving results to: {OUTPUT_FILE}")
    
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
        model.to(DEVICE)
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    except ImportError:
        print("❌ Error: open_clip not installed. Run: pip install open_clip_torch")
        return

    # Encode Text Queries
    text_tokens = tokenizer(list(QUERIES.values())).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    try:
        loader = NuScenesLoader(dataroot="../nuscenes_data", 
                                version="v1.0-trainval")
    except Exception as e:
        print(f"❌ NuScenes Loader Error: {e}")
        print("Check your config.py NUSCENES_DATAROOT path.")
        return

    # Let's sample 100 random frames + your gold set if available
    # Using a smaller subset for the benchmark speed
    samples = loader.get_sparse_samples(frames_per_scene=3)
    # Shuffle or just take first 200
    target_samples = samples[:]
    
    results = []
    
    print(f"📉 Benchmarking CLIP on {len(target_samples)} frames...")
    
    for token in tqdm(target_samples):
        # Load Images (Use Montage to give CLIP global context)
        paths = loader.get_camera_paths(token)
        # Filter for front cameras only for fairness
        front_paths = {k:v for k,v in paths.items() if "FRONT" in k}
        
        # Create visual
        montage = create_surround_montage(front_paths, resize_factor=0.5)
        if not montage: continue
        
        # Preprocess
        image_input = preprocess(montage).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Cosine Similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probs = similarity.cpu().numpy()[0]

        # Store top match
        entry = {"token": token, "scores": {}}
        for i, key in enumerate(QUERIES.keys()):
            entry["scores"][key] = float(probs[i])
            
        results.append(entry)

    # Save using the ABSOLUTE path
    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"✅ CLIP Benchmark Done. File saved at: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()