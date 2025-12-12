import json
import argparse
from tqdm import tqdm
from src.data.loader import NuScenesLoader
from src.model.vlm_client import VLMClient
from src.benchmark_deprecated import GROUND_TRUTH, ALL_TAGS

# Standard Prompt (The "Naive" Approach)
VANILLA_PROMPT = """
You are an AI assistant. Analyze these 3 driving camera images.
Identify if any of these categories are present: 
Construction, Intersection, VRU Hazard, Debris, Special Vehicles, Adverse Weather.
Return a JSON: {"tags": ["list", "of", "detected", "tags"]}
"""

def check_metadata(loader, token):
    """Baseline 1: Search NuScenes description."""
    try:
        desc = loader.get_scene_description(token).lower()
        found = []
        if "construction" in desc or "worker" in desc: found.append("construction")
        if "rain" in desc or "wet" in desc: found.append("weather_adverse")
        if "pedestrian" in desc or "bicycle" in desc: found.append("vru_hazard")
        if "intersection" in desc: found.append("intersection_complex")
        return found
    except: return []

def run_baselines():
    print("🚀 Running Baselines on Gold Set...")
    loader = NuScenesLoader()
    client = VLMClient(model_id="qwen3-vl-instruct") # Use standard model
    
    results = []

    for token in tqdm(GROUND_TRUTH.keys()):
        # 1. Metadata Baseline
        meta_tags = check_metadata(loader, token)
        
        # 2. Vanilla VLM Baseline
        paths = loader.get_camera_paths(token)
        # Load images (simple load)
        images = {k: client._load_image_from_path(v) for k,v in paths.items() if k in ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]} # You might need to adjust image loading logic here to match your main script
        
        # Call VLM (Simple)
        try:
            # Note: We reuse analyze_multiview but pass the simple prompt and NO inventory
            resp = client.analyze_multiview(images, VANILLA_PROMPT, object_inventory=None)
            vanilla_tags = resp.get("parsed_json", {}).get("tags", []) if resp else []
        except:
            vanilla_tags = []

        results.append({
            "token": token,
            "metadata_tags": meta_tags,
            "vanilla_tags": vanilla_tags
        })

    # Save for the benchmark script to read
    with open("output/baseline_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print("✅ Baselines Done.")

if __name__ == "__main__":
    # Note: Ensure you have your VLMClient image loading logic accessible or adapted
    run_baselines()