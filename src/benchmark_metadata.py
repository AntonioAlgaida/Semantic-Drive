import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import sys

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import NuScenesLoader

# --- CONFIGURATION ---
GOLD_FILE = "output/gold_annotations_master.json"
OUTPUT_FILE = "output/metadata_baseline.jsonl"

# Taxonomy Mapping: Metadata Keyword -> WOD-E2E Tag
KEYWORD_MAP = {
    "construction": ["construction", "road work", "worker", "cone", "barrier"],
    "weather_adverse": ["rain", "wet", "night", "glare", "dark", "storm", "fog"],
    "vru_hazard": ["pedestrian", "child", "bicycle", "cyclist", "jaywalk", "person"],
    "fod_debris": ["debris", "trash", "object on road"],
    "special_vehicle": ["police", "ambulance", "fire", "bus", "truck"],
    "lane_diversion": ["diversion", "lane shift", "merge"]
}

TARGET_TAGS = [
    "construction", "weather_adverse", "vru_hazard", 
    "fod_debris", "special_vehicle", "lane_diversion"
]

def check_keywords(description):
    """Returns a list of tags found in the description string."""
    found_tags = set()
    if not description: return []
    
    desc_lower = description.lower()
    
    for tag, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in desc_lower:
                found_tags.add(tag)
                break 
    return list(found_tags)

def main():
    print("📊 Running Metadata Keyword Baseline...")
    
    # 1. Load Gold Set
    if not os.path.exists(GOLD_FILE):
        print(f"❌ Gold file not found: {GOLD_FILE}")
        return
    with open(GOLD_FILE, 'r') as f:
        gold_data = json.load(f)
        
    # 2. Load NuScenes Metadata
    # We need the loader to get the scene description for each sample token
    # Ensure this path matches your setup
    loader = NuScenesLoader(dataroot="nuscenes_data", version="v1.0-trainval")
    
    y_true = []
    y_pred = []
    
    results_log = []
    
    print(f"🔎 Scanning metadata for {len(gold_data)} frames...")
    
    for token, truth in gold_data.items():
        # Get the Scene Description from NuScenes
        try:
            sample = loader.nusc.get('sample', token)
            scene = loader.nusc.get('scene', sample['scene_token'])
            description = scene['description']
        except:
            description = ""
            
        # "Simulate" the search
        pred_tags = check_keywords(description)
        
        # --- FIX IS HERE: Use correct key 'wod_e2e_tags' ---
        true_tags = truth.get('wod_e2e_tags', [])
        
        # Log for debugging
        results_log.append({
            "token": token,
            "description": description,
            "pred_tags": pred_tags,
            "true_tags": true_tags
        })
        
        # Build vectors
        gt_vec = [1 if t in true_tags else 0 for t in TARGET_TAGS]
        pr_vec = [1 if t in pred_tags else 0 for t in TARGET_TAGS]
        
        y_true.append(gt_vec)
        y_pred.append(pr_vec)

    # 3. Calculate Metrics
    micro_p = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_r = recall_score(y_true, y_pred, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    print("\n" + "="*60)
    print(f"METADATA SEARCH BASELINE")
    print("="*60)
    print(f"Precision: {micro_p:.3f}")
    print(f"Recall:    {micro_r:.3f}")
    print(f"F1-Score:  {micro_f1:.3f}")
    print("="*60)
    
    # Save for plotting later
    with open(OUTPUT_FILE, 'w') as f:
        for r in results_log:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()