import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# --- 1. THE GOLD STANDARD (MANUAL ANNOTATION) ---
# Format: "token": ["tag1", "tag2"]
# Select 10-20 interesting frames from your dashboard and tag them here.
GROUND_TRUTH = {
    # EXAMPLE 1: Construction Scene
    "eef55c...": ["construction", "lane_diversion"],
    # EXAMPLE 2: Rain
    "378a3a...": ["weather_adverse", "special_vehicle"],
    # EXAMPLE 3: Empty Road
    "token_3...": [],
    # ... Add your 20 frames here
}

# The vocabulary we are testing
ALL_TAGS = [
    "construction", "intersection_complex", "vru_hazard", 
    "fod_debris", "weather_adverse", "special_vehicle", 
    "lane_diversion", "sensor_failure"
]

def evaluate(prediction_file, run_name):
    print(f"📊 Evaluating: {run_name}")
    
    y_true = []
    y_pred = []
    
    # Load Predictions
    preds = {}
    with open(prediction_file, 'r') as f:
        for line in f:
            try:
                d = json.loads(line)
                # Combine WOD tags + other boolean flags if necessary
                tags = d.get('wod_e2e_tags', [])
                preds[d['token']] = set(tags)
            except: pass

    # Iterate over Ground Truth
    found_count = 0
    for token, gt_tags in GROUND_TRUTH.items():
        if token not in preds:
            print(f"⚠️ Warning: Token {token[:8]} not found in predictions.")
            continue
            
        found_count += 1
        pred_tags = preds[token]
        
        # Binary Vector for this frame
        gt_vector = [1 if t in gt_tags else 0 for t in ALL_TAGS]
        pred_vector = [1 if t in pred_tags else 0 for t in ALL_TAGS]
        
        y_true.append(gt_vector)
        y_pred.append(pred_vector)

    if found_count == 0:
        print("No matching tokens found.")
        return

    # Flatten for global micro-average
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    p = precision_score(y_true, y_pred, average='micro', zero_division=0)
    r = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    print(f"   Precision: {p:.3f}")
    print(f"   Recall:    {r:.3f}")
    print(f"   F1-Score:  {f1:.3f}")
    print("-" * 30)

if __name__ == "__main__":
    # Compare your Main Run vs. An Ablated Run (if you have one)
    # Or just validate your current best run
    evaluate("output/index_qwen_run.jsonl", "Ours (Neuro-Symbolic)")