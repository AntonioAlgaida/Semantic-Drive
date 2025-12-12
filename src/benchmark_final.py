import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# --- CONFIGURATION ---
GOLD_FILE = "output/gold_annotations_master.json"

# Define your experiments here. 
EXPERIMENTS = {
    "Baseline: CLIP (ViT-L/14)": "output/clip_baseline.jsonl",
    "Ablation: Qwen-VL (No YOLO)": "output/index_qwen3_noYOLO_run.jsonl", 
    "Single Scout: Qwen-VL + YOLO": "output/index_qwen3_run_reasoning.jsonl",
    "Single Scout: Gemma3 + YOLO": "output/index_gemma_run.jsonl",
    "Single Scout: Kimi + YOLO": "output/index_kimi_run_reasoning.jsonl",
    "Ours: Semantic-Drive (Consensus)": "output/consensus_final.jsonl"
}

# The specific tags to evaluate for F1 Score
TARGET_TAGS = [
    "construction", 
    "weather_adverse", 
    "vru_hazard", 
    "fod_debris", 
    "special_vehicle", 
    "lane_diversion"
]

def parse_clip_entry(item):
    """Converts CLIP probability scores into boolean tags."""
    scores = item.get("scores", {})
    tags = []
    THRESHOLD = 0.25 
    
    if (scores.get("vru_on_road_hazard", 0) > THRESHOLD or 
        scores.get("bicyclist_on_road_hazard", 0) > THRESHOLD or 
        scores.get("animal_crossing", 0) > THRESHOLD):
        tags.append("vru_hazard")
        
    if scores.get("construction_blocking", 0) > THRESHOLD:
        tags.append("construction")
        tags.append("lane_diversion")
        
    if (scores.get("weather_rain_night", 0) > THRESHOLD or 
        scores.get("fog_hazard", 0) > THRESHOLD):
        tags.append("weather_adverse")
        
    if (scores.get("special_police", 0) > THRESHOLD or 
        scores.get("special_ambulance", 0) > THRESHOLD):
        tags.append("special_vehicle")
        
    if scores.get("debris_hazard", 0) > THRESHOLD:
        tags.append("fod_debris")
        
    return tags, 0 

def parse_vlm_entry(item):
    """Parses standard VLM/Judge JSON output."""
    tags = item.get('wod_e2e_tags', [])
    if isinstance(tags, str): tags = [] 
    try:
        risk = int(item.get('scenario_criticality', {}).get('risk_score', 0))
    except:
        risk = 0
    return tags, risk

def calculate_metrics(name, pred_file, gold_data):
    if not os.path.exists(pred_file):
        print(f"⚠️ Skipping {name}: File not found ({pred_file})")
        return None

    # 1. Load Predictions
    preds_map = {}
    with open(pred_file, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                token = item['token']
                
                if "scores" in item:
                    tags, risk = parse_clip_entry(item)
                else:
                    tags, risk = parse_vlm_entry(item)
                    
                preds_map[token] = {"tags": tags, "risk": risk}
            except: pass

    # 2. Compare against Gold
    y_true = []
    y_pred = []
    risk_errors = []
    
    common_count = 0
    
    for token, truth in gold_data.items():
        if token not in preds_map:
            continue
            
        common_count += 1
        pred = preds_map[token]
        
        # --- FIX: Extract Ground Truth correctly from Full Schema ---
        gt_tags_list = truth.get('wod_e2e_tags', [])
        
        # Handle cases where risk might be missing or nested
        try:
            gt_risk = int(truth.get('scenario_criticality', {}).get('risk_score', 0))
        except:
            gt_risk = 0

        # Create Binary Vectors
        gt_vec = [1 if t in gt_tags_list else 0 for t in TARGET_TAGS]
        pr_vec = [1 if t in pred['tags'] else 0 for t in TARGET_TAGS]
        
        y_true.append(gt_vec)
        y_pred.append(pr_vec)
        
        # Risk Error
        if "CLIP" not in name:
            err = abs(gt_risk - pred['risk'])
            risk_errors.append(err)

    if common_count == 0:
        return None

    # 3. Compute Stats
    return {
        "Method": name,
        "Precision": precision_score(y_true, y_pred, average='micro', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='micro', zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average='micro', zero_division=0),
        "MAE Risk": np.mean(risk_errors) if risk_errors else np.nan,
        "Samples": common_count
    }

def main():
    print("📊 Running Final Benchmark...")
    
    if not os.path.exists(GOLD_FILE):
        print(f"❌ Critical Error: Gold file not found at {GOLD_FILE}")
        return
        
    with open(GOLD_FILE, 'r') as f:
        gold_data = json.load(f)
    print(f"✅ Loaded {len(gold_data)} Ground Truth annotations.")

    results = []
    for name, path in EXPERIMENTS.items():
        stats = calculate_metrics(name, path, gold_data)
        if stats:
            results.append(stats)
            
    if results:
        df = pd.DataFrame(results).set_index("Method")
        print("\n" + "="*80)
        print(df[["Precision", "Recall", "F1-Score", "MAE Risk", "Samples"]].round(3))
        print("="*80)
        df.to_csv("output/final_benchmark_results.csv")
        print("✅ Saved to CSV.")
    else:
        print("No valid experiments found.")

if __name__ == "__main__":
    main()