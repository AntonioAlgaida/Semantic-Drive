import json
import os
import matplotlib.pyplot as plt
import numpy as np
import glob

# CONFIG
OUTPUT_DIR = "output"
LOG_FILES = glob.glob(os.path.join(OUTPUT_DIR, "logs_*.jsonl"))

def analyze_logs():
    stats = {}

    for log_file in LOG_FILES:
        model_name = os.path.basename(log_file).replace("logs_", "").replace(".jsonl", "")
        print(f"Analyzing {model_name}...")
        
        input_toks = []
        output_toks = []
        latencies = [] # Note: You need to log 'elapsed_time' in main.py to get this perfectly, 
                       # but we can infer from 'usage' if needed or just use tokens.
        
        tags_found = {}

        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    
                    # 1. Token Usage
                    if entry.get("usage"):
                        input_toks.append(entry["usage"]["input_tokens"])
                        output_toks.append(entry["usage"]["output_tokens"])
                    
                    # 2. Mining Results (Tags)
                    # We need to look at the corresponding index file or parse the raw_response if valid
                    # Assuming we check the raw_response for tags if parsing failed, 
                    # but better to check the success entries in the index file usually.
                    # Here we just look at successful log entries.
                    if entry.get("success") and entry.get("raw_response"):
                        # Quick dirty parse for tags count
                        content = entry["raw_response"]
                        if "construction" in content.lower(): tags_found["Construction"] = tags_found.get("Construction", 0) + 1
                        if "rain" in content.lower(): tags_found["Rain"] = tags_found.get("Rain", 0) + 1
                        if "debris" in content.lower(): tags_found["Debris"] = tags_found.get("Debris", 0) + 1
                        if "police" in content.lower(): tags_found["Police"] = tags_found.get("Police", 0) + 1
                        if "pedestrian" in content.lower(): tags_found["VRU"] = tags_found.get("VRU", 0) + 1

                except: pass
        
        if input_toks:
            stats[model_name] = {
                "avg_input": np.mean(input_toks),
                "avg_output": np.mean(output_toks),
                "total_frames": len(input_toks),
                "tags": tags_found
            }

    # --- PLOTTING ---
    # Figure 1: Token Usage Comparison
    models = list(stats.keys())
    avg_in = [s["avg_input"] for s in stats.values()]
    avg_out = [s["avg_output"] for s in stats.values()]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, avg_in, width, label='Input Tokens (Context)')
    rects2 = ax.bar(x + width/2, avg_out, width, label='Output Tokens (Reasoning)')

    ax.set_ylabel('Token Count')
    ax.set_title('Cognitive Load per Model (Reasoning Depth)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # Save
    plt.savefig("assets/stats_efficiency.png")
    print("Saved assets/stats_efficiency.png")
    
    # Print Table Data
    print("\n--- TABLE DATA (For Paper) ---")
    print(f"{'Model':<15} | {'Input Avg':<10} | {'Output Avg':<10} | {'Reasoning Ratio'}")
    for m, s in stats.items():
        ratio = s['avg_output'] / s['avg_input']
        print(f"{m:<15} | {s['avg_input']:.1f}      | {s['avg_output']:.1f}       | {ratio:.2f}x")

if __name__ == "__main__":
    analyze_logs()