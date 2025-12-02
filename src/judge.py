# src/judge.py
import json
import os
import argparse
import time
from tqdm import tqdm
from openai import OpenAI

# CONFIGURATION
# ---------------------------------------------------------
# Recommendation: Use GPT-4o or Claude-3.5-Sonnet for the Judge.
# If local, use a LARGE model (70B+), otherwise the Judge is 
# no smarter than the Scouts.
JUDGE_API_URL = "https://api.openai.com/v1" 
JUDGE_API_KEY = os.getenv("OPENAI_API_KEY") 
JUDGE_MODEL = "gpt-4o" 
# ---------------------------------------------------------

client = OpenAI(base_url=JUDGE_API_URL, api_key=JUDGE_API_KEY)

SYSTEM_PROMPT_JUDGE = """
You are the **Chief Safety Officer** (The Judge) for an Autonomous Vehicle Dataset Curation project.
You are reviewing conflict reports from multiple AI Scouts (e.g., Kimi, Qwen) regarding the same driving scene.

### YOUR OBJECTIVE
Synthesize a single **"Ground Truth" JSON** for the scene.

### CONFLICT RESOLUTION RULES
1. **Safety Bias:** If ANY scout detects a High-Risk element (Pedestrian, Red Light, Construction) and provides valid reasoning, **mark it as TRUE**. Better to have a False Positive in a dataset than to miss a fatality.
2. **Hallucination Check:** If a scout detects something bizarre (e.g., "Elephant") with weak reasoning, and others disagree, **reject it**.
3. **YOLO Authority:** Use the provided YOLO Object Inventory as the tie-breaker for object existence.
4. **Waymo Taxonomy:** Ensure tags match the WOD-E2E categories (Construction, VRU, FOD, Weather).

### OUTPUT FORMAT
Return ONLY the consolidated JSON object.
"""

def load_jsonl_map(filename):
    """Loads a JSONL file into a Dict {token: data}."""
    data_map = {}
    if not os.path.exists(filename):
        print(f"⚠️ Warning: File not found {filename}")
        return data_map
        
    print(f"Loading {filename}...")
    with open(filename, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                if 'token' in item:
                    data_map[item['token']] = item
            except: pass
    return data_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs='+', required=True, help="List of index files (e.g. output/index_kimi.jsonl output/index_qwen.jsonl)")
    parser.add_argument("--output", type=str, default="consensus_dataset.jsonl", help="Filename for the final dataset")
    args = parser.parse_args()

    # 1. Load Data
    scout_maps = [load_jsonl_map(f) for f in args.files]
    
    # Get all unique tokens (intersection or union? Union is safer)
    all_tokens = set().union(*[d.keys() for d in scout_maps])
    print(f"👨‍⚖️ The Judge is ready. Processing {len(all_tokens)} unique frames.")

    # 2. Setup Output
    output_path = os.path.join("output", args.output)
    processed_tokens = set()
    
    # Resume logic
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try: processed_tokens.add(json.loads(line)['token'])
                except: pass
        print(f"Resuming... {len(processed_tokens)} already judged.")

    # 3. Judgment Loop
    with open(output_path, 'a') as f_out:
        for token in tqdm(all_tokens):
            if token in processed_tokens: continue

            # Gather Reports
            reports = []
            yolo_context = "Not Available"
            
            for i, scout_map in enumerate(scout_maps):
                if token in scout_map:
                    data = scout_map[token]
                    model_name = data.get('model_source', f'Scout_{i}')
                    trace = data.get('_reasoning_trace', 'No trace')
                    
                    # Clean up data for prompt (remove large fields)
                    clean_data = {k:v for k,v in data.items() if k not in ['token', '_reasoning_trace', 'model_source']}
                    
                    report = f"--- REPORT FROM {model_name.upper()} ---\n"
                    report += f"JSON: {json.dumps(clean_data)}\n"
                    report += f"REASONING: {trace}\n"
                    reports.append(report)
                    
                    # Try to grab YOLO inventory from one of them (they should be similar)
                    # Note: We need to pass YOLO inventory in index file if we want it here.
                    # Currently it is in the LOGS file, but the VLM Reasoning usually references it.

            if len(reports) == 0: continue

            # Construct the Case File
            user_content = f"### CASE FILE: Frame {token}\n\n"
            user_content += "\n".join(reports)
            user_content += "\n\n### INSTRUCTION\nCompare the reports above. Produce the FINAL CONSENSUS JSON."

            # Call The Judge
            try:
                response = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.0, # Deterministic
                    response_format={"type": "json_object"}
                )
                
                final_json = json.loads(response.choices[0].message.content)
                final_json['token'] = token
                final_json['consensus_source_count'] = len(reports)
                
                f_out.write(json.dumps(final_json) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"Judge Error on {token}: {e}")
                # Backoff
                time.sleep(2)

    print(f"✅ Consensus Reached. Final dataset saved to {output_path}")

if __name__ == "__main__":
    main()