# src/judge.py
import json
import os
import argparse
import re
from tqdm import tqdm
from openai import OpenAI
from src.reward import SymbolicVerifier

# IMPORT THE SCHEMA DEFINITIONS
# This ensures the Judge knows the allowed Enums (e.g., "jaywalking_hesitant")
from src.model.prompts import SCHEMA_GUIDE, OUTPUT_SKELETON

# --- CONFIGURATION ---
JUDGE_API_URL = "http://localhost:1234/v1" 
JUDGE_API_KEY = "lm-studio"
JUDGE_MODEL_ID = "local-model"

client = OpenAI(base_url=JUDGE_API_URL, api_key=JUDGE_API_KEY)

# --- SYSTEM PROMPT (Enforcing Uniformity) ---
SYSTEM_PROMPT = f"""
You are the **Chief Safety Officer** (The Judge) for an Autonomous Vehicle Data Mining system.
You have reports from 3 AI Scouts regarding a driving scene.

### YOUR GOAL
Synthesize a single **"Ground Truth" JSON** that resolves conflicts between scouts.

### RULES OF EVIDENCE
1. **Trust Grounding:** If YOLO detects an object, favor scouts that confirm it visually.
2. **Safety Bias:** In ambiguity, err on the side of caution (Higher Risk).
3. **Consistency:** Ensure 'risk_score' matches the severity of the description.

### SCHEMA ENFORCEMENT
You MUST output the JSON following this EXACT schema and vocabulary:
{SCHEMA_GUIDE}

{OUTPUT_SKELETON}

### OUTPUT
Return ONLY the final JSON object. Do not include markdown or reasoning text outside the JSON.
"""

def clean_json_string(content):
    """Robust cleaner for Local LLM outputs"""
    if "```" in content:
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            content = match.group(1)
    return content.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs='+', required=True, help="Input jsonl files")
    parser.add_argument("--output", type=str, default="output/consensus_final.jsonl")
    parser.add_argument("--n", type=int, default=3, help="Best-of-N attempts")
    args = parser.parse_args()

    # ... [Rest of the file remains the same] ...
    # (Load Data logic...)
    
    # Intersection of tokens
    data_maps = []
    print(f"📂 Loading {len(args.files)} scout files...")
    for f in args.files:
        d = {}
        with open(f, 'r') as file:
            for line in file:
                try:
                    obj = json.loads(line)
                    if obj.get('success'): 
                        d[obj['token']] = obj
                except: pass
        data_maps.append(d)

    all_tokens = set().union(*[d.keys() for d in data_maps])
    verifier = SymbolicVerifier()
    
    print(f"👨‍⚖️ Judge initialized. Processing {len(all_tokens)} frames...")

    with open(args.output, 'w') as f_out:
        for token in tqdm(all_tokens):
            
            # 1. Aggregate Reports
            reports = []
            yolo_context = "No YOLO Data"
            
            for i, d in enumerate(data_maps):
                if token in d:
                    item = d[token]
                    if yolo_context == "No YOLO Data" and "yolo_inventory" in item:
                        yolo_context = item["yolo_inventory"]
                    
                    # Clean the item for the prompt (remove bulky fields)
                    # We remove _reasoning_trace from the JSON dump because we pass it separately or summarize it
                    clean_obj = {k:v for k,v in item.items() if k not in ['token', '_reasoning_trace', 'yolo_inventory', 'raw_response', 'input_messages_log', 'usage']}
                    
                    # Get trace
                    trace = item.get('_reasoning_trace', 'No trace')[:500] 
                    
                    reports.append(f"--- SCOUT {i+1} ---\n[Trace]: {trace}...\n[JSON]: {json.dumps(clean_obj)}")

            if not reports: continue

            # 2. Construct Prompt
            user_content = f"### SYMBOLIC GROUNDING (YOLO):\n{yolo_context}\n\n"
            user_content += "### SCOUT REPORTS:\n" + "\n\n".join(reports)
            user_content += "\n\nSynthesize the Consensus JSON."

            # 3. Best-of-N Loop
            candidates = []
            for attempt in range(args.n):
                try:
                    response = client.chat.completions.create(
                        model=JUDGE_MODEL_ID,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=0.3,
                        max_tokens=16384
                    )
                    
                    json_text = clean_json_string(response.choices[0].message.content)
                    candidate_json = json.loads(json_text)
                    
                    score, reasons = verifier.calculate_score(candidate_json, yolo_context)
                    candidates.append({"json": candidate_json, "score": score, "reasons": reasons})
                except: pass

            if not candidates: continue

            # 4. Pick Winner
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best = candidates[0]
            
            final_record = best['json']
            final_record['token'] = token
            final_record['judge_score'] = best['score']
            final_record['judge_log'] = best['reasons']
            final_record['yolo_inventory'] = yolo_context
            
            f_out.write(json.dumps(final_record) + "\n")
            f_out.flush()

if __name__ == "__main__":
    main()
    
'''
# Example usage:

# Assuming you are in the project root

python -m src.judge \
  --files output/index_qwen_run.jsonl output/index_kimi_run.jsonl output/index_gemma_run.jsonl \
  --output output/consensus_final.jsonl \
  --n 3
'''