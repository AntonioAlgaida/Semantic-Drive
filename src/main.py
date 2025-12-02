# src/main.py
import json
import time
import os
import sys
import argparse
import traceback
from tqdm import tqdm
from PIL import Image

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))

from src.data.loader import NuScenesLoader
from src.model.vlm_client import VLMClient
from src.model.detector import ObjectDetector
from src.model.prompts import SYSTEM_PROMPT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model ID in LM Studio")
    parser.add_argument("--output_name", type=str, required=True, help="Suffix for output file")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Paths
    OUTPUT_DIR = "output"
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # FILE 1: The Clean Index (For Search)
    INDEX_FILE = os.path.join(OUTPUT_DIR, f"index_{args.output_name}.jsonl")
    # FILE 2: The Full Log (For Debugging/Paper Appendix)
    LOG_FILE = os.path.join(OUTPUT_DIR, f"logs_{args.output_name}.jsonl")

    # Initialize Components
    print("1. Loading NuScenes...")
    loader = NuScenesLoader()
    
    print("2. Loading YOLOE Detector...")
    detector = ObjectDetector() 

    print(f"3. Connecting to VLM ({args.model})...")
    client = VLMClient(model_id=args.model)

    # Resume Logic
    processed_tokens = set()
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'r') as f:
            for line in f:
                try: processed_tokens.add(json.loads(line)['token'])
                except: pass
    
    print(f"🚀 Starting Mining. Processed so far: {len(processed_tokens)}")
    samples = loader.get_all_samples()
    target_samples = samples[:args.limit] if args.limit else samples

    # Open both files
    with open(INDEX_FILE, 'a') as f_index, open(LOG_FILE, 'a') as f_log:
        
        for token in tqdm(target_samples):
            if token in processed_tokens: continue

            # 1. Load Images
            paths = loader.get_camera_paths(token)
            images = {}
            for cam, path in paths.items():
                try: 
                    img = Image.open(path)
                    img.thumbnail((1280, 1280))  # The image will be resized if too large
                    images[cam] = img
                except: pass
            
            if len(images) < 3: continue 

            # 2. Run YOLOE
            try:
                inventory = detector.detect_batch(images)
            except Exception as e:
                print(f"Detector Failed: {e}")
                inventory = "Detector Error"

            # 3. Run VLM Reasoning (Retry Logic)
            result = None
            for attempt in range(3):
                try:
                    result = client.analyze_multiview(
                        images, 
                        SYSTEM_PROMPT, 
                        object_inventory=inventory
                    )
                    
                    if result["success"]:
                        break # Exit retry loop on success
                    
                    # If JSON parse failed but we got text, wait and retry
                    time.sleep(1)
                except Exception as e:
                    print(f"API Error: {e}")
                    time.sleep(2)

            # 4. Prepare Data for Saving
            timestamp = time.time()
            
            # --- BUILD LOG ENTRY (Saves EVERYTHING) ---
            log_entry = {
                "token": token,
                "timestamp": timestamp,
                "model": args.model,
                "yolo_inventory": inventory,
                
                # Token Metrics
                "usage": result.get("usage") if result else None,
                
                "prompt_messages": result["input_messages_log"] if result else "API Call Failed",
                "raw_response": result["raw_response"] if result else None,
                "reasoning_trace": result["reasoning_trace"] if result else None,
                "error": result["error"] if result else "Loop Failed",
                "success": result["success"] if result else False
            }
            
            # Write Log
            f_log.write(json.dumps(log_entry) + "\n")
            f_log.flush()

            # --- BUILD INDEX ENTRY (Only if success) ---
            if result and result["success"]:
                clean_data = result["parsed_json"]
                clean_data['token'] = token
                clean_data['model_source'] = args.output_name
                
                # Write Index
                f_index.write(json.dumps(clean_data) + "\n")
                f_index.flush()

                # Print to console if verbose
                if args.verbose and result.get("reasoning_trace"):
                    print(f"\n[Token: {token}] 🧠 Reasoning:\n{result['reasoning_trace'][:300]}...\n")
            
            elif result and not result["success"]:
                # Print failure to console
                print(f"\n❌ Failed Token {token}: {result['error']}")

if __name__ == "__main__":
    main()