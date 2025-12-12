import os, sys
import shutil
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.loader import NuScenesLoader

# CONFIG
OUTPUT_DIR = "hf_demo_pack"
JSONL_FILE = "output/consensus_final.jsonl"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")

def main():
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
        
    loader = NuScenesLoader()
    
    # 1. Filter Interesting Frames
    # We don't want to upload 2,500 images. Let's pick top 200.
    interesting_tokens = []
    
    with open(JSONL_FILE, 'r') as f:
        for line in f:
            item = json.loads(line)
            risk = int(item.get('scenario_criticality', {}).get('risk_score', 0))
            tags = item.get('wod_e2e_tags', [])
            
            # Criteria: High Risk OR Rare Tags
            if risk >= 6 or len(tags) > 0:
                interesting_tokens.append(item)

    print(f"Packaging {len(interesting_tokens)} scenarios for the Web Demo...")
    
    # 2. Copy Images & Rewrite JSON
    demo_data = []
    
    for item in interesting_tokens[:]:  # Limit to top 200
        token = item['token']
        paths = loader.get_camera_paths(token)
        
        # We only keep Front-Center for the web demo to save space
        # (Or keep all 3 if you want the panorama)
        cam = "CAM_FRONT"
        if cam in paths:
            src_path = paths[cam]
            filename = f"{token}_{cam}.jpg"
            dst_path = os.path.join(IMAGES_DIR, filename)
            
            shutil.copy(src_path, dst_path)
            
            # Add image reference to the JSON object for the web app
            item["web_image_path"] = f"images/{filename}"
            demo_data.append(item)

    # 3. Save the "Web-Ready" JSONL
    with open(os.path.join(OUTPUT_DIR, "demo_data.jsonl"), 'w') as f:
        for item in demo_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Done! Upload the '{OUTPUT_DIR}' folder to your Hugging Face Space.")

if __name__ == "__main__":
    main()