import json

LOG_FILE = "output/consensus_final.jsonl"

def main():
    print("🔍 Hunting for 'Neuro-Symbolic Corrections'...")
    
    with open(LOG_FILE, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                yolo = item.get('yolo_inventory', '').lower()
                tags = item.get('wod_e2e_tags', [])
                
                # CASE 1: YOLO sees Person, but VLM says NO VRU Hazard
                if "person" in yolo and "vru_hazard" not in tags:
                    # Filter out nominal scenes where person might be on sidewalk (safe)
                    # We want cases where YOLO was confident but VLM rejected the risk
                    print(f"\nFOUND CANDIDATE (FP Rejection):")
                    print(f"Token: {item['token']}")
                    print(f"YOLO: {yolo[:100]}...")
                    print(f"Judge Description: {item.get('description')}")
                    print("-" * 40)
                    # Stop after finding a few
                    if "poster" in item.get('description', '').lower() or "reflection" in item.get('description', '').lower():
                        print("🏆 JACKPOT! Found a specific hallucination rejection.")
                        break
            except: pass

if __name__ == "__main__":
    main()