# src/reward.py

class SymbolicVerifier:
    def __init__(self):
        # Synonyms to help string matching
        self.synonyms = {
            "person": ["pedestrian", "child", "worker", "human", "person"],
            "construction": ["cone", "drum", "barrel", "barrier", "sign", "fence"],
            "vehicle": ["car", "truck", "bus", "van", "suv"],
            "debris": ["trash", "box", "rock", "object"]
        }

    def _check_keyword(self, text, category):
        """Returns True if any synonym of category is in text."""
        text = text.lower()
        for word in self.synonyms.get(category, [category]):
            if word in text:
                return True
        return False

    def calculate_score(self, json_output, yolo_text):
        score = 0.0
        reasons = []
        
        # If no YOLO text provided, we can't verify grounding.
        if not yolo_text:
            return 0.0, ["No YOLO context"]

        yolo_lower = yolo_text.lower()

        # --- 1. GROUNDING CONSISTENCY (The Anti-Hallucination Filter) ---
        # If the JSON claims a VRU Hazard, did YOLO see a person?
        vru_status = json_output.get("key_interacting_agents", {}).get("vru_status", "none")
        if vru_status not in ["none", "roadside_static"]:
            # Claiming active VRU
            if self._check_keyword(yolo_lower, "person"):
                score += 2.0
                reasons.append("✅ VRU Grounded")
            else:
                score -= 10.0
                reasons.append("❌ Hallucinated VRU (Not in YOLO)")

        # If JSON claims Construction, did YOLO see cones/barrels?
        tags = json_output.get("wod_e2e_tags", [])
        if "construction" in tags:
            if self._check_keyword(yolo_lower, "construction"):
                score += 2.0
                reasons.append("✅ Construction Grounded")
            else:
                score -= 5.0
                reasons.append("❌ Hallucinated Construction")

        # --- 2. CAUSAL LOGIC (The Planner Check) ---
        # If action is 'Stop', is there a blocker?
        crit = json_output.get("scenario_criticality", {})
        action = crit.get("ego_required_action", "lane_keep")
        blocker = crit.get("blocking_factor", "none")

        if action in ["stop", "emergency_brake", "nudge_around_static_obstacle"]:
            if blocker == "none":
                score -= 5.0
                reasons.append("❌ Action requires blocker")
            else:
                score += 3.0
                reasons.append("✅ Causal Link (Action -> Blocker)")

        # --- 3. SCHEMA COMPLETENESS ---
        # Penalize empty fields or "..." placeholders
        for k, v in json_output.items():
            if v == "...":
                score -= 10.0
                reasons.append(f"❌ Lazy Output ({k})")

        return score, reasons