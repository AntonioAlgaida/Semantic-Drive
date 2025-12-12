# src/model/vlm_client.py
import base64
import json
import os
import re
import copy
from io import BytesIO
from openai import OpenAI
from src.config import CAM_ORDER
import time

# Configuration for Local Inference
# Ensure LM Studio is running specifically on this port
# LM_STUDIO_URL = "http://192.168.1.67:1234/v1"
API_KEY = "lm-studio" # Placeholder, not used locally usually

class VLMClient:
    def __init__(self, model_id="qwen3-vl-30b", port=1234):
        # Allow dynamic port assignment
        base_url = f"http://localhost:{port}/v1"
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self.model_id = model_id
        print(f"✅ VLM Client connected to Port {port}")

    def _encode_image(self, pil_image):
        """
        Converts a PIL Image to a base64 string for the API.
        """
        buffered = BytesIO()
        # Convert to RGB to ensure no alpha channel issues
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
            
        # Resize if massive to save tokens (optional but recommended for 6 images)
        # Keeping max dimension around 1000px is usually a good balance
        if max(pil_image.size) > 1600:
            pil_image.thumbnail((1600, 1600))
            
        # DEBUG: Print exact size being sent
        # print(f"🔍 DEBUG: Encoding Image Size: {pil_image.size} (WxH)") 
        
        # ts = int(time.time() * 1000)
        # filename = f"debug_{ts}.jpg"
        # save_path = os.path.join("debug_images", filename)
        # pil_image.save(save_path, quality=95)
            
        pil_image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8'), pil_image.size

    def _sanitize_for_logging(self, messages):
        """
        Creates a copy of the messages but replaces massive Base64 strings 
        with <IMAGE_DATA> placeholders so logs are readable.
        """
        clean_msgs = copy.deepcopy(messages)
        for msg in clean_msgs:
            if isinstance(msg.get('content'), list):
                for item in msg['content']:
                    if item.get('type') == 'image_url':
                        item['image_url']['url'] = "<BASE64_IMAGE_DATA_REMOVED>"
        return clean_msgs

    def _extract_json(self, raw_text):
        """Aggressively hunts for a JSON block using Regex."""
        # 1. Try Markdown block
        code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(code_block_pattern, raw_text, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # 2. Fallback: Find outermost brackets
        try:
            start_idx = raw_text.find("{")
            end_idx = raw_text.rfind("}") + 1
            if start_idx != -1 and end_idx != -1:
                return raw_text[start_idx:end_idx]
        except:
            pass
        return None

    def _print_thought_process(self, raw_text):
        """
        Extracts thinking process even if the closing tag is missing.
        """
        start_tag = "◁think▷"
        
        if start_tag not in raw_text:
            return

        start_idx = raw_text.find(start_tag) + len(start_tag)
        
        # Try to find the closing tag
        end_tag = "◁/think▷"
        end_idx = raw_text.find(end_tag)
        
        # If closing tag MISSING, assume thought ends where the JSON block starts
        if end_idx == -1:
            # Look for the start of the code block
            code_start = raw_text.find("```")
            if code_start != -1:
                end_idx = code_start
            else:
                # Worst case: print everything (the JSON is mixed in, but helpful for debug)
                end_idx = len(raw_text)

        thought_content = raw_text[start_idx:end_idx].strip()
        
        if thought_content:
            print("\n" + "="*60)
            print("🧠 MODEL REASONING TRACE:")
            print("-" * 60)
            # Print first 500 chars to avoid spamming console if it's huge
            print(thought_content[:] + ("..." if len(thought_content) > 1000 else ""))
            print("-" * 60)
            print("="*60 + "\n")

    def _extract_reasoning(self, raw_text):
        """
        Extracts the text inside thinking tags.
        Fallback: Returns the entire raw text if no tags are found.
        """
        
        print("🔍 Extracting reasoning trace from response...")
        print(f" Raw text: {raw_text}")
        # Supports both DeepSeek/Kimi style tags
        start_patterns = ["◁think▷", "<think>"]
        end_patterns = ["◁/think▷", "</think>"]
        
        start_idx = -1
        
        for tag in start_patterns:
            if tag in raw_text:
                start_idx = raw_text.find(tag) + len(tag)
                break
        
        # --- MODIFICATION START ---
        # If no start tag is found, return the whole text (Fallback)
        if start_idx == -1:
            return raw_text.strip()
        # --- MODIFICATION END ---

        # Find closing tag
        end_idx = -1
        for tag in end_patterns:
            if tag in raw_text:
                end_idx = raw_text.find(tag)
                break
        
        # If closing tag missing, grab everything until JSON starts or end of string
        if end_idx == -1:
            code_start = raw_text.find("```")
            if code_start != -1:
                end_idx = code_start
            else:
                end_idx = len(raw_text)

        return raw_text[start_idx:end_idx].strip()
    
    def analyze_multiview(self, camera_images, system_prompt, object_inventory=None, verbose=False):
        # 1. Construct Prompt
        intro = "Here are the synchronized Front-View cameras."
        if object_inventory:
            intro = f"### DETECTED OBJECTS (YOLO) ###\n{object_inventory}\n\n{intro}"

        user_content = [{"type": "text", "text": intro}]
        img_sizes = [] # Track sizes for debug log

        # 2. Add Images (Using Config Order)
        for cam_name in CAM_ORDER:
            if cam_name in camera_images:
                base64_img, size = self._encode_image(camera_images[cam_name])
                img_sizes.append(f"{cam_name}: {size}")
                user_content.append({"type": "text", "text": f"\n### VIEW: {cam_name} ###\n"})
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})

        user_content.append({"type": "text", "text": "\nThink deeply inside <think> tags, then output valid JSON."})
        
        # DEBUG PRINT
        if verbose:
            print(f"🔍 INPUT AUDIT: Sending {len(img_sizes)} images. Dimensions: {img_sizes}")

        messages = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_content}
        ]
                
        # 2. Prepare Result Object
        result_pkg = {
            "success": False,
            "parsed_json": None,
            "raw_response": None,
            "reasoning_trace": None,
            "input_messages_log": self._sanitize_for_logging(messages), # Clean for saving
            "usage": None,  # <--- NEW FIELD
            "error": None
        }
        
        # 3. Call API
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.1,
                max_tokens=16384
            )
            raw = response.choices[0].message.content
            result_pkg["raw_response"] = raw

            # Extract Components
            reasoning =  response.choices[0].message.model_extra.get('reasoning_content') or None
            json_str = self._extract_json(raw)
            
            result_pkg["reasoning_trace"] = reasoning
            
            # --- CAPTURE TOKEN USAGE ---
            if response.usage:
                result_pkg["usage"] = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            # ---------------------------

            if json_str:
                data = json.loads(json_str)
                # Inject trace into JSON for the final index too
                data['_reasoning_trace'] = reasoning or "None"
                result_pkg["parsed_json"] = data
                result_pkg["success"] = True
            else:
                result_pkg["error"] = "No Valid JSON found in response"

        except Exception as e:
            result_pkg["error"] = str(e)
            
        return result_pkg