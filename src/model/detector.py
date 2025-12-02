# src/model/detector.py
from ultralytics import YOLOE
import torch

class ObjectDetector:
    def __init__(self, model_size='yoloe-11l-seg.pt', conf_threshold=0.40):
        print(f"🚀 Loading YOLOE-11 Open-Vocabulary Segmentor ({model_size})...")
        self.model = YOLOE(model_size)
        self.conf = conf_threshold
        
        # DEFINING THE LONG-TAIL TAXONOMY (WOD-E2E Optimized)
        # We include synonyms to boost recall for specific edge cases.
        self.custom_classes = [
            # 1. VRUs (Vulnerable Road Users)
            "person", "pedestrian", "child", 
            "cyclist", "bicyclist", "motorcyclist", "scooter rider",
            "construction worker", "worker in safety vest", "police officer",
            
            # 2. Vehicles (Specialized)
            "car", "pickup truck", "suv", "van", "sedan", "coupe",
            "truck", "semi truck", "trailer", "cement mixer",
            "bus", "school bus",
            "police car", "police vehicle", "ambulance", "fire truck",
            "construction vehicle", "bulldozer", "excavator", "forklift",
            "road sweeper", "street cleaner",
            
            # 3. Construction & Barriers
            "traffic cone", "orange cone",  "traffic drum",
            "construction barrel", "orange drum", # Crucial for Highway Construction
            "traffic barrier", "concrete barrier", "jersey barrier",
            "road work sign", "temporary sign",
            "construction fence", "safety fence",
            "scaffolding", "construction scaffolding",
            
            # 4. Hazards / Debris (FOD)
            "debris", "cardboard box", "tire", 
            "plastic bag", "tree branch", "large rock",
            "puddle", 
            
            # 5. Traffic Control
            "traffic light", "traffic signal", "red light", 
            "stop sign", "yield sign", "speed limit sign",
            "pedestrian crossing sign", "school zone sign",
            "crosswalk",
        ]
        
        # Compile prompts
        self.model.set_classes(self.custom_classes, self.model.get_text_pe(self.custom_classes))
        
    def detect_batch(self, images_dict):
        summary_lines = []
        cam_names = list(images_dict.keys())
        batch_images = [images_dict[k] for k in cam_names]
        
        # Run Inference
        results = self.model.predict(batch_images, verbose=False, conf=self.conf)
        
        for i, result in enumerate(results):
            cam_name = cam_names[i]
            detections = {}
            
            for box in result.boxes:
                cls_id = int(box.cls)
                class_name = self.custom_classes[cls_id]
                conf = float(box.conf)
                
                # Relative Size Calculation
                bbox = box.xyxy[0].cpu().numpy()
                area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                img_area = result.orig_shape[0] * result.orig_shape[1]
                rel_size = area / img_area
                
                if class_name not in detections: detections[class_name] = []
                detections[class_name].append({"conf": conf, "size": rel_size})
            
            # Format Output
            if detections:
                parts = []
                for name, items in detections.items():
                    items.sort(key=lambda x: x['size'], reverse=True)
                    desc_list = []
                    for item in items[:3]: # Top 3 only
                        size_str = "Large" if item['size'] > 0.1 else "Med" if item['size'] > 0.01 else "Small"
                        desc_list.append(f"{size_str}/{item['conf']:.2f}")
                    parts.append(f"{len(items)} {name}{'s' if len(items)>1 else ''} ({', '.join(desc_list)})")
                line = f"[{cam_name}]: " + "; ".join(parts)
            else:
                line = f"[{cam_name}]: Clear"
            summary_lines.append(line)
            
        return "\n".join(summary_lines)