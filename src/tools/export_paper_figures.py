import os
import sys
from PIL import Image, ImageDraw, ImageFont

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.loader import NuScenesLoader

# CONFIG
OUTPUT_DIR = "assets/figures"
SCENARIOS = {
    "crop_wheelchair": "8104e066e9964303ab23c07d88d209ed",
    "crop_dumpster": "dc73cecaa8ac47a6ab606c834829fee9"
}

def create_high_res_pano(loader, token, save_path):
    print(f"Processing {token}...")
    paths = loader.get_camera_paths(token)
    
    # 1. Load Images (Native Resolution)
    # Order: Left -> Center -> Right
    cam_order = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
    images = []
    
    for cam in cam_order:
        if cam in paths and os.path.exists(paths[cam]):
            img = Image.open(paths[cam])
            images.append(img)
        else:
            print(f"❌ Missing {cam} for {token}")
            return

    # 2. Stitch
    # NuScenes images are 1600x900. Total Pano: 4800x900
    w, h = images[0].size
    pano_w = w * 3
    pano_h = h
    
    # Create canvas
    pano = Image.new('RGB', (pano_w, pano_h))
    
    # Paste
    pano.paste(images[0], (0, 0))
    pano.paste(images[1], (w, 0))
    pano.paste(images[2], (w*2, 0))
    
    # 3. Add Subtle Borders/Separators (Optional, good for papers)
    draw = ImageDraw.Draw(pano)
    line_width = 4
    # Draw white lines between images to separate views visually
    draw.line([(w, 0), (w, h)], fill="white", width=line_width)
    draw.line([(w*2, 0), (w*2, h)], fill="white", width=line_width)

    # 4. Save
    # We save as PNG for lossless quality
    pano.save(save_path, quality=100)
    print(f"✅ Saved high-res figure to {save_path} ({pano_w}x{pano_h})")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("🚀 Loading NuScenes...")
    loader = NuScenesLoader(dataroot="nuscenes_data", version="v1.0-trainval")
    
    for name, token in SCENARIOS.items():
        filename = f"{name}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        create_high_res_pano(loader, token, path)

if __name__ == "__main__":
    main()