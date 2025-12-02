# src/data/visuals.py
from PIL import Image, ImageDraw, ImageFont
from src.config import CAM_ORDER, RESIZE_FACTOR

def create_surround_montage(camera_paths, resize_factor=RESIZE_FACTOR):
    """
    Stitches 6 camera images into a 3x2 grid.
    Layout:
    [Front Left] [Front] [Front Right]
    [Back Left]  [Back]  [Back Right]
    """
    images = {}
    
    # 1. Load and Resize
    for cam, path in camera_paths.items():
        try:
            img = Image.open(path)
            if resize_factor != 1.0:
                new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            images[cam] = img
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    # 2. Calculate Montage Dimensions
    # Assuming all images are same size (they are in NuScenes)
    w, h = list(images.values())[0].size
    montage_w = w * 3
    montage_h = h * 2
    
    montage = Image.new('RGB', (montage_w, montage_h))
    draw = ImageDraw.Draw(montage)
    
    # 3. Stitch and Label
    # Grid positions (col, row)
    grid_map = {
        "CAM_FRONT_LEFT":  (0, 0), "CAM_FRONT": (1, 0), "CAM_FRONT_RIGHT": (2, 0),
        "CAM_BACK_LEFT":   (0, 1), "CAM_BACK":  (1, 1), "CAM_BACK_RIGHT":  (2, 1)
    }

    for cam_name, img in images.items():
        col, row = grid_map[cam_name]
        x = col * w
        y = row * h
        montage.paste(img, (x, y))
        
        # 4. Add Label
        # Clean name: CAM_FRONT_LEFT -> FRONT LEFT
        label_text = cam_name.replace("CAM_", "").replace("_", " ")
        
        # Draw a small black box behind text for contrast
        text_x, text_y = x + 10, y + 10
        # Approximate text box size since we can't easily measure without loading a font file
        draw.rectangle([text_x, text_y, text_x + 120, text_y + 30], fill="black")
        draw.text((text_x + 5, text_y + 5), label_text, fill="white")

    return montage