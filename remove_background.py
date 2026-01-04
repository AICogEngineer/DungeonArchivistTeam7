import os
import pathlib
from tqdm import tqdm
from PIL import Image
from rembg import remove, new_session

# Configuration
INPUT_DIR = "./dataset_a_v3"
OUTPUT_DIR = "./dataset_a_v3_cleaned"

# Initialize Session
session = new_session()

def clean_dataset():
    # 1. Setup Output Directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Collect Paths
    path_obj = pathlib.Path(INPUT_DIR)
    all_images = list(path_obj.rglob("*.png")) + list(path_obj.rglob("*.jpg"))
    
    print(f"Found {len(all_images)} total images.")
    print(f"Processing {len(all_images)} images to {OUTPUT_DIR}...\n")

    # 4. Process loop
    for img_path in tqdm(all_images):
        try:
            # Load
            img = Image.open(img_path).convert("RGB")
            
            # Remove BG (Default settings as requested)
            no_bg_raw = remove(img, session=session)
            
            # Create Black Canvas (Matches original image size)
            clean_img = Image.new("RGB", no_bg_raw.size, (0, 0, 0))
            clean_img.paste(no_bg_raw, mask=no_bg_raw.split()[3])
            
            # Define Output Path (preserving filename)
            save_path = os.path.join(OUTPUT_DIR, f"cleaned_{img_path.name}")
            clean_img.save(save_path)
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    print(f"\nSuccess! Cleaned images are in: {OUTPUT_DIR}")

if __name__ == "__main__":
    clean_dataset()