import os
import shutil
import pathlib
from tqdm import tqdm

# --- Configuration ---
ORIGINAL_DATASET_A = "./dataset_a_v3"          # The source of the folder structure
FLAT_CLEANED_FOLDER = "./dataset_a_v3_cleaned"    # Where your "cleaned_*.png" files are
NEW_STRUCTURED_DATASET = "./dataset_a_v3_structured"
PREFIX = "cleaned_"

def rebuild_hierarchy():
    # 1. Setup the new root directory
    if not os.path.exists(NEW_STRUCTURED_DATASET):
        os.makedirs(NEW_STRUCTURED_DATASET)
    
    # 2. Map out the original structure
    original_path = pathlib.Path(ORIGINAL_DATASET_A)
    # Get all png files in the original folder (including subfolders)
    original_files = list(original_path.rglob("*.png"))
    
    print(f"Found {len(original_files)} original files to map.")
    
    copied_count = 0
    missing_count = 0

    for orig_file_path in tqdm(original_files, desc="Rebuilding Structure"):
        # Get the relative path (e.g., 'monster/goblin/image1.png')
        relative_path = orig_file_path.relative_to(original_path)
        
        # Determine what the cleaned filename should be
        cleaned_filename = f"{PREFIX}{orig_file_path.name}"
        cleaned_source_path = os.path.join(FLAT_CLEANED_FOLDER, cleaned_filename)
        
        if os.path.exists(cleaned_source_path):
            # Define the destination path in the new structured folder
            # This keeps the exact same subfolder path as the original
            dest_path = os.path.join(NEW_STRUCTURED_DATASET, relative_path)
            
            # Create the necessary subdirectories
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Copy the cleaned file to the new structured location
            shutil.copy2(cleaned_source_path, dest_path)
            copied_count += 1
        else:
            missing_count += 1

    print(f"\n--- Process Complete ---")
    print(f"Successfully structured: {copied_count} files")
    if missing_count > 0:
        print(f"Warning: {missing_count} files were not found in the cleaned folder.")
    print(f"New dataset ready at: {NEW_STRUCTURED_DATASET}")

if __name__ == "__main__":
    rebuild_hierarchy()