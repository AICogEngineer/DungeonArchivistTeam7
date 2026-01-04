import os
import shutil
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
ORIGINAL_DATASET = "./dataset_a_v3" 
SORTED_CHAOS_DATA = "./restored_archive"           
NEW_COMBINED_DATASET = "./combined_dataset_archivist" 

def consolidate_datasets():
    # Convert to Path objects
    base_dest = Path(NEW_COMBINED_DATASET)
    folders_to_merge = [
        Path(ORIGINAL_DATASET),
        Path(SORTED_CHAOS_DATA)
    ]

    # 1. Prepare the new folder
    if base_dest.exists():
        print(f"Note: {NEW_COMBINED_DATASET} already exists. Adding files to it.")
    else:
        print(f"Creating new dataset folder: {NEW_COMBINED_DATASET}")
        base_dest.mkdir(parents=True, exist_ok=True)

    # 2. Iterate through both source folders
    for source_root in folders_to_merge:
        if not source_root.exists():
            print(f"Warning: Source folder {source_root} not found. Skipping.")
            continue

        print(f"\nProcessing files from: {source_root}")
        
        # Gather all png files
        files = list(source_root.rglob("*.png"))
        
        for file_path in tqdm(files, desc=f"Copying to {NEW_COMBINED_DATASET}"):
            # Calculate the relative path from the current source root
            # e.g., 'monster/undead/zombie.png'
            relative_path = file_path.relative_to(source_root)
            
            # Define target path in the brand new folder
            target_path = base_dest / relative_path
            
            try:
                # Ensure the subfolder hierarchy is built in the new folder
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the file (shutil.copy2 preserves original metadata)
                shutil.copy2(str(file_path), str(target_path))
            except Exception as e:
                print(f"\nError copying {relative_path}: {e}")

    print(f"\n" + "="*50)
    print(f"Consolidation Complete!")
    print(f"New combined dataset: {NEW_COMBINED_DATASET}")
    print("Original folders remain unchanged.")
    print("="*50)

if __name__ == "__main__":
    consolidate_datasets()