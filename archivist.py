#  The main engine. 
#  Scans a target folder.
#  Queries Vector DB.
#  COPIES files to sorted subfolders (Preserving the sorting_Folder).

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model 
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from collections import Counter
import chromadb
import pathlib
from PIL import Image

# Configuration
# Point this to your CLEANED data folder for best accuracy
Sorting_Folder = "./dataset_c" 
Chroma_path = "./chroma_manual"
Restored_Folder = "./manual/restored_archive"
Review_Folder = "./manual/review_pile"
Top_Matches = 5
Confidence_Threshold = 50
Model_Path = "models/manual_model_v1.keras"
Embedding_Layer_Name = "embedding_out"



LABEL_TO_CATEGORY = {
    # Dungeon subcategories
    "dungeon": "dungeon",
    "dungeon_altars": "dungeon_altars",
    "dungeon_doors": "dungeon_doors",
    "dungeon_floor": "dungeon_floor",
    "dungeon_gateways": "dungeon_gateways",
    "dungeon_statues": "dungeon_statues",
    "dungeon_traps": "dungeon_traps",
    "dungeon_trees": "dungeon_trees",
    "dungeon_vaults": "dungeon_vaults",
    "dungeon_wall": "dungeon_wall",
    "dungeon_water": "dungeon_water",
    
    # Effect
    "effect": "effect",
    
    # Emissaries
    "emissaries": "emissaries",
    
    # GUI subcategories
    "gui": "gui",
    "gui_abilities": "gui_abilities",
    "gui_invocations": "gui_invocations",
    "gui_skills": "gui_skills",
    "gui_spells": "gui_spells",
    
    # Item subcategories
    "item_amulet": "item_amulet",
    "item_armor_artefact": "item_armor_artefact",
    "item_armor_back": "item_armor_back",
    "item_armor_bardings": "item_armor_bardings",
    "item_armor_feet": "item_armor_feet",
    "item_armor_hands": "item_armor_hands",
    "item_armor_headgear": "item_armor_headgear",
    "item_armor_shields": "item_armor_shields",
    "item_armor_torso": "item_armor_torso",
    "item_book": "item_book",
    "item_food": "item_food",
    "item_gold": "item_gold",
    "item_misc": "item_misc",
    "item_potion": "item_potion",
    "item_ring": "item_ring",
    "item_rod": "item_rod",
    "item_scroll": "item_scroll",
    "item_staff": "item_staff",
    "item_wand": "item_wand",
    "item_weapon": "item_weapon",
    
    # Misc subcategories
    "misc": "misc",
    "misc_blood": "misc_blood",
    "misc_brands": "misc_brands",
    "misc_numbers": "misc_numbers",
    
    # Monster subcategories
    "monster": "monster",
    "monster_panlord": "monster_panlord",
    "monster_tentacles": "monster_tentacles",
    
    # Player subcategories
    "player": "player",
    "player_barding": "player_barding",
    "player_base": "player_base",
    "player_beard": "player_beard",
    "player_body": "player_body",
    "player_boots": "player_boots",
    "player_cloak": "player_cloak",
    "player_draconic_head": "player_draconic_head",
    "player_draconic_wing": "player_draconic_wing",
    "player_felids": "player_felids",
    "player_gloves": "player_gloves",
    "player_hair": "player_hair",
    "player_hand": "player_hand",
    "player_head": "player_head",
    "player_legs": "player_legs",
    "player_transform": "player_transform",
}

def connect_to_vector_db():
    """Connect to ChromaDB with L2 distance metric"""
    try:
        client = chromadb.PersistentClient(path=Chroma_path)
        collection = client.get_collection(name="dungeondb")
        print(f"Connected to Vector DB (total vectors: {collection.count()})")
        return collection
    except Exception as e:
        print(f"Error connecting to Vector DB: {e}")
        raise

def load_archivist_model(model_path, input_shape=(32, 32, 3)):
    """Load the trained model and extract embedding layer"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load without compiling to avoid the ShuffleDataset error
        model = load_model(model_path, compile=False)
        
        input_tensor = Input(shape=input_shape)
        x = input_tensor
        embedding_found = False
        for layer in model.layers:
            x = layer(x)
            if layer.name == Embedding_Layer_Name:
                embedding_found = True
                break
        
        if not embedding_found:
            raise ValueError(f"Embedding layer '{Embedding_Layer_Name}' not found")
        
        embedding_model = Model(inputs=input_tensor, outputs=x)
        print(f"Model loaded successfully")
        return model, embedding_model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def preprocess_images_batch(image_paths, target_size=(32, 32)):
    """Load and preprocess a batch of images"""
    images = []
    valid_paths = []
    
    for path in image_paths:
        try:
            img = load_img(path, target_size=target_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            valid_paths.append(path)
        except Exception as e:
            print(f"✗ Error loading {os.path.basename(path)}: {e}")
            os.makedirs(Review_Folder, exist_ok=True)
            # CHANGED: Use copy2 instead of move for corrupted files
            shutil.copy2(path, os.path.join(Review_Folder, "corrupted_" + os.path.basename(path)))
    
    return np.array(images), valid_paths

def generate_embeddings_batch(embedding_model, images_batch):
    """Generate embeddings for a batch of images"""
    embeddings = embedding_model.predict(images_batch, verbose=0)
    return embeddings.reshape(len(embeddings), -1)

def nearest_neighbors_batch(collection, embeddings, top_matches=Top_Matches):
    """Query vector DB for nearest neighbors in batch"""
    results = collection.query(
        query_embeddings=embeddings.tolist(),
        n_results=top_matches
    )
    all_labels = []
    all_distances = []
    for i in range(len(embeddings)):
        labels = [m['label'] for m in results['metadatas'][i]]
        distances = results['distances'][i]
        all_labels.append(labels)
        all_distances.append(distances)
    return all_labels, all_distances

def decision(labels, distances, confidence_threshold=Confidence_Threshold):
    if not labels or not distances:
        return None
    top_distance = distances[0]
    if top_distance > confidence_threshold:
        return None 
    weighted_votes = {}
    for label, dist in zip(labels, distances):
        weight = 1.0 / (dist + 1e-6)
        weighted_votes[label] = weighted_votes.get(label, 0) + weight
    if not weighted_votes:
        return None
    winning_label = max(weighted_votes, key=weighted_votes.get)
    return winning_label # Return the label for move_file to handle hierarchy

# --- UPDATED: COPY VERSION OF MOVE_FILE ---
def move_file(file_path, label):
    """Copy file to appropriate nested destination folder"""
    try:
        if label is not None:
            parts = label.split('_', 1)
            if len(parts) == 2:
                category = parts[0]
                subcategory = parts[1]
                destination_folder = os.path.join(Restored_Folder, category, subcategory)
            else:
                destination_folder = os.path.join(Restored_Folder, label)
            
            os.makedirs(destination_folder, exist_ok=True)
            # CHANGED: Use copy2 instead of move
            shutil.copy2(file_path, os.path.join(destination_folder, os.path.basename(file_path)))
        else:
            # Send copy to review pile
            os.makedirs(Review_Folder, exist_ok=True)
            # CHANGED: Use copy2 instead of move
            shutil.copy2(file_path, os.path.join(Review_Folder, os.path.basename(file_path)))
    except Exception as e:
        print(f" Error copying {os.path.basename(file_path)}: {e}")

def process_sorting_folder_batch(sorting_folder, embedding_model, collection, batch_size=32):
    if not os.path.exists(sorting_folder):
        raise FileNotFoundError(f"Sorting folder not found: {sorting_folder}")
    
    filenames = [f for f in os.listdir(sorting_folder) if f.lower().endswith(".png")]
    if not filenames:
        print("No PNG files found in sorting folder")
        return

    print(f"Found {len(filenames)} images to process")
    total_batches = (len(filenames) + batch_size - 1) // batch_size
    sorted_count = 0
    review_count = 0

    for i in range(0, len(filenames), batch_size):
        batch_files = filenames[i:i + batch_size]
        batch_paths = [os.path.join(sorting_folder, f) for f in batch_files]
        print(f"Processing batch {i//batch_size + 1}/{total_batches}...", end=" ")

        images_batch, valid_paths = preprocess_images_batch(batch_paths)
        if len(images_batch) == 0:
            continue

        embeddings = generate_embeddings_batch(embedding_model, images_batch)
        all_labels, all_distances = nearest_neighbors_batch(collection, embeddings)

        batch_sorted = 0
        batch_review = 0
        for path, labels, distances in zip(valid_paths, all_labels, all_distances):
            final_label = decision(labels, distances)
            move_file(path, final_label)
            if final_label is not None:
                batch_sorted += 1
            else:
                batch_review += 1
        
        sorted_count += batch_sorted
        review_count += batch_review
        print(f" {batch_sorted} copied, {batch_review} to review")

    print(f"\nProcessing Complete!")
    print(f"Total entries in archive: {sorted_count}")
    print(f"Total in review pile: {review_count}")

def main():
    print("="*50)
    print("Dungeon Archivist - The Restoration (Safe Copy Mode)")
    print("="*50)
    try:
        collection = connect_to_vector_db()
        model, embedding_model = load_archivist_model(Model_Path)
        process_sorting_folder_batch(Sorting_Folder, embedding_model, collection)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()