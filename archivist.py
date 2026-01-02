#  The main engine. 
#  Scans a target folder.
#  Queries Vector DB.
#  Moves files to sorted subfolders.

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

# Configuration
Chaos_Folder = "C:\\Users\\jchac\\JaredChancey\\Group-7-AIENG\\DungeonArchivistTeam7\\Dataset_B\\chaos_data"
Restored_Folder = "C:\\Users\\jchac\\JaredChancey\\Group-7-AIENG\\DungeonArchivistTeam7\\restored_archive"
Review_Folder = "C:\\Users\\jchac\\JaredChancey\\Group-7-AIENG\\DungeonArchivistTeam7\\review_pile"
Top_Matches = 5
Confidence_Threshold = 2000
Model_Path = "models/dungeon_model_v1.keras"
Embedding_Layer_Name = "embedding_out"

# Label → Category mapping
LABEL_TO_CATEGORY = {
    # Dungeon category (17 labels)
    "dungeon": "dungeon",
    "dungeon_altars": "dungeon",
    "dungeon_doors": "dungeon",
    "dungeon_floor": "dungeon",
    "dungeon_floor_grass": "dungeon",
    "dungeon_floor_sigils": "dungeon",
    "dungeon_gateways": "dungeon",
    "dungeon_shops": "dungeon",
    "dungeon_statues": "dungeon",
    "dungeon_traps": "dungeon",
    "dungeon_trees": "dungeon",
    "dungeon_vaults": "dungeon",
    "dungeon_wall": "dungeon",
    "dungeon_wall_abyss": "dungeon",
    "dungeon_wall_banners": "dungeon",
    "dungeon_wall_torches": "dungeon",
    "dungeon_water": "dungeon",
    
    # Effect category (1 label)
    "effect": "effect",
    
    # Emissaries category (1 label)
    "emissaries": "emissaries",
    
    # GUI category (20 labels)
    "gui": "gui",
    "gui_abilities": "gui",
    "gui_commands": "gui",
    "gui_invocations": "gui",
    "gui_skills": "gui",
    "gui_spells": "gui",
    "gui_spells_air": "gui",
    "gui_spells_components": "gui",
    "gui_spells_conjuration": "gui",
    "gui_spells_disciplines": "gui",
    "gui_spells_divination": "gui",
    "gui_spells_earth": "gui",
    "gui_spells_enchantment": "gui",
    "gui_spells_fire": "gui",
    "gui_spells_ice": "gui",
    "gui_spells_monster": "gui",
    "gui_spells_necromancy": "gui",
    "gui_spells_poison": "gui",
    "gui_spells_summoning": "gui",
    "gui_spells_translocation": "gui",
    "gui_spells_transmutation": "gui",
    "gui_startup": "gui",
    "gui_tabs": "gui",
    
    # Item category (28 labels)
    "item_amulet": "item",
    "item_amulet_artefact": "item",
    "item_armor_artefact": "item",
    "item_armor_back": "item",
    "item_armor_bardings": "item",
    "item_armor_feet": "item",
    "item_armor_hands": "item",
    "item_armor_headgear": "item",
    "item_armor_shields": "item",
    "item_armor_torso": "item",
    "item_book": "item",
    "item_book_artefact": "item",
    "item_food": "item",
    "item_gold": "item",
    "item_misc": "item",
    "item_misc_runes": "item",
    "item_potion": "item",
    "item_ring": "item",
    "item_ring_artefact": "item",
    "item_rod": "item",
    "item_scroll": "item",
    "item_staff": "item",
    "item_wand": "item",
    "item_weapon": "item",
    "item_weapon_artefact": "item",
    "item_weapon_ranged": "item",
    
    # Misc category (8 labels)
    "misc": "misc",
    "misc_blood": "misc",
    "misc_brands_bottom_left": "misc",
    "misc_brands_bottom_right": "misc",
    "misc_brands_top_left": "misc",
    "misc_brands_top_right": "misc",
    "misc_numbers": "misc",
    
    # Monster category (31 labels)
    "monster": "monster",
    "monster_aberration": "monster",
    "monster_abyss": "monster",
    "monster_amorphous": "monster",
    "monster_animals": "monster",
    "monster_aquatic": "monster",
    "monster_demons": "monster",
    "monster_demonspawn": "monster",
    "monster_draconic": "monster",
    "monster_dragons": "monster",
    "monster_eyes": "monster",
    "monster_fungi_plants": "monster",
    "monster_holy": "monster",
    "monster_nonliving": "monster",
    "monster_panlord": "monster",
    "monster_spriggan": "monster",
    "monster_statues": "monster",
    "monster_tentacles_eldritch_corners": "monster",
    "monster_tentacles_eldritch_ends": "monster",
    "monster_tentacles_kraken_corners": "monster",
    "monster_tentacles_kraken_ends": "monster",
    "monster_tentacles_kraken_segments": "monster",
    "monster_tentacles_starspawn_corners": "monster",
    "monster_tentacles_starspawn_ends": "monster",
    "monster_tentacles_starspawn_segments": "monster",
    "monster_tentacles_vine_corners": "monster",
    "monster_tentacles_vine_ends": "monster",
    "monster_tentacles_vine_segments": "monster",
    "monster_undead": "monster",
    "monster_undead_simulacra": "monster",
    "monster_undead_skeletons": "monster",
    "monster_undead_spectrals": "monster",
    "monster_undead_zombies": "monster",
    "monster_unique": "monster",
    "monster_vault": "monster",
    
    # Player category (26 labels)
    "player_barding": "player",
    "player_base": "player",
    "player_beard": "player",
    "player_body": "player",
    "player_boots": "player",
    "player_cloak": "player",
    "player_draconic_head": "player",
    "player_draconic_wing": "player",
    "player_enchantment": "player",
    "player_felids": "player",
    "player_gloves": "player",
    "player_hair": "player",
    "player_halo": "player",
    "player_hand_left": "player",
    "player_hand_left_misc": "player",
    "player_hand_right": "player",
    "player_hand_right_artefact": "player",
    "player_hand_right_misc": "player",
    "player_head": "player",
    "player_legs": "player",
    "player_mutations": "player",
    "player_transform": "player",
}

# Connecting To Vector Database
def connect_to_vector_db():
    """Connect to ChromaDB with L2 distance metric"""
    try:
        # Use PersistentClient with absolute path
        client = chromadb.PersistentClient(
            path="C:\\Users\\jchac\\JaredChancey\\Group-7-AIENG\\DungeonArchivistTeam7\\chroma"
        )

        # Get the collection (don't use get_or_create, just get)
        collection = client.get_collection(name="dungeondb")

        print(f"Connected to Vector DB (total vectors: {collection.count()})")
        return collection
    
    except Exception as e:
        print(f"Error connecting to Vector DB: {e}")
        raise

# Load Model
def load_archivist_model(model_path, input_shape=(32, 32, 3)):
    """Load the trained model and extract embedding layer"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load the Sequential model
        model = load_model(model_path)
        
        # Create a functional Input
        input_tensor = Input(shape=input_shape)
        
        # Pass the input through each layer up to the embedding layer
        x = input_tensor
        embedding_found = False
        for layer in model.layers:
            x = layer(x)
            if layer.name == Embedding_Layer_Name:
                embedding_found = True
                break
        
        if not embedding_found:
            raise ValueError(f"Embedding layer '{Embedding_Layer_Name}' not found in model")
        
        embedding_model = Model(inputs=input_tensor, outputs=x)
        print(f"Model loaded successfully")
        return model, embedding_model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


# Preprocess Image
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
            # Move corrupted files to review pile
            os.makedirs(Review_Folder, exist_ok=True)
            shutil.move(path, os.path.join(Review_Folder, os.path.basename(path)))
    
    return np.array(images), valid_paths


# Generate Embedding 
def generate_embeddings_batch(embedding_model, images_batch):
    """Generate embeddings for a batch of images"""
    embeddings = embedding_model.predict(images_batch, verbose=0)
    return embeddings.reshape(len(embeddings), -1)


# Query Vector DB
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


# Decision Label
def decision(labels, distances, confidence_threshold=Confidence_Threshold):
    """
    Phase 2 Decision Logic:
    1. Confidence check on nearest neighbor
    2. Distance-weighted voting on LABEL level
    3. Map winning label to CATEGORY
    """
    if not labels or not distances:
        return None

    # Confidence check (nearest neighbor distance)
    top_distance = distances[0]
    if top_distance > confidence_threshold:
        return None  # send to review pile

    # Weighted voting on LABEL level (as per requirements)
    weighted_votes = {}

    for label, dist in zip(labels, distances):
        # Weight: closer neighbors vote stronger
        weight = 1.0 / (dist + 1e-6)
        weighted_votes[label] = weighted_votes.get(label, 0) + weight

    if not weighted_votes:
        return None

    # Get winning label
    winning_label = max(weighted_votes, key=weighted_votes.get)
    
    # Map label to category
    category = LABEL_TO_CATEGORY.get(winning_label)
    
    return category


# Move File
def move_file(file_path, label):
    """Move file to appropriate destination folder"""
    try:
        if label is not None:
            destination_folder = os.path.join(Restored_Folder, label)
            os.makedirs(destination_folder, exist_ok=True)
            shutil.move(file_path, os.path.join(destination_folder, os.path.basename(file_path)))
        else:
            os.makedirs(Review_Folder, exist_ok=True)
            shutil.move(file_path, os.path.join(Review_Folder, os.path.basename(file_path)))
    except Exception as e:
        print(f" Error moving {os.path.basename(file_path)}: {e}")


# Process Chaos Folder
def process_chaos_folder_batch(
    chaos_folder,
    embedding_model,
    collection,
    batch_size=32
):
    """Processing all images in chaos folder using batch processing"""
    
    if not os.path.exists(chaos_folder):
        raise FileNotFoundError(f"Chaos folder not found: {chaos_folder}")
    
    filenames = [
        f for f in os.listdir(chaos_folder)
        if f.lower().endswith(".png")
    ]

    if not filenames:
        print("No PNG files found in chaos folder")
        return

    print(f"Found {len(filenames)} images to process")
    
    total_batches = (len(filenames) + batch_size - 1) // batch_size
    sorted_count = 0
    review_count = 0

    for i in range(0, len(filenames), batch_size):
        batch_files = filenames[i:i + batch_size]
        batch_paths = [os.path.join(chaos_folder, f) for f in batch_files]

        print(f"Processing batch {i//batch_size + 1}/{total_batches}...", end=" ")

        # Load + preprocess batch
        images_batch, valid_paths = preprocess_images_batch(batch_paths)
        
        if len(images_batch) == 0:
            print("skipped (no valid images)")
            continue

        # Generate embeddings in ONE forward pass
        embeddings = generate_embeddings_batch(embedding_model, images_batch)

        # Batch vector search
        all_labels, all_distances = nearest_neighbors_batch(collection, embeddings)

        # Decide + move files
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
        
        print(f" {batch_sorted} sorted, {batch_review} to review")

    
    print(f"Processing Complete!")
    print(f"Total sorted: {sorted_count}")
    print(f"Total needing review: {review_count}")
   


# Main Function
def main():
    print("="*50)
    print("Dungeon Archivist - The Restoration")
    print("="*50)
    
    try:
        print("\n[1/3] Connecting to Vector Database...")
        collection = connect_to_vector_db()

        print("\n[2/3] Loading Archivist Model...")
        model, embedding_model = load_archivist_model(Model_Path)

        print("\n[3/3] Processing Chaos Folder...")
        process_chaos_folder_batch(
            Chaos_Folder,
            embedding_model,
            collection,
            batch_size=32
        )
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()