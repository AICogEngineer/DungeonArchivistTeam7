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
from PIL import Image

# Configuration
Chaos_Folder = "./Dataset_B/chaos_data"
Restored_Folder = "./restored_archive"
Review_Folder = "./review_pile"
Top_Matches = 5
Confidence_Threshold = 2000
Model_Path = "models/dungeon_model_v3.keras"
Embedding_Layer_Name = "embedding_out"

# Label → Category mapping
# Label → Category mapping (HIERARCHICAL VERSION)
# Labels stay as "dungeon_altars", "item_weapon" etc.
# The move_file function will create nested folders

LABEL_TO_CATEGORY = {
    # Dungeon subcategories
    "dungeon": "dungeon",
    "dungeon_altars": "dungeon_altars",
    "dungeon_doors": "dungeon_doors",
    "dungeon_floor": "dungeon_floor",
    "dungeon_floor_grass": "dungeon_floor_grass",
    "dungeon_floor_sigils": "dungeon_floor_sigils",
    "dungeon_gateways": "dungeon_gateways",
    "dungeon_shops": "dungeon_shops",
    "dungeon_statues": "dungeon_statues",
    "dungeon_traps": "dungeon_traps",
    "dungeon_trees": "dungeon_trees",
    "dungeon_vaults": "dungeon_vaults",
    "dungeon_wall": "dungeon_wall",
    "dungeon_wall_abyss": "dungeon_wall_abyss",
    "dungeon_wall_banners": "dungeon_wall_banners",
    "dungeon_wall_torches": "dungeon_wall_torches",
    "dungeon_water": "dungeon_water",
    
    # Effect
    "effect": "effect",
    
    # Emissaries
    "emissaries": "emissaries",
    
    # GUI subcategories
    "gui": "gui",
    "gui_abilities": "gui_abilities",
    "gui_commands": "gui_commands",
    "gui_invocations": "gui_invocations",
    "gui_skills": "gui_skills",
    "gui_spells": "gui_spells",
    "gui_spells_air": "gui_spells_air",
    "gui_spells_components": "gui_spells_components",
    "gui_spells_conjuration": "gui_spells_conjuration",
    "gui_spells_disciplines": "gui_spells_disciplines",
    "gui_spells_divination": "gui_spells_divination",
    "gui_spells_earth": "gui_spells_earth",
    "gui_spells_enchantment": "gui_spells_enchantment",
    "gui_spells_fire": "gui_spells_fire",
    "gui_spells_ice": "gui_spells_ice",
    "gui_spells_monster": "gui_spells_monster",
    "gui_spells_necromancy": "gui_spells_necromancy",
    "gui_spells_poison": "gui_spells_poison",
    "gui_spells_summoning": "gui_spells_summoning",
    "gui_spells_translocation": "gui_spells_translocation",
    "gui_spells_transmutation": "gui_spells_transmutation",
    "gui_startup": "gui_startup",
    "gui_tabs": "gui_tabs",
    
    # Item subcategories
    "item_amulet": "item_amulet",
    "item_amulet_artefact": "item_amulet_artefact",
    "item_armor_artefact": "item_armor_artefact",
    "item_armor_back": "item_armor_back",
    "item_armor_bardings": "item_armor_bardings",
    "item_armor_feet": "item_armor_feet",
    "item_armor_hands": "item_armor_hands",
    "item_armor_headgear": "item_armor_headgear",
    "item_armor_shields": "item_armor_shields",
    "item_armor_torso": "item_armor_torso",
    "item_book": "item_book",
    "item_book_artefact": "item_book_artefact",
    "item_food": "item_food",
    "item_gold": "item_gold",
    "item_misc": "item_misc",
    "item_misc_runes": "item_misc_runes",
    "item_potion": "item_potion",
    "item_ring": "item_ring",
    "item_ring_artefact": "item_ring_artefact",
    "item_rod": "item_rod",
    "item_scroll": "item_scroll",
    "item_staff": "item_staff",
    "item_wand": "item_wand",
    "item_weapon": "item_weapon",
    "item_weapon_artefact": "item_weapon_artefact",
    "item_weapon_ranged": "item_weapon_ranged",
    
    # Misc subcategories
    "misc": "misc",
    "misc_blood": "misc_blood",
    "misc_brands_bottom_left": "misc_brands_bottom_left",
    "misc_brands_bottom_right": "misc_brands_bottom_right",
    "misc_brands_top_left": "misc_brands_top_left",
    "misc_brands_top_right": "misc_brands_top_right",
    "misc_numbers": "misc_numbers",
    
    # Monster subcategories
    "monster": "monster",
    "monster_aberration": "monster_aberration",
    "monster_abyss": "monster_abyss",
    "monster_amorphous": "monster_amorphous",
    "monster_animals": "monster_animals",
    "monster_aquatic": "monster_aquatic",
    "monster_demons": "monster_demons",
    "monster_demonspawn": "monster_demonspawn",
    "monster_draconic": "monster_draconic",
    "monster_dragons": "monster_dragons",
    "monster_eyes": "monster_eyes",
    "monster_fungi_plants": "monster_fungi_plants",
    "monster_holy": "monster_holy",
    "monster_nonliving": "monster_nonliving",
    "monster_panlord": "monster_panlord",
    "monster_spriggan": "monster_spriggan",
    "monster_statues": "monster_statues",
    "monster_tentacles_eldritch_corners": "monster_tentacles_eldritch_corners",
    "monster_tentacles_eldritch_ends": "monster_tentacles_eldritch_ends",
    "monster_tentacles_kraken_corners": "monster_tentacles_kraken_corners",
    "monster_tentacles_kraken_ends": "monster_tentacles_kraken_ends",
    "monster_tentacles_kraken_segments": "monster_tentacles_kraken_segments",
    "monster_tentacles_starspawn_corners": "monster_tentacles_starspawn_corners",
    "monster_tentacles_starspawn_ends": "monster_tentacles_starspawn_ends",
    "monster_tentacles_starspawn_segments": "monster_tentacles_starspawn_segments",
    "monster_tentacles_vine_corners": "monster_tentacles_vine_corners",
    "monster_tentacles_vine_ends": "monster_tentacles_vine_ends",
    "monster_tentacles_vine_segments": "monster_tentacles_vine_segments",
    "monster_undead": "monster_undead",
    "monster_undead_simulacra": "monster_undead_simulacra",
    "monster_undead_skeletons": "monster_undead_skeletons",
    "monster_undead_spectrals": "monster_undead_spectrals",
    "monster_undead_zombies": "monster_undead_zombies",
    "monster_unique": "monster_unique",
    "monster_vault": "monster_vault",
    
    # Player subcategories
    "player_barding": "player_barding",
    "player_base": "player_base",
    "player_beard": "player_beard",
    "player_body": "player_body",
    "player_boots": "player_boots",
    "player_cloak": "player_cloak",
    "player_draconic_head": "player_draconic_head",
    "player_draconic_wing": "player_draconic_wing",
    "player_enchantment": "player_enchantment",
    "player_felids": "player_felids",
    "player_gloves": "player_gloves",
    "player_hair": "player_hair",
    "player_halo": "player_halo",
    "player_hand_left": "player_hand_left",
    "player_hand_left_misc": "player_hand_left_misc",
    "player_hand_right": "player_hand_right",
    "player_hand_right_artefact": "player_hand_right_artefact",
    "player_hand_right_misc": "player_hand_right_misc",
    "player_head": "player_head",
    "player_legs": "player_legs",
    "player_mutations": "player_mutations",
    "player_transform": "player_transform",
}

# Connecting To Vector Database
def connect_to_vector_db():
    """Connect to ChromaDB with L2 distance metric"""
    try:
        # Use PersistentClient with absolute path
        client = chromadb.PersistentClient(
            path="./chroma"
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
# Move File - HIERARCHICAL VERSION
def move_file(file_path, label):
    """Move file to appropriate nested destination folder"""
    try:
        if label is not None:
            # Split label: "dungeon_altars" → ["dungeon", "altars"]
            parts = label.split('_', 1)
            
            if len(parts) == 2:
                # Nested structure: restored_archive/dungeon/altars/
                category = parts[0]
                subcategory = parts[1]
                destination_folder = os.path.join(Restored_Folder, category, subcategory)
            else:
                # Single-level: restored_archive/effect/
                destination_folder = os.path.join(Restored_Folder, label)
            
            os.makedirs(destination_folder, exist_ok=True)
            shutil.move(file_path, os.path.join(destination_folder, os.path.basename(file_path)))
        else:
            # Send to review pile
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