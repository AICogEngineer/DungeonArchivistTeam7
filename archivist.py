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

# Configuration
Chaos_Folder = "C:\\Users\\jchac\\JaredChancey\\Group-7-AIENG\\DungeonArchivistTeam7\\Dataset_B\\chaos_data" # Do not have databas B yet
Restored_Folder = "C:\\Users\\jchac\\JaredChancey\\Group-7-AIENG\\DungeonArchivistTeam7\\restored_archive" # Stores all sorted files after script processes them
Review_Folder = "C:\\Users\\jchac\\JaredChancey\\Group-7-AIENG\\DungeonArchivistTeam7\\review_pile" # Files that need review after processing
Top_Matches = 5 # Number of top matches to consider for sorting
Confidence_Threshold = .75 # Minimum confidence level for a match to be considered valid
Model_Path = "models/dungeon_model_v1.keras"
# Path to the pre-trained model
Embedding_Layer_Name = "embedding_out" # Name of the embedding layer in the model

# Connecting To Vector Database
def connect_to_vector_db():
    client = chromadb.Client(
    settings=chromadb.Settings(
        persist_directory="./chroma"
    )
)

    collection = client.get_or_create_collection(name="dungeondb")
    return collection

# Load Model
def load_archivist_model(model_path, input_shape=(32,32,3)):
    # Load the Sequential model
    model = load_model(model_path)
    
    # Create a functional Input
    input_tensor = Input(shape=input_shape)
    
    # Pass the input through each layer of the model up to the embedding layer
    x = input_tensor
    for layer in model.layers:
        x = layer(x)
        if layer.name == Embedding_Layer_Name:
            break  # stop at embedding layer
    
    embedding_model = Model(inputs=input_tensor, outputs=x)
    return model, embedding_model


# Preprocess Image
def preprocess_image(image_path, target_size = (32, 32)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0 # Normalizes pixel values (0 - 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Generate Embedding 
def generate_embedding(embedding_model, image_array):
    embedding = embedding_model.predict(image_array)
    return embedding.flatten()

# Query Vector DB
def nearest_neighbors(collection, embedding, top_matches=Top_Matches):
    results = collection.query(query_embeddings=[embedding], n_results=top_matches)
    labels = [m['label'] for m in results['metadatas'][0]]
    distances = results['distances'][0]  # closest first
    return labels, distances


# Decision Label
def decision(labels, distances, confidence_threshold=Confidence_Threshold):
    """
    Decide which label to assign based on nearest neighbors and confidence threshold.
    """
    if not labels or not distances:
        return None  # safety check

    top_distance = distances[0]  # distance of the closest match

    
    if top_distance <= confidence_threshold:
        vote = Counter(labels)
        return vote.most_common(1)[0][0]  # most common label
    else:
        return None

    
# Move File
def move_file(file_path, label):
    if label is not None:
        destination_folder = os.path.join(Restored_Folder, label)
        os.makedirs(destination_folder, exist_ok=True)
        shutil.move(file_path, os.path.join(destination_folder, os.path.basename(file_path)))
    else:
        os.makedirs(Review_Folder, exist_ok=True)
        shutil.move(file_path, os.path.join(Review_Folder, os.path.basename(file_path)))

# Process Chaos Folder
def process_chaos_folder(Chaos_Folder, embedding_model, collection):
    for filename in os.listdir(Chaos_Folder):
        if not filename.endswith(".png"):
            continue
        file_path = os.path.join(Chaos_Folder, filename)
        image_array = preprocess_image(file_path)
        embedding = generate_embedding(embedding_model, image_array)
        labels, distances = nearest_neighbors(collection, embedding)
        final_label = decision(labels, distances)
        move_file(file_path, final_label)

# Main Function
def main():
    print("Connecting to Vector Database...")
    collection = connect_to_vector_db()

    print("Loading Archivist Model...")
    model, embedding_model = load_archivist_model(Model_Path)

    print("Processing Chaos Folder...")
    process_chaos_folder(Chaos_Folder, embedding_model, collection)

    print("Processing Complete.")

if __name__ == "__main__":
    main()






