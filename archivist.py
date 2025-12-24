#  The main engine. 
#  Scans a target folder.
#  Queries Vector DB.
#  Moves files to sorted subfolders.

import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model, Model 
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from collections import Counter
import chromadb

# Configuration
Chaos_Folder = "DatabaseB" # Do not have databas B yet
Restored_Folder = "./restored_archive" # Stores all sorted files after script processes them
Review_Folder = "./review_pile" # Files that need review after processing
Top_Matches = 5 # Number of top matches to consider for sorting
Confidence_Threshold = 0.2 # Minimum confidence level for a match to be considered valid
Model_Path = "path to model" # Path to the pre-trained model
Embedding_Layer_Name = "embedding" # Name of the embedding layer in the model

# Connecting To Vector Database
def connect_to_vector_db():
    client = chromadb.Client()
    collection = client.create_collection(name="Insert ChromaDB NAme Here") #Use CHROMADB Name Here
    return collection

# Load Model
def load_archivist_model(model_path):
    model = load_model(model_path)
    embedding_model = Model(inputs=model.input, outputs=model.get_layer("embedding").output)
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
    labels = [m['label'] for m in results['metadatas'][0]] # Extracts labels from the results
    distances = results['distances'][0]
    return labels, distances

# Decision Label
def decision(labels, distance, confidence_threshold = Confidence_Threshold):
    if distance <= confidence_threshold: # Cosine distance is used here, so a lower value indicates a better match
        return labels 
    else:
        return None 
    vote = Counter(labels) # Counts the occurrences of each label
    return vote.most_common(1)[0][0] # Returns the most common label

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






