#  The main engine. 
#  Scans a target folder.
#  Queries Vector DB.
#  Moves files to sorted subfolders.

import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model, Model 
from tensorflow.keras.preprocessing.image import img_to_array, load_img
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
def get_nearest_neighbors(collection, embedding, top_matches=Top_Matches):
    results = collection.query(query_embeddings=[embedding], n_results=top_matches)
    labels = results['metadata'][0]
    distances = results['distances'][0]
    return labels, distances

# Decision Label
def decision(label, distance, confidence_threshold = Confidence_Threshold):
    if distance < confidence_threshold:
        return label
    else:
        return None #return None if the distance is above the confidence threshold






