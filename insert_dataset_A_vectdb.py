# populate_vectordb.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import chromadb
import pathlib

# Configuration
DATASET_PATH = "./dataset_a_v3"
MODEL_PATH = "models/dungeon_model_v3.keras"
EMBEDDING_LAYER_NAME = "embedding_out"
BATCH_SIZE = 32
IMG_SIZE = (32, 32)
CHROMA_PATH = "./chroma"

def load_embedding_model(model_path):
    """Load trained model and extract embedding layer"""
    model = load_model(model_path)
    
    input_tensor = Input(shape=(32, 32, 3))
    x = input_tensor
    
    for layer in model.layers:
        x = layer(x)
        if layer.name == EMBEDDING_LAYER_NAME:
            break
    
    embedding_model = Model(inputs=input_tensor, outputs=x)
    print(f"Embedding model loaded")
    return embedding_model

def get_training_images_and_labels(dataset_path):
    """Collect all training images and their labels"""
    path_obj = pathlib.Path(dataset_path)
    image_paths = sorted([str(p) for p in path_obj.rglob("*.png")])
    
    labels = []
    for path in image_paths:
        rel_p = pathlib.Path(path).relative_to(path_obj)
        # Extract label from folder structure
        label = "_".join(rel_p.parent.parts) if str(rel_p.parent) != "." else "unlabeled"
        labels.append(label)
    
    print(f"Found {len(image_paths)} training images")
    return image_paths, labels

def preprocess_image_batch(image_paths):
    """Load and preprocess batch of images"""
    images = []
    for path in image_paths:
        img = load_img(path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
    return np.array(images)

def populate_chromadb(dataset_path, embedding_model):
    """Generate embeddings for all training data and store in ChromaDB"""
    
    # CRITICAL: Create directory first
    os.makedirs(CHROMA_PATH, exist_ok=True)
    print(f"Created/verified directory: {CHROMA_PATH}")
    
    # Connect to ChromaDB with persistence
    client = chromadb.PersistentClient(path=CHROMA_PATH)  # Use PersistentClient!
    print(f"Connected to ChromaDB at: {CHROMA_PATH}")
    
    # Delete existing collection if it exists (fresh start)
    try:
        client.delete_collection("dungeondb")
        print("Cleared existing database")
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name="dungeondb",
        metadata={"hnsw:space": "l2"}
    )
    print(f"Created collection: dungeondb")
    
    # Get all training images
    image_paths, labels = get_training_images_and_labels(dataset_path)
    
    # Process in batches
    total_batches = (len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch_labels = labels[i:i + BATCH_SIZE]
        
        batch_num = i // BATCH_SIZE + 1
        print(f"Processing batch {batch_num}/{total_batches}...", end=" ")
        
        # Load and preprocess images
        images_batch = preprocess_image_batch(batch_paths)
        
        # Generate embeddings
        embeddings = embedding_model.predict(images_batch, verbose=0)
        embeddings = embeddings.reshape(len(embeddings), -1)
        
        # Prepare data for ChromaDB
        ids = [f"train_{i+j}" for j in range(len(batch_paths))]
        metadatas = [{"label": label, "source": path} for label, path in zip(batch_labels, batch_paths)]
        
        # Add to collection
        collection.add(
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(batch_paths)} vectors")
    
    final_count = collection.count()
    print(f"\n Database populated with {final_count} vectors")
    print(f" Database saved to: {CHROMA_PATH}")
    
    # Verify it was saved
    if os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
        print(f"Verified: chroma.sqlite3 file created")
    else:
        print(f"WARNING: chroma.sqlite3 NOT found!")
    
    return collection

def main():
    print("="*50)
    print("Phase 1.5: Populating Vector Database")
    print("="*50)
    
    print("\n[1/2] Loading embedding model...")
    embedding_model = load_embedding_model(MODEL_PATH)
    
    print("\n[2/2] Generating and storing embeddings...")
    collection = populate_chromadb(DATASET_PATH, embedding_model)
    
    print("\n" + "="*50)
    print("SUCCESS!")
    print("="*50)
    print(f"Vector database ready at:")
    print(f"  {CHROMA_PATH}")
    print("\nNext step: Run Phase 2")
    print("  python archivist.py")
    print("="*50)

if __name__ == "__main__":
    main()