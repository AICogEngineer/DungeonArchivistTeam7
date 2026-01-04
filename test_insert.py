import os
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import chromadb

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# --- 1. CONFIGURATION ---
DATASET_PATH = "./dataset_a"
MODEL_PATH = "models/dungeon_model_v1.keras"
EMBEDDING_LAYER_NAME = "embedding_out"
CHROMA_PATH = "./chroma_db"
IMG_SIZE = (32, 32)
BATCH_SIZE = 32

# Suppress warnings in the console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def get_embedding_extractor(model_path):
    """Loads model and slices it at the embedding layer."""
    print(f"[*] Loading model from {model_path}...")
    full_model = load_model(model_path)
    
    # 'Wake up' the model to prevent 'layer not called' errors
    dummy_input = tf.zeros((1, *IMG_SIZE, 3))
    _ = full_model(dummy_input)

    try:
        embedding_layer = full_model.get_layer(EMBEDDING_LAYER_NAME)
        extractor = Model(inputs=full_model.inputs, outputs=embedding_layer.output)
        print(f"[+] Successfully extracted layer: {EMBEDDING_LAYER_NAME}")
        return extractor
    except ValueError:
        print(f"[!] Error: Layer '{EMBEDDING_LAYER_NAME}' not found.")
        print("Available layers:", [l.name for l in full_model.layers])
        exit()

def get_files_and_labels(data_dir):
    """Walks the directory and identifies labels based on folder names."""
    path_obj = pathlib.Path(data_dir)
    # Get all PNG files
    file_paths = sorted([str(p) for p in path_obj.rglob("*.png")])
    labels = []
    
    for p in file_paths:
        # Get the name of the folder the image is inside
        rel_path = pathlib.Path(p).relative_to(path_obj)
        label = str(rel_path.parent) if str(rel_path.parent) != "." else "unlabeled"
        labels.append(label)
        
    return file_paths, labels

def safe_preprocess(path, label):
    """The 'Secret Sauce' to prevent Segfaults: Force RGB and 3 channels."""
    img = tf.io.read_file(path)
    # channels=3 forces the decoder to strip transparency/palettes safely
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label, path

def run_indexing():
    # 1. Setup Model
    extractor = get_embedding_extractor(MODEL_PATH)

    # 2. Setup ChromaDB
    print(f"[*] Connecting to ChromaDB at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="dungeondb", 
        metadata={"hnsw:space": "l2"} # Euclidean distance
    )

    # 3. Prepare Data Pipeline
    file_paths, labels = get_files_and_labels(DATASET_PATH)
    print(f"[*] Found {len(file_paths)} images across {len(set(labels))} classes.")

    # Create a TF Dataset for high-speed batching
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(safe_preprocess, num_parallel_calls=1)
    ds = ds.batch(BATCH_SIZE)


    # 4. Process and Store
    print(f"[*] Starting embedding generation...")
    total_added = 0

    for batch_imgs, batch_labels, batch_paths in ds:
        embeddings = extractor(batch_imgs, training=False).numpy()

        curr_batch_size = embeddings.shape[0]
        ids = [f"id_{total_added + i}" for i in range(curr_batch_size)]

        meta_labels = [l.numpy().decode("utf-8") for l in batch_labels]
        meta_paths = [p.numpy().decode("utf-8") for p in batch_paths]

        metadatas = [
            {"label": lbl, "source": src}
            for lbl, src in zip(meta_labels, meta_paths)
        ]

        for i in range(0, curr_batch_size, 8):
            collection.upsert(
                embeddings=embeddings[i:i+8].tolist(),
                metadatas=metadatas[i:i+8],
                ids=ids[i:i+8],
            )

        total_added += curr_batch_size

        print(f"    > Progress: {total_added}/{len(file_paths)} images indexed", end='\r')

    print(f"\n[SUCCESS] Vector Database is ready at {CHROMA_PATH}")


if __name__ == "__main__":
    run_indexing()