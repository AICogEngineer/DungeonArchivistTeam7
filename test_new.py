import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers

# --- Global Reproducibility ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# --- Configuration ---
IMG_SIZE = (32, 32)
BATCH_SIZE = 32

def get_optimized_dataset(data_path):
    path_obj = pathlib.Path(data_path)
    image_paths = sorted([str(p) for p in path_obj.rglob("*.png")])
    labels = []
    for path in image_paths:
        rel_p = pathlib.Path(path).relative_to(path_obj)
        label = "_".join(rel_p.parent.parts) if str(rel_p.parent) != "." else "unlabeled"
        labels.append(label)
    label_lookup = layers.StringLookup(output_mode="int")
    label_lookup.adapt(labels)
    num_classes = label_lookup.vocabulary_size()

    def load_and_preprocess(filepath, label_str):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0 # CPU-safe normalization
        return img, label_lookup(label_str)

    # 4. Create the Pipeline
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.shuffle(len(image_paths), seed=SEED, reshuffle_each_iteration=False)
    
    # 80/20 Split
    train_size = int(0.8 * len(image_paths))
    train_ds = ds.take(train_size).map(load_and_preprocess).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = ds.skip(train_size).map(load_and_preprocess).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, num_classes

def build_simple_model(num_classes):
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),     
        layers.Dense(128, activation="relu", name="embedding_out"),
        layers.Dense(num_classes, activation="softmax")
    ])
    
    return model

def build_complex_model(num_classes):
    model = keras.Sequential(
        [
            # Input + Augmentation
            layers.Input(shape=(32, 32, 3)),
            layers.RandomFlip("horizontal"),

            # Not using bias because BatchNormalization layers have their own bias
            # Manually call ReLU activation fucntion after BatchNormalization
            # Max pool after the conv block 

            # ---- Conv Block 1 ----
            layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(32, (3, 3),padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.MaxPooling2D((2,2)),

            # ---- Conv Block 2 ----
            layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.MaxPooling2D((2,2)),
            layers.SpatialDropout2D(0.2),

            # ---- Conv Block 3 ----
            layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.GlobalAveragePooling2D(),

            # ---- Embedding Head ----
            layers.Dense(
                256,
                use_bias=False,
                kernel_regularizer=regularizers.l2(1e-4),
                name="embedding_dense",
            ),
            layers.BatchNormalization(),
            layers.ReLU(name="embedding_out"),

            layers.Dropout(0.4),

            # Classifier
            layers.Dense(num_classes, activation="softmax"),
        ],
    )

    return model

def run_training(data_path, model_save_path="dungeon_model_v1.keras"):
    """
    Executes the Phase 1 training pipeline: loads data, builds the model, 
    trains with callbacks, and saves the final artifacts.
    """
    
    train_ds, val_ds, num_classes = get_optimized_dataset(data_path)
    model = build_simple_model(num_classes)
    model.summary()
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )
    lr_schedule = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    adam_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    sgd_optimizer = keras.optimizers.SGD(learning_rate=.001, momentum=0.9)

    model.compile(
        optimizer=adam_optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[early_stop, lr_schedule]
    )

    # 5. Save Artifacts for Phase 2
    # We save the model and the training history for the 'analysis.md' report
    #model.save(model_save_path)
    #print(f"--- Model saved to {model_save_path} ---")
    
    return history

if __name__ == "__main__":
    # Ensure the path matches your project structure
    DATA_ROOT = "./dataset_a" 
    run_training(DATA_ROOT)