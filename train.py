import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pathlib
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Constants
SEED = 42
IMG_SIZE = (32, 32)
BATCH_SIZE = 32
DATASET = "./dataset_a" 

# Set seed for reproducibility
keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

def get_optimized_dataset(data_path):
    # Find image paths/labels
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
    class_names = label_lookup.get_vocabulary()

    # Function to load and normalize images
    def load_and_preprocess(filepath, label_str):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label_lookup(label_str)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.shuffle(len(image_paths), seed=SEED, reshuffle_each_iteration=False)
    
    # 80/20 Split
    train_size = int(0.8 * len(image_paths))
    train_ds = ds.take(train_size).map(load_and_preprocess).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = ds.skip(train_size).map(load_and_preprocess).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, num_classes, train_size, class_names

def plot_evaluation(model, val_ds, class_names):
    """Generates a confusion matrix and saves it as a PNG."""
    print("\n--- Generating Confusion Matrix ---")
    
    # 1. Get Predictions
    # model.predict(val_ds) respects the batch order
    y_probs = model.predict(val_ds)
    y_pred = np.argmax(y_probs, axis=1)
    
    # 2. Get True Labels
    # We iterate the dataset directly to extract the labels in order
    y_true = np.concatenate([y for x, y in val_ds], axis=0)
    
    # 3. Compute Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 4. Plot using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Save and Show
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"confusion_matrix_{timestamp}.png"
    plt.savefig(filename)
    print(f"Confusion matrix saved to {filename}")
    plt.show()

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
        layers.GlobalAveragePooling2D(),
 
        layers.Dense(256, activation="relu", name="embedding_out"),
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
            layers.Conv2D(32, (3, 3), padding="same", use_bias=False, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(32, (3, 3),padding="same", use_bias=False, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.MaxPooling2D((2,2)),

            # ---- Conv Block 2 ----
            layers.Conv2D(64, (3, 3), padding="same", use_bias=False, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(64, (3, 3), padding="same", use_bias=False, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.MaxPooling2D((2,2)),
            layers.SpatialDropout2D(0.3),

            # ---- Conv Block 3 ----
            layers.Conv2D(128, (3, 3), padding="same", use_bias=False, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(128, (3, 3), padding="same", use_bias=False, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.GlobalAveragePooling2D(),

            # ---- Embedding Head ----
            layers.Dense(
                256,
                use_bias=False,
                kernel_regularizer=regularizers.l2(1e-4),
                name="embedding_dense"
            ),
            layers.BatchNormalization(),
            layers.ReLU(name="embedding_out"),

            layers.Dropout(0.5),

            # Classifier
            layers.Dense(num_classes, activation="softmax"),
        ],
    )

    return model

def run_training(data_path, optimizer, complex_model=True, log=False, save_model=False, visualize_data=False):
    """
    Executes the Phase 1 training pipeline: loads data, builds the model, 
    trains with callbacks, and saves the final artifacts.
    """

    # Get datasets and parameters
    
    train_ds, val_ds, num_classes, train_size, class_names = get_optimized_dataset(data_path)

    # Build Model
    
    if complex_model:
        model = build_complex_model(num_classes)
    else:
        model = build_simple_model(num_classes)
    
    model.summary()

    # Custom Cosine Decay for SGD optimizer

    sgd_lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.1,
        decay_steps=100 * (train_size // BATCH_SIZE),
        alpha=0.01
    )  

    # Callback Definitions

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )

    lr_schedule = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # Optimizer Selection and Model Compilation

    if optimizer == "adam":    
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
                optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
        )
        callback_list = [early_stop, lr_schedule]

    elif optimizer == "adamw":
        optimizer = keras.optimizers.AdamW(learning_rate=0.001, weight_decay=.0001)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        callback_list = [early_stop, lr_schedule]
    
    elif optimizer == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=sgd_lr_schedule, momentum=0.9, nesterov=True)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        early_stop.patience = 15
        callback_list = [early_stop]

    #TensorBoard Logging Setup

    opt_name = optimizer.__class__.__name__ 
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/{opt_name}_{timestamp}"
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1,
        update_freq='epoch'
    )

    if log:
        callback_list += [tensorboard_callback]

    # Model Training

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=callback_list
    )

    # Confusion Matrix Generation

    if visualize_data:
        plot_evaluation(model, val_ds, class_names)

    # Save Artifacts for Phase 2

    if save_model:
        model.save("dungeon_model_v1.keras")
        print(f"--- Model saved to dungeon_model_v1.keras ---")
        with open("labels.txt", "w") as f:
            for name in class_names:
                f.write(f"{name}\n")

if __name__ == "__main__":
    run_training(DATASET, optimizer="adam")