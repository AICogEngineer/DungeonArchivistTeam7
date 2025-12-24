import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers


# Constants
IMG_SIZE = (32, 32)
BATCH_SIZE = 32
EMBEDDING_DIM = 128 
EPOCHS = 100
SEED = 42
DATASET_A_PATH = "./dataset_a" 



def get_path_labels(root_dir):

    # Crawls subdirectories and creates labels based on the relative path.
    # Example: 'Equipment/Weapons/Swords' -> 'Equipment_Weapons_Swords'
    
    file_paths = []
    labels = []
    class_names = []

    for root, dirs, files in os.walk(root_dir):
        if files:
            # Create a label by joining the path components relative to root
            rel_path = os.path.relpath(root, root_dir)
            label_name = rel_path.replace(os.sep, "_")
            
            if label_name not in class_names:
                class_names.append(label_name)
            
            label_idx = class_names.index(label_name)
            
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_paths.append(os.path.join(root, f))
                    labels.append(label_idx)

    return np.array(file_paths), np.array(labels), class_names

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMG_SIZE)
    return image, label

'''
def build_vision_model(num_classes):
    inputs = keras.Input(shape=(32, 32, 3))
    
    # Data Augmentation 
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomFlip("vertical")(inputs)
    
    x = layers.Rescaling(1./255)(x)
    
    # Conv Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Conv Block 2 + Spatial Dropout
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.2)(x) 
    
    # Conv Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    
    # Embedding Layer with L2 Regularization
    # kernel_regularizer prevents the weights from becoming too large
    embedding_layer = layers.Dense(EMBEDDING_DIM, activation='relu', name="embedding_out", kernel_regularizer=regularizers.l2(0.001))(x)
    
    x = layers.Dropout(0.5)(embedding_layer) 
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)
    '''

def build_simple_vision_model(num_classes):
    model = keras.Sequential(
        [
            # Input
            layers.Input(shape=(32, 32, 3)),
            
            # ---- Conv Block 1 ----
            layers.Conv2D(32, (3, 3), activation='relu', padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),

            # ---- Conv Block 2 ----
            layers.Conv2D(64, (3, 3), activation='relu', padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),

            # ---- Conv Block 3 ----
            layers.Conv2D(128, (3, 3), activation='relu', padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),

            #Flatten
            layers.Flatten(),

            # ---- Embedding Head ----
            layers.Dense(64, activation='relu', name="embedding_dense"),
            

            # Classifier
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model

def build_vision_model(num_classes):
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



def run_training():
    # 1. Generate Dynamic Labels
    paths, labels, class_names = get_path_labels(DATASET_A_PATH)
    num_classes = len(class_names) # Dynamically calculated
    print(f"Detected {num_classes} unique path-based classes.")

    # 2. Create TF Dataset
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=False)
    
    val_size = int(len(paths) * 0.2)
    train_ds = ds.skip(val_size)
    val_ds = ds.take(val_size)

    train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.cache()
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 3. Build & Train
    model = build_simple_vision_model(num_classes)
    adam_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    sgd_optimizer = keras.optimizers.SGD(learning_rate=.001, momentum=0.9)
    model.compile(optimizer=sgd_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3,
        restore_best_weights=True 
    )

    lr_schedule = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )


    model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS, 
        callbacks=[early_stop, lr_schedule]
    )

    # 4. Save artifacts
    # Commented out while testing
    #model.save("dungeon_model_v1.keras")
    #with open("labels.txt", "w") as f:
    #    for name in class_names:
    #        f.write(f"{name}\n")

if __name__ == "__main__":
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    run_training()

'''
Dataset A changes:

Philosophy of changes:
If there is a folder with only one image, move that image to its parent folder and delete the empty subfolder.
If there is a folder which is a subtype of the parent folder, but within the parent folder multiple subtypes exists without subtype folders,
merge the subtype folder with the parent folder.


1. Merged the subtype floor_grass with the parent folder floor.
2. Moved banner to parent folder wall because it was the only image in that folder.
3. Merged the subtype wall_abyss with the parent folder wall.

'''