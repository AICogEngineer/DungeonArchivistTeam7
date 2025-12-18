# Trains the Keras model on the known labeled data (Dataset_A).
import tensorflow as tf
from tensorflow.keras import layers, models

input_shape = (32, 32, 3)  # input shape for images
num_classes = 3  # number of classes for classification
embedding_size = 64  # dimension of the embedding layer

# Input layer
inputs = layers.Input(shape=input_shape)

# Convolutional layers
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)

# Flatten and embedding layer
x = layers.Flatten()(x)
embedding = layers.Dense(embedding_size, activation='relu', name='embedding')(x)

# Output layer
outputs = layers.Dense(num_classes, activation='softmax')(embedding)

# Model
model = models.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model Summary
model.summary()

# # Load dataset
# batch_size = 32
# img_size = (32, 32)
# train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     'Dataset_A',
#     image_size=(32, 32),
#     batch_size=32,
#     shuffle=True,
#     label_mode='int'
# )

# # Normalize values (0-255 to 0-1)
# normalization_layer = layers.Rescaling(1./255)
# train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))


# # Data -> Model
# model.fit(
#     train_dataset, 
#     epochs=10,
#     validation_data=None
#     )

# #Embedding Model
# embedding_model = tf.keras.Model(
#     inputs=model.input,
#     outputs=model.get_layer('embedding').output
# )








