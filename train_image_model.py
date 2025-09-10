# train_image_model.py
# This file trains a simple CNN model to classify slope images as Low Risk or High Risk

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset folders
train_dir = "dataset/train"
val_dir = "dataset/test"

# Preprocessing - scale pixel values (0-1)
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    batch_size=16,
    class_mode='binary'
)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=(128,128),
    batch_size=16,
    class_mode='binary'
)

# CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')   # 0 = Safe, 1 = Risk
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_gen, validation_data=val_gen, epochs=5)

# Save model
model.save("models/rockfall_image_model.h5")
print(" Model saved at models/rockfall_image_model.h5")
