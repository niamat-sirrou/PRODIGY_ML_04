import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to preprocess images
def preprocess_data(data_dir, image_size=(128, 128), batch_size=32):
    # Initialize ImageDataGenerator with rescaling and validation split
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Create training data generator
    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,  # Resize images to the specified size
        batch_size=batch_size,    # Set the batch size
        class_mode='sparse',      # Use sparse labels (integer labels)
        subset='training'         # Indicate this is the training set
    )

    # Create validation data generator
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,  # Resize images to the specified size
        batch_size=batch_size,    # Set the batch size
        class_mode='sparse',      # Use sparse labels (integer labels)
        subset='validation'       # Indicate this is the validation set
    )

    return train_gen, val_gen  # Return the training and validation generators
