import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load and prepare the training data
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'D:\Task_04\data',  # Replace with the path to your 'data' folder
    image_size=(64, 64),  # Resize images to a uniform size
    batch_size=32,        # Training batch size
    label_mode='int'      # Labels as integers (class numbers)
)

# Retrieve class names **before** applying normalization
class_names = dataset.class_names  # This works before applying .map()

# Normalize images between 0 and 1
def normalize_image(image, label):
    return image / 255.0, label

# Apply normalization using the `map` function
dataset = dataset.map(normalize_image)

# 2. Create a simple CNN model for gesture recognition
model = models.Sequential([
    layers.InputLayer(input_shape=(64, 64, 3)),  # Image size (64x64 RGB)
    layers.Conv2D(32, (3, 3), activation='relu'),  # 2D Convolution with 32 filters
    layers.MaxPooling2D((2, 2)),  # 2D Pooling to reduce the feature map size
    layers.Conv2D(64, (3, 3), activation='relu'),  # 2D Convolution with 64 filters
    layers.MaxPooling2D((2, 2)),  # 2D Pooling
    layers.Conv2D(128, (3, 3), activation='relu'), # 2D Convolution with 128 filters
    layers.MaxPooling2D((2, 2)),  # 2D Pooling
    layers.Flatten(),  # Flatten data to pass to the dense layer
    layers.Dense(128, activation='relu'),  # Dense layer with 128 neurons
    layers.Dense(len(class_names), activation='softmax')  # Output layer, one per class
])

# 3. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # For integer labels
              metrics=['accuracy'])

# 4. Train the model
history = model.fit(dataset, epochs=5)

# 5. Save the model
model.save('model.h5')

# 6. Plot the accuracy curve
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
