import os
import matplotlib.pyplot as plt
import numpy as np

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except ImportError:
    print("TensorFlow is not installed. Please install it using 'pip install tensorflow'")
    # We allow the script to continue without TF just to show dataset loading if possible,
    # but it will crash when building the model.

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
    print("--------------------------------------------------")
    print("Face Recognition using CNN in LFW Dataset")
    print("--------------------------------------------------")

    # 1. Load the LFW dataset
    print("\nDownloading/Loading LFW dataset... (This may take a minute)")
    # min_faces_per_person=70 ensures we only get people with at least 70 images
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    # Images arrays to find the shapes
    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.images
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print(f"Total dataset size:")
    print(f"n_samples: {n_samples}")
    print(f"n_classes: {n_classes} ({', '.join(target_names)})")
    print(f"Image dimensions: {h}x{w}")

    # 2. Preprocess the Data
    # CNNs expect the input shape to be (n_samples, height, width, channels)
    # Since LFW images are grayscale, channels = 1
    X = X.reshape((n_samples, h, w, 1))
    
    # Scale pixel values (LFW from sklearn is already scaled between 0 and 1 or 0-255, usually 0-255)
    if np.max(X) > 1.5:
        X = X / 255.0

    # Split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    
    print(f"\nTraining set: {X_train.shape[0]} images")
    print(f"Testing set: {X_test.shape[0]} images")

    # 3. Build the CNN Model
    print("\nBuilding the CNN model...")
    model = models.Sequential()

    # First Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(h, w, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Classification layers
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5)) # Dropout for regularization
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(n_classes, activation='softmax'))

    model.summary()

    # 4. Compile the Model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 5. Train the Model
    print("\nTraining the model...")
    history = model.fit(X_train, y_train, 
                        epochs=20, 
                        batch_size=32, 
                        validation_split=0.1)

    # 6. Evaluate the Model
    print("\nEvaluating the model on the test dataset...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=target_names))

    # 7. Plot and save training history
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('CNN Training and Validation Accuracy on LFW')
    plt.legend()
    plt.grid(True)
    plt.savefig('lfw_training_history.png', bbox_inches='tight')
    print("Training history plot saved as 'lfw_training_history.png'.")

if __name__ == "__main__":
    main()
