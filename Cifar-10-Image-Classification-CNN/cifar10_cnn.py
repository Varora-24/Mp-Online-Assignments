import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

def main():
    print("--------------------------------------------------")
    print("CIFAR-10 Image Classification using CNN")
    print("--------------------------------------------------")

    # 1. Load the CIFAR-10 dataset
    print("\nLoading CIFAR-10 dataset...")
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # 2. Normalize pixel values to be between 0 and 1
    # This helps the neural network converge faster
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Class names for CIFAR-10 corresponding to labels 0-9
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    print(f"Training data shape: {train_images.shape}")
    print(f"Testing data shape: {test_images.shape}")

    # 3. Build the Convolutional Neural Network (CNN) architecture
    print("\nBuilding the CNN model...")
    model = models.Sequential()
    
    # First Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten and add Dense layers for classification
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10)) # 10 output classes

    # Display the architecture
    model.summary()

    # 4. Compile the model
    print("\nCompiling the model...")
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # 5. Train the model
    print("\nTraining the model (this may take a few minutes depending on your hardware)...")
    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))

    # 6. Evaluate the model
    print("\nEvaluating the model on the test dataset...")
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    # 7. Plot and save training history
    print("\nGenerating training history plot...")
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title('CNN Training and Validation Accuracy on CIFAR-10')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_history.png', bbox_inches='tight')
    print("Training history plot saved as 'training_history.png'.")

if __name__ == "__main__":
    main()
