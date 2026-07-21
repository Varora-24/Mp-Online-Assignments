import os
import matplotlib.pyplot as plt

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.utils import image_dataset_from_directory
except ImportError:
    print("TensorFlow is not installed. Please install it using 'pip install tensorflow'")
    exit()

def main():
    print("--------------------------------------------------")
    print("Cancer Detection using MRI Images (CNN)")
    print("--------------------------------------------------")
    
    # Dataset path variables
    # This script assumes a standard directory structure: 
    # mri_dataset/tumor/
    # mri_dataset/no_tumor/
    dataset_dir = 'mri_dataset'
    
    if not os.path.exists(dataset_dir):
        print(f"\n[!] Dataset folder '{dataset_dir}' not found.")
        print("Please download an MRI dataset (e.g., Brain Tumor from Kaggle)")
        print("and extract it into a folder named 'mri_dataset' in this directory.")
        print("The folder should contain subfolders for each class (e.g., 'tumor', 'no_tumor').")
        return

    batch_size = 32
    img_height = 224
    img_width = 224

    print("\nLoading dataset...")
    # Use 80% for training and 20% for validation
    train_ds = image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"\nFound {num_classes} classes: {class_names}")

    # Optimize dataset performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("\nBuilding the CNN model...")
    model = models.Sequential([
        # Data Augmentation & Normalization
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        
        # Convolutional layers
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])

    model.summary()

    # Compile the model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy() if num_classes > 2 else tf.keras.losses.BinaryCrossentropy()
    
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    print("\nTraining the model...")
    epochs = 15
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    print("\nTraining complete.")

    # Plot training history
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('CNN Accuracy for MRI Cancer Detection')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('mri_training_history.png', bbox_inches='tight')
    print("Training history plot saved as 'mri_training_history.png'.")

if __name__ == "__main__":
    main()
