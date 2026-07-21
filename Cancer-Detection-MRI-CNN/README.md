# Cancer Detection using MRI Images (CNN)

## Objective
To develop a Convolutional Neural Network (CNN) that automatically classifies Magnetic Resonance Imaging (MRI) scans to detect the presence of cancer (e.g., Brain Tumors).

## Dataset Requirements
Since medical image datasets are typically large and come with strict licensing, the raw dataset is not included in this repository. 
To run this script, you must provide your own MRI dataset (such as the widely used "Brain MRI Images for Brain Tumor Detection" dataset from Kaggle).

1. Download the dataset.
2. Create a folder named `mri_dataset` in this directory.
3. Organize the images into class-specific subfolders. For example:
   ```
   mri_dataset/
   ├── tumor/
   │   ├── image1.jpg
   │   └── image2.jpg
   └── no_tumor/
       ├── image3.jpg
       └── image4.jpg
   ```

## Libraries Used
- **TensorFlow / Keras**: Used for the end-to-end deep learning pipeline, from loading the image dataset from the directory to building, training, and evaluating the CNN.
- **Matplotlib**: Used for visualizing the accuracy progression across epochs.

## Methodology
1. **Data Loading & Preprocessing**: 
   - The script uses `image_dataset_from_directory` to dynamically load images in batches (batch size = 32).
   - Images are uniformly resized to `224x224` pixels.
   - The dataset is split into 80% training data and 20% validation data.
   - For performance, the dataset leverages caching and prefetching (`tf.data.AUTOTUNE`).
2. **Data Augmentation**: 
   - Inside the model pipeline, images are scaled (`1./255`).
   - Random horizontal flipping and rotation are applied to artificially expand the training dataset and prevent overfitting.
3. **Architecture**: 
   - **Convolutional Base**: 3 blocks of `Conv2D` and `MaxPooling2D` layers (with 32, 64, and 128 filters respectively) designed to extract hierarchical visual features like edges, textures, and specific tumor shapes.
   - **Dropout**: A `Dropout(0.2)` layer mitigates overfitting.
   - **Classification Head**: A Flatten layer feeds into a `Dense(128)` layer, followed by the output classification layer.
4. **Training**: The model utilizes the `Adam` optimizer and dynamically switches between `BinaryCrossentropy` and `SparseCategoricalCrossentropy` depending on the number of classes detected in your dataset folder.

## How to Run
Ensure you have TensorFlow installed:
```bash
pip install tensorflow matplotlib
```
Then execute the script after setting up your `mri_dataset` folder:
```bash
python cancer_detection_cnn.py
```
