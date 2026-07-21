# CIFAR-10 Image Classification using CNN

## Objective
To develop a Convolutional Neural Network (CNN) capable of classifying images into one of ten distinct categories using the benchmark CIFAR-10 dataset.

## Dataset
**CIFAR-10** (Canadian Institute For Advanced Research) consists of 60,000 32x32 color images across 10 classes (airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks), with 6,000 images per class. There are 50,000 training images and 10,000 test images.

## Libraries Used
- **TensorFlow / Keras**: Deep learning framework used to build, compile, and train the neural network.
- **Matplotlib**: Used for plotting training and validation accuracy curves.

## Methodology
1. **Data Preprocessing**: The dataset is loaded directly via Keras. The RGB pixel values (ranging from 0 to 255) are normalized to floating-point values between `0.0` and `1.0` to ensure stable and faster convergence during training.
2. **Architecture**: 
   A standard sequential Convolutional Neural Network is built comprising:
   - **3 Convolutional (Conv2D) Layers**: Using ReLU activation to extract spatial hierarchies and features from the images.
   - **2 MaxPooling2D Layers**: Used to downsample the spatial dimensions, reducing computation and controlling overfitting.
   - **Flatten Layer**: Flattens the 3D output into a 1D vector.
   - **Dense Layers**: A fully connected layer with 64 units, followed by a final 10-unit Dense output layer representing the 10 classes.
3. **Training**: The model is compiled using the `Adam` optimizer and `SparseCategoricalCrossentropy` loss function. It is trained for 10 epochs.
4. **Evaluation**: The model evaluates its generalization on the 10,000 hold-out test images.

## Expected Results
- **Test Accuracy**: ~70% - 75%
- The model effectively learns distinct features of the 10 classes. A plot named `training_history.png` is generated upon execution, visualizing the accuracy improvement over the 10 epochs.

## How to Run
Ensure you have TensorFlow installed:
```bash
pip install tensorflow matplotlib
```
Then execute the script:
```bash
python cifar10_cnn.py
```
