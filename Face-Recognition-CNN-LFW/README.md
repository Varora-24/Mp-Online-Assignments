# Face Recognition using CNN in Wild Life (LFW Dataset)

## Objective
To build a Convolutional Neural Network (CNN) capable of performing facial recognition on the Labeled Faces in the Wild (LFW) dataset. The model aims to correctly identify specific public figures from images taken in unconstrained environments.

## Dataset
**Labeled Faces in the Wild (LFW)** is a public benchmark dataset for face verification and recognition. 
- The dataset is accessed directly using `scikit-learn`'s dataset loader (`fetch_lfw_people`).
- To ensure sufficient data for training the CNN, the dataset is filtered to only include individuals with a minimum of 70 images (`min_faces_per_person=70`).
- The images are resized (scaled down by 0.4) and converted to grayscale, yielding a shape of `(50, 37, 1)` per image.

## Libraries Used
- **TensorFlow / Keras**: To design, build, and train the deep Convolutional Neural Network.
- **Scikit-Learn**: For downloading the LFW dataset and generating the final classification report (Precision, Recall, F1).
- **Matplotlib**: For plotting the training and validation accuracy curves.
- **NumPy**: For array reshaping and manipulation.

## Methodology
1. **Data Preprocessing**: 
   - The LFW images are fetched and extracted.
   - Images are reshaped to add a channel dimension `(samples, height, width, 1)` as required by Keras CNNs.
   - The pixel intensities are normalized to a scale of `0.0` to `1.0`.
   - The data is split into 75% training and 25% testing sets using stratified sampling to maintain class distribution.
2. **Architecture**: 
   A custom sequential CNN is built:
   - **3 Conv2D + MaxPooling Blocks**: To extract deep facial features and spatial hierarchies (using 32, 64, and 128 filters respectively).
   - **Flattening & Dropout**: The output is flattened, and a Dropout layer (rate = 0.5) is applied to aggressively prevent overfitting given the small size of the dataset.
   - **Dense Layers**: A hidden layer with 128 neurons, followed by a Softmax output layer corresponding to the number of recognizable individuals (classes).
3. **Training & Evaluation**: 
   - The model is compiled with the `Adam` optimizer and `sparse_categorical_crossentropy` loss.
   - It is trained for 20 epochs with a validation split to monitor convergence.
   - Finally, the model generates predictions on the hold-out test set, and a full Classification Report is printed.

## Expected Results
- The network successfully learns to distinguish between the selected prominent individuals. 
- The training history plot (`lfw_training_history.png`) provides a visual representation of how accuracy improves across epochs. 
- *Note*: Given the highly unconstrained nature of "in-the-wild" images (varying lighting, poses, and expressions), the CNN leverages its multiple convolutional layers to isolate defining facial features over background noise.

## How to Run
Ensure you have TensorFlow and Scikit-Learn installed:
```bash
pip install tensorflow scikit-learn matplotlib
```
Then execute the script:
```bash
python lfw_face_recognition.py
```
