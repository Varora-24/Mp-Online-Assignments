# MP Online AI-ML Assignments Portfolio

Welcome to my comprehensive portfolio of AI and Machine Learning projects! This repository contains a diverse collection of 9 assignments and capstone projects spanning classical Machine Learning, Deep Learning (Computer Vision), Reinforcement Learning, and Full-Stack MLOps Deployment.

Each project is contained within its own dedicated directory with a detailed `README.md` and the associated source code.

## 🗂️ Projects Overview

### 1. Adult Census Income Classification
**Folder:** `Adult-Census-Income-Classification/`
- **Description:** A classical Machine Learning pipeline using models like Random Forest, Logistic Regression, and SVM to predict whether an individual's income exceeds $50K/yr based on demographic census data.
- **Highlights:** Feature Encoding, StandardScaler, and handling class imbalances.

### 2. Cifar-10 Image Classification using CNN
**Folder:** `Cifar-10-Image-Classification-CNN/`
- **Description:** A Deep Learning project utilizing TensorFlow/Keras to build a Convolutional Neural Network (CNN) from scratch to classify 32x32 images into 10 distinct categories.
- **Highlights:** Custom CNN Architecture, Image Normalization.

### 3. Face Recognition using CNN in Wild Life (LFW Dataset)
**Folder:** `Face-Recognition-CNN-LFW/`
- **Description:** A computer vision model built to perform facial recognition on prominent public figures using the Labeled Faces in the Wild (LFW) dataset. 
- **Highlights:** Grayscale Image Processing, Dropout Regularization, Precision/Recall optimization.

### 4. Cancer Detection using MRI Images
**Folder:** `Cancer-Detection-MRI-CNN/`
- **Description:** An automated medical imaging tool that utilizes a deep CNN to detect the presence of tumors in MRI scans.
- **Highlights:** `image_dataset_from_directory`, Random Data Augmentation (Rotation/Flipping), Binary vs Categorical Crossentropy handling.

### 5. Cart-Pole RL Agent Training
**Folder:** `Cart-Pole-RL-Agent/`
- **Description:** A Reinforcement Learning (RL) agent trained using a Deep Q-Network (DQN) to balance a pole on a moving cart inside the OpenAI Gym `CartPole-v1` environment.
- **Highlights:** Experience Replay, Epsilon-Greedy Strategy.

### 6. Lunar Lander RL Agent Training
**Folder:** `Lunar-Lander-RL-Agent/`
- **Description:** An advanced Reinforcement Learning challenge where a DQN agent learns to safely land a spacecraft on the moon (`LunarLander-v2`).
- **Highlights:** Target Networks, Continuous State Spaces, Complex Reward Structures.

### 7. Movie Recommendation System
**Folder:** `Movie-Recommendation-System/`
- **Description:** A collaborative filtering recommendation engine that predicts user ratings for unseen movies using the MovieLens dataset.
- **Highlights:** Utility Matrix formulation, Matrix Factorization via `TruncatedSVD`.

### 8. End-to-End Render Deployment Project
**Folder:** `Car-Value-Prediction-Render/`
- **Description:** A full-stack deployment project containing a pre-trained car value prediction model wrapped in a Flask API. It is configured to be deployed effortlessly on Render's cloud platform.
- **Highlights:** Flask API, `render.yaml` Blueprint, Scikit-Learn integration.

### 9. RAG Chatbot (Capstone Project)
**Folder:** `Rag-Chatbot-Capstone/`
- **Description:** The final capstone project showcasing an Exoplanet Explorer Chatbot. It leverages a Retrieval-Augmented Generation (RAG) engine powered by FastAPI and Groq LLMs to answer highly specific contextual questions.
- **Highlights:** FastAPI, LLM Integration, Custom RAG Engine, Vercel/Render frontend integration.
