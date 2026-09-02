# 🌱 Plant Disease Detection System

A deep learning-based Computer Vision application designed to detect and classify plant leaf diseases accurately using Convolutional Neural Networks (CNN) and deployed via an interactive Streamlit web interface.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Features](#-features)
3. [Dataset](#-dataset)
4. [Model Architecture](#-model-architecture)
5. [Training & Pipeline](#-training--pipeline)
6. [Evaluation & Performance](#-evaluation--performance)
7. [Project Structure](#-project-structure)
8. [Installation & Setup](#-installation--setup)
9. [Usage](#-usage)
10. [Limitations](#-limitations)
11. [Future Improvements](#-future-improvements)
12. [License](#-license)

---

## 🔍 Project Overview
Early detection of plant diseases is crucial for preventing major crop losses in agriculture. This project leverages Deep Learning to analyze leaf images and instantly identify potential diseases, providing farmers and researchers with a reliable diagnostic tool.

---

## ✨ Features
* **Real-time Prediction:** Instant classification via a user-friendly Streamlit web interface.
* **Human-Readable Outputs:** Maps raw model outputs directly to clear disease names and confidence scores.
* **Robust Preprocessing:** Automated image resizing and normalization matching the training pipeline.
* **Clean Architecture:** Well-structured codebase separating models, assets, and source code.

---

## 📊 Dataset
* **Source:** PlantVillage Dataset / Custom collected samples
* **Total Classes:** 5 Categories
  * Healthy Plant
  * Early Blight
  * Late Blight
  * Powdery Mildew
  * Leaf Spot
* **Data Split:** 
  * Training: 80%
  * Validation: 10%
  * Testing: 10%

---

## 🧠 Model Architecture
The system uses a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras, optimized for image classification tasks.
* **Input Shape:** `(128, 128, 3)`
* **Layers:** Convolutional layers with ReLU activation, Max Pooling, Dropout for regularization, and a Dense Softmax output layer.

---

## ⚙️ Training & Pipeline
* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Preprocessing:** Images resized to $128 \times 128$ pixels and normalized to scale pixel values.

---

## 📈 Evaluation & Performance
* Evaluated on a separate Test Dataset to ensure generalization and prevent overfitting.
* Metrics tracked: **Accuracy, Precision, Recall, F1-score**, and **Confusion Matrix**.

---

## 🗂️ Project Structure
```text
plant-disease-detection/
│
├── app.py                  # Streamlit web application
├── plant_disease_model.h5  # Trained CNN model weights
├── requirements.txt        # Required Python packages
├── .gitignore              # Git exclusion rules
├── LICENSE                 # Project license
└── README.md               # Project documentation

git clone [https://github.com/qamarsobhy7-source/plant-disease-detection.git](https://github.com/qamarsobhy7-source/plant-disease-detection.git)
cd plant-disease-detection
pip install -r requirements.txt
streamlit run app.py
