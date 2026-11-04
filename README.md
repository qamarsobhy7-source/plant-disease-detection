# 🌿 Verdant - Smart Crop Doctor
> **End-to-End Hierarchical Deep Learning Platform for Plant Disease Diagnosis with Explainable AI (Grad-CAM)**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red)
![Accuracy](https://img.shields.io/badge/Lab%20Accuracy-98.3%25-success)

## 📖 Overview
**Verdant** is an advanced AI platform that diagnoses plant diseases from leaf images. Unlike traditional single-model approaches, Verdant uses a **Hierarchical Architecture**:
1. **M1 (Crop Detector):** Identifies the plant species (20 supported crops).
2. **M2 (Disease Experts):** Routes the image to a specialized model for that specific crop to detect the exact disease.
3. **Explainable AI:** Uses **Grad-CAM** to show the user exactly *where* the disease is on the leaf.

## ✨ Features
- **20 Crops Supported:** Apple, Banana, Mango, Tomato, Potato, Grape, Corn, and more.
- **Top-3 Predictions:** As a safety net, the app shows the top 3 most likely crops if the AI is uncertain.
- **Detailed Treatment Plan:** Provides Immediate Actions, Organic, Chemical, and Preventive treatments.
- **Robust Training:** M1 was fine-tuned on real-world images (PlantDoc + AgroLeaf) to overcome background bias.

## 🏗️ Architecture
Input Image -> M1 (Crop Detector) -> M2 (Disease Expert) -> Grad-CAM + Treatment Plan

## ⚙️ How to Run Locally
1. git clone https://github.com/qamarsobhy7-source/plant-disease-detection.git
2. pip install -r requirements.txt
3. streamlit run app.py

## ⚠️ Limitations & Future Work
- The M2 Expert models for crops with limited data have lower accuracy.
- Future work includes adding a **Quality Gate (M0)** to reject non-leaf images.

## 📄 License
This project is licensed under the MIT License.