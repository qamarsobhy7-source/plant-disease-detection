# 🌿 Verdant - Smart Crop Doctor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red)
![Accuracy](https://img.shields.io/badge/Accuracy-98.66%25-success)

## 📖 Overview
**Verdant** is an end-to-end Deep Learning platform that diagnoses plant diseases from leaf images. It uses a fine-tuned **EfficientNet-B0** model to classify 38 different crop-disease combinations across 14 plant species.

Unlike traditional black-box models, Verdant integrates **Grad-CAM** to provide visual explainability, showing the user exactly *where* the model detected the disease on the leaf.

## ✨ Features
- **High Accuracy:** 98.66% validation accuracy.
- **Explainable AI (XAI):** Visual heatmaps using Grad-CAM.
- **Modern UI:** Built with Streamlit and custom CSS (Glassmorphism).
- **Treatment Recommendations:** Provides actionable treatment advice.

## 🏗️ Architecture
1. **Model:** EfficientNet-B0 (Pretrained on ImageNet, fine-tuned).
2. **Training Strategy:** Two-phase training.
3. **Data Augmentation:** Random Flip, Rotation, Color Jitter.
4. **Explainability:** PyTorch Grad-CAM.

## 🚀 Live Demo
[Click here to try the app!](YOUR_STREAMLIT_LINK_HERE)

## 👨‍💻 Author
**Qamar Sobhy**
- GitHub: [@qamarsobhy7-source](https://github.com/qamarsobhy7-source)