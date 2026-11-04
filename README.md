# 🌿 Verdant - Smart Crop Doctor

> **An End-to-End Hierarchical Deep Learning Platform for Plant Disease Diagnosis with Explainable AI (Grad-CAM).**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)
![Accuracy](https://img.shields.io/badge/Lab%20Accuracy-97.4%25-success.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🚀 Live Demo
Experience the application live: **[Verdant AI App](https://plant-disease-detection-8fez5r9xxtyvoc3end7ywc.streamlit.app/)**

## 📸 Application Screenshots
<table>
  <tr>
    <td align='center'><b>1. Landing Page</b></td>
    <td align='center'><b>2. Diagnosis & Explainable AI</b></td>
    <td align='center'><b>3. Treatment Plan</b></td>
  </tr>
  <tr>
    <td><img src='assets/home.png' width='250'/></td>
    <td><img src='assets/demo.png' width='250'/></td>
    <td><img src='assets/treatment.png' width='250'/></td>
  </tr>
</table>

## 📖 About The Project
**Verdant** is an advanced AI platform designed to assist farmers and agricultural experts in diagnosing plant diseases early and accurately. Traditional systems use a single, monolithic model to classify dozens of diseases, which often leads to confusion due to similar visual symptoms across different plants.

To solve this, Verdant introduces a **Hierarchical Architecture**:
1. **M1 (Crop Detector):** A robust EfficientNet-B0 model that first identifies the plant species (14 supported crops).
2. **M2 (Disease Experts):** The image is dynamically routed to a specialized, lightweight model trained *exclusively* on the diseases of that specific crop.
3. **Explainable AI (XAI):** The system utilizes **Grad-CAM** (Gradient-weighted Class Activation Mapping) to generate a heatmap, showing the user exactly *where* the AI detected the disease on the leaf.

## 🌱 Supported Crops & Diseases
The M1 model is trained on the PlantVillage dataset, supporting 14 primary crops and over 38 disease/healthy classes:
- 🍎 Apple (Scab, Black Rot, Cedar Apple Rust, Healthy)
- 🫐 Blueberry (Healthy)
- 🍒 Cherry (Powdery Mildew, Healthy)
- 🌽 Corn (Cercospora, Common Rust, Northern Leaf Blight, Healthy)
- 🍇 Grape (Black Rot, Esca, Leaf Blight, Healthy)
- 🍊 Orange (Haunglongbing/Citrus Greening)
- 🍑 Peach (Bacterial Spot, Healthy)
- 🫑 Pepper (Bacterial Spot, Healthy)
- 🥔 Potato (Early Blight, Late Blight, Healthy)
- 🍓 Strawberry (Leaf Scorch, Healthy)
- 🍅 Tomato (9 diseases including Late Blight, Mosaic Virus, etc.)
- 🌿 Soybean, Squash, and Raspberry.

## ✨ Key Features
- **Hierarchical Inference:** Combines a general crop classifier with specialized disease experts for higher precision.
- **Top-3 Predictions (Safety Net):** If the model is uncertain (e.g., due to real-world background noise), it displays the top 3 most likely crops. This prevents false diagnoses and acts as a safety net.
- **Actionable Treatment Plans:** Provides detailed, step-by-step recommendations for each detected disease (Immediate Actions, Organic, Chemical, and Preventive treatments).
- **Grad-CAM Visualizations:** Visual heatmaps highlight the infected areas on the leaf, building trust in the AI's decision.
- **Modern UI/UX:** A clean, responsive, and user-friendly interface built with Streamlit, featuring custom CSS and a mobile-friendly design.

## 🏗️ System Architecture
```text
[Input Image] 
      │
      ▼
[M1: Crop Detector (EfficientNet-B0)] ──> Identifies Crop (e.g., Tomato)
      │
      ▼
[M2: Crop-Specific Expert] ──> Identifies Disease (e.g., Late Blight)
      │
      ▼
[Grad-CAM Generator] + [Treatment Plan Database]
      │
      ▼
[Final Diagnosis & Recommendations to User]
```

## 📊 Results & Performance
- **M1 Crop Detector:** Achieved **97.4%** validation accuracy on the PlantVillage test set.
- **M2 Disease Experts:** Achieved **>95%** accuracy on their respective validation sets.
- **Real-World Robustness:** The system was fine-tuned with data augmentation to handle real-world images, and the Top-3 safety net ensures reliable output even in domain shift scenarios.

## 🛠️ Tech Stack
- **Deep Learning Framework:** PyTorch, Torchvision
- **Model Architecture:** EfficientNet-B0 (Transfer Learning)
- **Explainability:** PyTorch Grad-CAM
- **Frontend:** Streamlit (Custom CSS)
- **Data Handling:** Pandas, NumPy, PIL
- **Deployment:** Streamlit Community Cloud

## 📂 Project Structure
```text
├── app.py                  # Main Streamlit web application
├── requirements.txt        # Python dependencies
├── treatments.json         # Comprehensive treatment database (JSON)
├── models_14crops/         # M1: General Crop Detector (.pth)
├── models_v2/              # M2: Specialized Expert Models (.pth)
├── assets/                 # Images and screenshots for README
├── docs/                   # Additional technical documentation
├── src/                    # Source code for training and evaluation
└── tests/                  # Automated tests
```

## ⚙️ How to Run Locally
1. **Clone the repository:**
   ```bash
   git clone https://github.com/qamarsobhy7-source/plant-disease-detection.git
   cd plant-disease-detection
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## ⚠️ Limitations & Future Work
- **Domain Shift:** The model achieves **97.4%** accuracy on lab-controlled datasets (PlantVillage). However, real-world field images may show slight variations due to background noise. We implemented the *Top-3 Prediction* safety net to mitigate this.
- **Future Work:** 
  - Integrate an Out-of-Distribution (OOD) detector to reject non-leaf images.
  - Expand the M1 dataset to include 30+ crops using real-world datasets (e.g., PlantDoc).
  - Build a REST API using FastAPI for mobile app integration.
  - Add User Authentication to save diagnosis history.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.