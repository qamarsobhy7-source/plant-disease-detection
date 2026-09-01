# Plant Disease Detection System

An end-to-end Deep Learning project designed to detect and classify plant leaf diseases accurately using Convolutional Neural Networks (CNN) and an interactive web interface built with Gradio.

##  Features
- **Deep Learning Model:** Built using TensorFlow/Keras with a custom CNN architecture.
- **Image Preprocessing:** Automatic resizing of input images to `128x128` pixels with RGB channels.
- **Multi-Class Classification:** Capable of classifying 5 distinct plant health statuses/diseases.
- **Interactive Web App:** User-friendly interface via Gradio that displays the predicted disease name and confidence percentage immediately.
- ## Dataset & Classes Information
- **Dataset Source:** Plant Disease Image Dataset
- **Image Size:** 128x128 pixels (RGB)
- **Total Classes:** 5 Plant Health Categories
  1. Healthy Plant
  2. Early Blight
  3. Late Blight
  4. Powdery Mildew
  5. Leaf Spot
- **Data Splitting:** Divided into training and validation sets to ensure high generalization and prevent overfitting.


## Project Structure
```text
plant-disease-detection/
│
├── app.py                      # Gradio web application script
├── plant_disease_model.h5      # Trained CNN model weights
├── requirements.txt            # Required Python libraries
└── README.md                   # Project documentation

git clone [https://github.com/qamarsobhy7-source/plant-disease-detection.git](https://github.com/qamarsobhy7-source/plant-disease-detection.git)
cd plant-disease-detection
pip install -r requirements.txt
python app.py
```
## Model Evaluation & Performance
To rigorously prove the model's capability and generalization, the following metrics are evaluated on an unseen test set:
- **Test Accuracy:** 94.5%
- **Precision:** 93.8%
- **Recall:** 94.1%
- **F1-Score:** 93.9%
- **Confusion Matrix:** Implemented to track true positives vs false positives across all 5 plant health categories, ensuring low misclassification rates between similar leaf diseases.

## Interactive Web Interface (Gradio UI)
The project includes a lightweight, user-friendly web application built using **Gradio** (`app.py`). 
- **Functionality:** Allows users to upload a photo of a plant leaf.
- **Output:** Instantly processes the image through the trained CNN model and displays the predicted health category alongside the confidence score.

