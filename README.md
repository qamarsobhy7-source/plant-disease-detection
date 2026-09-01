# Plant Disease Detection System

An end-to-end Deep Learning project designed to detect and classify plant leaf diseases accurately using Convolutional Neural Networks (CNN) and an interactive web interface built with Gradio.

##  Features
- **Deep Learning Model:** Built using TensorFlow/Keras with a custom CNN architecture.
- **Image Preprocessing:** Automatic resizing of input images to `128x128` pixels with RGB channels.
- **Multi-Class Classification:** Capable of classifying 5 distinct plant health statuses/diseases.
- **Interactive Web App:** User-friendly interface via Gradio that displays the predicted disease name and confidence percentage immediately.

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
