# Plant Health Classifier

A lightweight computer vision tool built to inspect crop foliage, spot early signs of crop diseases, and output automated health diagnostics.

## Key Features
* **Image Processing Pipeline:** Standardizes input leaf photos for consistent model ingestion.
* **Core Neural Network:** Utilizes a sequential architecture built in TensorFlow and Keras to isolate visual anomalies.
* **Serialized Weights:** Stores trained parameters in `plant_disease_model.h5` for instant execution and evaluation.

## Tech Stack
* Python
* TensorFlow & Keras
* NumPy & OpenCV
* Google Colab

## Quick Start
Load the model weights directly into your environment for inference:
```python
from tensorflow.keras.models import load_model

model = load_model('plant_disease_model.h5')

