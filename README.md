Plant Disease Detection System

Overview

Plant Disease Detection System is a computer vision project that classifies plant leaf images using a trained deep learning model.

The project includes a TensorFlow/Keras model and a Streamlit application for testing the model through a web interface. Users can upload a leaf image and receive the predicted class along with the model confidence.

The application is intended for educational and experimental use.

---

Features

- Plant leaf image classification.
- Image upload through a Streamlit interface.
- Automatic image resizing before prediction.
- Deep learning model based on TensorFlow/Keras.
- Prediction confidence display.
- Simple web interface for testing the trained model.

---

Dataset

The model was trained on a labeled dataset of plant leaf images.

Dataset Details

Item| Details
Dataset| "[Dataset Name]"
Source| "[Dataset Source]"
Number of Images| "[Number of Images]"
Number of Classes| "[Number of Classes]"
Task| Multi-class image classification

Classes

The model predicts the classes included in the training dataset.

The class order used by the application must be the same as the order used during training. This is required to correctly map the model output to the corresponding class name.

Class 0 → [Class Name]
Class 1 → [Class Name]
Class 2 → [Class Name]
...

Image Processing

Before prediction, uploaded images are resized to the input dimensions expected by the model.

The same preprocessing steps used during training must be applied during prediction to keep the input consistent with the trained model.

---

Model Architecture

The project uses a TensorFlow/Keras image classification model saved in H5 format.

plant_disease_model.h5

The model receives a processed leaf image and produces a probability for each supported class.

The class with the highest probability is selected as the predicted class.

Prediction Flow

Leaf Image
    ↓
Image Preprocessing
    ↓
Model Input
    ↓
TensorFlow/Keras Model
    ↓
Class Probabilities
    ↓
Predicted Class
    ↓
Confidence

---

Training

The model training process follows these steps:

1. Load the labeled plant leaf dataset.
2. Prepare the images and class labels.
3. Split the data into training, validation, and test sets.
4. Resize the images to the required input size.
5. Apply the required image preprocessing.
6. Train the classification model.
7. Monitor the validation performance.
8. Evaluate the trained model using the test set.
9. Save the final model as "plant_disease_model.h5".

Training code or notebooks should be kept separately from the application code.

Recommended structure:

notebooks/
└── plant_disease_training.ipynb

If training is implemented using Python scripts:

src/
├── train.py
├── preprocessing.py
└── evaluate.py

---

Evaluation

The model should be evaluated using images that were not used during training.

The following metrics can be used to measure the classification performance:

Metric| Result
Accuracy| "[XX.XX%]"
Precision| "[XX.XX%]"
Recall| "[XX.XX%]"
F1-Score| "[XX.XX%]"

The values above should be replaced with the actual results from the final test set.

Confusion Matrix

A confusion matrix provides a class-by-class view of the model's predictions and helps identify classes that are frequently confused with each other.

Recommended file:

evaluation/confusion_matrix.png

Training Results

Training and validation accuracy and loss can be saved as figures:

evaluation/
├── accuracy.png
└── loss.png

These results can be used to check how the model performed during training and whether there are signs of overfitting or underfitting.

---

Installation

Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pillow
- Streamlit

The required packages are listed in "requirements.txt".

Setup

Clone the repository:

git clone https://github.com/qamarsobhy7-source/plant-disease-detection.git

Open the project directory:

cd plant-disease-detection

Create a virtual environment:

python -m venv .venv

Activate the environment on Windows:

.venv\Scripts\activate

On macOS/Linux:

source .venv/bin/activate

Install the dependencies:

pip install -r requirements.txt

Make sure the trained model file is available in the location expected by "app.py":

plant_disease_model.h5

---

Usage

Start the application with:

streamlit run app.py

After starting the application:

1. Open the Streamlit page in your browser.
2. Upload a plant leaf image.
3. The image is processed by the application.
4. The trained model generates the prediction.
5. The predicted class and confidence are displayed.

Example

Input:
Plant leaf image

Output:
Predicted Class: [Class Name]
Confidence: [XX.XX%]

---

Project Structure

plant-disease-detection/
│
├── app.py
├── plant_disease_model.h5
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── notebooks/
│   └── plant_disease_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── prediction.py
│   ├── train.py
│   └── evaluate.py
│
├── evaluation/
│   ├── confusion_matrix.png
│   ├── accuracy.png
│   ├── loss.png
│   └── classification_report.txt
│
├── screenshots/
│   ├── home.png
│   └── prediction.png
│
└── tests/
    └── test_prediction.py

Component| Description
"app.py"| Streamlit application
"plant_disease_model.h5"| Trained classification model
"requirements.txt"| Python dependencies
"notebooks/"| Model development and training notebooks
"src/"| Training, preprocessing, prediction, and evaluation code
"evaluation/"| Model performance results
"screenshots/"| Application screenshots
"tests/"| Automated tests

---

Example Predictions

The following format can be used to document predictions from the test images:

Image| Predicted Class| Confidence
Leaf Image 1| "[Class Name]"| "[XX.XX%]"
Leaf Image 2| "[Class Name]"| "[XX.XX%]"
Leaf Image 3| "[Class Name]"| "[XX.XX%]"

Application screenshots can be added to the repository under:

screenshots/
├── home.png
└── prediction.png

For example:

![Application Interface](screenshots/home.png)

![Prediction Result](screenshots/prediction.png)

---

Limitations

- The model can only recognize classes included in its training data.
- Prediction quality depends on image quality and the conditions under which the image was captured.
- Blurry, dark, low-resolution, or heavily obstructed images may produce less reliable results.
- Performance on real field images may differ from performance on the training and test datasets.
- Differences in plant varieties, cameras, lighting, and backgrounds can affect the prediction.
- The model's confidence should not be treated as a guarantee that the prediction is correct.

---

Future Improvements

- Add more plant species and disease classes.
- Increase the size and diversity of the training dataset.
- Include more real-world field images.
- Improve data augmentation.
- Handle class imbalance when necessary.
- Compare different CNN and transfer-learning architectures.
- Add Top-3 predictions.
- Add a confidence threshold for uncertain results.
- Add Grad-CAM for visual model interpretation.
- Add automated tests.
- Improve image validation.
- Improve the Streamlit interface.
- Deploy the application for public use.

---

License

This project is licensed under the MIT License.

The license file is available in the repository as:

LICENSE

Third-party datasets, images, or other external resources used by the project may have their own licenses and terms of use. These terms should be checked separately before redistribution or commercial use.

---

Disclaimer

This project is intended for educational and experimental purposes. The predictions should not be considered a professional agricultural diagnosis. Important decisions regarding plant treatment or crop management should be verified by a qualified agricultural specialist.
