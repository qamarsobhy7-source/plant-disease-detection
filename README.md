Plant Disease Detection System

A deep learning-based computer vision application for classifying plant leaf images into five predefined health and disease categories. The project uses a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras and provides a Streamlit interface for image-based prediction.

Overview

Plant diseases can affect crop quality and productivity, making early identification an important part of crop management.

This project uses image classification to analyze plant leaf images and predict their corresponding condition. The trained CNN receives a preprocessed leaf image and returns the most likely class together with a confidence score.

The project consists of a trained model, a training script, and a Streamlit application for running predictions.

Features

- Classifies plant leaf images into five predefined categories.
- Accepts JPG, JPEG, and PNG images.
- Resizes uploaded images to "128 × 128" pixels before prediction.
- Uses a TensorFlow/Keras CNN model.
- Maps model output indices to readable class names.
- Displays the predicted condition and confidence score.
- Uses Streamlit for the prediction interface.
- Includes sample images for testing the application.

Dataset

The project uses a labeled plant leaf image dataset for a five-class classification task.

Classes

The current application supports the following classes:

1. Healthy Plant
2. Early Blight
3. Late Blight
4. Powdery Mildew
5. Leaf Spot

The class order is defined in "app.py" and must remain consistent with the output order used when training the model.

Data Split

The model training workflow is intended to use separate training, validation, and test data.

If the dataset is split into subsets, the exact split ratios and random seed should be kept consistent between training and evaluation to make the results reproducible.

The full training dataset is not included in this repository.

Model Architecture

The project uses a custom Convolutional Neural Network implemented with TensorFlow/Keras.

Input

128 × 128 × 3

The model expects an RGB image resized to "128 × 128" pixels.

Architecture

Input Image
    ↓
Conv2D (32 filters, 3×3, ReLU)
    ↓
MaxPooling2D
    ↓
Conv2D (64 filters, 3×3, ReLU)
    ↓
MaxPooling2D
    ↓
Conv2D (128 filters, 3×3, ReLU)
    ↓
MaxPooling2D
    ↓
Flatten
    ↓
Dense (128, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (5, Softmax)

The final Softmax layer produces a probability for each of the five supported classes. The class with the highest probability is returned as the prediction.

The trained model is stored in:

plant_disease_model.h5

Training

The model architecture and compilation configuration are defined in "train_model.py".

The current model uses:

Setting| Value
Framework| TensorFlow / Keras
Architecture| Custom CNN
Input Size| "128 × 128 × 3"
Optimizer| Adam
Loss Function| Categorical Crossentropy
Output Classes| 5
Output Activation| Softmax
Dropout| 0.5

The training workflow should use the same image dimensions, class order, and preprocessing assumptions expected by the saved model.

The model definition is available in "train_model.py".

Evaluation

Model evaluation should be performed using a separate test set that was not used to update the model weights.

The recommended metrics for this multi-class classification problem are:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

No fixed performance numbers are reported here unless they are generated from the actual final test set.

This avoids presenting estimated or unverified accuracy values as measured results.

Evaluation Reports

For a complete experiment, the following files can be added to the repository:

evaluation/
├── classification_report.txt
├── confusion_matrix.png
├── accuracy.png
└── loss.png

Installation

Requirements

The application uses Python packages listed in "requirements.txt", including:

- TensorFlow
- Streamlit
- NumPy
- Pillow
- OpenCV
- Matplotlib

The repository currently provides these dependencies through "requirements.txt".

Setup

Clone the repository:

git clone https://github.com/qamarsobhy7-source/plant-disease-detection.git

Move into the project directory:

cd plant-disease-detection

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

On macOS/Linux:

source .venv/bin/activate

Install the required packages:

pip install -r requirements.txt

Usage

The project includes a Streamlit application in "app.py".

Start the application with:

streamlit run app.py

After launching the application:

1. Select a JPG, JPEG, or PNG image of a plant leaf.
2. The uploaded image is displayed in the application.
3. The image is resized to "128 × 128" pixels.
4. The processed image is passed to the trained CNN.
5. The application selects the class with the highest predicted probability.
6. The predicted condition and confidence score are displayed.

The application also includes error handling for image processing and model loading.

Example Predictions

The repository contains sample images under:

assets/sample_images/

These images can be used to test the Streamlit application.

A prediction is displayed in the following format:

Predicted Condition: Early Blight
Confidence Score: 94.25%

The actual class and confidence depend on the image provided to the model.

Project Structure

plant-disease-detection/
│
├── assets/
│   └── sample_images/
│       ├── sample image 1
│       ├── sample image 2
│       └── ...
│
├── app.py
├── train_model.py
├── plant_disease_model.h5
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md

Main Files

File| Purpose
"app.py"| Streamlit application used for image prediction
"train_model.py"| Defines and compiles the CNN architecture
"plant_disease_model.h5"| Saved trained Keras model
"requirements.txt"| Python dependencies
"assets/sample_images/"| Sample images for testing
".gitignore"| Git exclusion rules
"LICENSE"| Project license
"README.md"| Project documentation

Limitations

- The model is limited to the five classes included in its training configuration.
- Images outside these classes may still receive a prediction because the classifier always selects one of its available output classes.
- Prediction quality depends on image quality, lighting, background, camera conditions, and similarity to the training data.
- Performance on real-world field images may differ from performance on the dataset used for training and testing.
- The confidence score represents the model's output probability and does not guarantee that the prediction is correct.
- The system should not be used as the sole basis for agricultural treatment or crop-management decisions.

Future Improvements

- Add a clearly documented dataset source and dataset statistics.
- Add the complete training and evaluation pipeline.
- Generate and publish verified test-set metrics.
- Add a confusion matrix and training history plots.
- Add a confidence threshold for uncertain predictions.
- Add Top-3 predictions.
- Expand the number of plant species and disease classes.
- Include more real-world field images.
- Compare the custom CNN with transfer-learning models using the same test protocol.
- Add Grad-CAM for visual interpretation of model predictions.
- Add automated tests for preprocessing and prediction.
- Improve image validation and user feedback.
- Deploy the application as a public web service.

License

This project is licensed under the MIT License.

See the "LICENSE" file in the repository for the complete license text.

Disclaimer

This project is intended for educational and experimental purposes. Its predictions should not be considered a definitive agricultural diagnosis. For important crop-management or treatment decisions, the result should be reviewed by a qualified agricultural professional.
