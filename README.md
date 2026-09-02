Plant Disease Detection System

A deep learning-based computer vision system for classifying plant leaf images into predefined health and disease categories.

The project provides a complete machine learning pipeline covering assets preparation, stratified data splitting, model training, transfer learning, fine-tuning, model selection, evaluation, visualization, and interactive image classification through a Streamlit web application.

Project Overview

The system is designed to classify plant leaf images into five classes:

- Healthy Plant
- Early Blight
- Late Blight
- Powdery Mildew
- Leaf Spot

Three deep learning approaches are trained and compared:

1. Custom Convolutional Neural Network (CNN)
2. MobileNetV2 with transfer learning
3. EfficientNetB0 with transfer learning

The final model is selected using validation accuracy. The test set is kept separate and is used only for final evaluation.

Main Features

- Fixed and reproducible 80/10/10 train-validation-test split
- Stratified data splitting
- Fixed class ordering
- Class-weight calculation for imbalanced assetss
- Image augmentation during training
- Custom CNN architecture
- MobileNetV2 transfer learning
- EfficientNetB0 transfer learning
- Fine-tuning of transfer-learning models
- Early stopping
- Learning-rate reduction
- Best-checkpoint selection
- Automatic best-model selection using validation accuracy
- Classification reports
- Confusion matrices
- Accuracy, precision, recall, and F1-score
- Macro and weighted evaluation metrics
- Training history files
- Training accuracy and loss plots
- Model comparison results
- Interactive Streamlit prediction interface
- Top-3 prediction probabilities
- Confidence threshold warning
- Consistent preprocessing between training and inference

Project Structure

plant-disease-detection/
│
├── assets/
│   └── sample_images/
│
├── assets/
│   ├── Healthy Plant/
│   ├── Early Blight/
│   ├── Late Blight/
│   ├── Powdery Mildew/
│   └── Leaf Spot/
│
├── models/
│   ├── best_model.keras
│   ├── custom_cnn.keras
│   ├── mobilenetv2.keras
│   ├── efficientnetb0.keras
│   └── class_names.json
│
├── results/
│   ├── data_split.json
│   ├── class_weights.json
│   ├── model_comparison.json
│   ├── training_metadata.json
│   ├── model_summaries.txt
│   │
│   ├── training_history/
│   │   ├── custom_cnn_history.json
│   │   ├── custom_cnn_accuracy.png
│   │   ├── custom_cnn_loss.png
│   │   ├── mobilenetv2_history.json
│   │   ├── mobilenetv2_accuracy.png
│   │   ├── mobilenetv2_loss.png
│   │   ├── efficientnetb0_history.json
│   │   ├── efficientnetb0_accuracy.png
│   │   └── efficientnetb0_loss.png
│   │
│   └── evaluation/
│       ├── model_evaluation.json
│       ├── model_comparison.csv
│       ├── *_classification_report.txt
│       ├── *_evaluation.json
│       └── *_confusion_matrix.png
│
├── app.py
├── train_model.py
├── evaluate_model.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md

The "assets/", "models/", and "results/" directories are generated or populated according to the project workflow. The local assets itself is excluded from Git through ".gitignore".

Dataset Organization

The assets must be organized into one directory per class:

assets/
├── Healthy Plant/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
│
├── Early Blight/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
│
├── Late Blight/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
│
├── Powdery Mildew/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
│
└── Leaf Spot/
    ├── image_001.jpg
    ├── image_002.jpg
    └── ...

The directory names must match the five class names exactly because the training pipeline uses a fixed class order.

The supported image formats are:

- JPG
- JPEG
- PNG
- BMP
- GIF

The assets is not included in this repository.

Data Splitting

The project uses a deterministic stratified split:

Dataset Portion| Percentage
Training| 80%
Validation| 10%
Test| 10%

The split is created using a fixed random seed.

The exact paths and labels belonging to each split are stored in:

results/data_split.json

This prevents the evaluation script from creating a different test set.

Image Processing

All images are resized to:

128 × 128

The raw image is passed to each model, while model-specific preprocessing is embedded in the model architecture.

Custom CNN

The Custom CNN contains:

- Data augmentation
- Rescaling from "[0, 255]" to "[0, 1]"
- Four convolutional blocks
- Batch normalization
- Max pooling
- Global average pooling
- Dense classification layer
- Dropout
- Softmax output

MobileNetV2

MobileNetV2 uses ImageNet pretrained weights.

The pipeline contains:

- Data augmentation
- MobileNetV2 preprocessing
- Frozen feature extractor during initial training
- Global average pooling
- Dense classification layer
- Dropout
- Fine-tuning of the final portion of the feature extractor

Batch normalization layers remain frozen during fine-tuning for training stability.

EfficientNetB0

EfficientNetB0 uses ImageNet pretrained weights.

The pipeline contains:

- Data augmentation
- EfficientNetB0 built-in preprocessing
- Frozen feature extractor during initial training
- Global average pooling
- Dense classification layer
- Dropout
- Fine-tuning of the final portion of the feature extractor

Training Pipeline

The complete training process is implemented in:

train_model.py

Run:

python train_model.py

The script performs the following operations:

1. Validates the assets.
2. Checks all required classes.
3. Collects supported image files.
4. Creates a stratified 80/10/10 split.
5. Saves the exact split metadata.
6. Calculates class weights.
7. Saves the class mapping.
8. Creates TensorFlow assetss.
9. Trains the Custom CNN.
10. Trains MobileNetV2.
11. Fine-tunes MobileNetV2.
12. Trains EfficientNetB0.
13. Fine-tunes EfficientNetB0.
14. Compares model performance on the validation set.
15. Selects the best model using validation accuracy.
16. Saves the selected model as "best_model.keras".
17. Saves model-specific training histories.
18. Generates training plots.
19. Saves model comparison metadata.

Model Selection

The official best model is selected using:

Validation Accuracy

The test set is not used to select the best model.

This separation helps avoid test-set leakage.

The selected model is saved as:

models/best_model.keras

The individual models are also preserved:

models/custom_cnn.keras
models/mobilenetv2.keras
models/efficientnetb0.keras

Evaluation

Final model evaluation is performed using:

evaluate_model.py

Run:

python evaluate_model.py

The script loads the exact test split saved by the training pipeline.

The following metrics are calculated:

- Test Loss
- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1 Score
- Macro Precision
- Macro Recall
- Macro F1 Score
- Per-class precision
- Per-class recall
- Per-class F1 score
- Confusion matrix

Evaluation results are stored under:

results/evaluation/

Examples include:

model_evaluation.json
model_comparison.csv
custom_cnn_classification_report.txt
mobilenetv2_classification_report.txt
efficientnetb0_classification_report.txt
best_model_classification_report.txt

Confusion matrices are also generated for each evaluated model.

Important Evaluation Methodology

The project separates model selection from final evaluation.

During training:

Training Set
    ↓
Model Training
    ↓
Validation Set
    ↓
Model Selection

After the model has been selected:

Selected Model
    ↓
Test Set
    ↓
Final Evaluation

This prevents the test set from influencing model selection.

Training Outputs

After successful training, the following files are expected:

models/
├── best_model.keras
├── custom_cnn.keras
├── mobilenetv2.keras
├── efficientnetb0.keras
└── class_names.json

The results directory contains:

results/
├── data_split.json
├── class_weights.json
├── model_comparison.json
├── training_metadata.json
├── model_summaries.txt
├── training_history/
└── evaluation/

Installation

Create a Python virtual environment using a Python version compatible with TensorFlow 2.15.

Example:

python -m venv .venv

Activate the environment.

Windows

.venv\Scripts\activate

Linux/macOS

source .venv/bin/activate

Install the dependencies:

pip install -r requirements.txt

Requirements

The project uses:

- Python 3.9–3.11
- TensorFlow 2.15.0
- NumPy 1.26.4
- Scikit-learn 1.4.2
- Matplotlib 3.8.4
- Streamlit 1.32.0
- Pillow 10.2.0

The MobileNetV2 and EfficientNetB0 ImageNet weights may need to be downloaded during the first training run.

Running the Complete Pipeline

Step 1 — Prepare the Dataset

Place the images inside the required class directories:

assets/
├── Healthy Plant/
├── Early Blight/
├── Late Blight/
├── Powdery Mildew/
└── Leaf Spot/

Step 2 — Train the Models

python train_model.py

Wait until the training process completes.

Step 3 — Evaluate the Models

python evaluate_model.py

Step 4 — Start the Web Application

streamlit run app.py

The Streamlit application will load the trained models from:

models/

Web Application

The application provides:

- Model selection
- Image upload
- Image preview
- Predicted class
- Prediction confidence
- Top-3 predictions
- Probability for every supported class
- Confidence warning for predictions below 60%

The application supports:

.jpg
.jpeg
.png

Confidence Threshold

The application uses a 60% confidence threshold as an interface warning.

This does not mean that a prediction above 60% is guaranteed to be correct.

Model confidence should not be interpreted as a professional agricultural diagnosis.

Reproducibility

The project uses a fixed random seed:

123

The seed is applied to:

- Python random
- NumPy
- TensorFlow
- Dataset splitting

The exact train, validation, and test image lists are also stored in:

results/data_split.json

Limitations

The system is an image-classification model and its performance depends on factors such as:

- Dataset quality
- Dataset size
- Class balance
- Image quality
- Lighting conditions
- Background variation
- Leaf orientation
- Similarity between disease categories
- Distribution differences between training images and real-world images

Predictions should therefore be considered model outputs rather than definitive agricultural diagnoses.

Recommended Improvements

Possible future improvements include:

- Increasing assets size and diversity
- Collecting field images under real-world conditions
- External validation on an independent assets
- Duplicate-image and near-duplicate detection
- Advanced augmentation strategies
- Hyperparameter optimization
- Model explainability using Grad-CAM
- Confidence calibration
- Experiment tracking
- Deployment using a production inference service

License

This project is released under the MIT License.

See:

LICENSE

for the complete license text.
