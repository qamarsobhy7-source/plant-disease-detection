# Plant Disease Detection

A deep learning-based computer vision system for classifying tomato leaf images into five disease and health categories using TensorFlow, Keras, and EfficientNetB0.

## Overview

This project implements an end-to-end tomato plant disease classification pipeline:

- Dataset cleaning and duplicate removal
- Leakage-free train/validation/test splitting
- Image preprocessing and augmentation
- Transfer learning using EfficientNetB0
- Partial backbone fine-tuning
- Final test-set evaluation
- Confusion matrix and classification report
- Streamlit web application for image prediction

The project is designed for educational and research purposes.

## Supported Classes

The final model recognizes five classes:

1. `Early_Blight`
2. `Healthy`
3. `Late_Blight`
4. `Septoria_Leaf_Spot`
5. `Target_Spot`

## Dataset

The project uses the tomato disease portion of the PlantVillage dataset.

After duplicate removal:

| Class | Images |
|---|---:|
| Early_Blight | 1000 |
| Healthy | 1585 |
| Late_Blight | 1901 |
| Septoria_Leaf_Spot | 1771 |
| Target_Spot | 1404 |
| **Total** | **7661** |

### Dataset Split

The cleaned dataset was divided using a fixed random seed (`123`):

| Split | Images | Percentage |
|---|---:|---:|
| Training | 5360 | 70% |
| Validation | 765 | 10% |
| Test | 1536 | 20% |
| **Total** | **7661** | **100%** |

A leakage check confirmed that there are no overlapping images between the training, validation, and test sets.

## Model

The final model is **EfficientNetB0** initialized with ImageNet pretrained weights.

### Architecture

- Input: `128x128x3`
- EfficientNetB0 backbone
- Global Average Pooling
- Dense layer: `128` units with ReLU
- Dropout: `0.40`
- Output layer: `5` classes with Softmax

### Training Strategy

Training was performed in two stages:

1. Initial training with the EfficientNetB0 backbone frozen.
2. Partial backbone fine-tuning with Batch Normalization layers kept frozen.

Fine-tuning used:

- Optimizer: Adam
- Learning rate: `1e-5`
- Data augmentation: horizontal flip, rotation, zoom, and contrast
- Best validation weights restored before saving the final model

Best validation accuracy: **92.29%**

## Final Test Results

The final model was evaluated on the held-out test set containing **1536 images**.

| Metric | Score |
|---|---:|
| Accuracy | **92.19%** |
| Weighted Precision | **92.76%** |
| Weighted Recall | **92.19%** |
| Weighted F1 Score | **92.05%** |

The evaluation results and generated reports are stored in:

`results/evaluation/`

Including:

- Classification report
- Confusion matrix
- Evaluation metrics JSON
- Model comparison CSV

## Project Structure

```text
plant-disease-detection/
├── app.py
├── train_model.py
├── evaluate_model.py
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── models/
│   ├── efficientnetb0.keras
│   ├── efficientnetb0_initial_best.keras
│   └── class_names.json
├── results/
│   ├── data_split.json
│   ├── model_summaries.txt
│   └── evaluation/
└── data/
    └── split/
        ├── train/
        ├── validation/
        └── test/
```

The dataset directories are excluded from Git tracking through `.gitignore`.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training

The training pipeline is implemented in `train_model.py`.

Run:

```bash
python train_model.py
```

The script trains EfficientNetB0, performs fine-tuning, saves the best model, and stores training history and plots in `results/`.

## Evaluation

Run the final test evaluation with:

```bash
python evaluate_model.py
```

The evaluation script generates accuracy, precision, recall, F1 score, classification report, confusion matrix, JSON results, and CSV results.

## Streamlit Application

Run the application with:

```bash
streamlit run app.py
```

The application allows users to upload a tomato leaf image and view:

1. Predicted class
2. Prediction confidence
3. Top-3 predictions
4. Class probabilities

The application uses the final model stored at `models/efficientnetb0.keras`.

## Reproducibility

The dataset split uses random seed `123`.

The project records the dataset distribution, train/validation/test split, model architecture, training configuration, training history, evaluation metrics, classification report, and confusion matrix.

## Limitations

- The model is trained on specific tomato leaf disease categories.
- Performance may decrease on real-world field images with different lighting, backgrounds, or camera conditions.
- The model does not cover every possible tomato disease.
- Predictions should not be considered professional agricultural diagnosis.

## Future Improvements

- Testing on real-world field images
- Increasing image resolution
- Exploring additional transfer-learning architectures
- Advanced augmentation and class balancing
- Model calibration
- Grad-CAM explainability
- Production API or mobile deployment

## Disclaimer

This project is intended for educational and research purposes only.

Predictions should not be treated as professional agricultural diagnosis or used as the sole basis for crop-management decisions.

## License

This project is released under the license included in the `LICENSE` file.
