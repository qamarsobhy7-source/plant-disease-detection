import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

# ==========================================
# 1. Configuration & Model/Data Loading
# ==========================================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
DATA_DIR = "dataset"
MODEL_PATH = "plant_disease_model.h5"

print("--- Loading data and preparing test set ---")

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = val_ds.class_names

val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 2)

normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Please train the model first.")

# ==========================================
# 2. Extract Predictions and True Labels
# ==========================================
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    preds_classes = np.argmax(preds, axis=1)
    
    y_true.extend(labels.numpy())
    y_pred.extend(preds_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ==========================================
# 3. Calculate and Display Metrics
# ==========================================
print("\n" + "="*40)
print(" Classification Report ")
print("="*40)

report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

print("\n" + "="*40)
print(" Confusion Matrix ")
print("="*40)

conf_matrix = confusion_matrix(y_true, y_pred)
print(conf_matrix)

with open("evaluation_results.txt", "w", encoding="utf-8") as f:
    f.write("=== Classification Report ===\n")
    f.write(report + "\n\n")
    f.write("=== Confusion Matrix ===\n")
    f.write(str(conf_matrix))

print("\nEvaluation results successfully saved to: evaluation_results.txt")
