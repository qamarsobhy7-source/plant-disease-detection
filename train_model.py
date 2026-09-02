import os
import json
import shutil
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam


# ============================================================
# Configuration
# ============================================================

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10

RANDOM_SEED = 123

DATA_DIR = "dataset"

MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Reproducibility
tf.keras.utils.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print("=" * 70)
print("Plant Disease Detection - Model Training")
print("=" * 70)


# ============================================================
# Validate Dataset
# ============================================================

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(
        f"Dataset directory was not found: {DATA_DIR}"
    )


# ============================================================
# Load Dataset
# ============================================================

print("\nLoading dataset...")

full_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=RANDOM_SEED,
)

class_names = full_ds.class_names
num_classes = len(class_names)

print("\nClasses:")
for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")

print(f"\nNumber of classes: {num_classes}")


# Save class names for the application
with open(
    os.path.join(MODELS_DIR, "class_names.json"),
    "w",
    encoding="utf-8",
) as file:
    json.dump(class_names, file, ensure_ascii=False, indent=4)


# ============================================================
# Create Train / Validation / Test Split
# ============================================================

dataset_size = tf.data.experimental.cardinality(full_ds).numpy()

if dataset_size <= 2:
    raise ValueError(
        "Dataset is too small to create train/validation/test splits."
    )

train_size = int(dataset_size * 0.80)
val_size = int(dataset_size * 0.10)

test_size = dataset_size - train_size - val_size

print("\nDataset split:")
print(f"Train batches: {train_size}")
print(f"Validation batches: {val_size}")
print(f"Test batches: {test_size}")

full_ds = full_ds.shuffle(
    dataset_size,
    seed=RANDOM_SEED,
    reshuffle_each_iteration=False,
)

train_ds = full_ds.take(train_size)

remaining_ds = full_ds.skip(train_size)

val_ds = remaining_ds.take(val_size)

test_ds = remaining_ds.skip(val_size)


# ============================================================
# Data Augmentation
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(
            height_factor=0.1,
            width_factor=0.1,
        ),
        layers.RandomContrast(0.1),
    ],
    name="data_augmentation",
)


# ============================================================
# Dataset Performance
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

# Cache validation and test datasets.
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)


# ============================================================
# Custom CNN Dataset
# ============================================================

def prepare_custom_cnn(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    return image, label


def prepare_custom_train(image, label):
    image = tf.cast(image, tf.float32)

    image = data_augmentation(
        image,
        training=True,
    )

    image = image / 255.0

    return image, label


custom_train_ds = train_ds.map(
    prepare_custom_train,
    num_parallel_calls=AUTOTUNE,
)

custom_val_ds = val_ds.map(
    prepare_custom_cnn,
    num_parallel_calls=AUTOTUNE,
)

custom_test_ds = test_ds.map(
    prepare_custom_cnn,
    num_parallel_calls=AUTOTUNE,
)

custom_train_ds = custom_train_ds.prefetch(AUTOTUNE)


# ============================================================
# MobileNetV2 Dataset
# ============================================================

def prepare_mobilenet_train(image, label):
    image = tf.cast(image, tf.float32)

    image = data_augmentation(
        image,
        training=True,
    )

    # MobileNetV2 preprocessing:
    # [0, 255] -> [-1, 1]
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label


def prepare_mobilenet_eval(image, label):
    image = tf.cast(image, tf.float32)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label


mobilenet_train_ds = train_ds.map(
    prepare_mobilenet_train,
    num_parallel_calls=AUTOTUNE,
)

mobilenet_val_ds = val_ds.map(
    prepare_mobilenet_eval,
    num_parallel_calls=AUTOTUNE,
)

mobilenet_test_ds = test_ds.map(
    prepare_mobilenet_eval,
    num_parallel_calls=AUTOTUNE,
)

mobilenet_train_ds = mobilenet_train_ds.prefetch(AUTOTUNE)
mobilenet_val_ds = mobilenet_val_ds.prefetch(AUTOTUNE)
mobilenet_test_ds = mobilenet_test_ds.prefetch(AUTOTUNE)


# ============================================================
# EfficientNet Dataset
# ============================================================

def prepare_efficientnet_train(image, label):
    image = tf.cast(image, tf.float32)

    image = data_augmentation(
        image,
        training=True,
    )

    return image, label


def prepare_efficientnet_eval(image, label):
    image = tf.cast(image, tf.float32)

    return image, label


efficientnet_train_ds = train_ds.map(
    prepare_efficientnet_train,
    num_parallel_calls=AUTOTUNE,
)

efficientnet_val_ds = val_ds.map(
    prepare_efficientnet_eval,
    num_parallel_calls=AUTOTUNE,
)

efficientnet_test_ds = test_ds.map(
    prepare_efficientnet_eval,
    num_parallel_calls=AUTOTUNE,
)

efficientnet_train_ds = efficientnet_train_ds.prefetch(AUTOTUNE)
efficientnet_val_ds = efficientnet_val_ds.prefetch(AUTOTUNE)
efficientnet_test_ds = efficientnet_test_ds.prefetch(AUTOTUNE)


# ============================================================
# Callbacks
# ============================================================

def get_callbacks(model_path):

    checkpoint = ModelCheckpoint(
        model_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    )

    return [
        checkpoint,
        early_stopping,
        reduce_lr,
    ]


# ============================================================
# Custom CNN
# ============================================================

print("\n" + "=" * 70)
print("Training Custom CNN")
print("=" * 70)

custom_cnn = models.Sequential(
    [
        layers.Input(
            shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        ),

        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu",
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu",
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu",
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            256,
            (3, 3),
            padding="same",
            activation="relu",
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),

        layers.Dense(
            128,
            activation="relu",
        ),

        layers.Dropout(0.5),

        layers.Dense(
            num_classes,
            activation="softmax",
        ),
    ],
    name="CustomCNN",
)

custom_cnn.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

custom_cnn.summary()

custom_cnn_path = os.path.join(
    MODELS_DIR,
    "custom_cnn.keras",
)

history_cnn = custom_cnn.fit(
    custom_train_ds,
    validation_data=custom_val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=get_callbacks(custom_cnn_path),
)


# ============================================================
# MobileNetV2
# ============================================================

print("\n" + "=" * 70)
print("Training MobileNetV2")
print("=" * 70)

base_mobilenet = MobileNetV2(
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3,
    ),
    include_top=False,
    weights="imagenet",
)

base_mobilenet.trainable = False

mobilenet_inputs = layers.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3,
    )
)

x = base_mobilenet(
    mobilenet_inputs,
    training=False,
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    128,
    activation="relu",
)(x)

x = layers.Dropout(0.4)(x)

mobilenet_outputs = layers.Dense(
    num_classes,
    activation="softmax",
)(x)

mobilenet_model = Model(
    mobilenet_inputs,
    mobilenet_outputs,
    name="MobileNetV2",
)

mobilenet_model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

mobilenet_model.summary()

mobilenet_path = os.path.join(
    MODELS_DIR,
    "mobilenetv2.keras",
)

history_mobilenet = mobilenet_model.fit(
    mobilenet_train_ds,
    validation_data=mobilenet_val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=get_callbacks(mobilenet_path),
)


# ============================================================
# MobileNetV2 Fine-Tuning
# ============================================================

print("\n" + "=" * 70)
print("Fine-tuning MobileNetV2")
print("=" * 70)

base_mobilenet.trainable = True

# Freeze most layers and fine-tune only the last layers
for layer in base_mobilenet.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen
for layer in base_mobilenet.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


mobilenet_model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_mobilenet_fine = mobilenet_model.fit(
    mobilenet_train_ds,
    validation_data=mobilenet_val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=get_callbacks(mobilenet_path),
)


# ============================================================
# EfficientNetB0
# ============================================================

print("\n" + "=" * 70)
print("Training EfficientNetB0")
print("=" * 70)

base_efficientnet = EfficientNetB0(
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3,
    ),
    include_top=False,
    weights="imagenet",
)

base_efficientnet.trainable = False

efficientnet_inputs = layers.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3,
    )
)

x = base_efficientnet(
    efficientnet_inputs,
    training=False,
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    128,
    activation="relu",
)(x)

x = layers.Dropout(0.4)(x)

efficientnet_outputs = layers.Dense(
    num_classes,
    activation="softmax",
)(x)

efficientnet_model = Model(
    efficientnet_inputs,
    efficientnet_outputs,
    name="EfficientNetB0",
)

efficientnet_model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

efficientnet_model.summary()

efficientnet_path = os.path.join(
    MODELS_DIR,
    "efficientnetb0.keras",
)

history_efficientnet = efficientnet_model.fit(
    efficientnet_train_ds,
    validation_data=efficientnet_val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=get_callbacks(efficientnet_path),
)


# ============================================================
# EfficientNetB0 Fine-Tuning
# ============================================================

print("\n" + "=" * 70)
print("Fine-tuning EfficientNetB0")
print("=" * 70)

base_efficientnet.trainable = True

for layer in base_efficientnet.layers[:-30]:
    layer.trainable = False

for layer in base_efficientnet.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


efficientnet_model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_efficientnet_fine = efficientnet_model.fit(
    efficientnet_train_ds,
    validation_data=efficientnet_val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=get_callbacks(efficientnet_path),
)


# ============================================================
# Evaluation Function
# ============================================================

def evaluate_model(
    model_name,
    model_path,
    test_dataset,
):
    print("\n" + "=" * 70)
    print(f"Evaluating {model_name}")
    print("=" * 70)

    model = tf.keras.models.load_model(model_path)

    loss, accuracy = model.evaluate(
        test_dataset,
        verbose=0,
    )

    y_true = []
    y_pred = []

    for images, labels in test_dataset:

        predictions = model.predict(
            images,
            verbose=0,
        )

        predictions = np.argmax(
            predictions,
            axis=1,
        )

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            predictions
        )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print(f"\nTest Accuracy : {accuracy * 100:.2f}%")
    print(f"Test Loss     : {loss:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1 Score      : {f1:.4f}")

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(report)

    report_path = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower().replace(' ', '_')}_classification_report.txt",
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    # Confusion Matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(cm)

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(
        range(num_classes),
        class_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(num_classes),
        class_names,
    )

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()

    confusion_path = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png",
    )

    plt.savefig(
        confusion_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return {
        "model": model_name,
        "accuracy": float(accuracy),
        "loss": float(loss),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "path": model_path,
    }


# ============================================================
# Evaluate All Models
# ============================================================

results = []

results.append(
    evaluate_model(
        "Custom CNN",
        custom_cnn_path,
        custom_test_ds,
    )
)

results.append(
    evaluate_model(
        "MobileNetV2",
        mobilenet_path,
        mobilenet_test_ds,
    )
)

results.append(
    evaluate_model(
        "EfficientNetB0",
        efficientnet_path,
        efficientnet_test_ds,
    )
)


# ============================================================
# Compare Models
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for result in results:
    print(
        f"{result['model']:<20}"
        f"Accuracy: {result['accuracy'] * 100:>7.2f}% | "
        f"Precision: {result['precision']:>6.4f} | "
        f"Recall: {result['recall']:>6.4f} | "
        f"F1: {result['f1_score']:>6.4f}"
    )


# ============================================================
# Save Results
# ============================================================

results_path = os.path.join(
    RESULTS_DIR,
    "model_comparison.json",
)

with open(
    results_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        results,
        file,
        indent=4,
    )


# ============================================================
# Select Best Model
# ============================================================

best_result = max(
    results,
    key=lambda x: x["f1_score"],
)

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"Best Model: {best_result['model']}"
)

print(
    f"Accuracy: {best_result['accuracy'] * 100:.2f}%"
)

print(
    f"F1 Score: {best_result['f1_score']:.4f}"
)


# ============================================================
# Copy Best Model
# ============================================================

best_model_destination = os.path.join(
    MODELS_DIR,
    "best_model.keras",
)

shutil.copy2(
    best_result["path"],
    best_model_destination,
)

print(
    f"\nBest model saved to:"
    f" {best_model_destination}"
)


# ============================================================
# Plot Training History
# ============================================================

def plot_history(
    history,
    model_name,
):
    history_data = history.history

    # Accuracy
    plt.figure(figsize=(8, 5))

    plt.plot(
        history_data["accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        history_data["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.title(
        f"{model_name} - Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    accuracy_path = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower().replace(' ', '_')}_accuracy.png",
    )

    plt.savefig(
        accuracy_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))

    plt.plot(
        history_data["loss"],
        label="Training Loss",
    )

    plt.plot(
        history_data["val_loss"],
      
