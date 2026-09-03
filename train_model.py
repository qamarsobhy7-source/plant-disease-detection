from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from tensorflow.keras.optimizers import Adam

# ============================================================
# Configuration
# ============================================================

SEED = 123
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 5

PROJECT_DIR = Path(__file__).resolve().parent

SPLIT_DIR = PROJECT_DIR / "data" / "split"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Early_Blight",
    "Healthy",
    "Late_Blight",
    "Septoria_Leaf_Spot",
    "Target_Spot",
]

NUM_CLASSES = len(CLASS_NAMES)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# Dataset loading
# ============================================================

def load_dataset(directory, shuffle):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


# ============================================================
# EfficientNetB0 model
# ============================================================

def build_model():
    augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )

    base_model.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3))

    x = augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
    )(x)

    model = models.Model(
        inputs,
        outputs,
        name="EfficientNetB0",
    )

    return model, base_model


# ============================================================
# Training history
# ============================================================

def save_history(history, prefix):
    history_data = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }

    with open(
        RESULTS_DIR / f"{prefix}_history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history_data, file, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy",
    )
    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{prefix} Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f"{prefix}_accuracy.png",
        dpi=150,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        history.history["loss"],
        label="Training Loss",
    )
    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{prefix} Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f"{prefix}_loss.png",
        dpi=150,
    )
    plt.close()


# ============================================================
# Initial training
# ============================================================

def initial_training(model, train_ds, val_ds):
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint = MODELS_DIR / "efficientnetb0_initial_best.keras"

    callbacks = [
        ModelCheckpoint(
            checkpoint,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
    )

    save_history(
        history,
        "efficientnetb0_initial_training",
    )

    return model


# ============================================================
# Fine-tuning
# ============================================================

def fine_tuning(model, base_model, train_ds, val_ds):
    base_model.trainable = True

    freeze_until = int(len(base_model.layers) * 0.80)

    for index, layer in enumerate(base_model.layers):
        if index < freeze_until:
            layer.trainable = False
        elif isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint = MODELS_DIR / "efficientnetb0_finetune_best.keras"

    callbacks = [
        ModelCheckpoint(
            checkpoint,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-8,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks,
    )

    save_history(
        history,
        "efficientnetb0_finetune_training",
    )

    return model


# ============================================================
# Save final model
# ============================================================

def save_final_model(model):
    final_path = MODELS_DIR / "efficientnetb0.keras"

    model.save(final_path)

    print("Final model saved successfully.")
    print(f"Model path: {final_path}")

    return final_path


# ============================================================
# Main pipeline
# ============================================================

def main():
    print("=" * 70)
    print("PLANT DISEASE DETECTION - EfficientNetB0")
    print("=" * 70)

    train_dir = SPLIT_DIR / "train"
    val_dir = SPLIT_DIR / "validation"

    print("\nLoading datasets...")

    train_ds = load_dataset(
        train_dir,
        shuffle=True,
    )

    val_ds = load_dataset(
        val_dir,
        shuffle=False,
    )

    print("\nBuilding model...")

    model, base_model = build_model()

    print("\nLoading best initial model...")

    best_initial = MODELS_DIR / "efficientnetb0_initial_best.keras"

    if best_initial.exists():
        model = tf.keras.models.load_model(best_initial)
        base_model = model.get_layer("efficientnetb0")
    else:
        print("Initial model not found. Running initial training...")
        model = initial_training(
            model,
            train_ds,
            val_ds,
        )

    print("\nFine-tuning...")

    model = fine_tuning(
        model,
        base_model,
        train_ds,
        val_ds,
    )

    print("\nSaving final model...")

    save_final_model(model)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
