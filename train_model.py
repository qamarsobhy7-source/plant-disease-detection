import os
import json
import random
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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

DATA_DIR = Path("dataset")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

TRAINING_HISTORY_DIR = RESULTS_DIR / "training_history"

# The class order is fixed intentionally.
# The dataset folder names must match these names exactly.
CLASS_NAMES = [
    "Healthy Plant",
    "Early Blight",
    "Late Blight",
    "Powdery Mildew",
    "Leaf Spot",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
}


# ============================================================
# Reproducibility
# ============================================================

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# Directory Setup
# ============================================================

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utility Functions
# ============================================================

def save_json(data, path):
    """Save a Python object as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def load_image(path, label):
    """
    Load an image from disk, decode it as RGB,
    resize it to IMG_SIZE, and return it with its label.

    Preprocessing such as Rescaling or application-specific
    preprocessing is handled inside each model.
    """
    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMG_SIZE)

    image = tf.cast(image, tf.float32)

    label = tf.cast(label, tf.int32)

    return image, label


def create_dataset(paths, labels, training=False):
    """Create a tf.data.Dataset from image paths and integer labels."""

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            np.asarray(paths, dtype=str),
            np.asarray(labels, dtype=np.int32),
        )
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def collect_dataset():
    """
    Collect all image paths according to the fixed CLASS_NAMES order.
    """

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory was not found: {DATA_DIR}"
        )

    all_paths = []
    all_labels = []

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_dir = DATA_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Required class directory was not found: {class_dir}"
            )

        class_images = sorted(
            [
                path
                for path in class_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )

        if len(class_images) == 0:
            raise ValueError(
                f"No supported images were found in: {class_dir}"
            )

        print(
            f"{class_name}: {len(class_images)} images"
        )

        all_paths.extend(
            [str(path) for path in class_images]
        )

        all_labels.extend(
            [class_index] * len(class_images)
        )

    return all_paths, all_labels


def validate_dataset_distribution(labels):
    """Validate that every class has enough samples."""

    counts = np.bincount(
        labels,
        minlength=len(CLASS_NAMES),
    )

    print("\nClass distribution:")

    for class_name, count in zip(CLASS_NAMES, counts):
        print(f"  {class_name}: {count}")

    if np.any(counts < 10):
        raise ValueError(
            "Each class must contain at least 10 images "
            "to create a reliable stratified 80/10/10 split."
        )


def create_stratified_split(paths, labels):
    """
    Create deterministic 80/10/10 train/validation/test split.
    """

    train_paths, temp_paths, train_labels, temp_labels = (
        train_test_split(
            paths,
            labels,
            test_size=0.20,
            random_state=RANDOM_SEED,
            stratify=labels,
        )
    )

    validation_paths, test_paths, validation_labels, test_labels = (
        train_test_split(
            temp_paths,
            temp_labels,
            test_size=0.50,
            random_state=RANDOM_SEED,
            stratify=temp_labels,
        )
    )

    return (
        train_paths,
        train_labels,
        validation_paths,
        validation_labels,
        test_paths,
        test_labels,
    )


def save_split_metadata(
    train_paths,
    train_labels,
    validation_paths,
    validation_labels,
    test_paths,
    test_labels,
):
    """
    Save the exact split so evaluate_model.py can evaluate
    exactly the same test images used during training.
    """

    def relative_paths(paths):
        return [
            os.path.relpath(path, DATA_DIR)
            for path in paths
        ]

    metadata = {
        "random_seed": RANDOM_SEED,
        "image_size": list(IMG_SIZE),
        "class_names": CLASS_NAMES,
        "split_ratio": {
            "train": 0.80,
            "validation": 0.10,
            "test": 0.10,
        },
        "train": {
            "paths": relative_paths(train_paths),
            "labels": [int(x) for x in train_labels],
        },
        "validation": {
            "paths": relative_paths(validation_paths),
            "labels": [int(x) for x in validation_labels],
        },
        "test": {
            "paths": relative_paths(test_paths),
            "labels": [int(x) for x in test_labels],
        },
    }

    save_json(
        metadata,
        RESULTS_DIR / "data_split.json",
    )


def calculate_class_weights(labels):
    """Calculate balanced class weights."""

    counts = np.bincount(
        labels,
        minlength=len(CLASS_NAMES),
    )

    total = len(labels)
    num_classes = len(CLASS_NAMES)

    class_weights = {}

    for class_index, count in enumerate(counts):
        class_weights[class_index] = (
            total / (num_classes * count)
        )

    return class_weights


# ============================================================
# Data Augmentation
# ============================================================

def create_data_augmentation():
    return tf.keras.Sequential(
        [
            layers.RandomFlip(
                "horizontal",
                seed=RANDOM_SEED,
            ),
            layers.RandomRotation(
                0.10,
                seed=RANDOM_SEED,
            ),
            layers.RandomZoom(
                0.10,
                seed=RANDOM_SEED,
            ),
            layers.RandomTranslation(
                0.10,
                0.10,
                seed=RANDOM_SEED,
            ),
            layers.RandomContrast(
                0.10,
                seed=RANDOM_SEED,
            ),
        ],
        name="data_augmentation",
    )


# ============================================================
# Custom CNN
# ============================================================

def build_custom_cnn(num_classes):
    """Build the custom CNN model."""

    inputs = layers.Input(
        shape=(*IMG_SIZE, 3),
        name="input_image",
    )

    x = create_data_augmentation()(inputs)

    x = layers.Rescaling(
        1.0 / 255.0,
        name="rescaling",
    )(x)

    x = layers.Conv2D(
        32,
        3,
        padding="same",
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        64,
        3,
        padding="same",
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        128,
        3,
        padding="same",
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        256,
        3,
        padding="same",
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        128,
        activation="relu",
    )(x)

    x = layers.Dropout(0.50)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="CustomCNN",
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# Transfer Learning Models
# ============================================================

def build_mobilenetv2(num_classes):
    """Build MobileNetV2 transfer-learning model."""

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )

    base_model.trainable = False

    inputs = layers.Input(
        shape=(*IMG_SIZE, 3),
        name="input_image",
    )

    x = create_data_augmentation()(inputs)

    x = layers.Lambda(
        tf.keras.applications.mobilenet_v2.preprocess_input,
        name="mobilenetv2_preprocessing",
    )(x)

    x = base_model(
        x,
        training=False,
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        128,
        activation="relu",
    )(x)

    x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="MobileNetV2",
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def build_efficientnetb0(num_classes):
    """Build EfficientNetB0 transfer-learning model."""

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )

    base_model.trainable = False

    inputs = layers.Input(
        shape=(*IMG_SIZE, 3),
        name="input_image",
    )

    x = create_data_augmentation()(inputs)

    # EfficientNetB0 in this TensorFlow/Keras version
    # contains its preprocessing/rescaling internally.
    x = base_model(
        x,
        training=False,
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        128,
        activation="relu",
    )(x)

    x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="EfficientNetB0",
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


# ============================================================
# Callbacks
# ============================================================

def create_callbacks(checkpoint_path):
    return [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ============================================================
# Training History
# ============================================================

def combine_histories(history_one, history_two=None):
    """Combine history dictionaries from two training phases."""

    combined = {}

    for key, values in history_one.history.items():
        combined[key] = list(values)

    if history_two is not None:
        for key, values in history_two.history.items():
            combined.setdefault(key, [])
            combined[key].extend(values)

    return combined


def save_training_history(model_name, history):
    safe_name = model_name.lower().replace(" ", "_")

    save_json(
        history,
        TRAINING_HISTORY_DIR / f"{safe_name}_history.json",
    )


def plot_training_history(model_name, history):
    safe_name = model_name.lower().replace(" ", "_")

    epochs = range(
        1,
        len(history["loss"]) + 1,
    )

    # Accuracy plot
    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        history["accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        TRAINING_HISTORY_DIR / f"{safe_name}_accuracy.png",
        dpi=150,
    )

    plt.close()

    # Loss plot
    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        history["loss"],
        label="Training Loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        TRAINING_HISTORY_DIR / f"{safe_name}_loss.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# Model Summary
# ============================================================

def save_model_summary(model, model_name):
    summary_path = RESULTS_DIR / "model_summaries.txt"

    with open(
        summary_path,
        "a",
        encoding="utf-8",
    ) as file:

        file.write("\n")
        file.write("=" * 80)
        file.write("\n")
        file.write(f"{model_name}\n")
        file.write("=" * 80)
        file.write("\n")

        model.summary(
            print_fn=lambda line: file.write(line + "\n")
        )


# ============================================================
# Fine Tuning
# ============================================================

def fine_tune_model(
    model,
    base_model,
    model_name,
    train_dataset,
    validation_dataset,
    class_weights,
):
    """
    Fine-tune the last 30 layers of a transfer-learning model.

    BatchNormalization layers remain frozen for stability.
    """

    for layer in base_model.layers:
        layer.trainable = False

    total_layers = len(base_model.layers)

    fine_tune_from = max(
        0,
        total_layers - 30,
    )

    for layer in base_model.layers[fine_tune_from:]:
        if not isinstance(
            layer,
            layers.BatchNormalization,
        ):
            layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_path = (
        MODELS_DIR
        / f"{model_name}_finetune_best.keras"
    )

    callbacks = create_callbacks(
        checkpoint_path
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    return history, checkpoint_path


# ============================================================
# Model Training
# ============================================================

def train_single_model(
    model_name,
    model,
    train_dataset,
    validation_dataset,
    test_dataset,
    class_weights,
    base_model=None,
):
    """
    Train one model, optionally fine-tune it,
    compare initial and fine-tuned checkpoints on validation data,
    and save the globally best version.
    """

    safe_name = model_name.lower().replace(" ", "_")

    initial_checkpoint = (
        MODELS_DIR
        / f"{safe_name}_initial_best.keras"
    )

    initial_callbacks = create_callbacks(
        initial_checkpoint
    )

    print("\n")
    print("=" * 80)
    print(f"Training {model_name}")
    print("=" * 80)

    save_model_summary(
        model,
        model_name,
    )

    initial_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=INITIAL_EPOCHS,
        class_weight=class_weights,
        callbacks=initial_callbacks,
        verbose=1,
    )

    best_validation_accuracy = -1.0
    best_checkpoint = None

    # Evaluate the best initial checkpoint.
    if initial_checkpoint.exists():

        initial_best_model = tf.keras.models.load_model(
            initial_checkpoint
        )

        initial_metrics = initial_best_model.evaluate(
            validation_dataset,
            verbose=0,
        )

        initial_val_accuracy = float(
            initial_metrics[1]
        )

        best_validation_accuracy = initial_val_accuracy
        best_checkpoint = initial_checkpoint

        del initial_best_model

    fine_tune_history = None
    fine_tune_checkpoint = None

    # Fine-tuning is only performed for transfer-learning models.
    if base_model is not None:

        fine_tune_history, fine_tune_checkpoint = (
            fine_tune_model(
                model=model,
                base_model=base_model,
                model_name=safe_name,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                class_weights=class_weights,
            )
        )

        if fine_tune_checkpoint.exists():

            fine_tuned_model = tf.keras.models.load_model(
                fine_tune_checkpoint
            )

            fine_tune_metrics = (
                fine_tuned_model.evaluate(
                    validation_dataset,
                    verbose=0,
                )
            )

            fine_tune_val_accuracy = float(
                fine_tune_metrics[1]
            )

            if fine_tune_val_accuracy > best_validation_accuracy:

                best_validation_accuracy = (
                    fine_tune_val_accuracy
                )

                best_checkpoint = fin
