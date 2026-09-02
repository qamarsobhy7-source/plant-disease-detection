import os
import json
import random

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

DATA_DIR = "dataset"
MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

tf.keras.utils.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


print("=" * 70)
print("Plant Disease Detection - Model Training")
print("=" * 70)


# ============================================================
# Validate Dataset
# ============================================================

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"Dataset directory was not found: {DATA_DIR}"
    )


# ============================================================
# Supported Image Extensions
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
}


# ============================================================
# Collect Image Paths
# ============================================================

print("\nCollecting dataset images...")

class_names = sorted(
    [
        directory
        for directory in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, directory))
        and not directory.startswith(".")
    ]
)

if len(class_names) < 2:
    raise ValueError(
        "The dataset must contain at least two class directories."
    )


class_to_index = {
    class_name: index
    for index, class_name in enumerate(class_names)
}


image_paths = []
labels = []


for class_name in class_names:

    class_directory = os.path.join(
        DATA_DIR,
        class_name,
    )

    class_image_count = 0

    for root, _, files in os.walk(class_directory):

        for filename in files:

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension in IMAGE_EXTENSIONS:

                image_path = os.path.join(
                    root,
                    filename,
                )

                image_paths.append(image_path)

                labels.append(
                    class_to_index[class_name]
                )

                class_image_count += 1

    print(
        f"{class_name}: {class_image_count} images"
    )


image_paths = np.array(image_paths)
labels = np.array(labels)


if len(image_paths) == 0:
    raise ValueError(
        "No supported image files were found in the dataset."
    )


if len(set(labels.tolist())) < 2:
    raise ValueError(
        "At least two classes are required."
    )


num_classes = len(class_names)


print("\nClasses:")

for index, class_name in enumerate(class_names):
    print(
        f"{index}: {class_name}"
    )


print(
    f"\nTotal images: {len(image_paths)}"
)

print(
    f"Number of classes: {num_classes}"
)


# ============================================================
# Save Class Names
# ============================================================

class_names_path = os.path.join(
    MODELS_DIR,
    "class_names.json",
)

with open(
    class_names_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        class_names,
        file,
        ensure_ascii=False,
        indent=4,
    )


# ============================================================
# Dataset Split
# ============================================================

print("\nCreating stratified dataset split...")

# 80% train / 20% temporary
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=labels,
)


# Split remaining 20% into:
# 10% validation / 10% test
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_labels,
)


print("\nDataset split:")

print(
    f"Training images   : {len(train_paths)} "
    f"({len(train_paths) / len(image_paths) * 100:.1f}%)"
)

print(
    f"Validation images : {len(val_paths)} "
    f"({len(val_paths) / len(image_paths) * 100:.1f}%)"
)

print(
    f"Testing images    : {len(test_paths)} "
    f"({len(test_paths) / len(image_paths) * 100:.1f}%)"
)


# ============================================================
# Create TensorFlow Dataset
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image.set_shape(
        [
            None,
            None,
            3,
        ]
    )

    image = tf.image.resize(
        image,
        IMG_SIZE,
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    return image, label


def create_dataset(
    paths,
    labels,
    shuffle=False,
):

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels,
        )
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE,
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    return dataset


train_ds = create_dataset(
    train_paths,
    train_labels,
    shuffle=True,
)

val_ds = create_dataset(
    val_paths,
    val_labels,
    shuffle=False,
)

test_ds = create_dataset(
    test_paths,
    test_labels,
    shuffle=False,
)


# ============================================================
# Class Weights
# ============================================================

print("\nCalculating class weights...")

class_counts = np.bincount(
    train_labels,
    minlength=num_classes,
)

total_training_samples = len(
    train_labels
)


class_weights = {}

for class_index in range(num_classes):

    if class_counts[class_index] == 0:
        class_weights[class_index] = 1.0

    else:
        class_weights[class_index] = (
            total_training_samples
            / (
                num_classes
                * class_counts[class_index]
            )
        )


for class_index, weight in class_weights.items():

    print(
        f"{class_names[class_index]}: "
        f"{weight:.4f}"
    )


# ============================================================
# Data Augmentation
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.10
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomTranslation(
            height_factor=0.10,
            width_factor=0.10,
        ),

        layers.RandomContrast(
            0.10
        ),
    ],
    name="data_augmentation",
)


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
# Helper: Combine Histories
# ============================================================

def combine_histories(
    history_one,
    history_two=None,
):

    combined = {}

    for key, values in history_one.history.items():
        combined[key] = list(values)

    if history_two is not None:

        for key, values in history_two.history.items():

            if key not in combined:
                combined[key] = []

            combined[key].extend(
                list(values)
            )

    return combined


# ============================================================
# Custom CNN
# ============================================================

print("\n" + "=" * 70)
print("Training Custom CNN")
print("=" * 70)


custom_cnn = models.Sequential(
    [

        layers.Input(
            shape=(
                IMG_SIZE[0],
                IMG_SIZE[1],
                3,
            )
        ),

        data_augmentation,

        layers.Rescaling(
            1.0 / 255.0
        ),

        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu",
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu",
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu",
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Conv2D(
            256,
            (3, 3),
            padding="same",
            activation="relu",
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.GlobalAveragePooling2D(),

        layers.Dense(
            128,
            activation="relu",
        ),

        layers.Dropout(
            0.5
        ),

        layers.Dense(
            num_classes,
            activation="softmax",
        ),
    ],
    name="CustomCNN",
)


custom_cnn.compile(
    optimizer=Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


custom_cnn.summary()


custom_cnn_path = os.path.join(
    MODELS_DIR,
    "custom_cnn.keras",
)


history_cnn = custom_cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=get_callbacks(
        custom_cnn_path
    ),
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


x = data_augmentation(
    mobilenet_inputs
)


x = layers.Lambda(
    tf.keras.applications.mobilenet_v2.preprocess_input,
    name="mobilenetv2_preprocessing",
)(x)


x = base_mobilenet(
    x,
    training=False,
)


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dense(
    128,
    activation="relu",
)(x)


x = layers.Dropout(
    0.4
)(x)


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
    optimizer=Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


mobilenet_model.summary()


mobilenet_path = os.path.join(
    MODELS_DIR,
    "mobilenetv2.keras",
)


history_mobilenet = mobilenet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=get_callbacks(
        mobilenet_path
    ),
)


# ============================================================
# MobileNetV2 Fine-Tuning
# ============================================================

print("\n" + "=" * 70)
print("Fine-tuning MobileNetV2")
print("=" * 70)


base_mobilenet.trainable = True


for layer in base_mobilenet.layers[:-30]:
    layer.trainable = False


for layer in base_mobilenet.layers:

    if isinstance(
        layer,
        layers.BatchNormalization,
    ):
        layer.trainable = False


mobilenet_model.compile(
    optimizer=Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


history_mobilenet_fine = mobilenet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=get_callbacks(
        mobilenet_path
    ),
)


combined_mobilenet_history = combine_histories(
    history_mobilenet,
    history_mobilenet_fine,
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


x = data_augmentation(
    efficientnet_inputs
)


x = base_efficientnet(
    x,
    training=False,
)


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dense(
    128,
    activation="relu",
)(x)


x = layers.Dropout(
    0.4
)(x)


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
    optimizer=Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


efficientnet_model.summary()


efficientnet_path = os.path.join(
    MODELS_DIR,
    "efficientnetb0.keras",
)


history_efficientnet = efficientnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=get_callbacks(
        efficientnet_path
    ),
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

    if isinstance(
        layer,
        layers.BatchNormalization,
    ):
        layer.trainable = False


efficientnet_model.compile(
    optimizer=Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


history_efficientnet_fine = efficientnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=get_callbacks(
        efficientnet_path
    ),
)


combined_efficientnet_history = combine_histories(
    history_efficientnet,
    history_efficientnet_fine,
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
    print(
        f"Evaluating {model_name}"
    )
    print("=" * 70)


    model = tf.keras.models.load_model(
        model_path
    )


    loss, accuracy = model.evaluate(
        test_dataset,
        verbose=0,
    )


    y_true = []
    y_pred = []


    for images, labels_batch in test_dataset:

        predictions = model.predict(
            images,
            verbose=0,
        )

        predicted_labels = np.argmax(
            predictions,
            axis=1,
        )

        y_true.extend(
            labels_batch.numpy().tolist()
        )

        y_pred.extend(
            predicted_labels.tolist()
        )


    y_true = np.array(
        y_true
    )

    y_pred = np.array(
        y_pred
    )


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


    print(
        f"\nTest Accuracy : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Test Loss     : "
        f"{loss:.4f}"
    )

    print(
        f"Precision     : "
        f"{precision:.4f}"
    )

    print(
        f"Recall        : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score      : "
        f"{f1:.4f}"
    )


    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )


    print(
        "\nClassification Report:"
    )

    print(report)


    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )


    report_path = os.path.join(
        RESULTS_DIR,
        f"{safe_name}_classification_report.txt",
    )


    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(report)


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
    )


    plt.figure(
        figsize=(8, 6)
    )


    plt.imshow(cm)


    plt.title(
        f"{model_name} - Confusion Matrix"
    )


    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )


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
        f"{safe_name}_confusion_matrix.png",
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
# Validation Accuracy
# ============================================================

def get_validation_accuracy(
    
