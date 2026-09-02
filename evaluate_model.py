import csv
import json
from pathlib import Path

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


# ============================================================
# Configuration
# ============================================================

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

DATA_DIR = Path("assets")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

EVALUATION_DIR = RESULTS_DIR / "evaluation"

SPLIT_FILE = RESULTS_DIR / "data_split.json"
CLASS_NAMES_FILE = MODELS_DIR / "class_names.json"


MODEL_FILES = {
    "custom_cnn": MODELS_DIR / "custom_cnn.keras",
    "mobilenetv2": MODELS_DIR / "mobilenetv2.keras",
    "efficientnetb0": MODELS_DIR / "efficientnetb0.keras",
    "best_model": MODELS_DIR / "best_model.keras",
}


# ============================================================
# Setup
# ============================================================

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Utilities
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file was not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_image(path, label):
    """Load and resize an image without external normalization."""

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMG_SIZE,
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    label = tf.cast(
        label,
        tf.int32,
    )

    return image, label


def create_assets(paths, labels):
    assets = tf.data.Dataset.from_tensor_slices(
        (
            np.asarray(paths, dtype=str),
            np.asarray(labels, dtype=np.int32),
        )
    )

    assets = assets.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    assets = assets.batch(
        BATCH_SIZE
    )

    assets = assets.prefetch(
        tf.data.AUTOTUNE
    )

    return assets


def reconstruct_test_set(split_data):
    """Reconstruct the exact test split saved during training."""

    test_paths_relative = split_data["test"]["paths"]
    test_labels = split_data["test"]["labels"]

    test_paths = []

    for relative_path in test_paths_relative:

        absolute_path = DATA_DIR / relative_path

        if not absolute_path.exists():
            raise FileNotFoundError(
                "A test image from the saved split could not "
                f"be found: {absolute_path}"
            )

        test_paths.append(
            str(absolute_path)
        )

    return test_paths, test_labels


# ============================================================
# Confusion Matrix
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    model_name,
):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(len(CLASS_NAMES))
        ),
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(matrix)

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.colorbar()

    tick_positions = np.arange(
        len(CLASS_NAMES)
    )

    plt.xticks(
        tick_positions,
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        tick_positions,
        CLASS_NAMES,
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    threshold = matrix.max() / 2.0

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):

            value = matrix[row, column]

            plt.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white"
                if value > threshold
                else "black",
            )

    plt.tight_layout()

    output_path = (
        EVALUATION_DIR
        / f"{safe_name}_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


# ============================================================
# Evaluate One Model
# ============================================================

def evaluate_model(
    model_name,
    model_path,
    test_assets,
):
    print("\n")
    print("=" * 80)
    print(f"Evaluating: {model_name}")
    print("=" * 80)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not found: {model_path}"
        )

    model = tf.keras.models.load_model(
        model_path
    )

    # Verify model output shape.
    output_classes = model.output_shape[-1]

    if output_classes != len(CLASS_NAMES):
        raise ValueError(
            f"{model_name} outputs {output_classes} classes, "
            f"but class_names.json contains "
            f"{len(CLASS_NAMES)} classes."
        )

    loss, accuracy = model.evaluate(
        test_assets,
        verbose=1,
    )

    y_true = []
    y_pred = []
    probabilities = []

    for images, labels in test_assets:

        preds = model.predict(
            images,
            verbose=0,
        )

        predictions = np.argmax(
            preds,
            axis=1,
        )

        y_true.extend(
            labels.numpy().tolist()
        )

        y_pred.extend(
            predictions.tolist()
        )

        probabilities.extend(
            preds.tolist()
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

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    # --------------------------------------------------------
    # Text Report
    # --------------------------------------------------------

    report_path = (
        EVALUATION_DIR
        / f"{safe_name}_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"Model: {model_name}\n"
        )

        file.write(
            f"Model Path: {model_path}\n\n"
        )

        file.write(
            f"Test Loss: {loss:.6f}\n"
        )

        file.write(
            f"Test Accuracy: {accuracy:.6f}\n"
        )

        file.write(
            f"Weighted Precision: {precision:.6f}\n"
        )

        file.write(
            f"Weighted Recall: {recall:.6f}\n"
        )

        file.write(
            f"Weighted F1 Score: {f1:.6f}\n"
        )

        file.write(
            f"Macro Precision: {macro_precision:.6f}\n"
        )

        file.write(
            f"Macro Recall: {macro_recall:.6f}\n"
        )

        file.write(
            f"Macro F1 Score: {macro_f1:.6f}\n\n"
        )

        file.write(
            "Classification Report\n"
        )

        file.write(
            "=" * 60
        )

        file.write("\n")

        file.write(
            report_text
        )

    # --------------------------------------------------------
    # JSON Report
    # --------------------------------------------------------

    json_result = {
        "model": model_name,
        "model_path": str(model_path),
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "classification_report": report_dict,
    }

    json_path = (
        EVALUATION_DIR
        / f"{safe_name}_evaluation.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_result,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    confusion_matrix_path = (
        save_confusion_matrix(
            y_true,
            y_pred,
            model_name,
        )
    )

    del model

    return {
        "model": model_name,
        "model_path": str(model_path),
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "classification_report_path": str(
            report_path
        ),
        "confusion_matrix_path": str(
            confusion_matrix_path
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("Plant Disease Detection - Evaluation")
    print("=" * 80)

    # --------------------------------------------------------
    # Load class names
    # --------------------------------------------------------

    class_names = load_json(
        CLASS_NAMES_FILE
    )

    global CLASS_NAMES
    CLASS_NAMES = class_names

    # --------------------------------------------------------
    # Load exact training split
    # --------------------------------------------------------

    split_data = load_json(
        SPLIT_FILE
    )

    if split_data["class_names"] != CLASS_NAMES:
        raise ValueError(
            "Class names in data_split.json do not match "
            "class_names.json."
        )

    # --------------------------------------------------------
    # Reconstruct test set
    # --------------------------------------------------------

    test_paths, test_labels = (
        reconstruct_test_set(
            split_data
        )
    )

    print(
        f"\nTest images: {len(test_paths)}"
    )

    test_assets = create_assets(
        test_paths,
        test_labels,
    )

    # --------------------------------------------------------
    # Evaluate models
    # --------------------------------------------------------

    results = []

    for model_name, model_path in MODEL_FILES.items():

        result = evaluate_model(
            model_name=model_name,
            model_path=model_path,
            test_assets=test_assets,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Determine best test performer
    # --------------------------------------------------------
    # This is only a reporting comparison.
    # The official best_model was selected during training
    # using validation accuracy.

    best_test_model = max(
        results,
        key=lambda item: item[
            "test_accuracy"
        ],
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    evaluation_summary = {
        "evaluation_type": "final_test_evaluation",
        "test_size": len(test_paths),
        "class_names": CLASS_NAMES,
        "models": results,
        "highest_test_accuracy_model": (
            best_test_model["model"]
        ),
        "highest_test_accuracy": (
            best_test_model["test_accuracy"]
        ),
    }

    with open(
        EVALUATION_DIR
        / "model_evaluation.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evaluation_summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Save CSV comparison
    # --------------------------------------------------------

    csv_path = (
        EVALUATION_DIR
        / "model_comparison.csv"
    )

    fieldnames = [
        "model",
        "test_loss",
        "test_accuracy",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                {
                    field: result.get(
                        field,
                        "",
                    )
                    for field in fieldnames
                }
            )

    # --------------------------------------------------------
    # Print final comparison
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("Final Test Results")
    print("=" * 80)

    for result in results:

        print(
            f"\n{result['model']}"
        )

        print(
            f"  Accuracy:  "
            f"{result['test_accuracy']:.4f}"
        )

        print(
            f"  Precision: "
            f"{result['weighted_precision']:.4f}"
        )

        print(
            f"  Recall:    "
            f"{result['weighted_recall']:.4f}"
        )

        print(
            f"  F1 Score:  "
            f"{result['weighted_f1']:.4f}"
        )

    print("\n")
    print(
        "Highest test accuracy in this evaluation:"
    )

    print(
        f"{best_test_model['model']} "
        f"({best_test_model['test_accuracy']:.4f})"
    )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
