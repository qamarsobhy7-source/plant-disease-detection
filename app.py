import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon=None,
    layout="wide",
)

IMG_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.60

MODELS_DIR = Path("models")
CLASS_NAMES_FILE = MODELS_DIR / "class_names.json"

MODEL_FILES = {
    "EfficientNetB0": MODELS_DIR / "efficientnetb0.keras",
}


# ============================================================
# Load Class Names
# ============================================================

@st.cache_data
def load_class_names():
    if not CLASS_NAMES_FILE.exists():
        raise FileNotFoundError(
            "class_names.json was not found. "
            "Run train_model.py first."
        )

    with open(
        CLASS_NAMES_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        class_names = json.load(file)

    if not isinstance(class_names, list):
        raise ValueError(
            "class_names.json must contain a list."
        )

    if len(class_names) == 0:
        raise ValueError(
            "class_names.json contains no classes."
        )

    return class_names


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_selected_model(model_path):
    return tf.keras.models.load_model(
        model_path,
        compile=False,
    )


# ============================================================
# Prediction
# ============================================================

def preprocess_image(image):
    """
    Convert the uploaded PIL image into a raw float32 tensor.

    Model-specific preprocessing is embedded inside each model:
    - EfficientNetB0: built-in preprocessing
    """

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return image_array


def predict_image(model, image, class_names):
    image_array = preprocess_image(
        image
    )

    probabilities = model.predict(
        image_array,
        verbose=0,
    )[0]

    if len(probabilities) != len(
        class_names
    ):
        raise ValueError(
            "The model output does not match "
            "the number of classes."
        )

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    predicted_class = class_names[
        predicted_index
    ]

    ranked_indices = np.argsort(
        probabilities
    )[::-1]

    top_predictions = []

    for index in ranked_indices[:3]:

        top_predictions.append(
            {
                "class": class_names[
                    int(index)
                ],
                "probability": float(
                    probabilities[index]
                ),
            }
        )

    return (
        predicted_class,
        confidence,
        top_predictions,
        probabilities,
    )


# ============================================================
# Page Header
# ============================================================

st.title(
    "Plant Disease Detection System"
)

st.write(
    "Upload a plant leaf image to classify its "
    "health condition using a trained deep learning model."
)

st.divider()


# ============================================================
# Load Classes
# ============================================================

try:
    class_names = load_class_names()

except Exception as error:

    st.error(
        str(error)
    )

    st.stop()


# ============================================================
# Available Models
# ============================================================

available_models = {
    name: path
    for name, path in MODEL_FILES.items()
    if path.exists()
}

if not available_models:

    st.error(
        "No trained models were found in the models/ directory."
    )

    st.info(
        "Run `python train_model.py` first."
    )

    st.stop()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header(
    "Model Selection"
)

selected_model_name = st.sidebar.selectbox(
    "Choose a model",
    list(available_models.keys()),
)

selected_model_path = available_models[
    selected_model_name
]


st.sidebar.write(
    f"Model file: `{selected_model_path}`"
)

st.sidebar.write(
    f"Classes: {len(class_names)}"
)

st.sidebar.write(
    f"Image size: {IMG_SIZE[0]} × {IMG_SIZE[1]}"
)


# ============================================================
# Load Selected Model
# ============================================================

try:

    model = load_selected_model(
        str(selected_model_path)
    )

except Exception as error:

    st.error(
        f"Failed to load the selected model: {error}"
    )

    st.stop()


# ============================================================
# Model Validation
# ============================================================

try:

    output_classes = model.output_shape[-1]

    if output_classes != len(
        class_names
    ):
        st.error(
            "Model/class mismatch detected. "
            f"The model outputs {output_classes} classes, "
            f"while class_names.json contains "
            f"{len(class_names)} classes."
        )

        st.stop()

except Exception as error:

    st.error(
        f"Could not validate the model: {error}"
    )

    st.stop()


# ============================================================
# Image Upload
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
)


if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception as error:

        st.error(
            f"Could not read the uploaded image: {error}"
        )

        st.stop()

    # --------------------------------------------------------
    # Display Image
    # --------------------------------------------------------

    col1, col2 = st.columns(
        2
    )

    with col1:

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            image,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        (
            predicted_class,
            confidence,
            top_predictions,
            probabilities,
        ) = predict_image(
            model,
            image,
            class_names,
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )

        st.stop()

    # --------------------------------------------------------
    # Main Prediction
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "Prediction"
        )

        st.write(
            f"**Predicted Class:** {predicted_class}"
        )

        st.write(
            f"**Confidence:** "
            f"{confidence * 100:.2f}%"
        )

        if confidence >= CONFIDENCE_THRESHOLD:

            st.success(
                "The model produced a prediction "
                "above the configured confidence threshold."
            )

        else:

            st.warning(
                "The model confidence is below 60%. "
                "Treat this prediction as uncertain."
            )

    st.divider()

    # --------------------------------------------------------
    # Top 3 Predictions
    # --------------------------------------------------------

    st.subheader(
        "Top 3 Predictions"
    )

    top_columns = st.columns(
        len(top_predictions)
    )

    for column, prediction in zip(
        top_columns,
        top_predictions,
    ):

        with column:

            st.metric(
                prediction["class"],
                f"{prediction['probability'] * 100:.2f}%",
            )

    st.divider()

    # --------------------------------------------------------
    # All Class Probabilities
    # --------------------------------------------------------

    st.subheader(
        "Class Probabilities"
    )

    for class_name, probability in zip(
        class_names,
        probabilities,
    ):

        st.write(
            f"**{class_name}** — "
            f"{probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )


# ============================================================
# Supported Classes
# ============================================================

st.divider()

st.subheader(
    "Supported Classes"
)

for class_name in class_names:

    st.write(
        f"- {class_name}"
    )


# ============================================================
# Model Information
# ============================================================

st.divider()

st.subheader(
    "Model Information"
)

model_information = {
    "EfficientNetB0": (
        "EfficientNetB0 transfer-learning model "
        "initialized with ImageNet weights and fine-tuned."
    ),
}

st.write(
    model_information.get(
        selected_model_name,
        "Trained deep learning model.",
    )
)


# ============================================================
# Disclaimer
# ============================================================

st.divider()

st.caption(
    "This application is an image-classification system "
    "for educational and research purposes. Predictions "
    "should not be treated as a professional agricultural "
    "diagnosis."
    )
