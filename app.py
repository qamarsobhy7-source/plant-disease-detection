import os
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered",
)


# ============================================================
# Configuration
# ============================================================

MODELS_DIR = "models"

IMAGE_SIZE = (
    128,
    128,
)

CONFIDENCE_THRESHOLD = 60.0


# ============================================================
# Model Paths
# ============================================================

model_files = {
    "Best Model": "best_model.keras",
    "Custom CNN": "custom_cnn.keras",
    "MobileNetV2": "mobilenetv2.keras",
    "EfficientNetB0": "efficientnetb0.keras",
}


# ============================================================
# Load Class Names
# ============================================================

@st.cache_data
def load_class_names():

    class_names_path = os.path.join(
        MODELS_DIR,
        "class_names.json",
    )

    if not os.path.exists(
        class_names_path
    ):
        return None

    try:

        with open(
            class_names_path,
            "r",
            encoding="utf-8",
        ) as file:

            class_names = json.load(
                file
            )

        return class_names

    except Exception:
        return None


class_names = load_class_names()


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_model(
    model_path,
):

    if not os.path.exists(
        model_path
    ):
        return None

    try:

        model = tf.keras.models.load_model(
            model_path
        )

        return model

    except Exception as error:

        st.error(
            f"Unable to load model: {error}"
        )

        return None


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title(
    "Model Configuration"
)


available_models = []


for model_name, filename in model_files.items():

    path = os.path.join(
        MODELS_DIR,
        filename,
    )

    if os.path.exists(path):
        available_models.append(
            model_name
        )


if not available_models:

    st.error(
        "No trained models were found. "
        "Please run train_model.py first."
    )

    st.stop()


model_choice = st.sidebar.selectbox(
    "Select Model Architecture",
    available_models,
)


selected_filename = model_files[
    model_choice
]


selected_model_path = os.path.join(
    MODELS_DIR,
    selected_filename,
)


model = load_model(
    selected_model_path
)


# ============================================================
# Validate Class Names
# ============================================================

if class_names is None:

    st.error(
        "class_names.json was not found. "
        "Please run train_model.py first."
    )

    st.stop()


if model is not None:

    model_output_classes = (
        model.output_shape[-1]
    )

    if model_output_classes != len(
        class_names
    ):

        st.error(
            "Model output classes do not match "
            "class_names.json."
        )

        st.stop()


# ============================================================
# Application Header
# ============================================================

st.title(
    "Plant Disease Detection System"
)


st.write(
    f"Using **{model_choice}** "
    "for plant leaf classification."
)


st.markdown("---")


# ============================================================
# Image Upload
# ============================================================

uploaded_file = st.file_uploader(
    "Choose a plant leaf image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
)


# ============================================================
# Prediction
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # Open Image
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        st.image(
            image,
            caption="Uploaded Leaf Image",
            use_container_width=True,
        )


        # ----------------------------------------------------
        # Model Check
        # ----------------------------------------------------

        if model is None:

            st.error(
                f"Model file was not found: "
                f"{selected_model_path}"
            )

            st.stop()


        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        with st.spinner(
            f"Analyzing using {model_choice}..."
        ):

            resized_image = image.resize(
                IMAGE_SIZE
            )


            image_array = np.asarray(
                resized_image,
                dtype=np.float32,
            )


            image_array = np.expand_dims(
                image_array,
                axis=0,
            )


            # Important:
            # Preprocessing is already included
            # inside each trained model.
            #
            # Custom CNN:
            # 0-255 -> 0-1
            #
            # MobileNetV2:
            # MobileNetV2 preprocessing
            #
            # EfficientNetB0:
            # Uses its internal preprocessing.


            predictions = model.predict(
                image_array,
                verbose=0,
            )[0]


        # ----------------------------------------------------
        # Prediction Results
        # ----------------------------------------------------

        predicted_class_index = int(
            np.argmax(
                predictions
            )
        )


        confidence = float(
            predictions[
                predicted_class_index
            ]
        ) * 100


        predicted_label = class_names[
            predicted_class_index
        ]


        # ----------------------------------------------------
        # Analysis Complete
        # ----------------------------------------------------

        st.success(
            "Analysis Complete"
        )


        st.markdown(
            "### Prediction Result"
        )


        st.markdown(
            f"**Predicted Condition:** "
            f"{predicted_label}"
        )


        st.markdown(
            f"**Confidence:** "
            f"{confidence:.1f}%"
        )


        # ----------------------------------------------------
        # Confidence Warning
        # ----------------------------------------------------

        if confidence < CONFIDENCE_THRESHOLD:

            st.warning(
                "The model has low confidence "
                "in this prediction. "
                "Please upload a clearer image "
                "with the leaf visible."
            )


        # ----------------------------------------------------
        # Top Predictions
        # ----------------------------------------------------

        st.markdown("---")


        st.markdown(
            "### Top Predictions"
        )


        sorted_indices = np.argsort(
            predictions
        )[::-1]


        top_k = min(
            3,
            len(sorted_indices),
        )


        for rank in range(
            top_k
        ):

            index = int(
                sorted_indices[rank]
            )


            probability = float(
                predictions[index]
            ) * 100


            class_name = class_names[
                index
            ]


            st.write(
                f"{rank + 1}. "
                f"**{class_name}** — "
                f"{probability:.1f}%"
            )


        # ----------------------------------------------------
        # Probability Distribution
        # ----------------------------------------------------

        st.markdown("---")


        st.markdown(
            "### Class Probabilities"
        )


        for index in sorted_indices:

            probability = float(
                predictions[index]
            )


            st.progress(
                probability
            )


            st.caption(
                f"{class_names[index]}: "
                f"{probability * 100:.1f}%"
            )


    except Exception as error:

        st.error(
            f"An error occurred while processing "
            f"the image: {error}"
        )


# ============================================================
# Information Section
# ============================================================

st.markdown("---")


with st.expander(
    "About the Models"
):

    st.write(
        """
This application supports three trained deep learning architectures:

- Custom CNN
- MobileNetV2
- EfficientNetB0

The application can also load the automatically selected Best Model.

The Best Model is selected during training using validation performance.
The test dataset is used only for final model evaluation.
"""
    )


with st.expander(
    "Supported Classes"
):

    for index, class_name in enumerate(
        class_names
    ):

        st.write(
            f"{index}: {class_name}"
)
