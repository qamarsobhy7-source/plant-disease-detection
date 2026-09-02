import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Sidebar for Model Selection (Added to support the 3 comparative models)
st.sidebar.title("⚙️ Model Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model Architecture",
    ("Custom CNN", "MobileNetV2", "EfficientNetB0")
)

# Map choices to saved model filenames
model_files = {
    "Custom CNN": "custom_cnn_model.h5",
    "MobileNetV2": "mobilenetv2_model.h5",
    "EfficientNetB0": "efficientnet_model.h5"
}

selected_model_path = model_files[model_choice]

# Load the selected trained model with caching for performance
@st.cache_resource
def load_selected_model(path):
    try:
        if os.path.exists(path):
            model = tf.keras.models.load_model(path)
            return model
        else:
            # Fallback to legacy model name if specific one isn't trained yet
            if os.path.exists('plant_disease_model.h5'):
                return tf.keras.models.load_model('plant_disease_model.h5')
            return None
    except Exception as e:
        return None

model = load_selected_model(selected_model_path)

# Define readable class labels mapping
class_names = [
    "Healthy Plant",
    "Early Blight",
    "Late Blight",
    "Powdery Mildew",
    "Leaf Spot"
]

# App UI Design
st.title("🌿 Plant Disease Detection System")
st.write(f"Using **{model_choice}** architecture for instant analysis.")

st.markdown("---")

# File uploader with validation for images only
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

        if model is not None:
            with st.spinner(f"Analyzing using {model_choice}..."):
                # Preprocessing the image (Exact match with training pipeline)
                img = image.resize((128, 128))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                # Normalization
                img_array = img_array / 255.0

                # Make prediction
                predictions = model.predict(img_array)[0]
                predicted_class_index = np.argmax(predictions)
                confidence = float(predictions[predicted_class_index]) * 100

                predicted_label = class_names[predicted_class_index]

            # Display results professionally with confidence threshold check
            st.success("Analysis Complete!")

            if confidence < 60.0:
                st.warning("⚠️ Low Confidence Score.")
                st.warning("Please upload a clearer plant leaf image.")
            else:
                # Main Results Display
                st.markdown("### 🔍 Prediction Result")
                st.markdown(f"**Predicted Disease:** {predicted_label}")
                st.markdown(f"**Confidence:** {confidence:.1f}%")
                
                st.markdown("---")
                
                # Top Predictions List (Sorted from highest to lowest)
                st.markdown("### 📊 Top Predictions")
                sorted_indices = np.argsort(predictions)[::-1]
                
                for rank, idx in enumerate(sorted_indices, start=1):
                    c_name = class_names[idx]
                    c_prob = float(predictions[idx]) * 100
                    st.text(f"{rank}. {c_name} — {c_prob:.1f}%")
        else:
            st.error(f"Model file not found! Please ensure '{selected_model_path}' exists or run training first.")

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
        
