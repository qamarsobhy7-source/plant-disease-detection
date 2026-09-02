import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# Load the trained model with caching for performance
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('plant_disease_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Define readable class labels mapping
class_names = [
    "Healthy Plant",
    "Early Blight",
    "Late Blight",
    "Powdery Mildew",
    "Leaf Spot"
]

# App UI Design
st.title("🌱 Plant Disease Detection System")
st.write("Upload an image of a plant leaf to detect and classify potential diseases instantly using Deep Learning.")

st.markdown("---")

# File uploader with validation for images only
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Leaf Image', use_container_width=True)
        
        if model is not None:
            with st.spinner('Analyzing the leaf image...'):
                # Preprocessing the image
                img = image.resize((128, 128))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Optional: Normalization if required by your training pipeline
                # img_array = img_array / 255.0

                # Make prediction
                predictions = model.predict(img_array)
                predicted_class_index = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0])) * 100
                
                predicted_label = class_names[predicted_class_index]
                
            # Display results professionally
            st.success("Analysis Complete!")
            st.metric(label="Predicted Condition", value=predicted_label)
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
            
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
        
