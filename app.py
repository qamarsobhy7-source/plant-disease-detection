import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

@st.cache_resource
def get_model():
    return load_model("plant_disease_model.h5")

model = get_model()

st.title("🌱 Plant Disease Detection System")
st.write("ارفعي صورة ورقة النبات عشان النظام يفحصها!")

uploaded_file = st.file_uploader("اختر صورة ورقة نبات...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_obj = Image.open(uploaded_file)
    st.image(image_obj, caption='الصورة المرفوعة', use_column_width=True)

    img = image_obj.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    if st.button('فحص الصورة'):
        with st.spinner('جاري التحليل...'):
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0])) * 100

            st.success(f"النتيجة المتوقعة (Class Index): {predicted_class}")
            st.info(f"نسبة الثقة: {confidence:.2f}%")
          
