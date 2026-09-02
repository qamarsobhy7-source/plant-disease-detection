import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. قائمة أسماء الأمراض (Mapping)
class_names = [
    "Healthy Plant",
    "Early Blight",
    "Late Blight",
    "Powdery Mildew",
    "Leaf Spot"
]

@st.cache_resource
def get_model():
    return load_model("plant_disease_model.h5")

model = get_model()

# 2. دالة التنبؤ والتحويل لاسم المرض
def predict_plant_disease(image_obj):
    img = image_obj.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100
    predicted_label = class_names[predicted_class_index]
    
    return predicted_label, confidence

# 3. واجهة الاستخدام (Streamlit UI)
st.title("🌱 Plant Disease Detection System")
st.write("ارفعي صورة ورقة نبات عشان النظام يقوم بفحصها.")

uploaded_file = st.file_uploader("اختر صورة ورقة نبات...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_obj = Image.open(uploaded_file)
    st.image(image_obj, caption='الصورة المرفوعة', use_column_width=True)
    
    if st.button('افحص الصورة'):
        with st.spinner('جارٍ التحليل...'):
            # استخدام الدالة اللي بتطلع اسم المرض الحقيقي ونسبة الثقة
            predicted_label, confidence = predict_plant_disease(image_obj)
            
            st.success(f"🌿 النتيجة المتوقعة: {predicted_label}")
            st.info(f"📊 نسبة الثقة (Confidence): {confidence:.2f}%")
            
