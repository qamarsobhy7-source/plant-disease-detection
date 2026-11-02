
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

st.set_page_config(page_title="Verdant | Smart Crop Doctor", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); }
    .hero-title { font-size: 3.5rem; font-weight: 700; background: -webkit-linear-gradient(45deg, #166534, #22c55e); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0px; }
    .hero-subtitle { font-size: 1.2rem; color: #4b5563; text-align: center; margin-bottom: 2rem; }
    .glass-card { background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.3); box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1); padding: 2rem; margin-bottom: 2rem; }
    .metric-card { background: white; border-radius: 15px; padding: 1.5rem; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-left: 5px solid #22c55e; }
    .metric-card.danger { border-left: 5px solid #ef4444; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1f2937; }
    .metric-label { font-size: 1rem; color: #6b7280; font-weight: 500; }
    .stButton>button { background: linear-gradient(90deg, #16a34a 0%, #22c55e 100%); color: white; border: none; border-radius: 50px; padding: 12px 30px; font-size: 1.1rem; font-weight: 600; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3); width: 100%; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(34, 197, 94, 0.4); color: white; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    with open('class_names.json', 'r') as f:
        unique_classes = json.load(f)
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(unique_classes))
    model.load_state_dict(torch.load('best_model_final.pth', map_location=device))
    model.eval()
    cam = GradCAM(model=model, target_layers=[model.features[-1]])
    return model, cam, unique_classes, device

try:
    model, cam, unique_classes, device = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

treatments = {
    "Tomato___Early_blight": "Organic: Neem oil. Chemical: Chlorothalonil. Prevention: Crop rotation.",
    "Tomato___Late_blight": "Organic: Copper spray. Chemical: Mancozeb. Prevention: Avoid overhead watering.",
    "Apple___Apple_scab": "Organic: Sulfur spray. Chemical: Captan. Prevention: Remove fallen leaves.",
    "Potato___healthy": "Plant is healthy! Maintain regular watering.",
}

st.markdown("<h1 class='hero-title'>🌿 Verdant</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>AI-Powered Plant Disease Diagnosis with Explainable AI</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📸 Upload Image")
    uploaded_file = st.file_uploader("Choose a clear photo of a plant leaf", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        diagnose_btn = st.button("🔍 Analyze Leaf", type="primary")

if uploaded_file and diagnose_btn:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Image")
        st.image(image, use_column_width=True)
        
    with col2:
        st.markdown("#### AI Focus Area (Grad-CAM)")
        with st.spinner("Analyzing..."):
            input_tensor = val_transforms(image).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            conf, pred = torch.max(probs, 0)
            pred_class = unique_classes[pred.item()]
            
            crop, disease = pred_class.split('___')
            crop, disease = crop.replace('_', ' '), disease.replace('_', ' ')
            
            targets = [ClassifierOutputTarget(pred.item())]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            rgb_img = np.float32(image.resize((224, 224))) / 255
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            st.image(visualization, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### 📊 Diagnosis Results")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Detected Crop</div><div class='metric-value'>{crop}</div></div>", unsafe_allow_html=True)
    with m_col2:
        is_healthy = disease.lower() == 'healthy'
        card_class = "" if is_healthy else "danger"
        status_text = "✅ Healthy" if is_healthy else f"🦠 {disease}"
        st.markdown(f"<div class='metric-card {card_class}'><div class='metric-label'>Status</div><div class='metric-value'>{status_text}</div></div>", unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Confidence</div><div class='metric-value'>{conf.item()*100:.1f}%</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💊 Treatment Recommendations")
    st.info(treatments.get(pred_class, "Please consult a local agricultural expert for specific treatment."))
else:
    if not uploaded_file:
        st.info("👈 Please upload a leaf image from the sidebar to begin.")
