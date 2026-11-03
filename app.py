
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

st.set_page_config(page_title="Verdant AI | 34 Crops", page_icon="🌿", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background: radial-gradient(circle at top right, #e0f2fe 0%, #f0fdf4 100%); }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }
    .main-title { font-size: 3rem; font-weight: 800; color: #0f172a; text-align: center; margin-bottom: 0.5rem; letter-spacing: -1px; }
    .main-title span { color: #10b981; }
    .sub-title { font-size: 1.1rem; color: #64748b; text-align: center; margin-bottom: 3rem; font-weight: 400; }
    [data-testid="stFileUploader"] { background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(10px); border-radius: 24px; padding: 2rem; box-shadow: 0 20px 40px -10px rgba(16, 185, 129, 0.15); border: 2px dashed #10b981; margin: 0 auto 3rem auto; max-width: 600px; text-align: center; }
    [data-testid="stFileUploader"] section { background: transparent; border: none; }
    [data-testid="stFileUploader"] button { background: #10b981; color: white; border: none; border-radius: 50px; padding: 0.8rem 2rem; font-size: 1rem; font-weight: 600; width: 100%; max-width: 250px; margin: 0 auto; }
    .result-card { background: white; border-radius: 16px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 5px solid #10b981; text-align: center; }
    .result-card.danger { border-top: 5px solid #ef4444; }
    .result-card.warning { border-top: 5px solid #f59e0b; }
    .result-label { font-size: 0.85rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
    .result-value { font-size: 1.6rem; color: #0f172a; font-weight: 800; margin-top: 0.5rem; }
    .treat-box { background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 1rem; border-left: 5px solid #3b82f6; }
    .treat-title { font-weight: 700; color: #1e3a8a; margin-bottom: 0.5rem; font-size: 1.1rem; }
    .treat-desc { color: #475569; font-size: 0.95rem; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

device = torch.device("cpu")
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_m1():
    try:
        with open('models_v2/crop_classes.json', 'r') as f:
            classes = json.load(f)
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(classes))
        model.load_state_dict(torch.load('models_v2/best_crop_detector_m1.pth', map_location=device))
        model.eval()
        return model, classes
    except Exception as e:
        st.error(f"Error loading M1: {e}")
        st.stop()

@st.cache_resource
def load_expert(crop_name):
    safe_name = crop_name.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '').replace('/', '_')
    model_path = f'models_v2/expert_{safe_name}.pth'
    classes_path = f'models_v2/expert_{safe_name}_classes.json'
    
    if os.path.exists(model_path) and os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            classes = json.load(f)
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(classes))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, classes
    return None, None

try:
    with open('treatments.json', 'r') as f:
        treatments = json.load(f)
except:
    treatments = {}

st.markdown("<h1 class='main-title'>🌿 Verdant <span>AI</span></h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Advanced Plant Disease Diagnosis with Hierarchical AI</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader("Drag & Drop your leaf image here", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    
    col_img1, col_img2, col_btn = st.columns([2, 2, 1])
    with col_img1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    with col_btn:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        diagnose_btn = st.button("Analyze Leaf 🔍", use_container_width=True)
        
    if diagnose_btn:
        with st.spinner("Stage 1: Identifying Crop Type..."):
            m1_model, m1_classes = load_m1()
            input_tensor = val_transforms(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                m1_outputs = m1_model(input_tensor)
                m1_probs = torch.nn.functional.softmax(m1_outputs, dim=1)[0]
                m1_conf, m1_pred = torch.max(m1_probs, 0)
                crop_name = m1_classes[m1_pred.item()]
                crop_conf = m1_conf.item() * 100
                
        if crop_conf < 60.0:
            st.warning(f"⚠️ **Low Confidence ({crop_conf:.1f}%):** The AI is not completely sure, but its best guess is shown below.")

        with st.spinner(f"Stage 2: Analyzing {crop_name} Disease Expert..."):
            expert_model, expert_classes = load_expert(crop_name)
            
            if expert_model:
                with torch.no_grad():
                    outputs = expert_model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                    conf, pred = torch.max(probs, 0)
                    disease_name = expert_classes[pred.item()]
                    disease_conf = conf.item() * 100

                cam = GradCAM(model=expert_model, target_layers=[expert_model.features[-1]])
                targets = [ClassifierOutputTarget(pred.item())]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                rgb_img = np.float32(image.resize((224, 224))) / 255
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                final_conf = disease_conf
                
                is_healthy = 'healthy' in disease_name.lower()
                card_class = "" if is_healthy else "danger"
                status_icon = "✅" if is_healthy else "⚠️"
            else:
                disease_name = "Healthy / No specific disease detected"
                is_healthy = True
                card_class = ""
                status_icon = "✅"
                visualization = np.float32(image.resize((224, 224))) / 255
                final_conf = crop_conf
                
        with col_img2:
            st.image(visualization, caption="AI Focus (Grad-CAM)", use_column_width=True)
            
        st.markdown("---")
        st.markdown("<h3 style='text-align: center; color: #0f172a; margin-bottom: 2rem;'>📊 Hierarchical Diagnosis</h3>", unsafe_allow_html=True)
        
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"<div class='result-card'><div class='result-label'>Crop Detected</div><div class='result-value'>{crop_name} ({crop_conf:.1f}%)</div></div>", unsafe_allow_html=True)
        with r2:
            st.markdown(f"<div class='result-card {card_class}'><div class='result-label'>Disease Status</div><div class='result-value'>{status_icon} {disease_name}</div></div>", unsafe_allow_html=True)
        with r3:
            st.markdown(f"<div class='result-card'><div class='result-label'>Disease Confidence</div><div class='result-value'>{final_conf:.1f}%</div></div>", unsafe_allow_html=True)

        # عرض خطة العلاج التفصيلية
        st.markdown("### 💊 Step-by-Step Treatment Plan")
        
        # البحث عن العلاج
        treatment_key = f"{crop_name}___{disease_name}"
        steps = treatments.get(treatment_key)
        
        # لو ملقيناش بالاسم بالظبط، هندور على كلمة مفتاحية
        if not steps:
            for key, val in treatments.items():
                if crop_name in key and (disease_name.split()[0] in key or "healthy" in key.lower()):
                    steps = val
                    break
                    
        if not steps:
            # العلاج الافتراضي لو مش موجود
            steps = {
                "immediate": "Isolate the plant and remove visible infected parts.",
                "organic": "Consult a local expert for organic options.",
                "chemical": "Consult an agronomist before applying chemicals.",
                "prevention": "Ensure good air circulation and proper watering."
            }

        # عرض الخطوات في صناديق منفصلة
        st.markdown(f"<div class='treat-box'><div class='treat-title'>🚨 Immediate Actions</div><div class='treat-desc'>{steps['immediate']}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='treat-box' style='border-left-color: #10b981;'><div class='treat-title'>🌱 Organic Treatment</div><div class='treat-desc'>{steps['organic']}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='treat-box' style='border-left-color: #f59e0b;'><div class='treat-title'>🧪 Chemical Treatment</div><div class='treat-desc'>{steps['chemical']}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='treat-box' style='border-left-color: #6366f1;'><div class='treat-title'>🛡️ Prevention (Long-term)</div><div class='treat-desc'>{steps['prevention']}</div></div>", unsafe_allow_html=True)

else:
    st.markdown("<h3 style='text-align: center; margin-top: 3rem; margin-bottom: 2rem; color: #0f172a;'>How it works</h3>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("<div class='feature-card'><div style='font-size: 2.5rem;'>📸</div><h4 style='color:#0f172a;'>1. Capture</h4><p style='color:#64748b;'>Take a clear photo of the infected leaf.</p></div>", unsafe_allow_html=True)
    with s2:
        st.markdown("<div class='feature-card'><div style='font-size: 2.5rem;'>🧠</div><h4 style='color:#0f172a;'>2. Hierarchical AI</h4><p style='color:#64748b;'>Our system identifies the crop, then routes it to a specialized disease expert.</p></div>", unsafe_allow_html=True)
    with s3:
        st.markdown("<div class='feature-card'><div style='font-size: 2.5rem;'>💊</div><h4 style='color:#0f172a;'>3. Treat</h4><p style='color:#64748b;'>Get actionable treatment recommendations for 34+ crops.</p></div>", unsafe_allow_html=True)
