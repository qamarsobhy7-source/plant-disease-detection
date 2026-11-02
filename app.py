
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

# التغيير الأهم: استخدام شاشة كاملة (wide) بدل النص (centered)
st.set_page_config(page_title="Verdant AI", page_icon="🌿", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* خلفية متدرجة جميلة جداً */
    .stApp { 
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
    }

    /* تقليل المسافات الفاضية فوق وتحت */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }

    .main-title { font-size: 3rem; font-weight: 700; color: #0f172a; text-align: center; margin-bottom: 0.5rem; }
    .sub-title { font-size: 1.2rem; color: #64748b; text-align: center; margin-bottom: 3rem; }

    /* تكبير وتجميل مربع الرفع */
    [data-testid="stFileUploader"] {
        background-color: white;
        border-radius: 20px;
        padding: 3rem 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 2px dashed #10b981;
        margin-bottom: 3rem;
        text-align: center;
    }
    [data-testid="stFileUploader"] section { 
        background: #f0fdf4; 
        border-radius: 12px; 
        border: none; 
        display: flex; 
        justify-content: center; 
    }
    [data-testid="stFileUploader"] button { 
        background: #10b981; 
        color: white; 
        border: none; 
        border-radius: 8px; 
        padding: 0.8rem 2rem; 
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* كروت الخطوات */
    .step-box { 
        background: white;
        border-radius: 16px; 
        padding: 2rem 1.5rem; 
        text-align: center; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s;
    }
    .step-box:hover { transform: translateY(-5px); }
    .step-icon { font-size: 3rem; margin-bottom: 1rem; }
    .step-title { font-weight: 700; color: #0f172a; margin-bottom: 0.5rem; font-size: 1.2rem; }
    .step-desc { font-size: 0.95rem; color: #64748b; line-height: 1.5; }

    /* كروت النتائج */
    .result-card { background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 6px solid #10b981; text-align: center; }
    .result-card.danger { border-left: 6px solid #ef4444; }
    .result-label { font-size: 0.9rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .result-value { font-size: 1.8rem; color: #0f172a; font-weight: 700; margin-top: 0.5rem; }
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

# قاموس العلاجات (مختصر للكود)
treatments = {
    "Apple___Apple_scab": "Organic: Sulfur. Chemical: Captan. Prevention: Remove leaves.",
    "Strawberry___Leaf_scorch": "Organic: Neem oil. Chemical: Captan. Prevention: Air circulation.",
    "Tomato___Early_blight": "Organic: Neem oil. Chemical: Chlorothalonil. Prevention: Crop rotation.",
    "Potato___healthy": "Plant is healthy! Maintain regular watering."
}

# --- واجهة المستخدم ---
st.markdown("<h1 class='main-title'>🌿 Verdant AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Advanced Plant Disease Diagnosis with Explainable AI</p>", unsafe_allow_html=True)

# استخدام 3 أعمدة عشان نخلي المربع في النص وميكونش عريض أوي
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader("Drag and drop or click to upload a leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_img1, col_img2, col_btn = st.columns([2, 2, 1])
    
    with col_img1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
    with col_btn:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        diagnose_btn = st.button("Analyze Leaf 🔍", use_container_width=True)
        
    if diagnose_btn:
        with st.spinner("Analyzing leaf..."):
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
            
            with col_img2:
                st.image(visualization, caption="AI Focus (Grad-CAM)", use_column_width=True)
            
            st.markdown("---")
            st.markdown("<h3 style='text-align: center;'>📊 Diagnosis Results</h3>", unsafe_allow_html=True)
            
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"<div class='result-card'><div class='result-label'>Crop</div><div class='result-value'>{crop}</div></div>", unsafe_allow_html=True)
            with r2:
                is_healthy = disease.lower() == 'healthy'
                card_class = "" if is_healthy else "danger"
                status_icon = "✅" if is_healthy else "⚠️"
                st.markdown(f"<div class='result-card {card_class}'><div class='result-label'>Status</div><div class='result-value'>{status_icon} {disease}</div></div>", unsafe_allow_html=True)
            with r3:
                st.markdown(f"<div class='result-card'><div class='result-label'>Confidence</div><div class='result-value'>{conf.item()*100:.1f}%</div></div>", unsafe_allow_html=True)

            st.markdown("<h3 style='margin-top: 2rem;'>💊 Treatment</h3>", unsafe_allow_html=True)
            st.info(treatments.get(pred_class, "Consult a local agricultural expert for specific treatment."))

else:
    # قسم الخطوات في كروت واسعة
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-bottom: 2rem;'>How it works</h3>", unsafe_allow_html=True)
    
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("<div class='step-box'><div class='step-icon'>📸</div><div class='step-title'>1. Upload</div><div class='step-desc'>Take a clear photo of the infected leaf.</div></div>", unsafe_allow_html=True)
    with s2:
        st.markdown("<div class='step-box'><div class='step-icon'>🧠</div><div class='step-title'>2. Analyze</div><div class='step-desc'>Our AI scans the image for diseases.</div></div>", unsafe_allow_html=True)
    with s3:
        st.markdown("<div class='step-box'><div class='step-icon'>💊</div><div class='step-title'>3. Treat</div><div class='step-desc'>Get instant treatment recommendations.</div></div>", unsafe_allow_html=True)
