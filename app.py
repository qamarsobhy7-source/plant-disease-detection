
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
    /* إخفاء عناصر Streamlit الافتراضية */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    /* الخلفية */
    .stApp { background: radial-gradient(circle at top right, #e0f2fe 0%, #f0fdf4 100%); }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }

    /* العناوين (تعديل لون الخط عشان يبان) */
    h1, h2, h3, h4, p, span, div { color: #0f172a !important; }
    .main-title { font-size: 3rem; font-weight: 800; color: #0f172a !important; text-align: center; margin-bottom: 0.5rem; letter-spacing: -1px; }
    .main-title span { color: #10b981 !important; }
    .sub-title { font-size: 1.1rem; color: #64748b !important; text-align: center; margin-bottom: 3rem; font-weight: 400; }
    .section-title { text-align: center; font-weight: 800; color: #0f172a !important; margin-top: 2rem; margin-bottom: 1.5rem; font-size: 1.5rem; }

    /* مربع الرفع */
    [data-testid="stFileUploader"] { background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(10px); border-radius: 24px; padding: 2rem; box-shadow: 0 20px 40px -10px rgba(16, 185, 129, 0.15); border: 2px dashed #10b981; margin: 0 auto 3rem auto; max-width: 600px; text-align: center; }
    [data-testid="stFileUploader"] section { background: transparent; border: none; }
    [data-testid="stFileUploader"] button { background: #10b981; color: white !important; border: none; border-radius: 50px; padding: 0.8rem 2rem; font-size: 1rem; font-weight: 600; width: 100%; max-width: 250px; margin: 0 auto; }

    /* كروت النتائج */
    .result-card { background: white; border-radius: 16px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 5px solid #10b981; text-align: center; height: 100%; }
    .result-card.danger { border-top: 5px solid #ef4444; }
    .result-label { font-size: 0.85rem; color: #64748b !important; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
    .result-value { font-size: 1.5rem; color: #0f172a !important; font-weight: 800; margin-top: 0.5rem; }

    /* كروت الـ Top 3 */
    .top-card { background: #f8fafc; border-radius: 12px; padding: 1rem; text-align: center; border: 1px solid #e2e8f0; }
    .top-title { font-size: 0.8rem; color: #64748b !important; font-weight: 600; }
    .top-value { font-size: 1.1rem; color: #0f172a !important; font-weight: 700; }

    /* صناديق العلاج */
    .treat-box { background: white; border-radius: 12px; padding: 1.2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 1rem; border-left: 5px solid #3b82f6; }
    .treat-title { font-weight: 700; color: #1e3a8a !important; margin-bottom: 0.5rem; font-size: 1.1rem; }
    .treat-desc { color: #475569 !important; font-size: 0.95rem; line-height: 1.5; }
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
    with open('models_v2/crop_classes.json', 'r') as f:
        classes = json.load(f)
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(torch.load('models_v2/best_crop_detector_m1_robust.pth', map_location=device))
    model.eval()
    return model, classes

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
    
    # عرض الصورة الأصلية وزرار التحليل
    col_img, col_btn = st.columns([3, 1])
    with col_img:
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
                top3_probs, top3_indices = torch.topk(m1_probs, 3)
                
                crop_name = m1_classes[top3_indices[0].item()]
                crop_conf = top3_probs[0].item() * 100

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
                
        # عرض الصورتين جنب بعض
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.image(image, caption="Original Image", use_column_width=True)
        with c2:
            st.image(visualization, caption="AI Focus (Grad-CAM)", use_column_width=True)

        # عرض أفضل 3 احتمالات
        st.markdown("<h3 class='section-title'>🔍 Top 3 Crop Predictions</h3>", unsafe_allow_html=True)
        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown(f"<div class='top-card'><div class='top-title'>1st Guess</div><div class='top-value'>{m1_classes[top3_indices[0].item()]} ({top3_probs[0].item()*100:.1f}%)</div></div>", unsafe_allow_html=True)
        with t2:
            st.markdown(f"<div class='top-card'><div class='top-title'>2nd Guess</div><div class='top-value'>{m1_classes[top3_indices[1].item()]} ({top3_probs[1].item()*100:.1f}%)</div></div>", unsafe_allow_html=True)
        with t3:
            st.markdown(f"<div class='top-card'><div class='top-title'>3rd Guess</div><div class='top-value'>{m1_classes[top3_indices[2].item()]} ({top3_probs[2].item()*100:.1f}%)</div></div>", unsafe_allow_html=True)

        # عرض التشخيص النهائي
        st.markdown("<h3 class='section-title'>📊 Final Diagnosis</h3>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"<div class='result-card'><div class='result-label'>Crop Detected</div><div class='result-value'>{crop_name} ({crop_conf:.1f}%)</div></div>", unsafe_allow_html=True)
        with r2:
            st.markdown(f"<div class='result-card {card_class}'><div class='result-label'>Disease Status</div><div class='result-value'>{status_icon} {disease_name}</div></div>", unsafe_allow_html=True)
        with r3:
            st.markdown(f"<div class='result-card'><div class='result-label'>Disease Confidence</div><div class='result-value'>{final_conf:.1f}%</div></div>", unsafe_allow_html=True)

        # خطة العلاج
        st.markdown("<h3 class='section-title'>💊 Step-by-Step Treatment Plan</h3>", unsafe_allow_html=True)
        treatment_key = f"{crop_name}___{disease_name}"
        steps = treatments.get(treatment_key)
        
        if not steps:
            steps = {
                "immediate": "Isolate the plant and remove visible infected parts.",
                "organic": "Consult a local expert for organic options.",
                "chemical": "Consult an agronomist before applying chemicals.",
                "prevention": "Ensure good air circulation and proper watering."
            }

        st.markdown(f"<div class='treat-box'><div class='treat-title'>🚨 Immediate Actions</div><div class='treat-desc'>{steps['immediate']}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='treat-box' style='border-left-color: #10b981;'><div class='treat-title'>🌱 Organic Treatment</div><div class='treat-desc'>{steps['organic']}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='treat-box' style='border-left-color: #f59e0b;'><div class='treat-title'>🧪 Chemical Treatment</div><div class='treat-desc'>{steps['chemical']}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='treat-box' style='border-left-color: #6366f1;'><div class='treat-title'>🛡️ Prevention (Long-term)</div><div class='treat-desc'>{steps['prevention']}</div></div>", unsafe_allow_html=True)

else:
    st.info("👈 Please upload a clear image of a plant leaf from the sidebar.")
