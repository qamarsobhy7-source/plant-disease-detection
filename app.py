
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

# إعدادات الصفحة
st.set_page_config(page_title="Verdant AI", page_icon="🌿", layout="centered")

# --- CSS مخصص لتصميم Modern & Clean ---
st.markdown("""
<style>
    /* إخفاء عناصر Streamlit الافتراضية */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* تغيير الخط والخلفية */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f8fafc; }

    /* تنسيق العنوان الرئيسي */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* تنسيق المربع الزجاجي للرفع */
    .upload-box {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
        margin-bottom: 1rem;
    }

    /* تنسيق أزرار Streamlit */
    .stButton>button {
        background: #10b981;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2);
    }
    .stButton>button:hover { background: #059669; transform: translateY(-1px); }

    /* كروت النتائج */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        border-left: 4px solid #10b981;
        margin-bottom: 1rem;
    }
    .result-card.danger { border-left: 4px solid #ef4444; }
    .result-label { font-size: 0.8rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .result-value { font-size: 1.3rem; color: #0f172a; font-weight: 700; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# --- تحميل الموديل ---
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
    "Apple___Apple_scab": "Organic: Sulfur or copper sprays. Chemical: Captan or Myclobutanil. Prevention: Remove fallen leaves, prune for airflow.",
    "Apple___Black_rot": "Organic: Copper fungicides. Chemical: Captan. Prevention: Remove infected branches and mummified fruits.",
    "Apple___Cedar_apple_rust": "Organic: Neem oil. Chemical: Myclobutanil. Prevention: Remove nearby cedar/juniper trees.",
    "Apple___healthy": "Plant is healthy! Maintain regular watering and monitor for pests.",
    "Blueberry___healthy": "Plant is healthy! Ensure acidic soil (pH 4.5-5.5) and proper drainage.",
    "Cherry_(including_sour)___Powdery_mildew": "Organic: Potassium bicarbonate or Neem oil. Chemical: Sulfur. Prevention: Ensure good air circulation.",
    "Cherry_(including_sour)___healthy": "Plant is healthy! Prune annually and monitor for pests.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Organic: Crop rotation. Chemical: Strobilurin fungicides. Prevention: Use resistant hybrids, tillage.",
    "Corn_(maize)___Common_rust_": "Organic: Sulfur dust. Chemical: Azoxystrobin. Prevention: Plant resistant varieties early.",
    "Corn_(maize)___Northern_Leaf_Blight": "Organic: Neem oil. Chemical: Propiconazole. Prevention: Crop rotation and residue management.",
    "Corn_(maize)___healthy": "Plant is healthy! Maintain proper nitrogen levels and watering.",
    "Grape___Black_rot": "Organic: Copper fungicides. Chemical: Myclobutanil. Prevention: Remove mummified berries, canopy management.",
    "Grape___Esca_(Black_Measles)": "Organic: Pruning wounds protection. Chemical: No effective chemical cure; remove infected vines. Prevention: Avoid large pruning wounds.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Organic: Copper spray. Chemical: Mancozeb. Prevention: Improve air circulation.",
    "Grape___healthy": "Plant is healthy! Monitor for pests and maintain vine balance.",
    "Orange___Haunglongbing_(Citrus_greening)": "Organic: No cure, control psyllid vector. Chemical: Imidacloprid for psyllids. Prevention: Use disease-free nursery stock.",
    "Peach___Bacterial_spot": "Organic: Copper sprays. Chemical: Oxytetracycline. Prevention: Plant resistant varieties, avoid overhead watering.",
    "Peach___healthy": "Plant is healthy! Prune regularly for light penetration.",
    "Pepper,_bell___Bacterial_spot": "Organic: Copper + Mancozeb. Chemical: Fixed copper. Prevention: Use pathogen-free seeds, crop rotation.",
    "Pepper,_bell___healthy": "Plant is healthy! Maintain consistent soil moisture.",
    "Potato___Early_blight": "Organic: Neem oil. Chemical: Chlorothalonil. Prevention: Crop rotation, avoid overhead irrigation.",
    "Potato___Late_blight": "Organic: Copper spray. Chemical: Mancozeb or Chlorothalonil. Prevention: Plant certified seeds, destroy cull piles.",
    "Potato___healthy": "Plant is healthy! Hill soil around stems to protect tubers.",
    "Raspberry___healthy": "Plant is healthy! Prune old canes and ensure good drainage.",
    "Soybean___healthy": "Plant is healthy! Rotate crops and monitor for aphids.",
    "Squash___Powdery_mildew": "Organic: Potassium bicarbonate, milk spray. Chemical: Myclobutanil. Prevention: Plant in full sun, space plants well.",
    "Strawberry___Leaf_scorch": "Organic: Remove infected leaves, Neem oil. Chemical: Captan. Prevention: Ensure good air circulation, avoid overhead watering.",
    "Strawberry___healthy": "Plant is healthy! Renovate beds after harvest and maintain mulch.",
    "Tomato___Bacterial_spot": "Organic: Copper sprays. Chemical: Fixed copper. Prevention: Use certified seeds, crop rotation.",
    "Tomato___Early_blight": "Organic: Neem oil or Copper. Chemical: Chlorothalonil. Prevention: Mulch to prevent soil splash, crop rotation.",
    "Tomato___Late_blight": "Organic: Copper spray. Chemical: Mancozeb. Prevention: Avoid overhead watering, destroy infected plants immediately.",
    "Tomato___Leaf_Mold": "Organic: Improve ventilation. Chemical: Chlorothalonil. Prevention: Reduce humidity in greenhouse.",
    "Tomato___Septoria_leaf_spot": "Organic: Copper spray. Chemical: Mancozeb. Prevention: Remove lower infected leaves, mulch.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Organic: Insecticidal soap or Neem oil. Chemical: Abamectin. Prevention: Maintain humidity, wash plants with water.",
    "Tomato___Target_Spot": "Organic: Copper fungicides. Chemical: Azoxystrobin. Prevention: Crop rotation, remove debris.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Organic: Control whiteflies. Chemical: Imidacloprid for whiteflies. Prevention: Use virus-resistant varieties, row covers.",
    "Tomato___Tomato_mosaic_virus": "Organic: No cure, remove infected plants. Chemical: None (control aphids). Prevention: Disinfect tools, wash hands before handling.",
    "Tomato___healthy": "Plant is healthy! Provide support (stakes/cages) and consistent watering."
}


# --- واجهة المستخدم ---
st.markdown("<h1 class='main-title'>🌿 Verdant AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Advanced Plant Disease Diagnosis with Explainable AI</p>", unsafe_allow_html=True)

# منطقة رفع الصورة في منتصف الشاشة
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a clear photo of a plant leaf", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    
    col_img, col_btn = st.columns([2, 1])
    with col_img:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    with col_btn:
        st.markdown("<br><br>", unsafe_allow_html=True)
        diagnose_btn = st.button("Analyze Leaf 🔍")
        
    if diagnose_btn:
        with st.spinner("Analyzing leaf..."):
            # Prediction
            input_tensor = val_transforms(image).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            conf, pred = torch.max(probs, 0)
            pred_class = unique_classes[pred.item()]
            
            crop, disease = pred_class.split('___')
            crop, disease = crop.replace('_', ' '), disease.replace('_', ' ')
            
            # Grad-CAM
            targets = [ClassifierOutputTarget(pred.item())]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            rgb_img = np.float32(image.resize((224, 224))) / 255
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            # عرض الصور
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original**")
                st.image(image, use_column_width=True)
            with c2:
                st.markdown("**AI Focus (Grad-CAM)**")
                st.image(visualization, use_column_width=True)

            # عرض النتائج
            st.markdown("### 📊 Results")
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

            # التوصيات
            st.markdown("### 💊 Treatment")
            st.info(treatments.get(pred_class, "Consult a local agricultural expert for specific treatment."))
