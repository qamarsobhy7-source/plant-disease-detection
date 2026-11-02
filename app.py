
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

st.set_page_config(page_title="Verdant AI", page_icon="🌿", layout="centered")

st.markdown("""
<style>
    /* إخفاء عناصر Streamlit الافتراضية */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #f8fafc; }

    /* العناوين */
    .main-title { font-size: 2.5rem; font-weight: 700; color: #0f172a; text-align: center; margin-bottom: 0.2rem; margin-top: -2rem; }
    .sub-title { font-size: 1.1rem; color: #64748b; text-align: center; margin-bottom: 2rem; }

    /* تصميم مربع رفع الملفات (الأهم) */
    [data-testid="stFileUploader"] {
        background-color: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 2px dashed #10b981;
        margin-bottom: 2rem;
    }
    [data-testid="stFileUploader"] > label { color: #0f172a; font-weight: 600; font-size: 1.1rem; text-align: center; display: block; }
    [data-testid="stFileUploader"] section { background: #f0fdf4; border-radius: 10px; border: none; }
    [data-testid="stFileUploader"] button { background: #10b981; color: white; border: none; border-radius: 8px; padding: 0.5rem 1.5rem; }
    [data-testid="stFileUploader"] button:hover { background: #059669; }

    /* الأزرار */
    .stButton>button { background: #10b981; color: white; border: none; border-radius: 8px; padding: 0.75rem 2rem; font-weight: 600; font-size: 1rem; width: 100%; box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2); }
    .stButton>button:hover { background: #059669; }

    /* كروت النتائج */
    .result-card { background: white; border-radius: 12px; padding: 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #10b981; margin-bottom: 1rem; text-align: center; }
    .result-card.danger { border-left: 4px solid #ef4444; }
    .result-label { font-size: 0.8rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .result-value { font-size: 1.4rem; color: #0f172a; font-weight: 700; margin-top: 0.2rem; }

    /* خطوات كيف يعمل التطبيق */
    .step-box { text-align: center; padding: 1rem; }
    .step-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .step-title { font-weight: 600; color: #0f172a; margin-bottom: 0.2rem; }
    .step-desc { font-size: 0.85rem; color: #64748b; }
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

# نفس قاموس العلاجات الكامل اللي عملناه
treatments = {
    "Apple___Apple_scab": "Organic: Sulfur or copper sprays. Chemical: Captan. Prevention: Remove fallen leaves.",
    "Apple___Black_rot": "Organic: Copper fungicides. Chemical: Captan. Prevention: Remove infected branches.",
    "Apple___Cedar_apple_rust": "Organic: Neem oil. Chemical: Myclobutanil. Prevention: Remove nearby cedar trees.",
    "Apple___healthy": "Plant is healthy! Maintain regular watering.",
    "Blueberry___healthy": "Plant is healthy! Ensure acidic soil.",
    "Cherry_(including_sour)___Powdery_mildew": "Organic: Neem oil. Chemical: Sulfur. Prevention: Good air circulation.",
    "Cherry_(including_sour)___healthy": "Plant is healthy! Prune annually.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Organic: Crop rotation. Chemical: Strobilurin. Prevention: Resistant hybrids.",
    "Corn_(maize)___Common_rust_": "Organic: Sulfur dust. Chemical: Azoxystrobin. Prevention: Plant early.",
    "Corn_(maize)___Northern_Leaf_Blight": "Organic: Neem oil. Chemical: Propiconazole. Prevention: Residue management.",
    "Corn_(maize)___healthy": "Plant is healthy! Maintain nitrogen levels.",
    "Grape___Black_rot": "Organic: Copper. Chemical: Myclobutanil. Prevention: Canopy management.",
    "Grape___Esca_(Black_Measles)": "Organic: Wound protection. Chemical: No cure, remove vines. Prevention: Avoid large wounds.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Organic: Copper spray. Chemical: Mancozeb. Prevention: Air circulation.",
    "Grape___healthy": "Plant is healthy! Monitor for pests.",
    "Orange___Haunglongbing_(Citrus_greening)": "Organic: Control psyllid. Chemical: Imidacloprid. Prevention: Disease-free stock.",
    "Peach___Bacterial_spot": "Organic: Copper sprays. Chemical: Oxytetracycline. Prevention: Resistant varieties.",
    "Peach___healthy": "Plant is healthy! Prune regularly.",
    "Pepper,_bell___Bacterial_spot": "Organic: Copper. Chemical: Fixed copper. Prevention: Pathogen-free seeds.",
    "Pepper,_bell___healthy": "Plant is healthy! Consistent moisture.",
    "Potato___Early_blight": "Organic: Neem oil. Chemical: Chlorothalonil. Prevention: Crop rotation.",
    "Potato___Late_blight": "Organic: Copper. Chemical: Mancozeb. Prevention: Certified seeds.",
    "Potato___healthy": "Plant is healthy! Hill soil around stems.",
    "Raspberry___healthy": "Plant is healthy! Prune old canes.",
    "Soybean___healthy": "Plant is healthy! Rotate crops.",
    "Squash___Powdery_mildew": "Organic: Potassium bicarbonate. Chemical: Myclobutanil. Prevention: Full sun.",
    "Strawberry___Leaf_scorch": "Organic: Remove infected leaves, Neem oil. Chemical: Captan. Prevention: Air circulation.",
    "Strawberry___healthy": "Plant is healthy! Renovate beds.",
    "Tomato___Bacterial_spot": "Organic: Copper. Chemical: Fixed copper. Prevention: Crop rotation.",
    "Tomato___Early_blight": "Organic: Neem oil. Chemical: Chlorothalonil. Prevention: Mulch.",
    "Tomato___Late_blight": "Organic: Copper. Chemical: Mancozeb. Prevention: Avoid overhead watering.",
    "Tomato___Leaf_Mold": "Organic: Ventilation. Chemical: Chlorothalonil. Prevention: Reduce humidity.",
    "Tomato___Septoria_leaf_spot": "Organic: Copper. Chemical: Mancozeb. Prevention: Remove lower leaves.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Organic: Insecticidal soap. Chemical: Abamectin. Prevention: Maintain humidity.",
    "Tomato___Target_Spot": "Organic: Copper. Chemical: Azoxystrobin. Prevention: Crop rotation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Organic: Control whiteflies. Chemical: Imidacloprid. Prevention: Row covers.",
    "Tomato___Tomato_mosaic_virus": "Organic: Remove infected plants. Chemical: None. Prevention: Disinfect tools.",
    "Tomato___healthy": "Plant is healthy! Provide support and water consistently."
}

# --- واجهة المستخدم ---
st.markdown("<h1 class='main-title'>🌿 Verdant AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Advanced Plant Disease Diagnosis with Explainable AI</p>", unsafe_allow_html=True)

# مربع الرفع
uploaded_file = st.file_uploader("Drag and drop or click to upload a leaf image", type=["jpg", "jpeg", "png"])

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
            
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original**")
                st.image(image, use_column_width=True)
            with c2:
                st.markdown("**AI Focus (Grad-CAM)**")
                st.image(visualization, use_column_width=True)

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

            st.markdown("### 💊 Treatment")
            st.info(treatments.get(pred_class, "Consult a local agricultural expert for specific treatment."))

else:
    # القسم اللي تحت عشان يملأ الفراغ ويبين احترافي (How it works)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #0f172a; margin-bottom: 2rem;'>How it works</h4>", unsafe_allow_html=True)
    
    step1, step2, step3 = st.columns(3)
    with step1:
        st.markdown("<div class='step-box'><div class='step-icon'>📸</div><div class='step-title'>1. Upload</div><div class='step-desc'>Take a clear photo of the infected leaf.</div></div>", unsafe_allow_html=True)
    with step2:
        st.markdown("<div class='step-box'><div class='step-icon'>🧠</div><div class='step-title'>2. Analyze</div><div class='step-desc'>Our AI scans the image for diseases.</div></div>", unsafe_allow_html=True)
    with step3:
        st.markdown("<div class='step-box'><div class='step-icon'>💊</div><div class='step-title'>3. Treat</div><div class='step-desc'>Get instant treatment recommendations.</div></div>", unsafe_allow_html=True)
