
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

st.set_page_config(page_title="Verdant - Smart Crop Doctor", page_icon="🌿", layout="wide")

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
    "Tomato___Early_blight": "Organic: Neem oil spray. Chemical: Chlorothalonil. Prevention: Crop rotation.",
    "Tomato___Late_blight": "Organic: Copper spray. Chemical: Mancozeb. Prevention: Avoid overhead watering.",
    "Apple___Apple_scab": "Organic: Sulfur spray. Chemical: Captan. Prevention: Remove fallen leaves.",
    "Potato___healthy": "Plant is healthy! Maintain regular watering and monitor for pests.",
}

st.title("🌿 Verdant - Smart Crop Doctor")
st.write("Upload a leaf image to instantly diagnose diseases and get treatment recommendations using Explainable AI.")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        diagnose_btn = st.button("🔍 Diagnose Now", type="primary")

if uploaded_file is not None and diagnose_btn:
    with col2:
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
            
            st.image(visualization, caption="AI Focus Area (Grad-CAM)", use_column_width=True)
            
        st.markdown("---")
        st.markdown(f"### Results for: **{crop}**")
        if disease.lower() == 'healthy':
            st.success(f"✅ **Status:** Healthy (Confidence: {conf.item()*100:.1f}%)")
        else:
            st.error(f"🦠 **Disease Detected:** {disease} (Confidence: {conf.item()*100:.1f}%)")
        
        st.markdown("### 💊 Treatment Recommendation")
        st.info(treatments.get(pred_class, "Consult a local agricultural expert for specific treatment."))
