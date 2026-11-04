"""
Prediction script for Verdant AI.
Loads the trained PyTorch models and performs inference on plant leaf images.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json

def load_model(model_path: str, classes_path: str) -> tuple:
    """
    Loads a PyTorch model and its corresponding class names.
    
    Args:
        model_path (str): Path to the .pth model weights.
        classes_path (str): Path to the JSON file containing class names.
        
    Returns:
        tuple: (model, classes)
    """
    with open(classes_path, 'r') as f:
        classes = json.load(f)
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, classes

def predict(image_path: str, model: torch.nn.Module, classes: list) -> tuple:
    """
    Performs inference on a single image.
    
    Args:
        image_path (str): Path to the input image.
        model (torch.nn.Module): The loaded PyTorch model.
        classes (list): List of class names.
        
    Returns:
        tuple: (predicted_class, confidence_score)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, 0)
        
    return classes[pred.item()], conf.item() * 100

if __name__ == '__main__':
    print("Prediction script is ready for integration.")
