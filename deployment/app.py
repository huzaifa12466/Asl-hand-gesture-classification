import os
import gdown
from PIL import Image
import torch
import torchvision.transforms as transforms
import streamlit as st
from models.model import load_model  # EfficientNet-B3 loader

# ---------------- Model Setup ----------------
MODEL_PATH = "best_model.pth"
GDRIVE_FILE_ID = "158dfL0MYHEWUUNBNGyVAKrFAJTkflmjX"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 29

# Download model if missing
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded!")

# Load model
model = load_model(num_classes=NUM_CLASSES, model_path=MODEL_PATH, device=DEVICE)

# Class names
CLASS_NAMES = [
    'Q','U','A','del','nothing','F','V','B','S','H',
    'C','Z','D','N','L','Y','I','E','J','T',
    'O','R','P','W','G','space','M','K','X'
]

# Image transforms
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Prediction function
def predict(image, model, device):
    tensor = val_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
    return CLASS_NAMES[pred.item()]

# ---------------- Streamlit UI ----------------
st.title("✋ ASL Hand Gesture Recognition (Image Upload Only)")
st.write("Upload a hand image and the model will predict the ASL letter.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label = predict(image, model, DEVICE)
    st.success(f"Predicted ASL Letter: {label}")
