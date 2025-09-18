import os
import sys
import subprocess

# ---------------- Install required packages if missing ----------------
for package in ["streamlit", "gdown"]:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
       
import streamlit as st
import gdown
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.model import load_model  # EfficientNet-B3 loader


# ---------------- Download model if missing ----------------
MODEL_PATH = "best_model.pth"
DRIVE_FILE_ID = "158dfL0MYHEWUUNBNGyVAKrFAJTkflmjX"

if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# ---------------- Load model ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 29
model = load_model(num_classes=NUM_CLASSES, model_path=MODEL_PATH, device=DEVICE)

# ---------------- Class names ----------------
CLASS_NAMES = [
    'Q', 'U', 'A', 'del', 'nothing', 'F', 'V', 'B', 'S', 'H',
    'C', 'Z', 'D', 'N', 'L', 'Y', 'I', 'E', 'J', 'T',
    'O', 'R', 'P', 'W', 'G', 'space', 'M', 'K', 'X'
]

# ---------------- Image transforms ----------------
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- Prediction function ----------------
def predict(image, model, device):
    tensor = val_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
    return CLASS_NAMES[pred.item()]

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="ASL Sign Recognition", page_icon="✋", layout="centered")

st.title("✋ ASL Sign Language Recognition")
st.subheader("Detect American Sign Language letters (A–Z) + Special Signs")
st.write("Upload an image **or** take a photo with your webcam, then click **Predict**:")

uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("📸 Capture from Webcam")

image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
elif camera_image:
    image = Image.open(camera_image).convert("RGB")
    st.image(image, caption="Webcam Image", use_container_width=True)

if image and st.button("🔮 Predict"):
    label = predict(image, model, DEVICE)
    st.success(f"✅ Prediction: **{label}**")
