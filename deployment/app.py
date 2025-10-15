import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import streamlit as st
import cv2
import sys
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import gdown
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.model import load_model  # EfficientNet-B3 loader

# ---------------- Model Download & Setup ----------------
MODEL_PATH = "best_model.pth"
GDRIVE_FILE_ID = "158dfL0MYHEWUUNBNGyVAKrFAJTkflmjX"  # <-- Replace with your file ID
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 29

# Auto-download model if missing
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")

# Load model
model = load_model(num_classes=NUM_CLASSES, model_path=MODEL_PATH, device=DEVICE)

# ---------------- Class Names ----------------
CLASS_NAMES = [
    'Q','U','A','del','nothing','F','V','B','S','H',
    'C','Z','D','N','L','Y','I','E','J','T',
    'O','R','P','W','G','space','M','K','X'
]

# ---------------- Image Transforms ----------------
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict(image, model, device):
    tensor = val_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
    return CLASS_NAMES[pred.item()]

# ---------------- Haar Cascade Setup ----------------
# Replace with a proper hand cascade XML if available
hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'aGest.xml')  # Example placeholder

# ---------------- Video Transformer ----------------
class ASLVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hands = hand_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in hands:
            hand_crop = img[y:y+h, x:x+w]
            if hand_crop.size != 0:
                hand_image = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
                label = predict(hand_image, model, DEVICE)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(img, f"Letter: {label}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return img

# ---------------- Streamlit UI ----------------
st.title("✋ ASL Real-Time Sign Recognition (Haar Cascade + gdown)")
st.write("Mobile & Desktop compatible. Shows bounding box and letter prediction.")

webrtc_streamer(
    key="asl-haar-stream",
    video_transformer_factory=ASLVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)
