import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

import mediapipe as mp
from models.model import load_model  # Your EfficientNet-B3 loader

# ---------------- Model Setup ----------------
MODEL_PATH = "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 29

model = load_model(num_classes=NUM_CLASSES, model_path=MODEL_PATH, device=DEVICE)
CLASS_NAMES = [
    'Q','U','A','del','nothing','F','V','B','S','H',
    'C','Z','D','N','L','Y','I','E','J','T',
    'O','R','P','W','G','space','M','K','X'
]

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

# ---------------- MediaPipe Setup ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------------- Video Transformer ----------------
class ASLVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, c = img.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords)*w), int(max(x_coords)*w)
                y_min, y_max = int(min(y_coords)*h), int(max(y_coords)*h)

                hand_crop = img[y_min:y_max, x_min:x_max]
                if hand_crop.size != 0:
                    hand_image = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
                    label = predict(hand_image, model, DEVICE)
                    cv2.putText(img, f"Letter: {label}", (x_min, y_min-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return img

# ---------------- Streamlit UI ----------------
st.title("✋ ASL Real-Time Sign Recognition (Mobile & Desktop)")

webrtc_streamer(
    key="asl-stream",
    video_transformer_factory=ASLVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)

