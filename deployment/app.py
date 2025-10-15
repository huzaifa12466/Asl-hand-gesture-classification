import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import streamlit as st
import cv2
import gdown
import tempfile
from models.model import load_model  # EfficientNet-B3 loader

# ---------------- Model Download & Setup ----------------
MODEL_PATH = "best_model.pth"
GDRIVE_FILE_ID = "158dfL0MYHEWUUNBNGyVAKrFAJTkflmjX"  # Replace with your file ID
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

# ---------------- Streamlit UI ----------------
st.title("✋ ASL Video Hand Gesture Recognition")
st.write("Upload a video. Each frame will be analyzed, bounding box drawn, and letters detected. Sentence constructed at the end.")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    frames = []
    sentence = []

    # Optional: Haar cascade for hand detection
    hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'aGest.xml')  # replace with proper hand cascade

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hands = hand_cascade.detectMultiScale(gray, 1.1, 5)

        letters_in_frame = []

        for (x, y, w, h) in hands:
            hand_crop = frame[y:y+h, x:x+w]
            if hand_crop.size != 0:
                hand_image = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
                label = predict(hand_image, model, DEVICE)
                letters_in_frame.append(label)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, f"{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if letters_in_frame:
            # Take majority vote in frame
            sentence.append(max(set(letters_in_frame), key=letters_in_frame.count))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)
        frames.append(frame_rgb)

    cap.release()

    # Construct sentence
    final_sentence = "".join(sentence)
    st.success(f"Detected Sentence: {final_sentence}")

    # Optional: save output video with bounding boxes
    save_path = "output_video.mp4"
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    st.video(save_path)

