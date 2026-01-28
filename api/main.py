import os
import io
import base64
import logging
from io import BytesIO

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv

from training.model import create_model
from training.gradcam import generate_gradcam

# -----------------------
# Load environment variables
# -----------------------
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "training/best_model.pth")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 224))

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Logging
# -----------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------
# App
# -----------------------
app = FastAPI(
    title="Brain Tumor Detection API",
    version=MODEL_VERSION
)

# CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Load model
# -----------------------
model = create_model(num_classes=len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

logging.info(f"Model loaded from {MODEL_PATH} on {device}")

# -----------------------
# Image preprocessing
# -----------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor

def pil_to_base64(pil_img: Image.Image) -> str:
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# -----------------------
# Routes
# -----------------------

@app.get("/")
def root():
    return {
        "message": "Brain Tumor Detection API is running.",
        "model_version": MODEL_VERSION
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read & preprocess
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)

        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]

        confidence, predicted_class = torch.max(probs, 0)
        predicted_idx = predicted_class.item()

        probabilities = {
            CLASSES[i]: round(float(probs[i]), 4)
            for i in range(len(CLASSES))
        }

        # Grad-CAM
        gradcam_img = generate_gradcam(model, input_tensor, predicted_idx)
        gradcam_base64 = pil_to_base64(gradcam_img)

        result = {
            "prediction": CLASSES[predicted_idx],
            "confidence": round(float(confidence), 4),
            "probabilities": probabilities,
            "gradcam_image_base64": gradcam_base64
        }

        logging.info(f"Prediction: {result['prediction']} | Confidence: {result['confidence']}")

        return JSONResponse(content=result)

    except Exception as e:
        logging.exception("Prediction failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
