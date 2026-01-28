import io
import base64
import torch
import torch.nn.functional as F
from fastapi import APIRouter, File, UploadFile
from PIL import Image
from torchvision import transforms


from core.model_loader import ModelService
from core.logger import logger
from ml.gradcam import generate_gradcam
from api.schemas import PredictionResponse
from core.config import settings


router = APIRouter()


CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


model_service = ModelService(len(CLASSES))


transform = transforms.Compose([
transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(model_service.device)

    outputs = model_service.predict(tensor)
    probs_tensor = F.softmax(outputs, dim=1)[0]

    confidence, class_idx = torch.max(probs_tensor, 0)

    probabilities = {
        CLASSES[i]: round(probs_tensor[i].item(), 4)
        for i in range(len(CLASSES))
    }

    gradcam_img = generate_gradcam(model_service.model, tensor, class_idx.item())

    buffer = io.BytesIO()
    gradcam_img.save(buffer, format="PNG")
    gradcam_base64 = base64.b64encode(buffer.getvalue()).decode()

    logger.info(f"Prediction: {CLASSES[class_idx]} | Confidence: {confidence.item():.4f}")

    return PredictionResponse(
        prediction=CLASSES[class_idx],
        confidence=round(confidence.item(), 4),
        probabilities=probabilities,
        gradcam_image_base64=gradcam_base64
    )