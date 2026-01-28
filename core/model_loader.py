import torch
from ml.model import create_model
from core.config import settings


class ModelService:
    def __init__(self, num_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(num_classes)
        self.model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, tensor):
        with torch.no_grad():
            return self.model(tensor)