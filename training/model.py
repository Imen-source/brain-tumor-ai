import torch.nn as nn
from torchvision import models

def create_model(num_classes=4):

    model = models.resnet18(pretrained=True)

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    return model
