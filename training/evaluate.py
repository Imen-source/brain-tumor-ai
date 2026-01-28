import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import create_dataloaders
from model import create_model

DATA_DIR = "./data/raw"
MODEL_PATH = "./models/best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():

    _, _, test_loader, classes = create_dataloaders(DATA_DIR)

    model = create_model(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Brain Tumor Classification – Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate()
