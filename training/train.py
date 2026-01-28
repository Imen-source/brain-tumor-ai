import torch
import torch.nn as nn
import torch.optim as optim
import os

from dataset import create_dataloaders
from model import create_model

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/raw")
EPOCHS = 5                 # fine-tuning epochs
LR = 0.0001               # lower LR for fine-tuning
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/best_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("DATA_DIR exists:", os.path.exists(DATA_DIR))
print("Training folder exists:", os.path.exists(os.path.join(DATA_DIR, "Training")))


def train():

    train_loader, val_loader, _, classes = create_dataloaders(DATA_DIR)

    model = create_model(num_classes=len(classes))

    # 🔓 Unfreeze last ResNet block for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Only optimize trainable parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # -------- Training --------
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # -------- Validation --------
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}")
        print(f"Val Acc: {val_acc:.3f}")

        # -------- Save best model --------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("✅ Best model saved.")

    print("Training complete.")


if __name__ == "__main__":
    train()
