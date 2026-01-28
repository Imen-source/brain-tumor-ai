import os
from dataset import create_dataloaders

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

train_loader, val_loader, test_loader, classes = create_dataloaders(DATA_DIR)

print("Classes:", classes)
print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))
print("Test batches:", len(test_loader))

images, labels = next(iter(train_loader))
print("Batch shape:", images.shape)
