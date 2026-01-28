import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

IMAGE_SIZE = 224
BATCH_SIZE = 32

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform


def create_dataloaders(data_dir, val_split=0.15):

    train_transform, val_test_transform = get_transforms()

    full_train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "Training"),
        transform=train_transform
    )

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    val_dataset.dataset.transform = val_test_transform

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "Testing"),
        transform=val_test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, full_train_dataset.classes
