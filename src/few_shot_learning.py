import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import os
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image

def load_cinic10(data_root, split='train', few_shot_per_class=10, batch_size=16, dataset_name="cinic-10"):
    if split not in ['train', 'test']:
        data_dir = os.path.join(data_root, dataset_name)
    else:
        data_dir = os.path.join(data_root, dataset_name, split)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Ensure images are 32x32
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Few-shot sampling
    class_indices = {label: [] for label in range(10)}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    few_shot_indices = []
    for indices in class_indices.values():
        few_shot_indices.extend(random.sample(indices, min(len(indices),
                                                           few_shot_per_class)))  # Handle cases where a class has
        # fewer than few_shot_per_class images

    few_shot_dataset = Subset(dataset, few_shot_indices)
    dataloader = DataLoader(few_shot_dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def calculate_accuracy(model, data_root, split='test', batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loader = load_cinic10(data_root, split=split, few_shot_per_class=1000,
                               batch_size=batch_size)  # Use a larger subset

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the class with highest probability
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = (correct / total) * 100
    print(f"Accuracy on {split} set: {accuracy:.2f}%")
    return accuracy


def plot_confusion_matrix(model, test_loader, class_names=None, title="Confusion Matrix"):
    if class_names is None:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()


class FewShotConvNeXt(nn.Module):
    def __init__(self, num_classes=10, dropout=0.0):
        super(FewShotConvNeXt, self).__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=True)  # Load pretrained ConvNeXt-Tiny

        if dropout > 0.0:
            self.backbone.head.fc = nn.Sequential(
                nn.Linear(self.backbone.head.fc.in_features, num_classes),
                nn.Dropout(dropout)
            )
        else:
            self.backbone.head.fc = nn.Linear(self.backbone.head.fc.in_features, num_classes)  # Modify last layer

    def forward(self, x):
        return self.backbone(x)


def train_few_shot_convnext(model, dataloader, epochs=10, lr=0.001, weight_decay=None):
    # Freeze all layers
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Unfreeze the final classification layer (head.fc)
    for param in model.backbone.head.fc.parameters():
        param.requires_grad = True

    # Unfreeze specific stages (e.g., last two stages)
    for param in model.backbone.stages[2].parameters():  # Unfreeze Stage 3
        param.requires_grad = True
    for param in model.backbone.stages[3].parameters():  # Unfreeze Stage 4
        param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    if weight_decay is not None:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in dataloader:  # Load few-shot dataset
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = (total_correct / total_samples) * 100
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%")


def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)  # For CPU
    torch.cuda.manual_seed_all(seed_value)  # For GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict_image(model, image_path, class_names=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    if class_names is None:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Ensure size matches training input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path).convert("RGB")  # Ensure it's in RGB mode
    image = transform(image).unsqueeze(0)  # Add batch dimension

    model.to(device)
    model.eval()

    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        probabilities = F.softmax(output, dim=1)  # Convert to probabilities
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return class_names[predicted_class]