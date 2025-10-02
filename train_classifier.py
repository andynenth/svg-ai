#!/usr/bin/env python3
"""
Train a logo classifier to identify logo types
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from pathlib import Path

class LogoDataset(Dataset):
    def __init__(self, data_file="training_data.json"):
        with open(data_file) as f:
            self.data = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Map logo types to classes
        self.classes = ['simple_geometric', 'text_based', 'gradients', 'complex', 'abstract']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        image = self.transform(image)

        label = self.class_to_idx.get(item['logo_type'], 4)  # default to 'abstract'
        return image, label

class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_classifier():
    print("=" * 60)
    print("Training Logo Classifier")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = LogoDataset()
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Classes: {dataset.classes}")

    # Split dataset (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = SimpleClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    print("\nTraining...")
    best_acc = 0.0

    for epoch in range(20):  # Quick training
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()

        # Validation phase
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / len(train_dataset)
        val_acc = 100 * val_correct / len(val_dataset)

        print(f"Epoch {epoch+1}/20 - Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/classifier.pth")

    print(f"\n✅ Best validation accuracy: {best_acc:.1f}%")

    # Save as TorchScript
    os.makedirs("models/production", exist_ok=True)
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save("models/production/logo_classifier.torchscript")
    print("✅ Saved classifier to models/production/logo_classifier.torchscript")

    # Test on a sample
    print("\nTesting on a sample image...")
    test_image = "data/logos/simple_geometric/circle_00.png"
    img = Image.open(test_image).convert('RGB')
    img_tensor = dataset.transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = dataset.classes[predicted.item()]
        print(f"Sample: {test_image}")
        print(f"Predicted: {predicted_class}")

if __name__ == "__main__":
    train_classifier()