#!/usr/bin/env python3
"""
Train a quality predictor model that estimates conversion SSIM
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os
from PIL import Image
import cv2
from utils.image_utils import load_image_safe

class QualityPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: image features (10) + vtracer params (8) = 18
        self.network = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Output: predicted SSIM
            nn.Sigmoid()  # SSIM is 0-1
        )

    def forward(self, features, params):
        x = torch.cat([features, params], dim=1)
        return self.network(x)

def extract_features(image_path):
    """Extract simple features from image"""
    try:
        img = load_image_safe(image_path)
        cv_img = cv2.imread(image_path)

        if cv_img is None:
            # Return default features if image can't be loaded
            return [0.1] * 10

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # Extract various features
        features = [
            img.width / 1000,  # normalized width
            img.height / 1000,  # normalized height
            (img.width / img.height) / 3,  # aspect ratio normalized
            len(img.getcolors(maxcolors=256)) / 256 if img.getcolors(maxcolors=256) else 1,  # color count
            cv2.Laplacian(gray, cv2.CV_64F).var() / 10000,  # edge complexity
            np.std(cv_img) / 255,  # color variance
            np.mean(cv_img) / 255,  # mean brightness
            (cv_img.max() - cv_img.min()) / 255,  # contrast
            len(np.unique(gray)) / 256,  # unique gray levels
            gray.std() / 128,  # grayscale variance
        ]
        return features[:10]
    except Exception as e:
        print(f"Warning: Could not extract features from {image_path}: {e}")
        return [0.1] * 10

def train_quality_predictor():
    print("=" * 60)
    print("Training Quality Predictor")
    print("=" * 60)

    # Load training data
    with open("training_data.json") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} training samples")

    # Prepare features and labels
    X_features = []
    X_params = []
    y = []

    for item in data:
        features = extract_features(item['image_path'])

        # Normalize VTracer parameters
        params = [
            item['parameters'].get('color_precision', 6) / 10,
            item['parameters'].get('corner_threshold', 60) / 180,
            item['parameters'].get('segment_length', 4.0) / 10,
            item['parameters'].get('path_precision', 6) / 10,
            item['parameters'].get('layer_difference', 5) / 10,
            item['parameters'].get('mode', 0),  # 0 for default
            item['parameters'].get('filter_speckle', 4) / 10,
            item['parameters'].get('splice_threshold', 45) / 180,
        ][:8]  # Ensure exactly 8 parameters

        # Pad if needed
        while len(params) < 8:
            params.append(0.5)

        X_features.append(features)
        X_params.append(params)
        y.append(item['quality_score'])

    # Convert to tensors
    X_features = torch.FloatTensor(X_features)
    X_params = torch.FloatTensor(X_params)
    y = torch.FloatTensor(y).unsqueeze(1)

    print(f"Features shape: {X_features.shape}")
    print(f"Params shape: {X_params.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"SSIM range: {y.min().item():.3f} - {y.max().item():.3f}")

    # Split data
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_features_train = X_features[train_idx]
    X_params_train = X_params[train_idx]
    y_train = y[train_idx]

    X_features_val = X_features[val_idx]
    X_params_val = X_params[val_idx]
    y_val = y[val_idx]

    # Train model
    model = QualityPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining...")
    best_val_loss = float('inf')

    model.train()
    for epoch in range(200):
        # Training
        optimizer.zero_grad()
        predictions = model(X_features_train, X_params_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        with torch.no_grad():
            val_predictions = model(X_features_val, X_params_val)
            val_loss = criterion(val_predictions, y_val)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} - Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/quality_predictor.pth")

    print(f"\n✅ Best validation loss: {best_val_loss.item():.6f}")

    # Save as TorchScript
    os.makedirs("models/production", exist_ok=True)
    model.eval()

    # Use trace instead of script for this model
    dummy_features = torch.randn(1, 10)
    dummy_params = torch.randn(1, 8)
    traced_model = torch.jit.trace(model, (dummy_features, dummy_params))
    traced_model.save("models/production/quality_predictor.torchscript")
    print("✅ Saved quality predictor to models/production/quality_predictor.torchscript")

    # Test prediction
    print("\nTesting prediction...")
    test_features = X_features[0:1]
    test_params = X_params[0:1]
    actual_ssim = y[0].item()

    with torch.no_grad():
        predicted_ssim = model(test_features, test_params).item()

    print(f"Sample prediction:")
    print(f"  Actual SSIM: {actual_ssim:.3f}")
    print(f"  Predicted SSIM: {predicted_ssim:.3f}")
    print(f"  Error: {abs(actual_ssim - predicted_ssim):.3f}")

if __name__ == "__main__":
    train_quality_predictor()