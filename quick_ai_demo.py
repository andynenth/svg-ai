#!/usr/bin/env python3
"""
Quick AI demo - creates minimal working AI models in 1 minute
This activates all AI features with simple placeholder models
"""

import torch
import torch.nn as nn
import joblib
import os
from pathlib import Path

def create_minimal_ai_models():
    """Create minimal working AI models to activate AI features"""

    # Create models directory
    os.makedirs("models/production", exist_ok=True)
    print("📁 Created models/production directory")

    # 1. Logo Classifier (detects logo type)
    class MinimalClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            # Minimal CNN for image classification
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc = nn.Linear(32 * 56 * 56, 5)  # 5 logo types

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    classifier = MinimalClassifier()
    scripted_classifier = torch.jit.script(classifier)
    scripted_classifier.save("models/production/logo_classifier.torchscript")
    print("✅ Created logo classifier model")

    # 2. Quality Predictor (predicts SSIM score)
    class MinimalQualityPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            # Input: image features (224*224*3 flattened) + params (8)
            self.fc1 = nn.Linear(224*224*3, 100)
            self.fc2 = nn.Linear(100 + 8, 50)
            self.fc3 = nn.Linear(50, 1)

        def forward(self, image_features, params):
            # Flatten image
            image_flat = image_features.view(image_features.size(0), -1)
            x = torch.relu(self.fc1(image_flat))
            # Concatenate with parameters
            x = torch.cat([x, params], dim=1)
            x = torch.relu(self.fc2(x))
            return torch.sigmoid(self.fc3(x))  # SSIM is 0-1

    predictor = MinimalQualityPredictor()

    # Create dummy input for tracing
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_params = torch.randn(1, 8)

    traced_predictor = torch.jit.trace(predictor, (dummy_image, dummy_params))
    traced_predictor.save("models/production/quality_predictor.torchscript")
    print("✅ Created quality predictor model")

    # 3. Parameter Optimizer (correlation models)
    # Simple dictionary that suggests parameters based on logo type
    correlation_models = {
        'simple_geometric': {
            'color_precision': 4,
            'corner_threshold': 30,
            'segment_length': 5.0,
            'path_precision': 8
        },
        'text_based': {
            'color_precision': 2,
            'corner_threshold': 20,
            'segment_length': 10.0,
            'path_precision': 10
        },
        'gradient': {
            'color_precision': 8,
            'corner_threshold': 90,
            'segment_length': 3.0,
            'layer_difference': 8
        },
        'complex': {
            'color_precision': 10,
            'corner_threshold': 60,
            'max_iterations': 20,
            'splice_threshold': 60
        },
        'default': {
            'color_precision': 6,
            'corner_threshold': 60,
            'segment_length': 4.0,
            'path_precision': 6
        }
    }

    joblib.dump(correlation_models, "models/production/correlation_models.pkl")
    print("✅ Created parameter optimizer models")

    # 4. Export classifier to ONNX format (optional, for compatibility)
    try:
        import onnx
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            classifier,
            dummy_input,
            "models/production/logo_classifier.onnx",
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("✅ Exported classifier to ONNX format")
    except ImportError:
        print("⚠️  ONNX not installed, skipping ONNX export")

    print("\n" + "=" * 60)
    print("🎉 AI MODELS CREATED SUCCESSFULLY!")
    print("=" * 60)
    print("\nThe AI features are now activated with minimal models.")
    print("These models will:")
    print("  • Classify logos into 5 types")
    print("  • Predict conversion quality")
    print("  • Suggest optimal VTracer parameters")
    print("\nNext steps:")
    print("  1. Restart the server: pkill -f 'python -m backend.app' && python -m backend.app")
    print("  2. Test AI health: curl http://localhost:8001/api/ai-health")
    print("  3. Try AI conversion: curl -X POST http://localhost:8001/api/convert-ai ...")
    print("\nNote: These are placeholder models. For better results, train with real data.")

if __name__ == "__main__":
    create_minimal_ai_models()