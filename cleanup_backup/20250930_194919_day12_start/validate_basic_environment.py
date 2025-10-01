#!/usr/bin/env python3
"""
Basic Neural Network Environment Validation

Tests core functionality without requiring pre-trained model downloads.
"""

import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def test_basic_functionality():
    """Test basic PyTorch and EfficientNet functionality."""
    print("=== Basic Environment Test ===")

    # Test 1: PyTorch basics
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)

    # Test 2: Tensor operations
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = x + y
    print("✓ Tensor operations working")

    # Test 3: Model creation (without pretrained weights)
    try:
        model = models.efficientnet_b0(weights=None)  # No pretrained weights
        print("✓ EfficientNet-B0 architecture created")

        # Test model structure
        print(f"✓ Model classifier: {model.classifier}")

        # Modify classifier for our use case
        num_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(num_features, 4)
        )
        print("✓ Model classifier modified for 4 classes")

    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

    # Test 4: Image preprocessing
    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Create test image
        test_image = Image.new('RGB', (300, 300), color='red')
        input_tensor = transform(test_image)

        print(f"✓ Image preprocessing working - shape: {input_tensor.shape}")

    except Exception as e:
        print(f"✗ Image preprocessing failed: {e}")
        return False

    # Test 5: Model inference (with random weights)
    try:
        model.eval()
        batch_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        print(f"✓ Model inference working - output shape: {outputs.shape}")
        print(f"✓ Softmax probabilities: {probabilities.shape}")

    except Exception as e:
        print(f"✗ Model inference failed: {e}")
        return False

    print("\n✓ ALL BASIC TESTS PASSED - Core environment working!")
    return True

def test_pretrained_download():
    """Test pretrained model download separately."""
    print("\n=== Pretrained Model Download Test ===")

    try:
        print("Attempting to download pretrained EfficientNet-B0...")
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        print("✓ Pretrained model downloaded successfully")
        return True

    except Exception as e:
        print(f"⚠ Pretrained model download failed: {e}")
        print("This is not critical - we can train from scratch if needed")
        return False

def main():
    """Run validation tests."""
    print("Neural Network Basic Environment Validation")
    print("=" * 50)

    # Test basic functionality first
    basic_success = test_basic_functionality()

    if basic_success:
        # Only test pretrained if basic works
        pretrained_success = test_pretrained_download()

        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)

        if pretrained_success:
            print("✓ Environment fully ready - pretrained models working")
        else:
            print("⚠ Environment partially ready - can train from scratch")
            print("  Pretrained model download has issues but basic functionality works")

        print("✓ Ready to proceed with neural network implementation")
        return True
    else:
        print("\n✗ Basic environment test failed - needs attention")
        return False

if __name__ == "__main__":
    success = main()