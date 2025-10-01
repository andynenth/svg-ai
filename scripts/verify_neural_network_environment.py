#!/usr/bin/env python3
"""
Neural Network Environment Validation Script

This script verifies that the AI environment is properly set up for
EfficientNet-B0 neural network training and inference.
"""

import sys
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import psutil
import os
import tempfile
import tracemalloc

def test_pytorch_installation():
    """Test PyTorch installation and basic functionality."""
    print("=== PyTorch Installation Test ===")

    # Version check
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")

    # Basic tensor operations
    try:
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x + y
        print("✓ Basic tensor operations working")
    except Exception as e:
        print(f"✗ Basic tensor operations failed: {e}")
        return False

    # Device verification
    device = torch.device('cpu')
    print(f"✓ Default device: {device}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")

    return True

def test_model_loading():
    """Test EfficientNet-B0 model loading capabilities."""
    print("\n=== Model Loading Test ===")

    try:
        # Test loading pre-trained EfficientNet-B0
        print("Loading EfficientNet-B0...")
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        print("✓ EfficientNet-B0 loaded successfully")

        # Test model modification for classification
        num_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(num_features, 4)  # 4 logo classes
        )
        print("✓ Model classifier modified for 4 classes")

        # Test model to CPU device
        model.to('cpu')
        model.eval()
        print("✓ Model set to CPU and eval mode")

        return True, model

    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False, None

def test_image_preprocessing():
    """Test image preprocessing pipeline."""
    print("\n=== Image Preprocessing Test ===")

    try:
        # Create test transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        print("✓ Transform pipeline created")

        # Create test image
        test_image = Image.new('RGB', (300, 300), color='red')
        print("✓ Test image created")

        # Apply transforms
        input_tensor = transform(test_image)
        print(f"✓ Transform applied - tensor shape: {input_tensor.shape}")

        # Test batch processing
        batch_tensor = input_tensor.unsqueeze(0)
        print(f"✓ Batch tensor created - shape: {batch_tensor.shape}")

        return True, transform

    except Exception as e:
        print(f"✗ Image preprocessing failed: {e}")
        return False, None

def test_inference_pipeline(model, transform):
    """Test complete inference pipeline."""
    print("\n=== Inference Pipeline Test ===")

    if model is None or transform is None:
        print("✗ Cannot test inference - model or transform failed")
        return False

    try:
        # Create test image
        test_image = Image.new('RGB', (256, 256), color='blue')

        # Preprocess image
        input_tensor = transform(test_image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        print(f"✓ Inference successful")
        print(f"  Predicted class: {predicted_class}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Output shape: {outputs.shape}")

        return True

    except Exception as e:
        print(f"✗ Inference pipeline failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage for model operations."""
    print("\n=== Memory Usage Test ===")

    try:
        # Start memory tracking
        tracemalloc.start()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Load model and perform operations
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.eval()

        # Create multiple test tensors
        tensors = []
        for i in range(10):
            tensor = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(tensor)
            tensors.append(output)

        # Check memory after operations
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory

        print(f"✓ Memory test completed")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory used: {memory_used:.1f} MB")

        # Check if memory usage is reasonable (<200MB as per targets)
        if memory_used < 200:
            print(f"✓ Memory usage within target (<200MB)")
        else:
            print(f"⚠ Memory usage high: {memory_used:.1f} MB")

        # Clean up
        del model, tensors
        tracemalloc.stop()

        return True

    except Exception as e:
        print(f"✗ Memory usage test failed: {e}")
        return False

def test_batch_processing():
    """Test batch processing capabilities."""
    print("\n=== Batch Processing Test ===")

    try:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.eval()

        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16]

        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 3, 224, 224)

            with torch.no_grad():
                outputs = model(input_tensor)

            print(f"✓ Batch size {batch_size}: Output shape {outputs.shape}")

        return True

    except Exception as e:
        print(f"✗ Batch processing test failed: {e}")
        return False

def main():
    """Run all environment validation tests."""
    print("Neural Network Environment Validation")
    print("=" * 50)

    tests_passed = 0
    total_tests = 6

    # Test 1: PyTorch Installation
    if test_pytorch_installation():
        tests_passed += 1

    # Test 2: Model Loading
    model_success, model = test_model_loading()
    if model_success:
        tests_passed += 1

    # Test 3: Image Preprocessing
    preprocess_success, transform = test_image_preprocessing()
    if preprocess_success:
        tests_passed += 1

    # Test 4: Inference Pipeline
    if test_inference_pipeline(model, transform):
        tests_passed += 1

    # Test 5: Memory Usage
    if test_memory_usage():
        tests_passed += 1

    # Test 6: Batch Processing
    if test_batch_processing():
        tests_passed += 1

    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("✓ ALL TESTS PASSED - Environment ready for neural network training!")
        return True
    else:
        print(f"✗ {total_tests - tests_passed} tests failed - Environment needs attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)