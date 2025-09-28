#!/usr/bin/env python3
"""PyTorch Integration Tests"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import tempfile
import os
import cv2

def test_model_loading():
    """Test pre-trained model loading"""
    print("📦 Testing model loading...")

    # Test ResNet-50 (for quality prediction)
    resnet = models.resnet50(weights=None)
    print("✅ ResNet-50 loaded")

    # Test EfficientNet-B0 (for classification)
    efficientnet = models.efficientnet_b0(weights=None)
    print("✅ EfficientNet-B0 loaded")

    return True

def test_model_save_load():
    """Test saving and loading custom models"""
    print("💾 Testing model save/load...")

    # Create simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2)
    )

    # Save model
    temp_path = tempfile.mktemp(suffix='.pth')
    torch.save(model.state_dict(), temp_path)
    print(f"✅ Model saved to {temp_path}")

    # Load model
    loaded_model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2)
    )
    loaded_model.load_state_dict(torch.load(temp_path))
    print("✅ Model loaded successfully")

    # Test that models produce same output
    test_input = torch.randn(1, 10)
    original_output = model(test_input)
    loaded_output = loaded_model(test_input)

    if torch.allclose(original_output, loaded_output):
        print("✅ Save/load integrity verified")
    else:
        print("❌ Save/load integrity failed")
        return False

    # Clean up
    os.unlink(temp_path)
    return True

def test_image_processing():
    """Test image processing pipeline"""
    print("🖼️  Testing image processing...")

    # Create dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    print("✅ PIL image created")

    # Test transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    tensor_image = transform(dummy_image)
    print(f"✅ Image processed: {tensor_image.shape}")

    # Test with OpenCV
    cv_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    cv_tensor = transform(pil_image)
    print(f"✅ OpenCV→PIL→Tensor: {cv_tensor.shape}")

    return True

def test_tensor_conversions():
    """Test tensor conversions (NumPy ↔ PyTorch)"""
    print("🔄 Testing tensor conversions...")

    # NumPy to PyTorch
    np_array = np.random.randn(10, 10)
    torch_tensor = torch.from_numpy(np_array)
    print(f"✅ NumPy→PyTorch: {np_array.shape} → {torch_tensor.shape}")

    # PyTorch to NumPy
    torch_tensor2 = torch.randn(5, 5)
    np_array2 = torch_tensor2.numpy()
    print(f"✅ PyTorch→NumPy: {torch_tensor2.shape} → {np_array2.shape}")

    # Test data integrity
    if np.allclose(np_array, torch_tensor.numpy()):
        print("✅ NumPy→PyTorch conversion integrity verified")
    else:
        print("❌ NumPy→PyTorch conversion failed")
        return False

    if np.allclose(torch_tensor2.numpy(), np_array2):
        print("✅ PyTorch→NumPy conversion integrity verified")
    else:
        print("❌ PyTorch→NumPy conversion failed")
        return False

    return True

def test_ai_pipeline_functions():
    """Test functions needed for AI pipeline"""
    print("🤖 Testing AI pipeline functions...")

    # Test image feature extraction simulation
    dummy_image = torch.randn(3, 224, 224)  # CHW format

    # Simulate feature extraction with ResNet
    resnet = models.resnet50(weights=None)
    resnet.eval()

    with torch.no_grad():
        # Remove final classification layer to get features
        features = torch.nn.Sequential(*list(resnet.children())[:-1])
        feature_vector = features(dummy_image.unsqueeze(0))  # Add batch dimension
        feature_vector = feature_vector.squeeze()  # Remove batch and spatial dims

    print(f"✅ Feature extraction: {feature_vector.shape}")

    # Test quality prediction simulation
    class QualityPredictor(torch.nn.Module):
        def __init__(self, feature_dim=2048, param_dim=8):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(feature_dim + param_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1),
                torch.nn.Sigmoid()  # Quality score 0-1
            )

        def forward(self, features, parameters):
            combined = torch.cat([features, parameters], dim=-1)
            return self.network(combined)

    # Test quality predictor
    predictor = QualityPredictor()
    dummy_params = torch.randn(8)  # VTracer parameters
    quality_score = predictor(feature_vector, dummy_params)
    print(f"✅ Quality prediction: {quality_score.item():.3f}")

    return True

def main():
    """Run all integration tests"""
    print("🧪 PyTorch Integration Test Suite")
    print("=" * 40)

    tests = [
        ("Model Loading", test_model_loading),
        ("Model Save/Load", test_model_save_load),
        ("Image Processing", test_image_processing),
        ("Tensor Conversions", test_tensor_conversions),
        ("AI Pipeline Functions", test_ai_pipeline_functions)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print(f"\n📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)