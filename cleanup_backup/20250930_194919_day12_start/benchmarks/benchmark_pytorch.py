#!/usr/bin/env python3
"""PyTorch Performance Benchmark"""

import time
import torch
import torch.nn as nn
import numpy as np
import psutil

def benchmark_matrix_ops():
    """Test basic tensor operations"""
    print("🧮 Testing matrix operations...")
    start = time.time()
    x = torch.randn(1000, 1000)
    y = torch.mm(x, x.t())
    end = time.time()
    print(f"Matrix multiplication (1000x1000): {end-start:.3f}s")
    return end-start

def benchmark_neural_network():
    """Test neural network operations"""
    print("🧠 Testing neural network...")
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    start = time.time()
    x = torch.randn(32, 784)  # Batch of 32
    output = model(x)
    end = time.time()
    print(f"Neural network forward pass: {end-start:.3f}s")
    return end-start

def benchmark_model_loading():
    """Test model loading performance"""
    print("📦 Testing model loading...")

    # Test ResNet-18
    start = time.time()
    import torchvision.models as models
    resnet = models.resnet18(weights=None)
    resnet_time = time.time() - start
    print(f"ResNet-18 loading: {resnet_time:.3f}s")

    # Test EfficientNet-B0
    start = time.time()
    efficientnet = models.efficientnet_b0(weights=None)
    efficientnet_time = time.time() - start
    print(f"EfficientNet-B0 loading: {efficientnet_time:.3f}s")

    return resnet_time, efficientnet_time

def benchmark_memory_usage():
    """Test memory usage during operations"""
    print("💾 Testing memory usage...")

    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory: {initial_memory:.1f}MB")

    # Load models and perform operations
    x = torch.randn(1000, 1000)
    y = torch.mm(x, x.t())

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    input_data = torch.randn(32, 784)
    output = model(input_data)

    final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    print(f"Final memory: {final_memory:.1f}MB")
    print(f"Memory increase: {memory_increase:.1f}MB")

    return memory_increase

def compare_numpy_performance():
    """Compare PyTorch vs NumPy performance"""
    print("⚡ Comparing PyTorch vs NumPy...")

    # NumPy benchmark
    start = time.time()
    x_np = np.random.randn(1000, 1000)
    y_np = np.dot(x_np, x_np.T)
    numpy_time = time.time() - start
    print(f"NumPy matrix multiplication: {numpy_time:.3f}s")

    # PyTorch benchmark
    start = time.time()
    x_torch = torch.randn(1000, 1000)
    y_torch = torch.mm(x_torch, x_torch.t())
    pytorch_time = time.time() - start
    print(f"PyTorch matrix multiplication: {pytorch_time:.3f}s")

    ratio = pytorch_time / numpy_time
    print(f"PyTorch/NumPy ratio: {ratio:.2f}x")

    return numpy_time, pytorch_time, ratio

def main():
    """Run complete performance benchmark"""
    print("🚀 PyTorch Performance Benchmark")
    print("=" * 40)

    # Matrix operations
    matrix_time = benchmark_matrix_ops()
    print()

    # Neural network
    nn_time = benchmark_neural_network()
    print()

    # Model loading
    resnet_time, efficientnet_time = benchmark_model_loading()
    print()

    # Memory usage
    memory_increase = benchmark_memory_usage()
    print()

    # NumPy comparison
    numpy_time, pytorch_time, ratio = compare_numpy_performance()
    print()

    # Summary
    print("📊 Performance Summary")
    print("=" * 30)
    print(f"Matrix multiplication: {matrix_time:.3f}s")
    print(f"Neural network forward: {nn_time:.3f}s")
    print(f"ResNet-18 loading: {resnet_time:.3f}s")
    print(f"EfficientNet-B0 loading: {efficientnet_time:.3f}s")
    print(f"Memory increase: {memory_increase:.1f}MB")
    print(f"PyTorch vs NumPy: {ratio:.2f}x")

    # Performance validation
    print("\n✅ Performance Validation")
    print("=" * 30)

    if matrix_time < 1.0:
        print("✅ Matrix operations: PASS (<1s)")
    else:
        print("❌ Matrix operations: FAIL (>1s)")

    if nn_time < 0.5:
        print("✅ Neural network: PASS (<0.5s)")
    else:
        print("❌ Neural network: FAIL (>0.5s)")

    if memory_increase < 500:
        print("✅ Memory usage: PASS (<500MB)")
    else:
        print("❌ Memory usage: FAIL (>500MB)")

    print("\n🎉 Benchmark complete!")

if __name__ == "__main__":
    main()