#!/usr/bin/env python3
"""
Test script for LogoDataset PyTorch implementation.
"""

import sys
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ai_modules.training.logo_dataset import LogoDataset

def create_test_transforms():
    """Create transforms for testing."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def test_dataset_loading():
    """Test basic dataset loading."""
    print("=== Testing Dataset Loading ===")

    data_dirs = {
        'train': 'data/training/classification/train',
        'val': 'data/training/classification/val',
        'test': 'data/training/classification/test'
    }

    for split, data_dir in data_dirs.items():
        if not os.path.exists(data_dir):
            print(f"⚠ Data directory not found: {data_dir}")
            continue

        try:
            dataset = LogoDataset(data_dir)
            print(f"✓ {split} dataset loaded: {len(dataset)} samples")
            print(f"  Classes: {dataset.classes}")

            # Test class distribution
            stats = dataset.validate_dataset()
            print(f"  Distribution: {stats['class_distribution']}")

        except Exception as e:
            print(f"✗ Failed to load {split} dataset: {e}")
            return False

    return True

def test_dataset_with_transforms():
    """Test dataset with transforms."""
    print("\n=== Testing Dataset with Transforms ===")

    train_dir = 'data/training/classification/train'
    if not os.path.exists(train_dir):
        print(f"⚠ Train directory not found: {train_dir}")
        return False

    try:
        transform = create_test_transforms()
        dataset = LogoDataset(train_dir, transform=transform)

        print(f"✓ Dataset with transforms loaded: {len(dataset)} samples")

        # Test getting a sample
        if len(dataset) > 0:
            image, label = dataset[0]
            print(f"✓ Sample 0: image shape {image.shape}, label {label}")
            print(f"  Class: {dataset.classes[label]}")

        return True

    except Exception as e:
        print(f"✗ Transform test failed: {e}")
        return False

def test_dataloader():
    """Test PyTorch DataLoader integration."""
    print("\n=== Testing DataLoader Integration ===")

    train_dir = 'data/training/classification/train'
    if not os.path.exists(train_dir):
        print(f"⚠ Train directory not found: {train_dir}")
        return False

    try:
        transform = create_test_transforms()
        dataset = LogoDataset(train_dir, transform=transform)

        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )

        print(f"✓ DataLoader created: batch_size=4")

        # Test iteration
        for batch_idx, (images, labels) in enumerate(dataloader):
            print(f"✓ Batch {batch_idx}: images {images.shape}, labels {labels.shape}")

            if batch_idx >= 2:  # Test only first few batches
                break

        return True

    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        return False

def test_class_weights():
    """Test class weight calculation."""
    print("\n=== Testing Class Weights ===")

    train_dir = 'data/training/classification/train'
    if not os.path.exists(train_dir):
        print(f"⚠ Train directory not found: {train_dir}")
        return False

    try:
        dataset = LogoDataset(train_dir)
        class_weights = dataset.get_class_weights()

        print(f"✓ Class weights calculated: {class_weights}")
        print(f"  Weights per class:")
        for i, weight in enumerate(class_weights):
            print(f"    {dataset.classes[i]}: {weight:.4f}")

        return True

    except Exception as e:
        print(f"✗ Class weights test failed: {e}")
        return False

def test_sample_info():
    """Test sample information retrieval."""
    print("\n=== Testing Sample Information ===")

    train_dir = 'data/training/classification/train'
    if not os.path.exists(train_dir):
        print(f"⚠ Train directory not found: {train_dir}")
        return False

    try:
        dataset = LogoDataset(train_dir)

        if len(dataset) > 0:
            # Test first few samples
            for i in range(min(3, len(dataset))):
                info = dataset.get_sample_info(i)
                print(f"✓ Sample {i}: {info['filename']} -> {info['class_name']}")
                print(f"  Size: {info.get('width', 'unknown')}x{info.get('height', 'unknown')}")

        return True

    except Exception as e:
        print(f"✗ Sample info test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("LogoDataset Test Suite")
    print("=" * 50)

    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Dataset with Transforms", test_dataset_with_transforms),
        ("DataLoader Integration", test_dataloader),
        ("Class Weights", test_class_weights),
        ("Sample Information", test_sample_info)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✓ {test_name}: PASSED")
            else:
                print(f"✗ {test_name}: FAILED")
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}")

        print()

    # Summary
    print("=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ ALL TESTS PASSED - LogoDataset working correctly!")
        return True
    else:
        print("✗ Some tests failed - needs attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)