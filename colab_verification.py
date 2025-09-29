
# COLAB UPLOAD VERIFICATION
# Run this after dataset organization to verify everything is ready

def verify_colab_dataset():
    """Comprehensive verification of Colab dataset setup"""

    import os
    from pathlib import Path

    print("🔍 VERIFYING COLAB DATASET SETUP")
    print("=" * 40)

    # Check directory structure
    required_dirs = [
        'data/training/classification/train',
        'data/training/classification/val',
        'data/training/classification/test'
    ]

    structure_ok = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - MISSING")
            structure_ok = False

    if not structure_ok:
        print("\n❌ Directory structure incomplete")
        return False

    # Check class distribution
    class_names = ['simple', 'text', 'gradient', 'complex']
    splits = ['train', 'val', 'test']

    print("\n📊 CLASS DISTRIBUTION:")
    total_images = 0

    for split in splits:
        split_total = 0
        class_counts = {}

        for class_name in class_names:
            class_dir = Path(f'data/training/classification/{split}/{class_name}')
            count = len(list(class_dir.glob('*.png'))) if class_dir.exists() else 0
            class_counts[class_name] = count
            split_total += count

        total_images += split_total
        print(f"{split.upper()}: {split_total} images {dict(class_counts)}")

    print(f"\nTOTAL: {total_images} images")

    # Validation checks
    checks = {
        'Total images ≥ 600': total_images >= 600,
        'All classes present': all(
            any(Path(f'data/training/classification/{split}/{cls}').glob('*.png'))
            for split in splits for cls in class_names
        ),
        'Train set largest': len(list(Path('data/training/classification/train').rglob('*.png'))) >
                           len(list(Path('data/training/classification/val').rglob('*.png'))),
        'Test set smallest': len(list(Path('data/training/classification/test').rglob('*.png'))) <
                           len(list(Path('data/training/classification/val').rglob('*.png')))
    }

    print("\n✅ VALIDATION CHECKS:")
    all_passed = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 DATASET SETUP COMPLETE - READY FOR TRAINING!")
        print("Expected performance with Colab GPU:")
        print("  • Training time: 2-3 hours")
        print("  • Target accuracy: >90%")
        print("  • Batch size: 64 (vs 4 locally)")
        print("  • Speed improvement: 10-20x faster")
        return True
    else:
        print("\n❌ Dataset setup issues detected")
        return False

# Run verification
verify_colab_dataset()
