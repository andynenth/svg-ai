#!/usr/bin/env python3
"""
Quick Dataset Setup - Manually organize a subset for training
"""

import os
import shutil
import random
from glob import glob

def quick_organize_dataset():
    """Quickly organize a subset of raw logos for training"""

    raw_dir = "data/raw_logos"
    output_dir = "data/training/classification"

    print("🚀 Quick dataset organization...")

    # Get all PNG files
    png_files = glob(os.path.join(raw_dir, "*.png"))
    print(f"Found {len(png_files)} PNG files")

    if len(png_files) < 100:
        print("❌ Not enough images found")
        return False

    # Randomly sample images for each class
    random.shuffle(png_files)

    # Simple distribution: divide into 4 classes
    images_per_class = min(200, len(png_files) // 4)

    classes = ['simple', 'text', 'gradient', 'complex']

    # Clear existing data and create fresh structure
    for split in ['train', 'val', 'test']:
        for cls in classes:
            target_dir = os.path.join(output_dir, split, cls)
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            os.makedirs(target_dir, exist_ok=True)

    class_idx = 0
    for i, png_file in enumerate(png_files[:images_per_class * 4]):
        # Assign to class
        cls = classes[class_idx]

        # Determine split (70% train, 20% val, 10% test)
        class_position = i % images_per_class
        if class_position < int(images_per_class * 0.7):
            split = 'train'
        elif class_position < int(images_per_class * 0.9):
            split = 'val'
        else:
            split = 'test'

        # Copy file
        src_path = png_file
        filename = os.path.basename(png_file)
        dst_path = os.path.join(output_dir, split, cls, filename)

        shutil.copy2(src_path, dst_path)

        # Move to next class after processing all images for current class
        if (i + 1) % images_per_class == 0:
            class_idx += 1

    # Count results
    for split in ['train', 'val', 'test']:
        split_total = 0
        for cls in classes:
            cls_dir = os.path.join(output_dir, split, cls)
            cls_count = len([f for f in os.listdir(cls_dir) if f.endswith('.png')])
            split_total += cls_count
            print(f"   {split}/{cls}: {cls_count} images")
        print(f"   {split} total: {split_total}")

    return True

if __name__ == "__main__":
    quick_organize_dataset()