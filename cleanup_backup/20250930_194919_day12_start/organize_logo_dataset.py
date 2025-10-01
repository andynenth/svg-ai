#!/usr/bin/env python3
"""
Logo Dataset Organization Script
Automatically categorizes and organizes logo images for training
"""

import os
import shutil
import random
import json
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse

def analyze_logo_image(image_path: str) -> Dict:
    """
    Analyze a logo image to extract features for classification

    Returns:
        Dict with image analysis features
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Basic image properties
            width, height = img.size
            aspect_ratio = width / height

            # Convert to numpy for analysis
            img_array = np.array(img)

            # Color analysis
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))

            # Calculate color variance (higher = more colorful/complex)
            color_variance = np.var(img_array)

            # Calculate edge density (higher = more complex shapes)
            gray = np.mean(img_array, axis=2)
            edges = np.abs(np.gradient(gray)).mean()

            # Text detection heuristics
            # High contrast between adjacent pixels often indicates text
            contrast_score = np.std(gray)

            return {
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'unique_colors': unique_colors,
                'color_variance': color_variance,
                'edge_density': edges,
                'contrast_score': contrast_score,
                'file_size': os.path.getsize(image_path)
            }

    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return None

def classify_logo_type(image_path: str, analysis: Dict) -> str:
    """
    Classify logo type based on image analysis

    Args:
        image_path: Path to the image file
        analysis: Image analysis results

    Returns:
        Logo type: 'simple', 'text', 'gradient', or 'complex'
    """
    if analysis is None:
        return 'simple'  # Default fallback

    filename = os.path.basename(image_path).lower()

    # Text-based logo detection
    # High contrast, rectangular aspect ratio, lower edge complexity
    if (analysis['contrast_score'] > 50 and
        0.5 < analysis['aspect_ratio'] < 5.0 and
        analysis['edge_density'] < 30):
        return 'text'

    # Simple geometric logo detection
    # Low color count, simple shapes, low edge density
    if (analysis['unique_colors'] < 50 and
        analysis['edge_density'] < 25 and
        analysis['color_variance'] < 2000):
        return 'simple'

    # Gradient logo detection
    # Medium color count, higher color variance, smooth transitions
    if (50 <= analysis['unique_colors'] <= 200 and
        analysis['color_variance'] > 3000 and
        analysis['edge_density'] < 40):
        return 'gradient'

    # Complex logo detection
    # High color count, high edge density, detailed
    if (analysis['unique_colors'] > 100 or
        analysis['edge_density'] > 35 or
        analysis['color_variance'] > 5000):
        return 'complex'

    # Default to simple if no other criteria met
    return 'simple'

def organize_dataset(raw_dir: str, output_dir: str,
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.2,
                    test_ratio: float = 0.1,
                    max_images_per_class: int = None) -> bool:
    """
    Organize raw logo images into structured train/val/test dataset

    Args:
        raw_dir: Directory containing raw logo images
        output_dir: Output directory for organized dataset
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        max_images_per_class: Maximum images per class (None for no limit)

    Returns:
        Success status
    """

    print(f"🔧 Organizing logo dataset from {raw_dir}")
    print(f"📊 Target splits: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")

    # Verify input directory
    if not os.path.exists(raw_dir):
        print(f"❌ Raw directory not found: {raw_dir}")
        return False

    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg'}
    image_files = [
        f for f in os.listdir(raw_dir)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]

    if len(image_files) == 0:
        print(f"❌ No image files found in {raw_dir}")
        return False

    print(f"📂 Found {len(image_files)} images to process")

    # Analyze and classify images
    print("🔍 Analyzing images for classification...")
    categorized_images = defaultdict(list)
    analysis_results = {}

    for i, img_file in enumerate(image_files):
        if i % 100 == 0:
            print(f"   Processed {i}/{len(image_files)} images...")

        img_path = os.path.join(raw_dir, img_file)

        # Analyze image
        analysis = analyze_logo_image(img_path)
        if analysis is None:
            continue

        # Classify logo type
        logo_type = classify_logo_type(img_path, analysis)

        categorized_images[logo_type].append(img_file)
        analysis_results[img_file] = {
            'logo_type': logo_type,
            'analysis': analysis
        }

    print(f"\n📋 Classification results:")
    total_classified = 0
    for logo_type, images in categorized_images.items():
        count = len(images)
        total_classified += count
        print(f"   {logo_type.title():8s}: {count:4d} images")

    print(f"   Total classified: {total_classified}")

    # Apply per-class limits if specified
    if max_images_per_class:
        print(f"\n⚖️  Applying per-class limit: {max_images_per_class}")
        for logo_type in categorized_images:
            if len(categorized_images[logo_type]) > max_images_per_class:
                # Randomly sample images
                random.shuffle(categorized_images[logo_type])
                categorized_images[logo_type] = categorized_images[logo_type][:max_images_per_class]
                print(f"   {logo_type.title()}: limited to {max_images_per_class} images")

    # Create output directory structure
    splits = ['train', 'val', 'test']
    logo_types = ['simple', 'text', 'gradient', 'complex']

    for split in splits:
        for logo_type in logo_types:
            os.makedirs(os.path.join(output_dir, split, logo_type), exist_ok=True)

    # Split and copy images
    print(f"\n🔄 Creating dataset splits...")

    dataset_stats = {
        'total_images': total_classified,
        'splits': {},
        'class_distribution': {}
    }

    for logo_type, images in categorized_images.items():
        if len(images) == 0:
            continue

        # Shuffle images
        random.shuffle(images)

        # Calculate split sizes
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val  # Remaining goes to test

        # Ensure minimum 1 image per split if we have enough images
        if n_total >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n_test)

            # Adjust if totals don't match
            while n_train + n_val + n_test != n_total:
                if n_train + n_val + n_test < n_total:
                    n_train += 1
                else:
                    n_train -= 1

        # Split the images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        # Copy images to appropriate directories
        split_counts = {}
        for split, split_images in zip(['train', 'val', 'test'],
                                     [train_images, val_images, test_images]):
            split_counts[split] = len(split_images)

            for img_file in split_images:
                src_path = os.path.join(raw_dir, img_file)
                dst_path = os.path.join(output_dir, split, logo_type, img_file)
                shutil.copy2(src_path, dst_path)

        dataset_stats['class_distribution'][logo_type] = {
            'total': n_total,
            'train': n_train,
            'val': n_val,
            'test': n_test
        }

        print(f"   {logo_type.title():8s}: {n_train:3d} train, {n_val:3d} val, {n_test:3d} test")

    # Calculate split totals
    for split in splits:
        split_total = sum(dataset_stats['class_distribution'][logo_type][split]
                         for logo_type in dataset_stats['class_distribution'])
        dataset_stats['splits'][split] = split_total

    # Save dataset statistics
    stats_path = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(dataset_stats, f, indent=2)

    # Save analysis results
    analysis_path = os.path.join(output_dir, 'image_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\n✅ Dataset organized successfully!")
    print(f"   📊 Train: {dataset_stats['splits']['train']} images")
    print(f"   📊 Val:   {dataset_stats['splits']['val']} images")
    print(f"   📊 Test:  {dataset_stats['splits']['test']} images")
    print(f"   📄 Statistics saved: {stats_path}")
    print(f"   📄 Analysis saved: {analysis_path}")

    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Organize logo dataset for training')
    parser.add_argument('--raw-dir', default='data/raw_logos',
                       help='Directory containing raw logo images')
    parser.add_argument('--output-dir', default='data/training/classification',
                       help='Output directory for organized dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Proportion for training set (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Proportion for validation set (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Proportion for test set (default: 0.1)')
    parser.add_argument('--max-per-class', type=int, default=None,
                       help='Maximum images per class (default: no limit)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Verify ratios sum to 1
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        print("❌ Error: train_ratio + val_ratio + test_ratio must equal 1.0")
        return False

    print("🏗️  Logo Dataset Organization")
    print("=" * 40)

    # Run organization
    success = organize_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_images_per_class=args.max_per_class
    )

    if success:
        print("\n🎉 Dataset organization completed successfully!")
        return True
    else:
        print("\n❌ Dataset organization failed!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)