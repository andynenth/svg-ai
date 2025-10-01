#!/usr/bin/env python3
"""
Training Data Preparation Script

Organizes existing logo dataset into train/val/test splits for neural network training.
Maps 5 source categories to 4 target classes as required by EfficientNet classifier.
"""

import os
import shutil
import json
from pathlib import Path
from PIL import Image
import random
from typing import Dict, List, Tuple
import argparse

def setup_logging():
    """Setup basic logging."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class DatasetPreparer:
    """Handles dataset organization and validation."""

    def __init__(self, source_dir: str = 'data/logos', target_dir: str = 'data/training/classification'):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)

        # Mapping from source categories to target classes
        self.category_mapping = {
            'simple_geometric': 'simple',
            'text_based': 'text',
            'gradients': 'gradient',
            'complex': 'complex',
            'abstract': 'complex'  # Merge abstract with complex
        }

        # Split ratios
        self.train_ratio = 0.7
        self.val_ratio = 0.2
        self.test_ratio = 0.1

        # Random seed for reproducibility
        random.seed(42)

    def discover_source_images(self) -> Dict[str, List[Path]]:
        """
        Discover all images in source directories.

        Returns:
            Dictionary mapping source categories to image paths
        """
        images_by_category = {}

        logger.info(f"Discovering images in {self.source_dir}")

        for category in self.category_mapping.keys():
            category_dir = self.source_dir / category

            if not category_dir.exists():
                logger.warning(f"Category directory not found: {category_dir}")
                continue

            # Find all PNG images
            image_paths = list(category_dir.glob('*.png'))
            images_by_category[category] = image_paths

            logger.info(f"Found {len(image_paths)} images in {category}")

        return images_by_category

    def validate_images(self, image_paths: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        Validate images and separate valid from corrupted ones.

        Args:
            image_paths: List of image file paths

        Returns:
            Tuple of (valid_images, corrupted_images)
        """
        valid_images = []
        corrupted_images = []

        for img_path in image_paths:
            try:
                # Try to open and verify the image
                with Image.open(img_path) as img:
                    img.verify()  # Verify image integrity

                # Re-open to check basic properties
                with Image.open(img_path) as img:
                    width, height = img.size
                    if width > 0 and height > 0:
                        valid_images.append(img_path)
                    else:
                        corrupted_images.append(img_path)
                        logger.warning(f"Invalid dimensions: {img_path}")

            except Exception as e:
                corrupted_images.append(img_path)
                logger.warning(f"Corrupted image {img_path}: {e}")

        return valid_images, corrupted_images

    def split_dataset(self, images: List[Path]) -> Dict[str, List[Path]]:
        """
        Split images into train/val/test sets.

        Args:
            images: List of image paths

        Returns:
            Dictionary with train/val/test splits
        """
        # Shuffle images for random splits
        shuffled_images = images.copy()
        random.shuffle(shuffled_images)

        total = len(shuffled_images)
        train_size = int(total * self.train_ratio)
        val_size = int(total * self.val_ratio)

        splits = {
            'train': shuffled_images[:train_size],
            'val': shuffled_images[train_size:train_size + val_size],
            'test': shuffled_images[train_size + val_size:]
        }

        logger.info(f"Split {total} images: train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, test={len(splits['test'])}")

        return splits

    def copy_images_to_split(self, images: List[Path], target_class: str, split: str) -> int:
        """
        Copy images to target directory structure.

        Args:
            images: List of image paths to copy
            target_class: Target class name (simple, text, gradient, complex)
            split: Data split (train, val, test)

        Returns:
            Number of images successfully copied
        """
        target_split_dir = self.target_dir / split / target_class
        target_split_dir.mkdir(parents=True, exist_ok=True)

        copied_count = 0

        for img_path in images:
            try:
                # Create unique filename to avoid conflicts
                source_category = img_path.parent.name
                new_filename = f"{source_category}_{img_path.name}"
                target_path = target_split_dir / new_filename

                shutil.copy2(img_path, target_path)
                copied_count += 1

            except Exception as e:
                logger.error(f"Failed to copy {img_path}: {e}")

        logger.info(f"Copied {copied_count} images to {split}/{target_class}")
        return copied_count

    def organize_dataset(self) -> Dict[str, any]:
        """
        Main function to organize the complete dataset.

        Returns:
            Dataset statistics
        """
        logger.info("Starting dataset organization...")

        # Clean target directory
        if self.target_dir.exists():
            logger.info(f"Cleaning existing target directory: {self.target_dir}")
            shutil.rmtree(self.target_dir)

        # Discover source images
        source_images = self.discover_source_images()

        # Initialize statistics
        stats = {
            'source_categories': {},
            'target_classes': {},
            'splits': {'train': {}, 'val': {}, 'test': {}},
            'corrupted_images': [],
            'total_images': 0,
            'total_valid': 0
        }

        # Process each source category
        for source_category, image_paths in source_images.items():
            target_class = self.category_mapping[source_category]

            logger.info(f"Processing {source_category} -> {target_class}")

            # Validate images
            valid_images, corrupted_images = self.validate_images(image_paths)

            # Update statistics
            stats['source_categories'][source_category] = {
                'total': len(image_paths),
                'valid': len(valid_images),
                'corrupted': len(corrupted_images)
            }

            stats['corrupted_images'].extend([str(p) for p in corrupted_images])
            stats['total_images'] += len(image_paths)
            stats['total_valid'] += len(valid_images)

            if not valid_images:
                logger.warning(f"No valid images in {source_category}")
                continue

            # Split valid images
            splits = self.split_dataset(valid_images)

            # Copy images to target structure
            for split_name, split_images in splits.items():
                copied = self.copy_images_to_split(split_images, target_class, split_name)

                # Update statistics
                if target_class not in stats['target_classes']:
                    stats['target_classes'][target_class] = {'train': 0, 'val': 0, 'test': 0}
                if target_class not in stats['splits'][split_name]:
                    stats['splits'][split_name][target_class] = 0

                stats['target_classes'][target_class][split_name] += copied
                stats['splits'][split_name][target_class] += copied

        return stats

    def generate_report(self, stats: Dict[str, any]) -> None:
        """
        Generate and save dataset preparation report.

        Args:
            stats: Dataset statistics
        """
        report_path = self.target_dir.parent / 'dataset_preparation_report.json'

        # Add summary information
        stats['summary'] = {
            'total_source_images': stats['total_images'],
            'total_valid_images': stats['total_valid'],
            'corruption_rate': (stats['total_images'] - stats['total_valid']) / stats['total_images'] if stats['total_images'] > 0 else 0,
            'target_classes': list(set(self.category_mapping.values())),
            'splits_ratio': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            }
        }

        # Calculate split totals
        for split in ['train', 'val', 'test']:
            stats['summary'][f'{split}_total'] = sum(stats['splits'][split].values())

        # Save report
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Dataset report saved to: {report_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("DATASET PREPARATION SUMMARY")
        print("=" * 60)
        print(f"Source images: {stats['total_images']}")
        print(f"Valid images: {stats['total_valid']}")
        print(f"Corrupted images: {stats['total_images'] - stats['total_valid']}")
        print(f"Corruption rate: {stats['summary']['corruption_rate']:.2%}")

        print(f"\nTarget classes: {', '.join(stats['summary']['target_classes'])}")

        print(f"\nDataset splits:")
        for split in ['train', 'val', 'test']:
            total = stats['summary'][f'{split}_total']
            print(f"  {split}: {total} images")
            for class_name, count in stats['splits'][split].items():
                print(f"    {class_name}: {count}")

        if stats['corrupted_images']:
            print(f"\nCorrupted images found:")
            for img in stats['corrupted_images']:
                print(f"  {img}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Prepare training dataset for neural network')
    parser.add_argument('--source-dir', default='data/logos',
                       help='Source directory containing logo categories')
    parser.add_argument('--target-dir', default='data/training/classification',
                       help='Target directory for organized dataset')
    parser.add_argument('--clean', action='store_true',
                       help='Clean target directory before organizing')

    args = parser.parse_args()

    # Initialize preparer
    preparer = DatasetPreparer(args.source_dir, args.target_dir)

    # Organize dataset
    try:
        stats = preparer.organize_dataset()
        preparer.generate_report(stats)

        logger.info("Dataset preparation completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)