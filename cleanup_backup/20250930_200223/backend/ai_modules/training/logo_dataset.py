"""
PyTorch Dataset Implementation for Logo Classification

Custom dataset class for loading and preprocessing logo images
for neural network training.
"""

from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from typing import List, Tuple, Optional, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogoDataset(Dataset):
    """
    PyTorch Dataset for logo classification.

    Loads images from organized directory structure:
    data_dir/
    ├── simple/
    ├── text/
    ├── gradient/
    └── complex/
    """

    def __init__(self, data_dir: str, split: Optional[str] = None, transform: Optional[Callable] = None):
        """
        Initialize dataset.

        Args:
            data_dir: Path to data directory containing class folders or split folders
            split: Optional split name ('train', 'val', 'test') for organized datasets
            transform: Optional transform to apply to images
        """
        # Handle both organized (with splits) and flat directory structures
        if split and os.path.exists(os.path.join(data_dir, split)):
            self.data_dir = os.path.join(data_dir, split)
        elif split is None and os.path.exists(data_dir):
            self.data_dir = data_dir
        else:
            self.data_dir = data_dir

        self.transform = transform
        self.classes = ['simple', 'text', 'gradient', 'complex']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()

        logger.info(f"Loaded dataset from {self.data_dir}")
        logger.info(f"Classes: {self.classes}")
        logger.info(f"Total samples: {len(self.samples)}")
        self._log_class_distribution()

    def _load_samples(self) -> List[Tuple[str, int]]:
        """
        Load all image samples with their labels.

        Returns:
            List of (image_path, label_index) tuples
        """
        samples = []

        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)

            if not os.path.exists(class_dir):
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            # Get all image files
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    label = self.class_to_idx[class_name]
                    samples.append((img_path, label))

        return samples

    def _log_class_distribution(self):
        """Log the distribution of samples across classes."""
        class_counts = {}
        for _, label in self.samples:
            class_name = self.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        logger.info("Class distribution:")
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name}: {count} samples")

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, label)
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")

        img_path, label = self.samples[idx]

        try:
            # Load image and convert to RGB
            image = Image.open(img_path).convert('RGB')

            # Apply transforms if provided
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image with the correct label as fallback
            if self.transform:
                # Create a dummy image that matches transform output
                dummy_image = Image.new('RGB', (224, 224), color='black')
                image = self.transform(dummy_image)
            else:
                # Return raw PIL image
                image = Image.new('RGB', (224, 224), color='black')

            return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced datasets.

        Returns:
            Tensor of class weights
        """
        class_counts = torch.zeros(len(self.classes))

        for _, label in self.samples:
            class_counts[label] += 1

        # Avoid division by zero
        class_counts = torch.clamp(class_counts, min=1)

        # Calculate inverse frequency weights
        total_samples = len(self.samples)
        class_weights = total_samples / (len(self.classes) * class_counts)

        logger.info(f"Class weights: {class_weights}")
        return class_weights

    def get_sample_info(self, idx: int) -> dict:
        """
        Get detailed information about a sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with sample information
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range")

        img_path, label = self.samples[idx]
        class_name = self.classes[label]

        try:
            with Image.open(img_path) as img:
                width, height = img.size
                mode = img.mode

            return {
                'index': idx,
                'path': img_path,
                'class_name': class_name,
                'label': label,
                'width': width,
                'height': height,
                'mode': mode,
                'filename': os.path.basename(img_path)
            }

        except Exception as e:
            return {
                'index': idx,
                'path': img_path,
                'class_name': class_name,
                'label': label,
                'error': str(e),
                'filename': os.path.basename(img_path)
            }

    def get_samples_by_class(self, class_name: str) -> List[Tuple[str, int]]:
        """
        Get all samples for a specific class.

        Args:
            class_name: Name of the class

        Returns:
            List of (image_path, label) tuples for the class
        """
        if class_name not in self.classes:
            raise ValueError(f"Class {class_name} not found in {self.classes}")

        target_label = self.class_to_idx[class_name]
        return [(path, label) for path, label in self.samples if label == target_label]

    def validate_dataset(self) -> dict:
        """
        Validate the dataset and return statistics.

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating dataset...")

        stats = {
            'total_samples': len(self.samples),
            'valid_samples': 0,
            'corrupted_samples': 0,
            'class_distribution': {},
            'corrupted_files': []
        }

        for idx in range(len(self.samples)):
            img_path, label = self.samples[idx]
            class_name = self.classes[label]

            # Update class distribution
            if class_name not in stats['class_distribution']:
                stats['class_distribution'][class_name] = 0
            stats['class_distribution'][class_name] += 1

            # Try to load the image
            try:
                with Image.open(img_path) as img:
                    # Basic validation
                    if img.size[0] > 0 and img.size[1] > 0:
                        stats['valid_samples'] += 1
                    else:
                        stats['corrupted_samples'] += 1
                        stats['corrupted_files'].append(img_path)

            except Exception as e:
                stats['corrupted_samples'] += 1
                stats['corrupted_files'].append(img_path)
                logger.warning(f"Corrupted image {img_path}: {e}")

        stats['corruption_rate'] = stats['corrupted_samples'] / stats['total_samples'] if stats['total_samples'] > 0 else 0

        logger.info(f"Dataset validation complete: {stats['valid_samples']}/{stats['total_samples']} valid")
        logger.info(f"Corruption rate: {stats['corruption_rate']:.2%}")

        return stats