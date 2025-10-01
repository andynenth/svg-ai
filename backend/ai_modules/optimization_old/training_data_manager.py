# backend/ai_modules/optimization/training_data_manager.py
"""Comprehensive training data preparation and organization system"""

import os
import json
import shutil
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Metadata for training images"""
    file_path: str
    category: str
    file_size: int
    dimensions: Tuple[int, int]
    color_channels: int
    bit_depth: int
    quality_score: float
    complexity_score: float
    has_transparency: bool
    dominant_colors: List[Tuple[int, int, int]]
    file_hash: str
    validation_status: str  # 'valid', 'invalid', 'warning'
    validation_messages: List[str]


@dataclass
class DatasetSplit:
    """Dataset split configuration"""
    training_ratio: float = 0.7
    validation_ratio: float = 0.2
    test_ratio: float = 0.1
    stratified: bool = True
    random_seed: int = 42


@dataclass
class BatchConfig:
    """Batch loading configuration"""
    batch_size: int = 32
    shuffle: bool = True
    drop_last: bool = False
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True


class ImageQualityAnalyzer:
    """Analyzes image quality and complexity"""

    @staticmethod
    def calculate_quality_score(image_path: str) -> float:
        """Calculate image quality score based on multiple factors"""
        try:
            image = Image.open(image_path)
            img_array = np.array(image)

            # Convert to grayscale for analysis
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Contrast (standard deviation)
            contrast = gray.std()

            # Signal-to-noise ratio approximation
            mean_val = gray.mean()
            noise_estimate = np.abs(gray - mean_val).mean()
            snr = mean_val / max(noise_estimate, 1e-6)

            # Normalize and combine metrics
            normalized_sharpness = min(sharpness / 1000.0, 1.0)
            normalized_contrast = min(contrast / 128.0, 1.0)
            normalized_snr = min(snr / 100.0, 1.0)

            quality_score = (normalized_sharpness * 0.4 +
                           normalized_contrast * 0.4 +
                           normalized_snr * 0.2)

            return min(quality_score, 1.0)

        except Exception as e:
            logger.warning(f"Quality analysis failed for {image_path}: {e}")
            return 0.5  # Default moderate quality

    @staticmethod
    def calculate_complexity_score(image_path: str) -> float:
        """Calculate image complexity score"""
        try:
            image = Image.open(image_path)
            img_array = np.array(image)

            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Edge density (Canny edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2).mean()

            # Texture analysis (local binary patterns approximation)
            texture_score = np.std(gray) / 128.0

            # Normalize and combine
            normalized_edges = min(edge_density * 10, 1.0)
            normalized_gradient = min(gradient_magnitude / 100.0, 1.0)
            normalized_texture = min(texture_score, 1.0)

            complexity_score = (normalized_edges * 0.5 +
                              normalized_gradient * 0.3 +
                              normalized_texture * 0.2)

            return min(complexity_score, 1.0)

        except Exception as e:
            logger.warning(f"Complexity analysis failed for {image_path}: {e}")
            return 0.5  # Default moderate complexity


class ImageValidator:
    """Validates training images for quality and suitability"""

    def __init__(self):
        self.min_width = 32
        self.min_height = 32
        self.max_width = 4096
        self.max_height = 4096
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        self.min_quality_score = 0.1
        self.max_file_size_mb = 50

    def validate_image(self, image_path: str) -> Tuple[str, List[str]]:
        """
        Validate single image

        Returns:
            (status, messages) where status is 'valid', 'invalid', or 'warning'
        """
        messages = []
        status = 'valid'

        try:
            path = Path(image_path)

            # Check file existence
            if not path.exists():
                return 'invalid', ['File does not exist']

            # Check file extension
            if path.suffix.lower() not in self.supported_formats:
                return 'invalid', [f'Unsupported format: {path.suffix}']

            # Check file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                messages.append(f'Large file size: {file_size_mb:.1f}MB')
                status = 'warning'

            # Check image properties
            try:
                with Image.open(image_path) as img:
                    width, height = img.size

                    # Check dimensions
                    if width < self.min_width or height < self.min_height:
                        return 'invalid', [f'Image too small: {width}x{height}']

                    if width > self.max_width or height > self.max_height:
                        messages.append(f'Very large image: {width}x{height}')
                        status = 'warning'

                    # Check if image can be loaded properly
                    img.verify()

            except Exception as e:
                return 'invalid', [f'Cannot read image: {str(e)}']

            # Quality check
            quality_score = ImageQualityAnalyzer.calculate_quality_score(image_path)
            if quality_score < self.min_quality_score:
                messages.append(f'Low quality score: {quality_score:.3f}')
                if quality_score < 0.05:
                    status = 'invalid'
                else:
                    status = 'warning'

            if not messages:
                messages.append('Image passed all validations')

            return status, messages

        except Exception as e:
            return 'invalid', [f'Validation error: {str(e)}']


class TrainingDataManager:
    """Comprehensive training data preparation and organization system"""

    def __init__(self,
                 data_root: str,
                 cache_dir: Optional[str] = None,
                 enable_caching: bool = True):
        """
        Initialize training data manager

        Args:
            data_root: Root directory containing training data
            cache_dir: Directory for caching metadata and processed data
            enable_caching: Whether to enable metadata caching
        """
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_root / '.cache'
        self.enable_caching = enable_caching

        # Create cache directory
        if self.enable_caching:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Category mappings (map directory names to standardized categories)
        self.category_mappings = {
            'simple_geometric': 'simple',
            'text_based': 'text',
            'gradients': 'gradient',
            'complex': 'complex',
            'abstract': 'complex'  # Map abstract to complex
        }

        # Initialize components
        self.validator = ImageValidator()
        self.quality_analyzer = ImageQualityAnalyzer()

        # Data storage
        self.image_metadata: Dict[str, ImageMetadata] = {}
        self.category_images: Dict[str, List[str]] = {}
        self.dataset_splits: Dict[str, Dict[str, List[str]]] = {}

        logger.info(f"TrainingDataManager initialized with data root: {self.data_root}")

    def scan_and_organize_data(self,
                              force_rescan: bool = False,
                              max_workers: int = 4) -> Dict[str, Any]:
        """
        Scan and organize all training data

        Args:
            force_rescan: Force rescan even if cache exists
            max_workers: Number of parallel workers for processing

        Returns:
            Summary of organized data
        """
        logger.info("🔍 Scanning and organizing training data...")
        start_time = time.time()

        # Check for cached metadata
        cache_file = self.cache_dir / 'metadata_cache.json'
        if not force_rescan and self.enable_caching and cache_file.exists():
            try:
                self._load_cached_metadata(cache_file)
                logger.info(f"Loaded cached metadata for {len(self.image_metadata)} images")
                return self._generate_data_summary()
            except Exception as e:
                logger.warning(f"Failed to load cached metadata: {e}")

        # Scan directory structure
        image_files = self._discover_image_files()
        logger.info(f"Discovered {len(image_files)} image files")

        # Process images in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            metadata_list = list(executor.map(self._process_single_image, image_files))

        # Store metadata
        for metadata in metadata_list:
            if metadata:
                self.image_metadata[metadata.file_path] = metadata

        # Organize by categories
        self._organize_by_categories()

        # Cache metadata
        if self.enable_caching:
            self._save_metadata_cache(cache_file)

        processing_time = time.time() - start_time
        logger.info(f"✅ Data organization completed in {processing_time:.2f}s")

        return self._generate_data_summary()

    def _discover_image_files(self) -> List[str]:
        """Discover all image files in data directory"""
        supported_extensions = {'*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff'}
        image_files = []

        for ext in supported_extensions:
            image_files.extend(self.data_root.rglob(ext))
            image_files.extend(self.data_root.rglob(ext.upper()))

        return [str(f) for f in image_files]

    def _process_single_image(self, image_path: str) -> Optional[ImageMetadata]:
        """Process single image and extract metadata"""
        try:
            path = Path(image_path)

            # Determine category from directory structure
            category = self._determine_category(path)
            if not category:
                logger.warning(f"Cannot determine category for: {image_path}")
                return None

            # Validate image
            validation_status, validation_messages = self.validator.validate_image(image_path)

            # Skip invalid images
            if validation_status == 'invalid':
                logger.warning(f"Invalid image {image_path}: {validation_messages}")
                return None

            # Extract basic metadata
            with Image.open(image_path) as img:
                width, height = img.size
                color_channels = len(img.getbands())
                has_transparency = img.mode in ('RGBA', 'LA') or 'transparency' in img.info

                # Get dominant colors (simplified)
                img_small = img.resize((50, 50))
                img_array = np.array(img_small)
                if len(img_array.shape) == 3:
                    dominant_colors = [tuple(map(int, color)) for color in
                                     img_array.reshape(-1, img_array.shape[-1])[:5]]
                else:
                    dominant_colors = [(128, 128, 128)]  # Default gray

            # Calculate quality and complexity
            quality_score = self.quality_analyzer.calculate_quality_score(image_path)
            complexity_score = self.quality_analyzer.calculate_complexity_score(image_path)

            # File metadata
            file_size = path.stat().st_size
            file_hash = self._calculate_file_hash(image_path)

            return ImageMetadata(
                file_path=image_path,
                category=category,
                file_size=file_size,
                dimensions=(width, height),
                color_channels=color_channels,
                bit_depth=8,  # Assume 8-bit for now
                quality_score=quality_score,
                complexity_score=complexity_score,
                has_transparency=has_transparency,
                dominant_colors=dominant_colors,
                file_hash=file_hash,
                validation_status=validation_status,
                validation_messages=validation_messages
            )

        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return None

    def _determine_category(self, image_path: Path) -> Optional[str]:
        """Determine image category from directory structure"""
        # Look for category in parent directories
        for parent in image_path.parents:
            parent_name = parent.name.lower()
            if parent_name in self.category_mappings:
                return self.category_mappings[parent_name]

        # Try direct mapping
        parent_name = image_path.parent.name.lower()
        return self.category_mappings.get(parent_name)

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()[:16]  # Use first 16 characters

    def _organize_by_categories(self) -> None:
        """Organize images by categories"""
        self.category_images = {'simple': [], 'text': [], 'gradient': [], 'complex': []}

        for metadata in self.image_metadata.values():
            if metadata.validation_status in ('valid', 'warning'):
                self.category_images[metadata.category].append(metadata.file_path)

        # Sort by quality score (best first)
        for category in self.category_images:
            self.category_images[category].sort(
                key=lambda path: self.image_metadata[path].quality_score,
                reverse=True
            )

    def create_dataset_splits(self,
                            split_config: DatasetSplit,
                            split_name: str = 'default') -> Dict[str, Dict[str, List[str]]]:
        """
        Create training/validation/test splits

        Args:
            split_config: Split configuration
            split_name: Name for this split configuration

        Returns:
            Dictionary with split data organized by category
        """
        logger.info(f"Creating dataset split: {split_name}")

        np.random.seed(split_config.random_seed)

        splits = {
            'train': {'simple': [], 'text': [], 'gradient': [], 'complex': []},
            'validation': {'simple': [], 'text': [], 'gradient': [], 'complex': []},
            'test': {'simple': [], 'text': [], 'gradient': [], 'complex': []}
        }

        for category, images in self.category_images.items():
            if not images:
                continue

            # Shuffle images
            shuffled_images = np.random.permutation(images).tolist()

            # Calculate split indices
            total = len(shuffled_images)
            train_end = int(total * split_config.training_ratio)
            val_end = train_end + int(total * split_config.validation_ratio)

            # Split data
            splits['train'][category] = shuffled_images[:train_end]
            splits['validation'][category] = shuffled_images[train_end:val_end]
            splits['test'][category] = shuffled_images[val_end:]

            logger.info(f"{category}: {len(splits['train'][category])} train, "
                       f"{len(splits['validation'][category])} val, "
                       f"{len(splits['test'][category])} test")

        self.dataset_splits[split_name] = splits
        return splits

    def get_training_batches(self,
                           split_name: str = 'default',
                           split_type: str = 'train',
                           batch_config: Optional[BatchConfig] = None) -> List[List[str]]:
        """
        Create training batches from dataset split

        Args:
            split_name: Name of dataset split to use
            split_type: Type of split ('train', 'validation', 'test')
            batch_config: Batch configuration

        Returns:
            List of image path batches
        """
        if split_name not in self.dataset_splits:
            raise ValueError(f"Dataset split '{split_name}' not found")

        if batch_config is None:
            batch_config = BatchConfig()

        # Get all images from split
        all_images = []
        for category_images in self.dataset_splits[split_name][split_type].values():
            all_images.extend(category_images)

        if not all_images:
            return []

        # Shuffle if requested
        if batch_config.shuffle:
            np.random.shuffle(all_images)

        # Create batches
        batches = []
        for i in range(0, len(all_images), batch_config.batch_size):
            batch = all_images[i:i + batch_config.batch_size]

            # Skip incomplete batch if drop_last is True
            if batch_config.drop_last and len(batch) < batch_config.batch_size:
                continue

            batches.append(batch)

        logger.info(f"Created {len(batches)} batches for {split_type} split")
        return batches

    def validate_dataset_quality(self) -> Dict[str, Any]:
        """Validate overall dataset quality"""
        logger.info("🔍 Validating dataset quality...")

        quality_report = {
            'total_images': len(self.image_metadata),
            'valid_images': 0,
            'warning_images': 0,
            'invalid_images': 0,
            'category_distribution': {},
            'quality_statistics': {},
            'complexity_statistics': {},
            'recommendations': []
        }

        # Count validation statuses
        for metadata in self.image_metadata.values():
            if metadata.validation_status == 'valid':
                quality_report['valid_images'] += 1
            elif metadata.validation_status == 'warning':
                quality_report['warning_images'] += 1
            else:
                quality_report['invalid_images'] += 1

        # Category distribution
        for category, images in self.category_images.items():
            quality_report['category_distribution'][category] = len(images)

        # Quality and complexity statistics
        if self.image_metadata:
            qualities = [m.quality_score for m in self.image_metadata.values()]
            complexities = [m.complexity_score for m in self.image_metadata.values()]

            quality_report['quality_statistics'] = {
                'mean': np.mean(qualities),
                'std': np.std(qualities),
                'min': np.min(qualities),
                'max': np.max(qualities),
                'median': np.median(qualities)
            }

            quality_report['complexity_statistics'] = {
                'mean': np.mean(complexities),
                'std': np.std(complexities),
                'min': np.min(complexities),
                'max': np.max(complexities),
                'median': np.median(complexities)
            }

        # Generate recommendations
        recommendations = []

        # Check category balance
        category_counts = list(quality_report['category_distribution'].values())
        if category_counts:
            min_count = min(category_counts)
            max_count = max(category_counts)
            if max_count > min_count * 3:
                recommendations.append("Dataset is imbalanced across categories")

        # Check overall quality
        if quality_report['quality_statistics']:
            avg_quality = quality_report['quality_statistics']['mean']
            if avg_quality < 0.3:
                recommendations.append("Overall image quality is low")
            elif avg_quality < 0.5:
                recommendations.append("Consider improving image quality")

        # Check for insufficient data
        total_valid = quality_report['valid_images'] + quality_report['warning_images']
        if total_valid < 50:
            recommendations.append("Dataset too small - consider adding more images")

        quality_report['recommendations'] = recommendations

        logger.info(f"Dataset validation completed: {total_valid} usable images")
        return quality_report

    def _load_cached_metadata(self, cache_file: Path) -> None:
        """Load metadata from cache"""
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)

        self.image_metadata = {}
        for path, metadata_dict in cached_data['image_metadata'].items():
            self.image_metadata[path] = ImageMetadata(**metadata_dict)

        self.category_images = cached_data['category_images']
        self.dataset_splits = cached_data.get('dataset_splits', {})

    def _save_metadata_cache(self, cache_file: Path) -> None:
        """Save metadata to cache"""
        cache_data = {
            'image_metadata': {path: asdict(metadata)
                             for path, metadata in self.image_metadata.items()},
            'category_images': self.category_images,
            'dataset_splits': self.dataset_splits,
            'cache_version': '1.0',
            'created_at': time.time()
        }

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

        logger.info(f"Metadata cached to: {cache_file}")

    def _generate_data_summary(self) -> Dict[str, Any]:
        """Generate summary of organized data"""
        return {
            'total_images': len(self.image_metadata),
            'categories': {cat: len(imgs) for cat, imgs in self.category_images.items()},
            'validation_status': {
                status: sum(1 for m in self.image_metadata.values()
                          if m.validation_status == status)
                for status in ['valid', 'warning', 'invalid']
            },
            'quality_range': {
                'min': min((m.quality_score for m in self.image_metadata.values()), default=0),
                'max': max((m.quality_score for m in self.image_metadata.values()), default=0),
                'avg': np.mean([m.quality_score for m in self.image_metadata.values()]) if self.image_metadata else 0
            },
            'splits_created': list(self.dataset_splits.keys())
        }

    def export_dataset_info(self, output_file: str) -> None:
        """Export comprehensive dataset information"""
        dataset_info = {
            'summary': self._generate_data_summary(),
            'quality_report': self.validate_dataset_quality(),
            'metadata': {path: asdict(metadata)
                        for path, metadata in self.image_metadata.items()},
            'splits': self.dataset_splits
        }

        with open(output_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)

        logger.info(f"Dataset information exported to: {output_file}")

    def get_category_images(self, category: str,
                          quality_threshold: float = 0.0,
                          max_images: Optional[int] = None) -> List[str]:
        """
        Get images from specific category with quality filtering

        Args:
            category: Category name ('simple', 'text', 'gradient', 'complex')
            quality_threshold: Minimum quality score
            max_images: Maximum number of images to return

        Returns:
            List of image paths
        """
        if category not in self.category_images:
            return []

        # Filter by quality
        filtered_images = []
        for image_path in self.category_images[category]:
            metadata = self.image_metadata[image_path]
            if metadata.quality_score >= quality_threshold:
                filtered_images.append(image_path)

        # Limit number if requested
        if max_images:
            filtered_images = filtered_images[:max_images]

        return filtered_images

    def get_balanced_dataset(self,
                           images_per_category: int = 10,
                           quality_threshold: float = 0.3) -> Dict[str, List[str]]:
        """
        Get balanced dataset with equal representation from each category

        Args:
            images_per_category: Number of images per category
            quality_threshold: Minimum quality threshold

        Returns:
            Dictionary mapping categories to image lists
        """
        balanced_dataset = {}

        for category in ['simple', 'text', 'gradient', 'complex']:
            category_images = self.get_category_images(
                category,
                quality_threshold=quality_threshold,
                max_images=images_per_category
            )
            balanced_dataset[category] = category_images

            if len(category_images) < images_per_category:
                logger.warning(f"Only {len(category_images)} images available for {category} "
                             f"(requested {images_per_category})")

        return balanced_dataset


# Factory function for easy creation
def create_training_data_manager(data_root: str,
                               cache_dir: Optional[str] = None,
                               auto_scan: bool = True) -> TrainingDataManager:
    """
    Factory function to create and initialize training data manager

    Args:
        data_root: Root directory containing training data
        cache_dir: Directory for caching (optional)
        auto_scan: Whether to automatically scan data on creation

    Returns:
        Initialized TrainingDataManager
    """
    manager = TrainingDataManager(data_root, cache_dir)

    if auto_scan:
        manager.scan_and_organize_data()

    return manager