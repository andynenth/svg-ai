#!/usr/bin/env python3
"""
Image loading utilities for quality metrics calculations.

This module provides functions to load PNG and SVG files as numpy arrays,
fixing the issue where QualityMetrics expects arrays but receives file paths.
"""

import os
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import numpy as np
    from PIL import Image
    import cairosvg
    from io import BytesIO
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    logger.warning(f"Missing dependencies for image loading: {e}")


class ImageLoader:
    """Utility class for loading images in various formats."""

    @staticmethod
    def load_png(png_path: str) -> Optional[np.ndarray]:
        """
        Load a PNG file as a numpy array.

        Args:
            png_path: Path to PNG file

        Returns:
            Numpy array of image data or None if failed
        """
        if not DEPENDENCIES_AVAILABLE:
            logger.error("Dependencies not available. Install with: pip install numpy pillow")
            return None

        if not os.path.exists(png_path):
            logger.error(f"PNG file not found: {png_path}")
            return None

        try:
            # Load image and convert to RGBA
            img = Image.open(png_path).convert('RGBA')
            return np.array(img)
        except Exception as e:
            logger.error(f"Failed to load PNG {png_path}: {e}")
            return None

    @staticmethod
    def load_svg(svg_path: str, width: Optional[int] = None,
                 height: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Load an SVG file as a numpy array by rendering it to PNG.

        Args:
            svg_path: Path to SVG file
            width: Target width for rendering (optional)
            height: Target height for rendering (optional)

        Returns:
            Numpy array of rendered image or None if failed
        """
        if not DEPENDENCIES_AVAILABLE:
            logger.error("Dependencies not available. Install with: pip install numpy pillow cairosvg")
            return None

        if not os.path.exists(svg_path):
            logger.error(f"SVG file not found: {svg_path}")
            return None

        try:
            # Render SVG to PNG bytes
            png_bytes = cairosvg.svg2png(
                url=svg_path,
                output_width=width,
                output_height=height
            )

            # Convert to PIL Image and then numpy array
            img = Image.open(BytesIO(png_bytes)).convert('RGBA')
            return np.array(img)

        except Exception as e:
            logger.error(f"Failed to load SVG {svg_path}: {e}")
            return None

    @staticmethod
    def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """
        Load an image file (PNG, JPG, or SVG) as a numpy array.

        Args:
            image_path: Path to image file
            target_size: Optional (width, height) tuple for resizing

        Returns:
            Numpy array of image data or None if failed
        """
        if not DEPENDENCIES_AVAILABLE:
            logger.error("Dependencies not available")
            return None

        file_ext = Path(image_path).suffix.lower()

        if file_ext in ['.png', '.jpg', '.jpeg']:
            img_array = ImageLoader.load_png(image_path)
        elif file_ext == '.svg':
            width, height = target_size if target_size else (None, None)
            img_array = ImageLoader.load_svg(image_path, width, height)
        else:
            logger.error(f"Unsupported file type: {file_ext}")
            return None

        # Resize if needed
        if img_array is not None and target_size is not None:
            img = Image.fromarray(img_array)
            img = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(img)

        return img_array

    @staticmethod
    def ensure_same_size(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensure two images have the same dimensions.

        Args:
            img1: First image array
            img2: Second image array

        Returns:
            Tuple of images with matching dimensions
        """
        if img1.shape == img2.shape:
            return img1, img2

        # Get target size (use smaller dimensions)
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])

        # Resize both images
        img1_resized = Image.fromarray(img1).resize((width, height), Image.LANCZOS)
        img2_resized = Image.fromarray(img2).resize((width, height), Image.LANCZOS)

        return np.array(img1_resized), np.array(img2_resized)

    @staticmethod
    def normalize_images(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize two images for comparison.

        Args:
            img1: First image array
            img2: Second image array

        Returns:
            Tuple of normalized images
        """
        # Ensure same size
        img1, img2 = ImageLoader.ensure_same_size(img1, img2)

        # Convert to same number of channels
        if len(img1.shape) != len(img2.shape):
            # One is grayscale, one is color
            if len(img1.shape) == 2:
                img1 = np.stack([img1] * 3, axis=2)
            if len(img2.shape) == 2:
                img2 = np.stack([img2] * 3, axis=2)

        # Ensure same number of channels (handle RGBA vs RGB)
        if img1.shape[2] != img2.shape[2]:
            target_channels = min(img1.shape[2], img2.shape[2])
            img1 = img1[:, :, :target_channels]
            img2 = img2[:, :, :target_channels]

        # Convert to float32 and normalize to 0-1
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

        return img1, img2


class QualityMetricsWrapper:
    """
    Wrapper for QualityMetrics that handles file loading.

    This class provides methods that accept file paths and handle
    the image loading before calling the actual quality metrics.
    """

    def __init__(self):
        """Initialize the wrapper with an ImageLoader."""
        self.loader = ImageLoader()

        # Try to import QualityMetrics
        try:
            from utils.quality_metrics import QualityMetrics
            self.metrics = QualityMetrics()
        except ImportError:
            logger.error("Could not import QualityMetrics")
            self.metrics = None

    def calculate_ssim_from_paths(self, path1: str, path2: str) -> float:
        """
        Calculate SSIM between two image files.

        Args:
            path1: Path to first image (PNG or SVG)
            path2: Path to second image (PNG or SVG)

        Returns:
            SSIM score (0-1) or -1 if failed
        """
        if self.metrics is None:
            return -1.0

        # Load first image
        img1 = self.loader.load_image(path1)
        if img1 is None:
            logger.error(f"Failed to load {path1}")
            return -1.0

        # Load second image with same size as first
        height, width = img1.shape[:2]
        img2 = self.loader.load_image(path2, target_size=(width, height))
        if img2 is None:
            logger.error(f"Failed to load {path2}")
            return -1.0

        # Normalize images
        img1, img2 = self.loader.normalize_images(img1, img2)

        # Calculate SSIM
        try:
            # Convert back to 0-255 range for metrics
            img1 = (img1 * 255).astype(np.uint8)
            img2 = (img2 * 255).astype(np.uint8)
            ssim = self.metrics.calculate_ssim(img1, img2)
            return ssim
        except Exception as e:
            logger.error(f"Failed to calculate SSIM: {e}")
            return -1.0

    def calculate_mse_from_paths(self, path1: str, path2: str) -> float:
        """
        Calculate MSE between two image files.

        Args:
            path1: Path to first image
            path2: Path to second image

        Returns:
            MSE value or -1 if failed
        """
        if self.metrics is None:
            return -1.0

        # Load images
        img1 = self.loader.load_image(path1)
        img2 = self.loader.load_image(path2)

        if img1 is None or img2 is None:
            return -1.0

        # Normalize
        img1, img2 = self.loader.normalize_images(img1, img2)

        # Calculate MSE
        try:
            mse = np.mean((img1 - img2) ** 2) * 255 * 255  # Scale back to 0-255 range
            return float(mse)
        except Exception as e:
            logger.error(f"Failed to calculate MSE: {e}")
            return -1.0

    def calculate_psnr_from_paths(self, path1: str, path2: str) -> float:
        """
        Calculate PSNR between two image files.

        Args:
            path1: Path to first image
            path2: Path to second image

        Returns:
            PSNR in dB or -1 if failed
        """
        mse = self.calculate_mse_from_paths(path1, path2)
        if mse <= 0:
            return -1.0 if mse < 0 else 100.0  # Perfect match

        try:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            return float(psnr)
        except Exception as e:
            logger.error(f"Failed to calculate PSNR: {e}")
            return -1.0


def test_loader():
    """Test the image loader with sample files."""
    print("Testing ImageLoader...")

    # Test with actual files if they exist
    test_png = "data/logos/text_based/text_tech_00.png"
    test_svg = "data/logos/text_based/text_tech_00.optimized.svg"

    loader = ImageLoader()

    if os.path.exists(test_png):
        print(f"\nLoading PNG: {test_png}")
        png_array = loader.load_png(test_png)
        if png_array is not None:
            print(f"  ✅ Loaded PNG: shape={png_array.shape}, dtype={png_array.dtype}")
        else:
            print("  ❌ Failed to load PNG")

    if os.path.exists(test_svg):
        print(f"\nLoading SVG: {test_svg}")
        svg_array = loader.load_svg(test_svg)
        if svg_array is not None:
            print(f"  ✅ Loaded SVG: shape={svg_array.shape}, dtype={svg_array.dtype}")
        else:
            print("  ❌ Failed to load SVG")

    # Test quality metrics wrapper
    if os.path.exists(test_png) and os.path.exists(test_svg):
        print("\nTesting QualityMetricsWrapper...")
        wrapper = QualityMetricsWrapper()

        ssim = wrapper.calculate_ssim_from_paths(test_png, test_svg)
        mse = wrapper.calculate_mse_from_paths(test_png, test_svg)
        psnr = wrapper.calculate_psnr_from_paths(test_png, test_svg)

        print(f"  SSIM: {ssim:.4f}")
        print(f"  MSE: {mse:.2f}")
        print(f"  PSNR: {psnr:.2f} dB")

    print("\n✅ Image loader test complete!")


if __name__ == "__main__":
    test_loader()