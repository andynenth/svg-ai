#!/usr/bin/env python3
"""
Image processing utilities for PNG to SVG conversion.

This module provides centralized image processing functions to eliminate
code duplication across converters.
"""

import logging
import numpy as np
from PIL import Image
from typing import Tuple, Union, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageUtils:
    """
    Centralized image processing utilities for converters.

    This class consolidates common image processing operations that were
    previously duplicated across multiple converter classes.
    """

    @staticmethod
    def convert_to_rgba(image_path: Union[str, Path]) -> Image.Image:
        """
        Load image and convert to RGBA format.

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image in RGBA format

        Raises:
            FileNotFoundError: If image file doesn't exist
            IOError: If image cannot be loaded or is corrupted
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            img = Image.open(image_path)

            # Convert to RGBA if not already
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
                logger.debug(f"Converted image from {img.mode} to RGBA: {image_path.name}")

            return img

        except Exception as e:
            logger.error(f"Failed to load image as RGBA: {image_path} - {e}")
            raise

    @staticmethod
    def composite_on_background(
        image: Image.Image,
        bg_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """
        Composite RGBA image onto a solid background color.

        This method handles alpha channel compositing properly, which is commonly
        needed when converting transparent images for converters that don't
        support transparency.

        Args:
            image: PIL Image (should be RGBA format)
            bg_color: Background color as RGB tuple (default: white)

        Returns:
            PIL Image in RGB format with composited background

        Raises:
            ValueError: If background color is invalid
        """
        try:
            # Validate background color
            if not (isinstance(bg_color, (tuple, list)) and len(bg_color) == 3):
                raise ValueError(f"Background color must be RGB tuple, got: {bg_color}")

            if not all(0 <= c <= 255 for c in bg_color):
                raise ValueError(f"RGB values must be 0-255, got: {bg_color}")

            # Handle different image modes
            if image.mode == 'RGBA':
                # Create background with specified color
                background = Image.new('RGB', image.size, bg_color)
                # Composite using alpha channel as mask
                background.paste(image, mask=image.split()[3])
                logger.debug(f"Composited RGBA image on {bg_color} background")
                return background

            elif image.mode == 'P':
                # Convert palette to RGBA first, then composite
                rgba_img = image.convert('RGBA')
                return ImageUtils.composite_on_background(rgba_img, bg_color)

            elif image.mode in ['RGB', 'L']:
                # Already opaque, just ensure RGB format
                if image.mode == 'L':
                    image = image.convert('RGB')
                logger.debug(f"Image already opaque ({image.mode}), returning as-is")
                return image

            else:
                # Convert other modes to RGB
                rgb_img = image.convert('RGB')
                logger.debug(f"Converted {image.mode} to RGB")
                return rgb_img

        except Exception as e:
            logger.error(f"Failed to composite image on background: {e}")
            raise

    @staticmethod
    def convert_to_grayscale(image: Image.Image) -> Image.Image:
        """
        Convert image to grayscale format.

        Handles various input formats and ensures proper grayscale conversion
        while preserving image quality.

        Args:
            image: PIL Image in any format

        Returns:
            PIL Image in grayscale ('L') format
        """
        try:
            original_mode = image.mode

            # Handle RGBA images by compositing first
            if image.mode == 'RGBA':
                # Composite on white background before grayscale conversion
                image = ImageUtils.composite_on_background(image, (255, 255, 255))

            # Convert to grayscale
            if image.mode != 'L':
                grayscale_img = image.convert('L')
                logger.debug(f"Converted {original_mode} to grayscale")
                return grayscale_img
            else:
                logger.debug("Image already in grayscale format")
                return image

        except Exception as e:
            logger.error(f"Failed to convert image to grayscale: {e}")
            raise

    @staticmethod
    def apply_alpha_threshold(
        image: Image.Image,
        threshold: int = 128
    ) -> Image.Image:
        """
        Apply threshold to alpha channel, creating binary transparency.

        This is useful for converting anti-aliased transparency to hard edges
        suitable for converters that work better with binary masks.

        Args:
            image: PIL Image (preferably RGBA)
            threshold: Alpha threshold value (0-255, default: 128)

        Returns:
            PIL Image with thresholded alpha channel

        Raises:
            ValueError: If threshold is out of valid range
        """
        try:
            # Validate threshold
            if not (0 <= threshold <= 255):
                raise ValueError(f"Threshold must be 0-255, got: {threshold}")

            # Ensure image has alpha channel
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
                logger.debug("Converted image to RGBA for alpha threshold")

            # Convert to numpy array for efficient processing
            img_array = np.array(image)

            # Apply threshold to alpha channel
            alpha_channel = img_array[:, :, 3]
            thresholded_alpha = np.where(alpha_channel >= threshold, 255, 0).astype(np.uint8)

            # Update alpha channel
            img_array[:, :, 3] = thresholded_alpha

            # Convert back to PIL Image
            result_img = Image.fromarray(img_array, 'RGBA')

            # Log statistics
            transparent_pixels = np.sum(thresholded_alpha == 0)
            total_pixels = alpha_channel.size
            transparency_ratio = transparent_pixels / total_pixels

            logger.debug(f"Applied alpha threshold {threshold}: "
                        f"{transparency_ratio:.1%} pixels became transparent")

            return result_img

        except Exception as e:
            logger.error(f"Failed to apply alpha threshold: {e}")
            raise

    @staticmethod
    def get_image_mode_info(image: Image.Image) -> dict:
        """
        Get detailed information about image mode and characteristics.

        Useful for debugging and understanding image processing requirements.

        Args:
            image: PIL Image

        Returns:
            Dictionary with image mode information
        """
        try:
            info = {
                'mode': image.mode,
                'size': image.size,
                'has_alpha': image.mode in ['RGBA', 'LA', 'PA'],
                'is_grayscale': image.mode in ['L', 'LA'],
                'is_palette': image.mode in ['P', 'PA'],
                'bands': len(image.getbands()),
                'format': getattr(image, 'format', 'Unknown')
            }

            # Add alpha channel statistics if present
            if info['has_alpha']:
                if image.mode == 'RGBA':
                    alpha_array = np.array(image.split()[3])
                    info['alpha_stats'] = {
                        'min': int(alpha_array.min()),
                        'max': int(alpha_array.max()),
                        'mean': float(alpha_array.mean()),
                        'has_transparency': alpha_array.min() < 255,
                        'fully_opaque_pixels': int(np.sum(alpha_array == 255)),
                        'transparent_pixels': int(np.sum(alpha_array == 0))
                    }

            return info

        except Exception as e:
            logger.error(f"Failed to get image mode info: {e}")
            return {'mode': 'unknown', 'error': str(e)}

    @staticmethod
    def create_binary_mask(
        image: Image.Image,
        threshold: int = 128,
        invert: bool = False
    ) -> Image.Image:
        """
        Create binary (1-bit) mask from image.

        Useful for creating masks for potrace-style converters that need
        binary input.

        Args:
            image: PIL Image
            threshold: Threshold for binarization (0-255)
            invert: Whether to invert the binary result

        Returns:
            PIL Image in binary ('1') format
        """
        try:
            # Convert to grayscale first
            grayscale_img = ImageUtils.convert_to_grayscale(image)

            # Apply threshold to create binary image
            if invert:
                binary_img = grayscale_img.point(lambda x: 0 if x > threshold else 255, mode='1')
            else:
                binary_img = grayscale_img.point(lambda x: 255 if x > threshold else 0, mode='1')

            logger.debug(f"Created binary mask with threshold {threshold}, inverted: {invert}")
            return binary_img

        except Exception as e:
            logger.error(f"Failed to create binary mask: {e}")
            raise

    @staticmethod
    def safe_image_load(image_path: Union[str, Path]) -> Optional[Image.Image]:
        """
        Safely load image with comprehensive error handling.

        Args:
            image_path: Path to image file

        Returns:
            PIL Image or None if loading failed
        """
        try:
            return ImageUtils.convert_to_rgba(image_path)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None

    @staticmethod
    def validate_image_for_conversion(image: Image.Image) -> Tuple[bool, str]:
        """
        Validate that image is suitable for conversion.

        Args:
            image: PIL Image to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check basic requirements
            if image.size[0] == 0 or image.size[1] == 0:
                return False, "Image has zero dimensions"

            if image.size[0] > 10000 or image.size[1] > 10000:
                return False, "Image too large (max 10000x10000)"

            # Check if image has any content
            if image.mode in ['RGBA', 'LA']:
                # For images with alpha, check if all pixels are transparent
                alpha_array = np.array(image.split()[-1])
                if alpha_array.max() == 0:
                    return False, "Image is completely transparent"

            # Warn about very small images
            if image.size[0] < 10 or image.size[1] < 10:
                return True, "Warning: Image is very small, conversion quality may be poor"

            return True, "Image is valid for conversion"

        except Exception as e:
            return False, f"Image validation failed: {e}"