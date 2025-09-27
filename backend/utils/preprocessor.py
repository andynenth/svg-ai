from PIL import Image
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """Preprocess PNG images for optimal SVG conversion."""

    @staticmethod
    def prepare_logo(
        image_path: str,
        target_size: int = 512,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """
        Prepare logo image for conversion.

        Args:
            image_path: Path to input image
            target_size: Maximum dimension for resizing
            background_color: RGB color for background (default white)

        Returns:
            Preprocessed PIL Image
        """
        img = Image.open(image_path)

        # Convert RGBA to RGB with specified background
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, background_color)
            background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize maintaining aspect ratio
        img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

        return img

    @staticmethod
    def remove_background(img: Image.Image, threshold: int = 240) -> Image.Image:
        """
        Remove white/light background from image.

        Args:
            img: Input PIL Image
            threshold: Brightness threshold for background (0-255)

        Returns:
            Image with transparent background
        """
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        data = np.array(img)

        # Create mask for non-white pixels
        rgb_sum = data[:, :, :3].sum(axis=2)
        mask = rgb_sum < (threshold * 3)

        # Apply mask to alpha channel
        data[:, :, 3] = mask * 255

        return Image.fromarray(data, 'RGBA')

    @staticmethod
    def enhance_edges(img: Image.Image, factor: float = 1.5) -> Image.Image:
        """
        Enhance edges for better tracing.

        Args:
            img: Input PIL Image
            factor: Enhancement factor (1.0 = no change)

        Returns:
            Edge-enhanced image
        """
        from PIL import ImageFilter, ImageEnhance

        # Apply edge enhancement
        img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)

        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(factor)

        return img

    @staticmethod
    def quantize_colors(img: Image.Image, num_colors: int = 16) -> Image.Image:
        """
        Reduce number of colors in image.

        Args:
            img: Input PIL Image
            num_colors: Target number of colors

        Returns:
            Color-quantized image
        """
        # Convert to P mode (palette) with specified number of colors
        img_quant = img.quantize(colors=num_colors, method=Image.MEDIANCUT)

        # Convert back to RGB for compatibility
        return img_quant.convert('RGB')

    @staticmethod
    def denoise(img: Image.Image, radius: int = 2) -> Image.Image:
        """
        Remove noise from image.

        Args:
            img: Input PIL Image
            radius: Blur radius for denoising

        Returns:
            Denoised image
        """
        from PIL import ImageFilter

        # Apply median filter to remove noise
        return img.filter(ImageFilter.MedianFilter(size=radius * 2 + 1))

    @staticmethod
    def count_colors(img: Image.Image) -> int:
        """
        Count unique colors in image.

        Args:
            img: Input PIL Image

        Returns:
            Number of unique colors
        """
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Get unique colors
        colors = img.getcolors(maxcolors=256 * 256 * 256)
        return len(colors) if colors else 256 * 256 * 256

    @staticmethod
    def analyze_complexity(image_path: str) -> dict:
        """
        Analyze image complexity for routing decisions.

        Args:
            image_path: Path to image

        Returns:
            Dictionary with complexity metrics
        """
        img = Image.open(image_path)

        # Convert to RGB for analysis
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Calculate metrics
        data = np.array(img)

        return {
            'dimensions': img.size,
            'aspect_ratio': img.size[0] / img.size[1],
            'unique_colors': min(ImagePreprocessor.count_colors(img), 10000),
            'has_transparency': Image.open(image_path).mode == 'RGBA',
            'brightness_mean': data.mean(),
            'brightness_std': data.std(),
            'file_size_kb': os.path.getsize(image_path) / 1024
        }


import os