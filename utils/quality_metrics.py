"""
Advanced quality metrics for PNG to SVG conversion.
"""

import os
import io
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple
import time


class QualityMetrics:
    """Calculate visual quality metrics for SVG conversion."""

    @staticmethod
    def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate Mean Squared Error.

        Args:
            img1, img2: Input images as numpy arrays

        Returns:
            MSE value (lower is better, 0 = identical)
        """
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions")

        # Handle RGBA images
        if len(img1.shape) == 3 and img1.shape[2] == 4:
            alpha1 = img1[:,:,3:4] / 255.0
            rgb1 = img1[:,:,:3]
            white = np.ones_like(rgb1) * 255
            img1 = (rgb1 * alpha1 + white * (1 - alpha1)).astype(np.uint8)

        if len(img2.shape) == 3 and img2.shape[2] == 4:
            alpha2 = img2[:,:,3:4] / 255.0
            rgb2 = img2[:,:,:3]
            white = np.ones_like(rgb2) * 255
            img2 = (rgb2 * alpha2 + white * (1 - alpha2)).astype(np.uint8)

        return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

    @staticmethod
    def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.

        Args:
            img1, img2: Input images as numpy arrays

        Returns:
            PSNR value in dB (higher is better, >30 is good)
        """
        mse = QualityMetrics.calculate_mse(img1, img2)

        if mse == 0:
            return float('inf')

        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    @staticmethod
    def calculate_perceptual_loss(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate simple perceptual loss based on edge and color differences.

        Args:
            img1, img2: Input images as numpy arrays

        Returns:
            Perceptual loss value (lower is better)
        """
        from scipy import ndimage

        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions")

        # Handle RGBA
        if len(img1.shape) == 3 and img1.shape[2] == 4:
            alpha1 = img1[:,:,3:4] / 255.0
            rgb1 = img1[:,:,:3]
            white = np.ones_like(rgb1) * 255
            img1 = (rgb1 * alpha1 + white * (1 - alpha1)).astype(np.uint8)

        if len(img2.shape) == 3 and img2.shape[2] == 4:
            alpha2 = img2[:,:,3:4] / 255.0
            rgb2 = img2[:,:,:3]
            white = np.ones_like(rgb2) * 255
            img2 = (rgb2 * alpha2 + white * (1 - alpha2)).astype(np.uint8)

        # Convert to grayscale for edge detection
        if len(img1.shape) == 3:
            gray1 = np.mean(img1, axis=2)
            gray2 = np.mean(img2, axis=2)
        else:
            gray1 = img1
            gray2 = img2

        # Edge detection using Sobel filters
        edges1 = ndimage.sobel(gray1)
        edges2 = ndimage.sobel(gray2)

        # Edge difference
        edge_diff = np.mean(np.abs(edges1 - edges2))

        # Color difference (if color image)
        if len(img1.shape) == 3:
            color_diff = np.mean(np.abs(img1.astype(float) - img2.astype(float)))
        else:
            color_diff = np.mean(np.abs(gray1.astype(float) - gray2.astype(float)))

        # Combined perceptual loss (weighted sum)
        perceptual_loss = 0.7 * edge_diff + 0.3 * color_diff

        return perceptual_loss

    @staticmethod
    def calculate_unified_score(ssim: float, psnr: float, perceptual: float,
                              file_size_ratio: float) -> float:
        """
        Calculate unified quality score combining all metrics.

        Args:
            ssim: SSIM score (0-1)
            psnr: PSNR value in dB
            perceptual: Perceptual loss (lower is better)
            file_size_ratio: SVG/PNG size ratio

        Returns:
            Unified score (0-100)
        """
        # Normalize PSNR (30-50 dB range to 0-1)
        psnr_norm = min(max((psnr - 30) / 20, 0), 1) if psnr != float('inf') else 1.0

        # Normalize perceptual loss (inverse, assume 0-100 range)
        perceptual_norm = max(1 - perceptual / 100, 0)

        # Normalize file size (bonus for smaller files)
        size_norm = max(1 - file_size_ratio, 0) if file_size_ratio < 1 else 0.5 / file_size_ratio

        # Weighted combination
        unified = (
            0.4 * ssim +           # Visual similarity most important
            0.2 * psnr_norm +      # Signal quality
            0.2 * perceptual_norm + # Perceptual quality
            0.2 * size_norm        # File size efficiency
        ) * 100

        return min(max(unified, 0), 100)

    @staticmethod
    def calculate_ssim(img1: np.ndarray, img2: np.ndarray,
                       k1: float = 0.01, k2: float = 0.03, win_size: int = 11) -> float:
        """
        Calculate Structural Similarity Index (SSIM).

        Args:
            img1, img2: Input images as numpy arrays
            k1, k2: SSIM constants
            win_size: Window size for local statistics

        Returns:
            SSIM score between 0 and 1 (1 = identical)
        """
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions")

        # Handle RGBA images with transparency by compositing on white
        if len(img1.shape) == 3 and img1.shape[2] == 4:
            # Composite on white background
            alpha1 = img1[:,:,3:4] / 255.0
            rgb1 = img1[:,:,:3]
            white = np.ones_like(rgb1) * 255
            img1 = (rgb1 * alpha1 + white * (1 - alpha1)).astype(np.uint8)

        if len(img2.shape) == 3 and img2.shape[2] == 4:
            # Composite on white background
            alpha2 = img2[:,:,3:4] / 255.0
            rgb2 = img2[:,:,:3]
            white = np.ones_like(rgb2) * 255
            img2 = (rgb2 * alpha2 + white * (1 - alpha2)).astype(np.uint8)

        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1 = np.mean(img1, axis=2)
        if len(img2.shape) == 3:
            img2 = np.mean(img2, axis=2)

        # Normalize to 0-1 range
        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0

        # Constants
        C1 = (k1 * 1) ** 2
        C2 = (k2 * 1) ** 2

        # Mean
        mu1 = QualityMetrics._window_mean(img1, win_size)
        mu2 = QualityMetrics._window_mean(img2, win_size)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Variance and covariance
        sigma1_sq = QualityMetrics._window_mean(img1 ** 2, win_size) - mu1_sq
        sigma2_sq = QualityMetrics._window_mean(img2 ** 2, win_size) - mu2_sq
        sigma12 = QualityMetrics._window_mean(img1 * img2, win_size) - mu1_mu2

        # SSIM
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / denominator
        return float(np.mean(ssim_map))

    @staticmethod
    def _window_mean(img: np.ndarray, win_size: int) -> np.ndarray:
        """Calculate mean using sliding window."""
        from scipy.ndimage import uniform_filter
        return uniform_filter(img, size=win_size, mode='constant')

    @staticmethod
    def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate Mean Squared Error.

        Args:
            img1, img2: Input images as numpy arrays

        Returns:
            MSE value (lower is better)
        """
        if img1.shape != img2.shape:
            # Resize to match
            img2_pil = Image.fromarray(img2.astype('uint8'))
            img2_pil = img2_pil.resize((img1.shape[1], img1.shape[0]), Image.LANCZOS)
            img2 = np.array(img2_pil)

        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        return float(mse)

    @staticmethod
    def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.

        Args:
            img1, img2: Input images as numpy arrays

        Returns:
            PSNR in dB (higher is better)
        """
        mse = QualityMetrics.calculate_mse(img1, img2)
        if mse == 0:
            return 100.0  # Identical images

        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return float(psnr)

    @staticmethod
    def calculate_edge_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate edge similarity using Sobel edge detection.

        Args:
            img1, img2: Input images as numpy arrays

        Returns:
            Edge similarity score (0-1, higher is better)
        """
        from scipy import ndimage

        # Convert to grayscale
        if len(img1.shape) == 3:
            gray1 = np.mean(img1, axis=2)
        else:
            gray1 = img1

        if len(img2.shape) == 3:
            gray2 = np.mean(img2, axis=2)
        else:
            gray2 = img2

        # Sobel edge detection
        edges1 = QualityMetrics._sobel_edges(gray1)
        edges2 = QualityMetrics._sobel_edges(gray2)

        # Normalize edges
        edges1 = edges1 / (np.max(edges1) + 1e-10)
        edges2 = edges2 / (np.max(edges2) + 1e-10)

        # Calculate similarity
        similarity = 1 - np.mean(np.abs(edges1 - edges2))
        return float(similarity)

    @staticmethod
    def _sobel_edges(img: np.ndarray) -> np.ndarray:
        """Apply Sobel edge detection."""
        from scipy import ndimage

        sx = ndimage.sobel(img, axis=0, mode='constant')
        sy = ndimage.sobel(img, axis=1, mode='constant')
        return np.hypot(sx, sy)

    @staticmethod
    def calculate_color_accuracy(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate color distribution similarity.

        Args:
            img1, img2: Input images as numpy arrays

        Returns:
            Color accuracy score (0-1, higher is better)
        """
        if len(img1.shape) == 2:
            return 1.0  # Grayscale, skip color comparison

        # Calculate histograms for each channel
        score = 0
        for channel in range(min(img1.shape[2], 3)):  # RGB channels only
            hist1, _ = np.histogram(img1[:, :, channel], bins=256, range=(0, 255))
            hist2, _ = np.histogram(img2[:, :, channel], bins=256, range=(0, 255))

            # Normalize histograms
            hist1 = hist1.astype(float) / hist1.sum()
            hist2 = hist2.astype(float) / hist2.sum()

            # Calculate correlation
            correlation = np.corrcoef(hist1, hist2)[0, 1]
            score += max(0, correlation)  # Ensure non-negative

        return score / 3.0  # Average across channels


class SVGRenderer:
    """Render SVG to PNG for comparison."""

    @staticmethod
    def render_to_array(svg_content: str, width: int, height: int) -> np.ndarray:
        """Render SVG to numpy array at specified size."""
        result = SVGRenderer.svg_to_png(svg_content, (width, height))
        if result is None:
            # Return blank array if rendering fails
            return np.zeros((height, width, 3), dtype=np.uint8)
        return result

    @staticmethod
    def svg_to_png(svg_content: str, target_size: Tuple[int, int] = (256, 256)) -> Optional[np.ndarray]:
        """
        Convert SVG content to PNG array.

        Args:
            svg_content: SVG file content as string
            target_size: Target image size

        Returns:
            Image as numpy array or None if conversion fails
        """
        try:
            import cairosvg

            # Convert SVG to PNG bytes
            png_bytes = cairosvg.svg2png(
                bytestring=svg_content.encode('utf-8'),
                output_width=target_size[0],
                output_height=target_size[1]
            )

            # Load as PIL Image and convert to numpy array
            img = Image.open(io.BytesIO(png_bytes))
            # Ensure RGB format
            if img.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if len(img.split()) > 3 else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)

        except ImportError:
            # Fallback: try using svglib if cairosvg not available
            try:
                from svglib.svglib import svg2rlg
                from reportlab.graphics import renderPM

                drawing = svg2rlg(io.BytesIO(svg_content.encode('utf-8')))
                img_data = renderPM.drawToString(drawing, fmt='PNG')
                img = Image.open(io.BytesIO(img_data))
                img = img.resize(target_size, Image.LANCZOS)
                return np.array(img)

            except:
                return None

        except Exception as e:
            print(f"Warning: Could not render SVG: {e}")
            return None


class ComprehensiveMetrics:
    """Calculate all metrics for conversion evaluation."""

    def __init__(self):
        self.quality_calc = QualityMetrics()
        self.renderer = SVGRenderer()

    def compare_images(self, original_path: str, svg_path: str) -> Dict[str, float]:
        """
        Compare original PNG with SVG file.

        Args:
            original_path: Path to original PNG
            svg_path: Path to SVG file

        Returns:
            Dictionary with SSIM, MSE, and PSNR metrics
        """
        # Load original image
        original = Image.open(original_path).convert('RGB')
        original_array = np.array(original)

        # Read SVG content
        with open(svg_path, 'r') as f:
            svg_content = f.read()

        # Render SVG to same size as original
        rendered_array = self.renderer.render_to_array(
            svg_content,
            width=original.width,
            height=original.height
        )

        # Ensure both arrays have the same shape (RGB, no alpha)
        if rendered_array.shape[-1] == 4:
            # Remove alpha channel if present
            rendered_array = rendered_array[:, :, :3]

        if original_array.shape != rendered_array.shape:
            # Resize if needed
            from PIL import Image as PILImage
            rendered_img = PILImage.fromarray(rendered_array.astype('uint8'))
            rendered_img = rendered_img.resize((original.width, original.height), PILImage.LANCZOS)
            rendered_array = np.array(rendered_img)

        # Calculate metrics
        return {
            'ssim': self.quality_calc.calculate_ssim(original_array, rendered_array),
            'mse': self.quality_calc.calculate_mse(original_array, rendered_array),
            'psnr': self.quality_calc.calculate_psnr(original_array, rendered_array)
        }

    def evaluate(self, png_path: str, svg_content: str,
                conversion_time: float) -> Dict[str, Any]:
        """
        Evaluate conversion with all available metrics.

        Args:
            png_path: Path to original PNG
            svg_content: Generated SVG content
            conversion_time: Time taken for conversion

        Returns:
            Dictionary with comprehensive metrics
        """
        # Load original image
        original = Image.open(png_path)
        if original.mode == 'RGBA':
            # Convert to RGB with white background
            background = Image.new('RGB', original.size, (255, 255, 255))
            background.paste(original, mask=original.split()[3])
            original = background
        original_array = np.array(original)

        # Render SVG for comparison
        rendered_array = self.renderer.svg_to_png(svg_content, original.size)

        metrics = {
            'file': {
                'png_size_kb': os.path.getsize(png_path) / 1024 if os.path.exists(png_path) else 0,
                'svg_size_kb': len(svg_content.encode('utf-8')) / 1024,
                'compression_ratio': len(svg_content.encode('utf-8')) / os.path.getsize(png_path)
                                   if os.path.exists(png_path) and os.path.getsize(png_path) > 0 else 0
            },
            'performance': {
                'conversion_time_s': conversion_time,
                'svg_complexity': self._analyze_svg_complexity(svg_content)
            },
            'visual': {}
        }

        # Calculate visual metrics only if rendering succeeded
        if rendered_array is not None:
            # Ensure same shape
            if rendered_array.shape != original_array.shape:
                rendered_pil = Image.fromarray(rendered_array)
                rendered_pil = rendered_pil.resize(original.size, Image.LANCZOS)
                rendered_array = np.array(rendered_pil)

            try:
                metrics['visual'] = {
                    'ssim': self.quality_calc.calculate_ssim(original_array, rendered_array),
                    'mse': self.quality_calc.calculate_mse(original_array, rendered_array),
                    'psnr': self.quality_calc.calculate_psnr(original_array, rendered_array),
                    'edge_similarity': self.quality_calc.calculate_edge_similarity(original_array, rendered_array),
                    'color_accuracy': self.quality_calc.calculate_color_accuracy(original_array, rendered_array)
                }
            except Exception as e:
                print(f"Warning: Could not calculate visual metrics: {e}")
                metrics['visual'] = {
                    'ssim': 0.0,
                    'mse': float('inf'),
                    'psnr': 0.0,
                    'edge_similarity': 0.0,
                    'color_accuracy': 0.0
                }
        else:
            metrics['visual'] = {
                'error': 'Could not render SVG for comparison'
            }

        return metrics

    def _analyze_svg_complexity(self, svg_content: str) -> Dict[str, int]:
        """Analyze SVG complexity."""
        import re

        return {
            'total_size': len(svg_content),
            'num_paths': svg_content.count('<path'),
            'num_groups': svg_content.count('<g'),
            'num_colors': len(set(re.findall(r'fill="([^"]+)"', svg_content))),
            'num_commands': sum(svg_content.count(cmd) for cmd in ['M', 'L', 'C', 'Q', 'A', 'Z'])
        }


# Simple fallback if scipy not available
try:
    from scipy import ndimage
except ImportError:
    print("Warning: scipy not available, some metrics will be limited")