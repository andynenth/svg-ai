"""
Unified Quality Module
Quality measurement and tracking system
"""
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import json
import numpy as np

class QualitySystem:
    """Complete quality measurement and tracking system"""

    def __init__(self) -> None:
        self.metrics_cache = {}

    def calculate_ssim(self, original_path: str, converted_path: str) -> float:
        """Calculate Structural Similarity Index"""
        try:
            from skimage.metrics import structural_similarity as ssim
            original = cv2.imread(original_path)
            converted = cv2.imread(converted_path)
            if original is None or converted is None:
                return 0.0
            if original.shape != converted.shape:
                converted = cv2.resize(converted, (original.shape[1], original.shape[0]))
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            converted_gray = cv2.cvtColor(converted, cv2.COLOR_BGR2GRAY)
            score = ssim(original_gray, converted_gray, data_range=255)
            return score
        except Exception:
            return 0.0

    def calculate_comprehensive_metrics(self, original_path: str, svg_path: str) -> Dict:
        """Calculate all quality metrics"""
        # Calculate real metrics
        try:
            # Render SVG to PNG for comparison
            import cairosvg
            import tempfile
            import os

            # Create temp file for rendered PNG
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
                tmp_png_path = tmp_png.name

            # Render SVG to PNG
            cairosvg.svg2png(url=svg_path, write_to=tmp_png_path)

            # Calculate real SSIM
            ssim_value = self.calculate_ssim(original_path, tmp_png_path)

            # Calculate MSE and PSNR
            original = cv2.imread(original_path)
            rendered = cv2.imread(tmp_png_path)

            if original is not None and rendered is not None:
                # Resize if needed
                if original.shape != rendered.shape:
                    rendered = cv2.resize(rendered, (original.shape[1], original.shape[0]))

                # Calculate MSE
                mse_value = np.mean((original.astype(float) - rendered.astype(float)) ** 2)

                # Calculate PSNR
                if mse_value > 0:
                    psnr_value = 20 * np.log10(255.0 / np.sqrt(mse_value))
                else:
                    psnr_value = float('inf')  # Perfect match
            else:
                mse_value = 0.0
                psnr_value = 0.0

            # Clean up temp file
            os.unlink(tmp_png_path)

        except Exception as e:
            # Fallback to defaults if calculation fails
            ssim_value = 0.0
            mse_value = 0.0
            psnr_value = 0.0

        metrics = {
            'ssim': ssim_value,
            'mse': mse_value,
            'psnr': psnr_value,
            'file_size_original': Path(original_path).stat().st_size if Path(original_path).exists() else 0,
            'file_size_svg': Path(svg_path).stat().st_size if Path(svg_path).exists() else 0
        }
        if metrics['file_size_original'] > 0 and metrics['file_size_svg'] > 0:
            metrics['compression_ratio'] = metrics['file_size_original'] / metrics['file_size_svg']
        else:
            metrics['compression_ratio'] = 1.0
        metrics['quality_score'] = metrics['ssim'] * 0.7 + metrics['compression_ratio'] / 10.0 * 0.3
        return metrics

    def calculate_metrics(self, original_path: str, converted_path: str) -> dict:
        """
        Compatibility wrapper for integration tests.

        Maps the expected calculate_metrics API to the existing
        calculate_comprehensive_metrics implementation.

        Args:
            original_path: Path to the original image file
            converted_path: Path to the converted SVG file

        Returns:
            dict: Quality metrics including SSIM, MSE, PSNR, file sizes,
                  compression ratio, and overall quality score
        """
        return self.calculate_comprehensive_metrics(original_path, converted_path)


# Legacy compatibility
ENHANCEDMETRICS = QualitySystem
QUALITYTRACKER = QualitySystem
ABTesting = QualitySystem