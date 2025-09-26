#!/usr/bin/env python3
"""Standalone iterative optimizer without click dependency."""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
from PIL import Image

from converters.vtracer_converter import VTracerConverter
from utils.quality_metrics import ComprehensiveMetrics

class IterativeOptimizer:
    """Iteratively optimize VTracer parameters to achieve target quality."""

    def __init__(self, input_path: str, output_dir: str,
                 target_ssim: float = 0.98, max_iterations: int = 10):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_ssim = target_ssim
        self.max_iterations = max_iterations
        self.converter = VTracerConverter()
        self.metrics = ComprehensiveMetrics()

        # Preset configurations for different logo types
        self.presets = {
            'simple': {
                'color_precision': 4,
                'layer_difference': 32,
                'corner_threshold': 30,
                'path_precision': 8,
                'length_threshold': 2.0,
                'splice_threshold': 45,
                'max_iterations': 10
            },
            'text': {
                'color_precision': 6,
                'layer_difference': 16,
                'corner_threshold': 20,
                'path_precision': 10,
                'length_threshold': 1.0,
                'splice_threshold': 30,
                'max_iterations': 12
            },
            'gradient': {
                'color_precision': 8,
                'layer_difference': 8,
                'corner_threshold': 60,
                'path_precision': 6,
                'length_threshold': 3.0,
                'splice_threshold': 60,
                'max_iterations': 15
            },
            'complex': {
                'color_precision': 10,
                'layer_difference': 4,
                'corner_threshold': 90,
                'path_precision': 4,
                'length_threshold': 4.0,
                'splice_threshold': 90,
                'max_iterations': 20
            }
        }

    def detect_logo_type(self) -> str:
        """Detect the type of logo based on image characteristics."""
        try:
            img = Image.open(self.input_path).convert('RGBA')
            pixels = np.array(img)

            # Calculate various metrics
            unique_colors = len(np.unique(pixels.reshape(-1, pixels.shape[-1]), axis=0))
            has_transparency = np.any(pixels[:, :, 3] < 255) if pixels.shape[-1] == 4 else False

            # Calculate edge complexity (simplified without scipy)
            gray = np.mean(pixels[:, :, :3], axis=2) if pixels.shape[-1] >= 3 else pixels
            # Simple edge detection using gradients
            h_edges = np.abs(np.diff(gray, axis=0))
            v_edges = np.abs(np.diff(gray, axis=1))
            edge_magnitude = np.mean(h_edges) + np.mean(v_edges)
            edge_ratio = min(edge_magnitude / 255.0, 1.0)  # Normalize

            # Detect gradients
            gradient_score = self._detect_gradients(pixels)

            # Classification logic
            if unique_colors <= 10 and edge_ratio < 0.1:
                return 'simple'
            elif gradient_score > 0.3 or unique_colors > 100:
                return 'gradient'
            elif edge_ratio > 0.2 and unique_colors < 50:
                return 'text'
            else:
                return 'complex'

        except Exception as e:
            print(f"Error detecting logo type: {e}")
            return 'complex'  # Default to complex

    def _detect_gradients(self, pixels: np.ndarray) -> float:
        """Detect presence of gradients in the image."""
        if pixels.shape[-1] < 3:
            return 0.0

        rgb = pixels[:, :, :3]
        # Check horizontal and vertical color changes
        h_changes = np.mean(np.abs(np.diff(rgb, axis=1)))
        v_changes = np.mean(np.abs(np.diff(rgb, axis=0)))

        # High but smooth changes indicate gradients
        gradient_score = (h_changes + v_changes) / 510.0  # Normalize

        return min(gradient_score, 1.0)

    def optimize(self) -> Dict[str, Any]:
        """Run iterative optimization."""
        print(f"\nStarting optimization for: {self.input_path.name}")

        # Detect logo type
        logo_type = self.detect_logo_type()
        print(f"Detected logo type: {logo_type}")

        # Get initial parameters
        best_params = self.presets[logo_type].copy()
        best_ssim = 0
        best_svg_path = None
        best_metrics = None

        for iteration in range(1, self.max_iterations + 1):
            print(f"\nIteration {iteration}/{self.max_iterations}")

            # Convert with current parameters
            svg_path = self.output_dir / f"{self.input_path.stem}.iter_{iteration}.svg"

            # Create temporary file for VTracer output requirement
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_svg:
                tmp_svg_path = tmp_svg.name

            try:
                # Convert using VTracer
                self.converter.convert(
                    str(self.input_path),
                    tmp_svg_path,
                    **best_params
                )

                # Move to final location
                import shutil
                shutil.move(tmp_svg_path, str(svg_path))

            except Exception as e:
                print(f"Conversion error: {e}")
                if os.path.exists(tmp_svg_path):
                    os.remove(tmp_svg_path)
                continue

            # Calculate metrics
            try:
                metrics_result = self.metrics.compare_images(str(self.input_path), str(svg_path))
                current_ssim = metrics_result['ssim']

                print(f"  SSIM: {current_ssim:.4f} (target: {self.target_ssim:.4f})")
                print(f"  MSE: {metrics_result['mse']:.2f}")
                print(f"  PSNR: {metrics_result['psnr']:.2f}")

                # Update best if improved
                if current_ssim > best_ssim:
                    best_ssim = current_ssim
                    best_svg_path = str(svg_path)
                    best_metrics = metrics_result
                    print(f"  ✓ New best SSIM: {best_ssim:.4f}")

                # Check if target reached
                if current_ssim >= self.target_ssim:
                    print(f"✅ Target SSIM reached!")
                    break

                # Adjust parameters if not at target
                if iteration < self.max_iterations:
                    best_params = self._adjust_parameters(
                        best_params, current_ssim, logo_type
                    )

            except Exception as e:
                print(f"Metrics calculation error: {e}")
                continue

        # Save final optimized version
        if best_svg_path:
            final_path = self.output_dir / f"{self.input_path.stem}.optimized.svg"
            import shutil
            shutil.copy(best_svg_path, final_path)

            # Clean up iteration files
            for iter_file in self.output_dir.glob(f"{self.input_path.stem}.iter_*.svg"):
                if str(iter_file) != best_svg_path:
                    iter_file.unlink()

            return {
                'file': str(self.input_path),
                'success': best_ssim >= self.target_ssim,
                'ssim': best_ssim,
                'logo_type': logo_type,
                'iterations': iteration,
                'svg_path': str(final_path),
                'best_params': best_params,
                'metrics': best_metrics
            }
        else:
            return {
                'file': str(self.input_path),
                'success': False,
                'error': 'No successful conversion',
                'logo_type': logo_type,
                'iterations': iteration
            }

    def _adjust_parameters(self, params: Dict, current_ssim: float,
                          logo_type: str) -> Dict:
        """Adjust parameters based on current quality."""
        gap = self.target_ssim - current_ssim
        new_params = params.copy()

        if gap > 0.1:  # Large gap - major adjustments
            scale = 1.5
        elif gap > 0.05:  # Medium gap
            scale = 1.2
        else:  # Small gap - fine tuning
            scale = 1.1

        # Adjust based on logo type
        if logo_type == 'text':
            # Text needs sharp edges
            new_params['corner_threshold'] = max(10, int(params['corner_threshold'] / scale))
            new_params['path_precision'] = min(12, int(params['path_precision'] * scale))
            new_params['length_threshold'] = max(0.5, params['length_threshold'] / scale)

        elif logo_type == 'gradient':
            # Gradients need smooth transitions
            new_params['color_precision'] = min(12, int(params['color_precision'] * scale))
            new_params['layer_difference'] = max(2, int(params['layer_difference'] / scale))

        elif logo_type == 'simple':
            # Simple shapes need clean paths
            new_params['path_precision'] = min(12, int(params['path_precision'] * scale))
            new_params['splice_threshold'] = max(20, int(params['splice_threshold'] / scale))

        else:  # complex
            # Complex needs balance
            new_params['color_precision'] = min(12, int(params['color_precision'] * scale))
            new_params['path_precision'] = min(10, int(params['path_precision'] * scale))
            new_params['corner_threshold'] = max(20, int(params['corner_threshold'] / scale))

        return new_params