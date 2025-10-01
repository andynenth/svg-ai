"""Quality measurement system for parameter optimization"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Tuple, Optional, List, Any
import tempfile
import os
import time
import logging
from pathlib import Path
import cairosvg
from PIL import Image
import io
import json
from scipy import stats

from .vtracer_test import VTracerTestHarness
from .parameter_bounds import VTracerParameterBounds

logger = logging.getLogger(__name__)


class OptimizationQualityMetrics:
    """Measure optimization quality improvements"""

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.harness = VTracerTestHarness(timeout=30)
        self.bounds = VTracerParameterBounds()
        self.logger = logging.getLogger(__name__)

    def measure_improvement(self,
                           image_path: str,
                           default_params: Dict,
                           optimized_params: Dict,
                           runs: int = 3) -> Dict[str, Any]:
        """
        Compare quality between parameter sets.

        Args:
            image_path: Path to source image
            default_params: Default VTracer parameters
            optimized_params: Optimized VTracer parameters
            runs: Number of runs for averaging (default 3)

        Returns:
            Dictionary with improvement metrics
        """
        result = {
            "image_path": str(image_path),
            "default_metrics": {},
            "optimized_metrics": {},
            "improvements": {},
            "statistical_significance": {},
            "visual_quality": {}
        }

        try:
            # Test both parameter sets
            default_results = []
            optimized_results = []

            for _ in range(runs):
                # Clear cache to get fresh measurements
                self.harness.clear_cache()

                # Test default parameters
                default_result = self.harness.test_parameters(image_path, default_params)
                default_results.append(default_result)

                # Test optimized parameters
                optimized_result = self.harness.test_parameters(image_path, optimized_params)
                optimized_results.append(optimized_result)

            # Calculate average metrics
            result["default_metrics"] = self._average_metrics(default_results)
            result["optimized_metrics"] = self._average_metrics(optimized_results)

            # Calculate improvements
            result["improvements"] = self._calculate_improvements(
                result["default_metrics"],
                result["optimized_metrics"]
            )

            # Statistical significance testing
            result["statistical_significance"] = self._test_significance(
                default_results,
                optimized_results
            )

            # Visual quality assessment
            if default_results[0]["success"] and optimized_results[0]["success"]:
                result["visual_quality"] = self._assess_visual_quality(
                    image_path,
                    default_results[0].get("svg_path"),
                    optimized_results[0].get("svg_path")
                )

        except Exception as e:
            logger.error(f"Failed to measure improvement: {e}", exc_info=True)
            result["error"] = str(e)

        return result

    def _average_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Calculate average metrics from multiple runs.

        Args:
            results: List of test results

        Returns:
            Averaged metrics
        """
        successful_results = [r for r in results if r.get("success", False)]

        if not successful_results:
            return {
                "success_rate": 0.0,
                "error": "No successful conversions"
            }

        avg_metrics = {
            "success_rate": len(successful_results) / len(results),
            "ssim": np.mean([r["metrics"]["ssim"] for r in successful_results]),
            "mse": np.mean([r["metrics"]["mse"] for r in successful_results]),
            "psnr": np.mean([r["metrics"]["psnr"] for r in successful_results]),
            "conversion_time": np.mean([r["performance"]["conversion_time"]
                                       for r in successful_results]),
            "file_size_reduction": np.mean([r["performance"]["file_size_reduction"]
                                           for r in successful_results]),
            "svg_size_bytes": np.mean([r["performance"]["svg_size_bytes"]
                                      for r in successful_results])
        }

        # Add standard deviations
        if len(successful_results) > 1:
            avg_metrics["ssim_std"] = np.std([r["metrics"]["ssim"]
                                             for r in successful_results])
            avg_metrics["time_std"] = np.std([r["performance"]["conversion_time"]
                                             for r in successful_results])

        return avg_metrics

    def _calculate_improvements(self,
                               default_metrics: Dict,
                               optimized_metrics: Dict) -> Dict[str, float]:
        """
        Calculate improvement percentages.

        Args:
            default_metrics: Default parameter metrics
            optimized_metrics: Optimized parameter metrics

        Returns:
            Dictionary of improvement percentages
        """
        improvements = {}

        # SSIM improvement
        if "ssim" in default_metrics and "ssim" in optimized_metrics:
            old_ssim = default_metrics["ssim"]
            new_ssim = optimized_metrics["ssim"]
            if old_ssim > 0:
                improvements["ssim_improvement"] = ((new_ssim - old_ssim) / old_ssim) * 100
            else:
                improvements["ssim_improvement"] = 100.0 if new_ssim > 0 else 0.0
            improvements["ssim_absolute"] = new_ssim - old_ssim

        # File size improvement (smaller is better)
        if "svg_size_bytes" in default_metrics and "svg_size_bytes" in optimized_metrics:
            old_size = default_metrics["svg_size_bytes"]
            new_size = optimized_metrics["svg_size_bytes"]
            if old_size > 0:
                improvements["file_size_improvement"] = ((old_size - new_size) / old_size) * 100
            improvements["file_size_difference_bytes"] = old_size - new_size

        # Speed improvement (faster is better)
        if "conversion_time" in default_metrics and "conversion_time" in optimized_metrics:
            old_time = default_metrics["conversion_time"]
            new_time = optimized_metrics["conversion_time"]
            if old_time > 0:
                improvements["speed_improvement"] = ((old_time - new_time) / old_time) * 100
            improvements["time_difference_seconds"] = old_time - new_time

        # MSE improvement (lower is better)
        if "mse" in default_metrics and "mse" in optimized_metrics:
            old_mse = default_metrics["mse"]
            new_mse = optimized_metrics["mse"]
            if old_mse > 0:
                improvements["mse_improvement"] = ((old_mse - new_mse) / old_mse) * 100

        # PSNR improvement (higher is better)
        if "psnr" in default_metrics and "psnr" in optimized_metrics:
            old_psnr = default_metrics["psnr"]
            new_psnr = optimized_metrics["psnr"]
            if old_psnr > 0:
                improvements["psnr_improvement"] = ((new_psnr - old_psnr) / old_psnr) * 100

        return improvements

    def _test_significance(self,
                          default_results: List[Dict],
                          optimized_results: List[Dict]) -> Dict[str, Any]:
        """
        Test statistical significance of improvements.

        Args:
            default_results: List of default parameter test results
            optimized_results: List of optimized parameter test results

        Returns:
            Statistical test results
        """
        significance = {}

        # Extract successful results
        default_success = [r for r in default_results if r.get("success", False)]
        optimized_success = [r for r in optimized_results if r.get("success", False)]

        if len(default_success) > 1 and len(optimized_success) > 1:
            # SSIM significance
            default_ssim = [r["metrics"]["ssim"] for r in default_success]
            optimized_ssim = [r["metrics"]["ssim"] for r in optimized_success]

            if len(default_ssim) >= 2 and len(optimized_ssim) >= 2:
                t_stat, p_value = stats.ttest_ind(optimized_ssim, default_ssim)
                significance["ssim_t_statistic"] = t_stat
                significance["ssim_p_value"] = p_value
                significance["ssim_significant"] = p_value < 0.05

                # Calculate confidence interval for SSIM improvement
                ssim_diff = np.mean(optimized_ssim) - np.mean(default_ssim)
                ssim_se = np.sqrt(np.var(optimized_ssim)/len(optimized_ssim) +
                                 np.var(default_ssim)/len(default_ssim))
                significance["ssim_improvement_ci"] = [
                    ssim_diff - 1.96 * ssim_se,
                    ssim_diff + 1.96 * ssim_se
                ]

            # Conversion time significance
            default_times = [r["performance"]["conversion_time"] for r in default_success]
            optimized_times = [r["performance"]["conversion_time"] for r in optimized_success]

            if len(default_times) >= 2 and len(optimized_times) >= 2:
                t_stat, p_value = stats.ttest_ind(optimized_times, default_times)
                significance["time_t_statistic"] = t_stat
                significance["time_p_value"] = p_value
                significance["time_significant"] = p_value < 0.05

        return significance

    def _assess_visual_quality(self,
                              original_path: str,
                              default_svg_path: Optional[str],
                              optimized_svg_path: Optional[str]) -> Dict[str, float]:
        """
        Assess visual quality metrics.

        Args:
            original_path: Path to original image
            default_svg_path: Path to SVG with default parameters
            optimized_svg_path: Path to SVG with optimized parameters

        Returns:
            Visual quality assessment metrics
        """
        quality_metrics = {}

        try:
            # Load original image
            original = cv2.imread(original_path)
            if original is None:
                return {"error": "Failed to load original image"}

            # Edge preservation metric
            if default_svg_path and optimized_svg_path:
                edge_preservation_default = self._measure_edge_preservation(
                    original_path, default_svg_path
                )
                edge_preservation_optimized = self._measure_edge_preservation(
                    original_path, optimized_svg_path
                )

                quality_metrics["edge_preservation_default"] = edge_preservation_default
                quality_metrics["edge_preservation_optimized"] = edge_preservation_optimized
                quality_metrics["edge_improvement"] = (
                    edge_preservation_optimized - edge_preservation_default
                )

                # Color accuracy
                color_accuracy_default = self._measure_color_accuracy(
                    original_path, default_svg_path
                )
                color_accuracy_optimized = self._measure_color_accuracy(
                    original_path, optimized_svg_path
                )

                quality_metrics["color_accuracy_default"] = color_accuracy_default
                quality_metrics["color_accuracy_optimized"] = color_accuracy_optimized
                quality_metrics["color_improvement"] = (
                    color_accuracy_optimized - color_accuracy_default
                )

                # Shape fidelity
                shape_fidelity_default = self._measure_shape_fidelity(
                    original_path, default_svg_path
                )
                shape_fidelity_optimized = self._measure_shape_fidelity(
                    original_path, optimized_svg_path
                )

                quality_metrics["shape_fidelity_default"] = shape_fidelity_default
                quality_metrics["shape_fidelity_optimized"] = shape_fidelity_optimized
                quality_metrics["shape_improvement"] = (
                    shape_fidelity_optimized - shape_fidelity_default
                )

        except Exception as e:
            logger.warning(f"Visual quality assessment failed: {e}")
            quality_metrics["error"] = str(e)

        return quality_metrics

    def _measure_edge_preservation(self,
                                  original_path: str,
                                  svg_path: str) -> float:
        """
        Measure how well edges are preserved in conversion.

        Args:
            original_path: Path to original image
            svg_path: Path to SVG file

        Returns:
            Edge preservation score (0-1)
        """
        try:
            # Load images
            original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
            if original is None:
                return 0.0

            # Render SVG to same size
            svg_png_data = cairosvg.svg2png(
                url=svg_path,
                output_width=original.shape[1],
                output_height=original.shape[0]
            )
            svg_image = Image.open(io.BytesIO(svg_png_data)).convert('L')
            svg_array = np.array(svg_image)

            # Apply edge detection
            original_edges = cv2.Canny(original, 50, 150)
            svg_edges = cv2.Canny(svg_array, 50, 150)

            # Calculate edge overlap
            overlap = np.logical_and(original_edges > 0, svg_edges > 0).sum()
            original_edge_pixels = (original_edges > 0).sum()

            if original_edge_pixels > 0:
                return overlap / original_edge_pixels
            return 1.0

        except Exception as e:
            logger.warning(f"Edge preservation measurement failed: {e}")
            return 0.0

    def _measure_color_accuracy(self,
                               original_path: str,
                               svg_path: str) -> float:
        """
        Measure color accuracy between original and SVG.

        Args:
            original_path: Path to original image
            svg_path: Path to SVG file

        Returns:
            Color accuracy score (0-1)
        """
        try:
            # Load original
            original = Image.open(original_path).convert('RGB')
            original_array = np.array(original)

            # Render SVG
            svg_png_data = cairosvg.svg2png(
                url=svg_path,
                output_width=original.width,
                output_height=original.height
            )
            svg_image = Image.open(io.BytesIO(svg_png_data)).convert('RGB')
            svg_array = np.array(svg_image)

            # Calculate color difference in LAB space
            original_lab = cv2.cvtColor(original_array, cv2.COLOR_RGB2LAB)
            svg_lab = cv2.cvtColor(svg_array, cv2.COLOR_RGB2LAB)

            # Delta E color difference
            delta_e = np.sqrt(np.sum((original_lab - svg_lab) ** 2, axis=2))

            # Normalize to 0-1 (Delta E < 2 is imperceptible)
            accuracy = 1.0 - np.clip(np.mean(delta_e) / 100.0, 0, 1)
            return accuracy

        except Exception as e:
            logger.warning(f"Color accuracy measurement failed: {e}")
            return 0.0

    def _measure_shape_fidelity(self,
                               original_path: str,
                               svg_path: str) -> float:
        """
        Measure shape fidelity using contour matching.

        Args:
            original_path: Path to original image
            svg_path: Path to SVG file

        Returns:
            Shape fidelity score (0-1)
        """
        try:
            # Load images
            original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
            if original is None:
                return 0.0

            # Render SVG
            svg_png_data = cairosvg.svg2png(
                url=svg_path,
                output_width=original.shape[1],
                output_height=original.shape[0]
            )
            svg_image = Image.open(io.BytesIO(svg_png_data)).convert('L')
            svg_array = np.array(svg_image)

            # Find contours
            _, original_binary = cv2.threshold(original, 127, 255, cv2.THRESH_BINARY)
            _, svg_binary = cv2.threshold(svg_array, 127, 255, cv2.THRESH_BINARY)

            original_contours, _ = cv2.findContours(
                original_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            svg_contours, _ = cv2.findContours(
                svg_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not original_contours or not svg_contours:
                return 0.0

            # Use Hu moments for shape comparison
            original_moments = cv2.moments(original_binary)
            svg_moments = cv2.moments(svg_binary)

            original_hu = cv2.HuMoments(original_moments).flatten()
            svg_hu = cv2.HuMoments(svg_moments).flatten()

            # Log transform for scale invariance
            original_hu_log = -np.sign(original_hu) * np.log10(np.abs(original_hu) + 1e-10)
            svg_hu_log = -np.sign(svg_hu) * np.log10(np.abs(svg_hu) + 1e-10)

            # Calculate similarity (closer to 0 is better)
            distance = np.linalg.norm(original_hu_log - svg_hu_log)

            # Convert to 0-1 score (lower distance = higher score)
            fidelity = 1.0 / (1.0 + distance)
            return fidelity

        except Exception as e:
            logger.warning(f"Shape fidelity measurement failed: {e}")
            return 0.0

    def generate_quality_report(self,
                               improvement_data: Dict,
                               output_format: str = "json") -> str:
        """
        Generate detailed quality report.

        Args:
            improvement_data: Data from measure_improvement()
            output_format: Format for report ("json" or "html")

        Returns:
            Formatted report string
        """
        if output_format == "json":
            return json.dumps(improvement_data, indent=2, default=str)
        elif output_format == "html":
            return self._generate_html_report(improvement_data)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def _generate_html_report(self, data: Dict) -> str:
        """
        Generate HTML report for quality metrics.

        Args:
            data: Improvement data

        Returns:
            HTML string
        """
        html = """
        <html>
        <head>
            <title>Optimization Quality Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric { margin: 10px 0; padding: 10px; background: #f0f0f0; }
                .improvement { color: green; font-weight: bold; }
                .regression { color: red; font-weight: bold; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
            </style>
        </head>
        <body>
            <h1>Parameter Optimization Quality Report</h1>
        """

        # Image info
        html += f"<h2>Image: {data.get('image_path', 'Unknown')}</h2>"

        # Quality improvements
        if "improvements" in data:
            html += "<h2>Quality Improvements</h2><table>"
            html += "<tr><th>Metric</th><th>Improvement</th></tr>"

            for key, value in data["improvements"].items():
                if "improvement" in key:
                    css_class = "improvement" if value > 0 else "regression"
                    html += f"<tr><td>{key}</td><td class='{css_class}'>{value:.2f}%</td></tr>"
            html += "</table>"

        # Metrics comparison
        if "default_metrics" in data and "optimized_metrics" in data:
            html += "<h2>Metrics Comparison</h2><table>"
            html += "<tr><th>Metric</th><th>Default</th><th>Optimized</th></tr>"

            for key in data["default_metrics"]:
                if not key.endswith("_std"):
                    default_val = data["default_metrics"].get(key, "N/A")
                    optimized_val = data["optimized_metrics"].get(key, "N/A")

                    if isinstance(default_val, float):
                        html += f"<tr><td>{key}</td><td>{default_val:.4f}</td>"
                        html += f"<td>{optimized_val:.4f}</td></tr>"
            html += "</table>"

        # Statistical significance
        if "statistical_significance" in data:
            html += "<h2>Statistical Significance</h2><ul>"
            sig = data["statistical_significance"]
            if "ssim_significant" in sig:
                html += f"<li>SSIM improvement significant: {sig['ssim_significant']}"
                if "ssim_p_value" in sig:
                    html += f" (p={sig['ssim_p_value']:.4f})"
                html += "</li>"
            html += "</ul>"

        html += "</body></html>"
        return html

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.debug(f"Cleaned up temp directory: {self.temp_dir}")