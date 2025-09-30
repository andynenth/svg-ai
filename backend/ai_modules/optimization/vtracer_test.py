"""VTracer test harness for safe parameter testing"""

import time
import tempfile
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
from datetime import datetime
import hashlib

import vtracer
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cairosvg

from .parameter_bounds import VTracerParameterBounds
from .validator import ParameterValidator

logger = logging.getLogger(__name__)


class VTracerTestHarness:
    """Safe testing environment for VTracer parameters"""

    def __init__(self, timeout: int = 30):
        """
        Initialize test harness.

        Args:
            timeout: Maximum seconds for each conversion
        """
        self.timeout = timeout
        self.results_cache = {}
        self.validator = ParameterValidator()
        self.executor = ThreadPoolExecutor(max_workers=1)

    def test_parameters(self, image_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test VTracer with given parameters safely.

        Args:
            image_path: Path to input image
            params: VTracer parameters to test

        Returns:
            Dictionary with conversion results and metrics
        """
        result = {
            "image_path": str(image_path),
            "parameters": params.copy(),
            "success": False,
            "error": None,
            "metrics": {},
            "performance": {},
            "timestamp": datetime.now().isoformat()
        }

        # Generate cache key
        cache_key = self._generate_cache_key(image_path, params)
        if cache_key in self.results_cache:
            logger.debug(f"Returning cached result for {cache_key}")
            return self.results_cache[cache_key]

        # Validate parameters
        is_valid, errors = self.validator.validate_parameters(params)
        if not is_valid:
            result["error"] = f"Invalid parameters: {'; '.join(errors)}"
            return result

        try:
            # Run conversion with timeout
            conversion_result = self._run_conversion_with_timeout(image_path, params)

            if conversion_result["success"]:
                result["success"] = True
                result["metrics"] = conversion_result["metrics"]
                result["performance"] = conversion_result["performance"]
                result["svg_path"] = conversion_result["svg_path"]
            else:
                result["error"] = conversion_result["error"]

        except Exception as e:
            result["error"] = f"Test harness error: {str(e)}"
            logger.error(f"Test harness error: {e}", exc_info=True)

        # Cache result
        self.results_cache[cache_key] = result
        return result

    def _run_conversion_with_timeout(
        self, image_path: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run VTracer conversion with timeout protection.

        Args:
            image_path: Path to input image
            params: VTracer parameters

        Returns:
            Conversion result dictionary
        """
        future = self.executor.submit(self._run_conversion, image_path, params)

        try:
            result = future.result(timeout=self.timeout)
            return result
        except TimeoutError:
            future.cancel()
            return {
                "success": False,
                "error": f"Conversion timeout after {self.timeout} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Conversion failed: {str(e)}"
            }

    def _run_conversion(self, image_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute VTracer conversion.

        Args:
            image_path: Path to input image
            params: VTracer parameters

        Returns:
            Conversion result with metrics
        """
        start_time = time.time()
        result = {
            "success": False,
            "error": None,
            "metrics": {},
            "performance": {}
        }

        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
                svg_path = tmp.name

            # Run VTracer conversion
            vtracer.convert_image_to_svg_py(
                str(image_path),
                svg_path,
                colormode=params.get('mode', 'spline'),
                hierarchical='stacked',
                color_precision=params.get('color_precision', 6),
                layer_difference=params.get('layer_difference', 10),
                corner_threshold=params.get('corner_threshold', 60),
                length_threshold=params.get('length_threshold', 5.0),
                max_iterations=params.get('max_iterations', 10),
                splice_threshold=params.get('splice_threshold', 45),
                path_precision=params.get('path_precision', 8)
            )

            conversion_time = time.time() - start_time

            # Calculate quality metrics
            metrics = self._calculate_quality_metrics(image_path, svg_path)

            # Calculate file sizes
            original_size = Path(image_path).stat().st_size if Path(image_path).exists() else 0
            svg_size = Path(svg_path).stat().st_size if Path(svg_path).exists() else 0

            result["success"] = True
            result["svg_path"] = svg_path
            result["metrics"] = metrics
            result["performance"] = {
                "conversion_time": conversion_time,
                "original_size_bytes": original_size,
                "svg_size_bytes": svg_size,
                "file_size_reduction": 1 - (svg_size / original_size) if original_size > 0 else 0,
                "compression_ratio": original_size / svg_size if svg_size > 0 else 0
            }

        except Exception as e:
            result["error"] = f"Conversion error: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Conversion failed: {e}", exc_info=True)

        return result

    def _calculate_quality_metrics(
        self, original_path: str, svg_path: str
    ) -> Dict[str, float]:
        """
        Calculate quality metrics by comparing original and SVG.

        Args:
            original_path: Path to original image
            svg_path: Path to SVG file

        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            "ssim": 0.0,
            "mse": 0.0,
            "psnr": 0.0
        }

        try:
            # Load original image
            original = Image.open(original_path).convert('RGBA')
            original_array = np.array(original)

            # Render SVG to PNG at same size
            svg_png_data = cairosvg.svg2png(
                url=svg_path,
                output_width=original.width,
                output_height=original.height
            )

            # Convert to PIL Image
            import io
            svg_image = Image.open(io.BytesIO(svg_png_data)).convert('RGBA')
            svg_array = np.array(svg_image)

            # Calculate SSIM
            metrics["ssim"] = ssim(
                original_array,
                svg_array,
                multichannel=True,
                channel_axis=2,
                data_range=255
            )

            # Calculate MSE
            mse = np.mean((original_array.astype(float) - svg_array.astype(float)) ** 2)
            metrics["mse"] = mse

            # Calculate PSNR
            if mse > 0:
                metrics["psnr"] = 20 * np.log10(255.0 / np.sqrt(mse))
            else:
                metrics["psnr"] = float('inf')

        except Exception as e:
            logger.warning(f"Failed to calculate quality metrics: {e}")

        return metrics

    def batch_test(
        self,
        image_paths: List[str],
        param_sets: List[Dict[str, Any]],
        parallel: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Test multiple images with multiple parameter sets.

        Args:
            image_paths: List of image paths
            param_sets: List of parameter dictionaries
            parallel: Number of parallel workers

        Returns:
            List of all test results
        """
        results = []
        total_tests = len(image_paths) * len(param_sets)

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = []

            for image_path in image_paths:
                for params in param_sets:
                    future = executor.submit(self.test_parameters, image_path, params)
                    futures.append((image_path, params, future))

            for idx, (image_path, params, future) in enumerate(futures, 1):
                try:
                    result = future.result(timeout=self.timeout * 2)
                    results.append(result)
                    logger.info(f"Completed test {idx}/{total_tests}")
                except Exception as e:
                    logger.error(f"Batch test failed for {image_path}: {e}")
                    results.append({
                        "image_path": str(image_path),
                        "parameters": params,
                        "success": False,
                        "error": str(e)
                    })

        return results

    def _generate_cache_key(self, image_path: str, params: Dict[str, Any]) -> str:
        """
        Generate cache key for result.

        Args:
            image_path: Path to image
            params: Parameter dictionary

        Returns:
            Cache key string
        """
        # Create stable string representation
        param_str = json.dumps(params, sort_keys=True)
        key_str = f"{image_path}:{param_str}"

        # Generate hash
        return hashlib.md5(key_str.encode()).hexdigest()

    def clear_cache(self):
        """Clear results cache"""
        self.results_cache.clear()
        logger.info("Cleared test harness cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.results_cache),
            "cache_entries": list(self.results_cache.keys())
        }

    def save_results(self, results: List[Dict[str, Any]], filepath: str):
        """
        Save test results to JSON file.

        Args:
            results: List of test results
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved {len(results)} results to {filepath}")

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze batch test results.

        Args:
            results: List of test results

        Returns:
            Analysis summary
        """
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        if successful:
            ssim_scores = [r["metrics"]["ssim"] for r in successful]
            conversion_times = [r["performance"]["conversion_time"] for r in successful]
            file_reductions = [r["performance"]["file_size_reduction"] for r in successful]

            analysis = {
                "total_tests": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(results),
                "quality": {
                    "avg_ssim": np.mean(ssim_scores),
                    "min_ssim": np.min(ssim_scores),
                    "max_ssim": np.max(ssim_scores),
                    "std_ssim": np.std(ssim_scores)
                },
                "performance": {
                    "avg_conversion_time": np.mean(conversion_times),
                    "min_conversion_time": np.min(conversion_times),
                    "max_conversion_time": np.max(conversion_times),
                    "avg_file_reduction": np.mean(file_reductions)
                }
            }
        else:
            analysis = {
                "total_tests": len(results),
                "successful": 0,
                "failed": len(failed),
                "success_rate": 0.0
            }

        if failed:
            error_types = {}
            for result in failed:
                error = result.get("error", "Unknown error")
                error_type = error.split(":")[0] if ":" in error else "Unknown"
                error_types[error_type] = error_types.get(error_type, 0) + 1

            analysis["error_summary"] = error_types

        return analysis