#!/usr/bin/env python3
"""
AI-Enhanced Iterative PNG to SVG Optimizer.

This script uses CLIP for intelligent logo type detection and iteratively optimizes
VTracer parameters to achieve target quality metrics.
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import converters and utilities
from converters.vtracer_converter import VTracerConverter
from utils.quality_metrics import QualityMetrics
from utils.svg_optimizer import SVGOptimizer
from utils.ai_detector import create_detector, AILogoDetector, FallbackDetector
from utils.image_loader import QualityMetricsWrapper


class AIEnhancedOptimizer:
    """
    AI-Enhanced iterative optimizer for PNG to SVG conversion.
    Uses CLIP for intelligent logo detection and optimizes VTracer parameters.
    """

    # Parameter presets based on AI-detected logo type
    LOGO_PRESETS = {
        'text': {
            'color_precision': 6,
            'corner_threshold': 20,
            'path_precision': 10,
            'layer_difference': 10,
            'mode': 'spline',
            'filter_speckle': 4,
            'max_iterations': 15
        },
        'simple': {
            'color_precision': 3,
            'corner_threshold': 30,
            'path_precision': 6,
            'layer_difference': 12,
            'mode': 'spline',
            'filter_speckle': 8,
            'max_iterations': 10
        },
        'gradient': {
            'color_precision': 8,
            'corner_threshold': 60,
            'path_precision': 4,
            'layer_difference': 8,
            'mode': 'spline',
            'filter_speckle': 2,
            'max_iterations': 20
        },
        'complex': {
            'color_precision': 10,
            'corner_threshold': 90,
            'path_precision': 3,
            'layer_difference': 5,
            'mode': 'spline',
            'filter_speckle': 1,
            'max_iterations': 25
        }
    }

    def __init__(self, input_path: str, output_dir: str = None,
                 target_ssim: float = 0.98, max_iterations: int = 30,
                 use_ai: bool = True):
        """
        Initialize the AI-enhanced optimizer.

        Args:
            input_path: Path to input PNG file
            output_dir: Output directory for SVG files
            target_ssim: Target SSIM quality (0-1)
            max_iterations: Maximum optimization iterations
            use_ai: Whether to use AI detection (True) or fallback (False)
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir) if output_dir else self.input_path.parent
        self.target_ssim = target_ssim
        self.max_iterations = max_iterations

        # Initialize components
        self.converter = VTracerConverter()
        self.metrics = QualityMetricsWrapper()  # Use wrapper that handles file paths
        self.svg_optimizer = SVGOptimizer()

        # Initialize AI detector
        logger.info("Initializing AI detector...")
        self.detector = create_detector(use_fallback=not use_ai)
        if isinstance(self.detector, AILogoDetector):
            logger.info("✅ Using CLIP AI for logo detection")
        else:
            logger.info("⚠️ Using fallback detection (install AI dependencies for better results)")

        # Tracking
        self.iteration_history = []
        self.best_params = None
        self.best_ssim = 0
        self.best_output = None

    def detect_logo_type(self) -> Tuple[str, float, Dict[str, float]]:
        """
        Detect logo type using AI.

        Returns:
            Tuple of (logo_type, confidence, all_scores)
        """
        logger.info(f"Analyzing {self.input_path.name}...")
        logo_type, confidence, scores = self.detector.detect_logo_type(str(self.input_path))

        logger.info(f"  Detected type: {logo_type} (confidence: {confidence:.2%})")

        if scores and logger.isEnabledFor(logging.DEBUG):
            logger.debug("  All scores:")
            for type_name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                logger.debug(f"    - {type_name}: {score:.2%}")

        return logo_type, confidence, scores

    def get_initial_params(self, logo_type: str) -> Dict:
        """
        Get initial parameters based on detected logo type.

        Args:
            logo_type: Detected logo type

        Returns:
            Initial VTracer parameters
        """
        params = self.LOGO_PRESETS.get(logo_type, self.LOGO_PRESETS['complex']).copy()
        logger.info(f"Using {logo_type} preset parameters")
        return params

    def optimize_single(self, params: Dict) -> Tuple[float, str]:
        """
        Run a single optimization iteration.

        Args:
            params: VTracer parameters

        Returns:
            Tuple of (ssim_score, output_path)
        """
        # Generate output path
        output_path = self.output_dir / f"{self.input_path.stem}_iter.svg"

        try:
            # Convert (VTracer converter needs params passed differently)
            result = self.converter.convert(str(self.input_path), **params)

            # Save the result (skip SVG optimization for now)
            with open(output_path, 'w') as f:
                f.write(result)

            # Measure quality using the wrapper that handles file paths
            ssim = self.metrics.calculate_ssim_from_paths(str(self.input_path), str(output_path))

            return ssim, str(output_path)

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return 0.0, None

    def adjust_params(self, params: Dict, ssim: float, iteration: int) -> Dict:
        """
        Adjust parameters based on current quality.

        Args:
            params: Current parameters
            ssim: Current SSIM score
            iteration: Current iteration number

        Returns:
            Adjusted parameters
        """
        new_params = params.copy()

        # Calculate quality gap
        gap = self.target_ssim - ssim

        if gap > 0.1:  # Need significant improvement
            # Increase quality parameters
            new_params['color_precision'] = min(12, params['color_precision'] + 2)
            new_params['path_precision'] = max(2, params['path_precision'] - 1)
            new_params['corner_threshold'] = min(120, params['corner_threshold'] + 10)

        elif gap > 0.05:  # Need moderate improvement
            new_params['color_precision'] = min(12, params['color_precision'] + 1)
            new_params['layer_difference'] = max(3, params['layer_difference'] - 1)

        elif gap > 0.02:  # Need small improvement
            new_params['filter_speckle'] = max(0, params['filter_speckle'] - 1)
            new_params['path_precision'] = max(2, params['path_precision'] - 1)

        elif gap < -0.02:  # Over-optimized, can reduce quality
            new_params['color_precision'] = max(2, params['color_precision'] - 1)
            new_params['path_precision'] = min(10, params['path_precision'] + 1)

        return new_params

    def optimize(self) -> Dict:
        """
        Run the full AI-enhanced optimization process.

        Returns:
            Optimization results dictionary
        """
        start_time = time.time()

        # Step 1: AI Detection
        logo_type, confidence, scores = self.detect_logo_type()

        # Step 2: Get initial parameters
        params = self.get_initial_params(logo_type)

        # Step 3: Iterative optimization
        logger.info(f"Starting optimization (target SSIM: {self.target_ssim})")

        for iteration in range(self.max_iterations):
            # Run conversion
            ssim, output_path = self.optimize_single(params)

            # Track history
            self.iteration_history.append({
                'iteration': iteration + 1,
                'ssim': ssim,
                'params': params.copy()
            })

            # Update best if improved
            if ssim > self.best_ssim:
                self.best_ssim = ssim
                self.best_params = params.copy()
                self.best_output = output_path

            # Log progress
            logger.info(f"  Iteration {iteration + 1}: SSIM = {ssim:.4f} "
                       f"(best: {self.best_ssim:.4f})")

            # Check if target reached
            if ssim >= self.target_ssim:
                logger.info(f"✅ Target quality achieved!")
                break

            # Adjust parameters for next iteration
            if iteration < self.max_iterations - 1:
                params = self.adjust_params(params, ssim, iteration)

        # Step 4: Save best result
        if self.best_output:
            final_output = self.output_dir / f"{self.input_path.stem}.optimized.svg"
            shutil.copy(self.best_output, final_output)
            logger.info(f"✅ Best result saved to: {final_output}")

        # Calculate metrics
        elapsed_time = time.time() - start_time

        return {
            'success': self.best_ssim >= self.target_ssim * 0.95,
            'detected_type': logo_type,
            'detection_confidence': confidence,
            'detection_scores': scores,
            'ssim': self.best_ssim,
            'target_ssim': self.target_ssim,
            'iterations': len(self.iteration_history),
            'best_params': self.best_params,
            'output_file': str(final_output) if self.best_output else None,
            'elapsed_time': elapsed_time,
            'history': self.iteration_history
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="AI-Enhanced PNG to SVG optimizer using CLIP detection"
    )
    parser.add_argument("input", help="Input PNG file path")
    parser.add_argument("-o", "--output-dir", help="Output directory")
    parser.add_argument("-t", "--target-ssim", type=float, default=0.98,
                       help="Target SSIM quality (default: 0.98)")
    parser.add_argument("-m", "--max-iterations", type=int, default=30,
                       help="Maximum optimization iterations (default: 30)")
    parser.add_argument("--no-ai", action="store_true",
                       help="Use fallback detection instead of AI")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--save-history", action="store_true",
                       help="Save optimization history to JSON")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check input file
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Create optimizer
    optimizer = AIEnhancedOptimizer(
        input_path=args.input,
        output_dir=args.output_dir,
        target_ssim=args.target_ssim,
        max_iterations=args.max_iterations,
        use_ai=not args.no_ai
    )

    # Run optimization
    logger.info("=" * 60)
    logger.info("AI-ENHANCED PNG TO SVG OPTIMIZATION")
    logger.info("=" * 60)

    result = optimizer.optimize()

    # Save history if requested
    if args.save_history:
        history_path = Path(args.input).parent / f"{Path(args.input).stem}_history.json"
        with open(history_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"History saved to: {history_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Detected Type: {result['detected_type']} "
               f"(confidence: {result['detection_confidence']:.2%})")
    logger.info(f"Final SSIM: {result['ssim']:.4f}")
    logger.info(f"Target SSIM: {result['target_ssim']:.4f}")
    logger.info(f"Success: {'✅ Yes' if result['success'] else '❌ No'}")
    logger.info(f"Iterations: {result['iterations']}")
    logger.info(f"Time: {result['elapsed_time']:.2f} seconds")

    if result['output_file']:
        logger.info(f"Output: {result['output_file']}")

    return 0 if result['success'] else 1


if __name__ == "__main__":
    sys.exit(main())