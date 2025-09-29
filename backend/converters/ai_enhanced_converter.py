#!/usr/bin/env python3
"""
AI-Enhanced SVG Converter using Feature Extraction and Classification

This converter combines the Day 1-2 AI pipeline (feature extraction + classification)
with VTracer conversion to automatically optimize parameters based on logo type.

Features:
- Automatic logo type detection (simple, text, gradient, complex)
- AI-driven VTracer parameter optimization
- Intelligent fallback to standard conversion
- Comprehensive AI metadata collection
- Full backward compatibility with existing BaseConverter interface
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from backend.converters.base import BaseConverter
from backend.converters.vtracer_converter import VTracerConverter
from backend.utils.validation import validate_file_path

logger = logging.getLogger(__name__)


class AIEnhancedSVGConverter(BaseConverter):
    """AI-enhanced SVG converter using feature extraction and logo classification.

    Automatically analyzes logos using the Day 1-2 AI pipeline to extract features
    and classify logo types, then optimizes VTracer parameters accordingly.

    Features:
        - 6-feature extraction pipeline (edge density, colors, entropy, corners, gradients, complexity)
        - 4-class logo classification (simple, text, gradient, complex)
        - Confidence-based parameter optimization
        - Intelligent fallback mechanisms for reliability
        - Comprehensive AI metadata collection
        - Full BaseConverter interface compliance

    Architecture:
        - Extends BaseConverter for interface compliance
        - Integrates FeaturePipeline for AI analysis
        - Wraps VTracerConverter for actual conversion
        - Provides graceful degradation on AI failures

    Example:
        Basic AI-enhanced conversion:

        converter = AIEnhancedSVGConverter()
        svg = converter.convert("logo.png")

        With detailed AI analysis:

        result = converter.convert_with_ai_analysis("logo.png")
        print(f"Logo type: {result['classification']['logo_type']}")
        print(f"Confidence: {result['classification']['confidence']:.2%}")
    """

    def __init__(self, enable_ai: bool = True, ai_timeout: float = 5.0):
        """Initialize AI-enhanced converter.

        Args:
            enable_ai (bool): Whether to enable AI features. If False, falls back to standard VTracer.
            ai_timeout (float): Maximum time allowed for AI analysis in seconds.

        Raises:
            ImportError: If AI dependencies are not available (falls back to standard mode).
        """
        super().__init__(name="AI-Enhanced SVG Converter")

        self.enable_ai = enable_ai
        self.ai_timeout = ai_timeout
        self.ai_available = False
        self.feature_pipeline = None

        # Initialize standard VTracer converter for fallback
        self.vtracer_converter = VTracerConverter()

        # Track AI usage statistics
        self.ai_stats = {
            'total_conversions': 0,
            'ai_enhanced_conversions': 0,
            'fallback_conversions': 0,
            'ai_failures': 0,
            'average_ai_time': 0.0,
            'classification_history': []
        }

        # Initialize AI pipeline if enabled
        if self.enable_ai:
            self._initialize_ai_pipeline()

        logger.info(f"AIEnhancedSVGConverter initialized (AI {'enabled' if self.ai_available else 'disabled'})")

    def _initialize_ai_pipeline(self) -> None:
        """Initialize the AI feature extraction and classification pipeline.

        Attempts to import and initialize the FeaturePipeline from Day 2.
        Falls back to standard conversion if AI modules are not available.
        """
        try:
            # Import AI modules (may fail if not installed)
            from backend.ai_modules.feature_pipeline import FeaturePipeline

            # Initialize feature pipeline
            self.feature_pipeline = FeaturePipeline()
            self.ai_available = True

            logger.info("AI pipeline initialized successfully")

        except ImportError as e:
            logger.warning(f"AI modules not available: {e}")
            logger.info("Falling back to standard VTracer conversion")
            self.ai_available = False

        except Exception as e:
            logger.error(f"Failed to initialize AI pipeline: {e}")
            logger.info("Falling back to standard VTracer conversion")
            self.ai_available = False

    def get_name(self) -> str:
        """Get the human-readable name of this converter.

        Returns:
            str: Converter name with AI status indicator.
        """
        ai_status = "AI-Enhanced" if self.ai_available else "Standard"
        return f"{ai_status} SVG Converter"

    @validate_file_path(param_name="image_path", allowed_extensions=['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'])
    def convert(self, image_path: str, **kwargs) -> str:
        """Convert image to SVG using AI-enhanced parameter optimization.

        Analyzes the input image using the AI pipeline to extract features and
        classify logo type, then optimizes VTracer parameters accordingly.
        Falls back to standard VTracer conversion on AI failures.

        Args:
            image_path (str): Path to image file to convert.
            **kwargs: VTracer parameters. User-provided parameters override AI recommendations.
                Common parameters:
                - colormode (str): 'color' or 'binary'
                - color_precision (int): Color reduction level (1-10)
                - layer_difference (int): Layer separation threshold (1-32)
                - path_precision (int): Path coordinate precision (0-10)
                - corner_threshold (int): Corner detection threshold (0-180)
                - ai_disable (bool): Disable AI analysis for this conversion

        Returns:
            str: SVG content with optional AI metadata embedded as comments.

        Raises:
            FileNotFoundError: If input image file doesn't exist.
            ValueError: If image format is not supported.
            RuntimeError: If both AI and fallback conversion fail.
        """
        start_time = time.time()
        self.ai_stats['total_conversions'] += 1

        # Check if AI is disabled for this conversion
        if kwargs.pop('ai_disable', False):
            logger.info(f"AI disabled by parameter, using standard conversion")
            result = self._fallback_conversion(image_path, **kwargs)
            self.ai_stats['fallback_conversions'] += 1
            return result

        # Use AI enhancement if available
        if self.ai_available:
            try:
                return self._ai_enhanced_conversion(image_path, **kwargs)
            except Exception as e:
                logger.warning(f"AI conversion failed: {e}")
                self.ai_stats['ai_failures'] += 1
                # Fall back to standard conversion
                logger.info("Falling back to standard VTracer conversion")

        # Fallback to standard conversion
        result = self._fallback_conversion(image_path, **kwargs)
        self.ai_stats['fallback_conversions'] += 1
        return result

    def _ai_enhanced_conversion(self, image_path: str, **kwargs) -> str:
        """Perform AI-enhanced conversion with feature analysis and parameter optimization.

        Args:
            image_path (str): Path to input image.
            **kwargs: User-provided VTracer parameters (override AI recommendations).

        Returns:
            str: SVG content with AI metadata.

        Raises:
            Exception: If AI analysis or conversion fails.
        """
        ai_start_time = time.time()

        logger.info(f"Starting AI analysis for {Path(image_path).name}")

        # Extract features and classify logo type
        pipeline_result = self.feature_pipeline.process_image(image_path)

        ai_analysis_time = time.time() - ai_start_time

        # Update AI timing statistics
        if self.ai_stats['ai_enhanced_conversions'] > 0:
            # Running average
            old_avg = self.ai_stats['average_ai_time']
            n = self.ai_stats['ai_enhanced_conversions']
            self.ai_stats['average_ai_time'] = (old_avg * n + ai_analysis_time) / (n + 1)
        else:
            self.ai_stats['average_ai_time'] = ai_analysis_time

        # Extract classification results
        classification = pipeline_result['classification']
        features = pipeline_result['features']

        logger.info(f"AI analysis complete: {classification['logo_type']} "
                   f"(confidence: {classification['confidence']:.2%}, "
                   f"time: {ai_analysis_time*1000:.1f}ms)")

        # Optimize VTracer parameters based on classification
        optimized_params = self._optimize_parameters(classification, features)

        # User parameters override AI recommendations
        final_params = {**optimized_params, **kwargs}

        logger.info(f"Using optimized parameters: {optimized_params}")
        if kwargs:
            logger.info(f"User overrides: {kwargs}")

        # Convert using optimized parameters
        conversion_start = time.time()
        svg_content = self.vtracer_converter.convert(image_path, **final_params)
        conversion_time = time.time() - conversion_start

        # Add AI metadata to SVG
        svg_content = self._add_ai_metadata(
            svg_content, classification, features, optimized_params,
            ai_analysis_time, conversion_time
        )

        # Track successful AI conversion
        self.ai_stats['ai_enhanced_conversions'] += 1
        self.ai_stats['classification_history'].append({
            'image': Path(image_path).name,
            'logo_type': classification['logo_type'],
            'confidence': classification['confidence'],
            'ai_time': ai_analysis_time,
            'conversion_time': conversion_time
        })

        total_time = time.time() - (ai_start_time - ai_analysis_time)  # Include conversion time
        logger.info(f"AI-enhanced conversion complete (total: {total_time*1000:.1f}ms)")

        return svg_content

    def _optimize_parameters(self, classification: Dict[str, Any], features: Dict[str, float]) -> Dict[str, Any]:
        """Optimize VTracer parameters based on AI classification and features.

        Maps the 4 logo types to optimal VTracer parameter sets, with confidence-based
        adjustments and feature-driven fine-tuning.

        Args:
            classification (Dict[str, Any]): Logo classification results with type and confidence.
            features (Dict[str, float]): Extracted features for fine-tuning.

        Returns:
            Dict[str, Any]: Optimized VTracer parameters.
        """
        logo_type = classification['logo_type']
        confidence = classification['confidence']

        # Get base parameters for logo type
        base_params = self._get_base_parameters_for_type(logo_type)

        # Apply confidence-based adjustments
        adjusted_params = self._apply_confidence_adjustments(base_params, confidence)

        # Apply feature-based fine-tuning
        final_params = self._apply_feature_adjustments(adjusted_params, features)

        return final_params

    def _get_base_parameters_for_type(self, logo_type: str) -> Dict[str, Any]:
        """Get base VTracer parameters optimized for specific logo type.

        Parameter sets are based on Day 1-2 analysis and VTracer documentation:
        - Simple geometric: Clean parameters for sharp edges
        - Text-based: High precision for readable text
        - Gradient: Maximum precision for smooth transitions
        - Complex: Balanced parameters for detail preservation

        Args:
            logo_type (str): One of 'simple', 'text', 'gradient', 'complex'.

        Returns:
            Dict[str, Any]: Base VTracer parameters for the logo type.
        """
        parameter_sets = {
            'simple': {
                'colormode': 'color',
                'color_precision': 3,          # Fewer colors for clean output
                'layer_difference': 32,        # High separation for distinct regions
                'path_precision': 6,           # High precision for sharp edges
                'corner_threshold': 30,        # Lower threshold for sharp corners
                'length_threshold': 3.0,       # Keep small details
                'max_iterations': 10,
                'splice_threshold': 45
            },
            'text': {
                'colormode': 'color',
                'color_precision': 2,          # Minimal colors for text clarity
                'layer_difference': 24,        # Good separation for legibility
                'path_precision': 8,           # Maximum precision for text quality
                'corner_threshold': 20,        # Sharp corners for letter forms
                'length_threshold': 2.0,       # Preserve text details
                'max_iterations': 12,
                'splice_threshold': 40
            },
            'gradient': {
                'colormode': 'color',
                'color_precision': 8,          # High precision for smooth gradients
                'layer_difference': 8,         # Fine layers for smooth transitions
                'path_precision': 6,           # Good precision for curves
                'corner_threshold': 60,        # Higher threshold for smooth curves
                'length_threshold': 4.0,       # Balance detail vs smoothness
                'max_iterations': 15,
                'splice_threshold': 60
            },
            'complex': {
                'colormode': 'color',
                'color_precision': 6,          # Balanced color handling
                'layer_difference': 16,        # Medium separation
                'path_precision': 5,           # Standard precision
                'corner_threshold': 45,        # Balanced corner detection
                'length_threshold': 5.0,       # Standard detail level
                'max_iterations': 20,          # More iterations for complexity
                'splice_threshold': 50
            }
        }

        return parameter_sets.get(logo_type, parameter_sets['complex'])

    def _apply_confidence_adjustments(self, params: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        """Apply confidence-based parameter adjustments.

        Lower confidence indicates uncertain classification, so use more conservative
        parameters that work well across logo types.

        Args:
            params (Dict[str, Any]): Base parameters to adjust.
            confidence (float): Classification confidence (0.0 to 1.0).

        Returns:
            Dict[str, Any]: Confidence-adjusted parameters.
        """
        adjusted_params = params.copy()

        if confidence < 0.6:
            # Low confidence - use more conservative parameters
            logger.info(f"Low confidence ({confidence:.2%}), applying conservative adjustments")

            # More conservative color precision
            adjusted_params['color_precision'] = min(6, max(4, params['color_precision']))

            # Standard layer difference
            adjusted_params['layer_difference'] = 16

            # Standard corner threshold
            adjusted_params['corner_threshold'] = 50

        elif confidence < 0.8:
            # Medium confidence - slight adjustments toward middle ground
            logger.info(f"Medium confidence ({confidence:.2%}), applying moderate adjustments")

            # Moderate adjustments
            if params['color_precision'] <= 3:
                adjusted_params['color_precision'] = 4
            elif params['color_precision'] >= 7:
                adjusted_params['color_precision'] = 6

        # High confidence (>=0.8) - use parameters as-is
        return adjusted_params

    def _apply_feature_adjustments(self, params: Dict[str, Any], features: Dict[str, float]) -> Dict[str, Any]:
        """Apply feature-based fine-tuning to parameters.

        Uses individual feature values to fine-tune parameters beyond just logo type:
        - High edge density: Adjust corner threshold
        - High color count: Adjust color precision
        - High entropy: Adjust complexity handling

        Args:
            params (Dict[str, Any]): Base parameters to fine-tune.
            features (Dict[str, float]): Extracted features (all in [0,1] range).

        Returns:
            Dict[str, Any]: Feature-tuned parameters.
        """
        tuned_params = params.copy()

        # Edge density adjustments
        edge_density = features.get('edge_density', 0.5)
        if edge_density > 0.3:  # High edge density
            # Lower corner threshold for better edge detection
            tuned_params['corner_threshold'] = max(20, tuned_params['corner_threshold'] - 10)
        elif edge_density < 0.1:  # Very low edge density
            # Higher corner threshold for smoother curves
            tuned_params['corner_threshold'] = min(80, tuned_params['corner_threshold'] + 15)

        # Color complexity adjustments
        unique_colors = features.get('unique_colors', 0.5)
        if unique_colors > 0.7:  # Many colors
            # Increase color precision to preserve detail
            tuned_params['color_precision'] = min(8, tuned_params['color_precision'] + 1)
        elif unique_colors < 0.2:  # Few colors
            # Decrease color precision for cleaner output
            tuned_params['color_precision'] = max(2, tuned_params['color_precision'] - 1)

        # Complexity adjustments
        complexity_score = features.get('complexity_score', 0.5)
        if complexity_score > 0.7:  # High complexity
            # More iterations and finer layer difference
            tuned_params['max_iterations'] = min(25, tuned_params['max_iterations'] + 5)
            tuned_params['layer_difference'] = max(8, tuned_params['layer_difference'] - 4)

        # Gradient strength adjustments
        gradient_strength = features.get('gradient_strength', 0.5)
        if gradient_strength > 0.6:  # Strong gradients detected
            # Optimize for smooth gradients
            tuned_params['color_precision'] = min(8, max(6, tuned_params['color_precision']))
            tuned_params['layer_difference'] = min(12, tuned_params['layer_difference'])

        return tuned_params

    def _fallback_conversion(self, image_path: str, **kwargs) -> str:
        """Perform standard VTracer conversion without AI enhancement.

        Args:
            image_path (str): Path to input image.
            **kwargs: VTracer parameters.

        Returns:
            str: SVG content from standard VTracer conversion.
        """
        logger.info(f"Using standard VTracer conversion for {Path(image_path).name}")
        return self.vtracer_converter.convert(image_path, **kwargs)

    def _add_ai_metadata(self, svg_content: str, classification: Dict[str, Any],
                        features: Dict[str, float], params: Dict[str, Any],
                        ai_time: float, conversion_time: float) -> str:
        """Add AI analysis metadata to SVG content as comments.

        Args:
            svg_content (str): Original SVG content.
            classification (Dict[str, Any]): Logo classification results.
            features (Dict[str, float]): Extracted features.
            params (Dict[str, Any]): VTracer parameters used.
            ai_time (float): AI analysis time in seconds.
            conversion_time (float): Conversion time in seconds.

        Returns:
            str: SVG content with AI metadata embedded.
        """
        # Create comprehensive metadata comment
        metadata = f"""
<!-- AI-Enhanced SVG Converter Metadata
Logo Classification:
  Type: {classification['logo_type']}
  Confidence: {classification['confidence']:.3f}

Extracted Features:
  Edge Density: {features.get('edge_density', 0):.3f}
  Unique Colors: {features.get('unique_colors', 0):.3f}
  Entropy: {features.get('entropy', 0):.3f}
  Corner Density: {features.get('corner_density', 0):.3f}
  Gradient Strength: {features.get('gradient_strength', 0):.3f}
  Complexity Score: {features.get('complexity_score', 0):.3f}

Optimized VTracer Parameters:
  Color Mode: {params.get('colormode', 'color')}
  Color Precision: {params.get('color_precision', 6)}
  Layer Difference: {params.get('layer_difference', 16)}
  Path Precision: {params.get('path_precision', 5)}
  Corner Threshold: {params.get('corner_threshold', 60)}

Performance:
  AI Analysis Time: {ai_time*1000:.1f}ms
  Conversion Time: {conversion_time*1000:.1f}ms
  Total Enhancement Time: {(ai_time + conversion_time)*1000:.1f}ms
-->"""

        # Insert metadata after the first line (XML declaration or SVG tag)
        lines = svg_content.split('\n')
        if len(lines) > 1:
            return lines[0] + '\n' + metadata + '\n' + '\n'.join(lines[1:])
        else:
            return metadata + '\n' + svg_content

    def convert_with_ai_analysis(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Convert image and return detailed AI analysis results.

        Provides comprehensive conversion results including SVG content, AI analysis,
        classification details, and performance metrics for evaluation and debugging.

        Args:
            image_path (str): Path to image file to convert.
            **kwargs: VTracer parameters (override AI recommendations).

        Returns:
            Dict[str, Any]: Comprehensive conversion results:
                - svg (str): SVG content with AI metadata.
                - features (Dict[str, float]): Extracted features.
                - classification (Dict[str, Any]): Logo classification results.
                - parameters_used (Dict[str, Any]): Final VTracer parameters.
                - ai_analysis_time (float): AI analysis time in seconds.
                - conversion_time (float): SVG conversion time in seconds.
                - total_time (float): Total processing time in seconds.
                - ai_enhanced (bool): Whether AI enhancement was used.
                - success (bool): Whether conversion succeeded.

        Example:
            Detailed AI-enhanced conversion:

            result = converter.convert_with_ai_analysis("logo.png")
            print(f"Logo type: {result['classification']['logo_type']}")
            print(f"AI confidence: {result['classification']['confidence']:.2%}")
            print(f"Processing time: {result['total_time']*1000:.1f}ms")
        """
        start_time = time.time()

        try:
            # Check if AI enhancement is available and enabled
            if not self.ai_available or kwargs.get('ai_disable', False):
                # Standard conversion without AI
                svg_content = self._fallback_conversion(image_path, **kwargs)
                total_time = time.time() - start_time

                return {
                    'svg': svg_content,
                    'features': {},
                    'classification': {'logo_type': 'unknown', 'confidence': 0.0},
                    'parameters_used': kwargs,
                    'ai_analysis_time': 0.0,
                    'conversion_time': total_time,
                    'total_time': total_time,
                    'ai_enhanced': False,
                    'success': True
                }

            # AI-enhanced conversion with detailed analysis
            ai_start_time = time.time()

            # Extract features and classify
            pipeline_result = self.feature_pipeline.process_image(image_path)
            classification = pipeline_result['classification']
            features = pipeline_result['features']

            ai_analysis_time = time.time() - ai_start_time

            # Optimize parameters
            optimized_params = self._optimize_parameters(classification, features)
            final_params = {**optimized_params, **kwargs}

            # Convert
            conversion_start = time.time()
            svg_content = self.vtracer_converter.convert(image_path, **final_params)
            conversion_time = time.time() - conversion_start

            # Add metadata
            svg_content = self._add_ai_metadata(
                svg_content, classification, features, final_params,
                ai_analysis_time, conversion_time
            )

            total_time = time.time() - start_time

            return {
                'svg': svg_content,
                'features': features,
                'classification': classification,
                'parameters_used': final_params,
                'ai_analysis_time': ai_analysis_time,
                'conversion_time': conversion_time,
                'total_time': total_time,
                'ai_enhanced': True,
                'success': True
            }

        except Exception as e:
            # Fallback on any error
            logger.error(f"AI analysis failed: {e}")
            svg_content = self._fallback_conversion(image_path, **kwargs)
            total_time = time.time() - start_time

            return {
                'svg': svg_content,
                'features': {},
                'classification': {'logo_type': 'error', 'confidence': 0.0, 'error': str(e)},
                'parameters_used': kwargs,
                'ai_analysis_time': 0.0,
                'conversion_time': total_time,
                'total_time': total_time,
                'ai_enhanced': False,
                'success': True,
                'error': str(e)
            }

    def get_ai_stats(self) -> Dict[str, Any]:
        """Get comprehensive AI usage statistics and performance metrics.

        Returns:
            Dict[str, Any]: AI usage statistics:
                - total_conversions (int): Total conversions processed.
                - ai_enhanced_conversions (int): Conversions using AI enhancement.
                - fallback_conversions (int): Conversions using standard VTracer.
                - ai_failures (int): Number of AI analysis failures.
                - ai_success_rate (float): Percentage of successful AI enhancements.
                - average_ai_time (float): Average AI analysis time in seconds.
                - classification_breakdown (Dict): Count of each logo type classified.
        """
        stats = self.ai_stats.copy()

        # Calculate derived statistics
        if stats['total_conversions'] > 0:
            stats['ai_success_rate'] = (stats['ai_enhanced_conversions'] / stats['total_conversions']) * 100
            stats['fallback_rate'] = (stats['fallback_conversions'] / stats['total_conversions']) * 100
        else:
            stats['ai_success_rate'] = 0.0
            stats['fallback_rate'] = 0.0

        # Classification breakdown
        classification_counts = {}
        for entry in stats['classification_history']:
            logo_type = entry['logo_type']
            classification_counts[logo_type] = classification_counts.get(logo_type, 0) + 1

        stats['classification_breakdown'] = classification_counts

        return stats


def test_ai_enhanced_converter():
    """Test the AI-enhanced converter on sample images."""
    import os
    from pathlib import Path

    print("\n" + "="*70)
    print("Testing AI-Enhanced SVG Converter")
    print("="*70)

    converter = AIEnhancedSVGConverter()

    # Test directory
    test_dir = Path("data/logos")
    if not test_dir.exists():
        print("Test directory not found. Please run from project root.")
        return

    # Test different logo types
    test_images = []
    for category in ['simple_geometric', 'text_based', 'gradients', 'complex']:
        category_dir = test_dir / category
        if category_dir.exists():
            images = list(category_dir.glob("*.png"))[:2]  # First 2 from each category
            test_images.extend([(str(img), category) for img in images])

    if not test_images:
        print("No test images found.")
        return

    results = []

    for image_path, category in test_images[:6]:  # Test first 6 images
        print(f"\n[{Path(image_path).name}] ({category})")
        print("-" * 50)

        try:
            result = converter.convert_with_ai_analysis(image_path)

            if result['ai_enhanced']:
                print(f"✅ AI-Enhanced Conversion")
                print(f"   Logo Type: {result['classification']['logo_type']}")
                print(f"   Confidence: {result['classification']['confidence']:.2%}")
                print(f"   Key Features:")
                for feature, value in list(result['features'].items())[:3]:
                    print(f"     {feature}: {value:.3f}")
                print(f"   Parameters Used:")
                params = result['parameters_used']
                print(f"     color_precision: {params.get('color_precision', 'default')}")
                print(f"     layer_difference: {params.get('layer_difference', 'default')}")
                print(f"   AI Analysis: {result['ai_analysis_time']*1000:.1f}ms")
                print(f"   Conversion: {result['conversion_time']*1000:.1f}ms")
            else:
                print(f"⚠️  Standard Conversion (AI not available)")

            print(f"   SVG Size: {len(result['svg'])} bytes")
            print(f"   Total Time: {result['total_time']*1000:.1f}ms")

            results.append(result)

        except Exception as e:
            print(f"❌ Error: {e}")

    # Print AI statistics
    print("\n" + "="*70)
    print("AI Enhancement Statistics")
    print("="*70)

    stats = converter.get_ai_stats()
    print(f"Total Conversions: {stats['total_conversions']}")
    print(f"AI Enhanced: {stats['ai_enhanced_conversions']} ({stats['ai_success_rate']:.1f}%)")
    print(f"Standard Fallback: {stats['fallback_conversions']} ({stats['fallback_rate']:.1f}%)")
    print(f"AI Failures: {stats['ai_failures']}")
    print(f"Average AI Time: {stats['average_ai_time']*1000:.1f}ms")

    if stats['classification_breakdown']:
        print("\nLogo Type Classification:")
        for logo_type, count in stats['classification_breakdown'].items():
            print(f"  {logo_type}: {count}")


if __name__ == "__main__":
    test_ai_enhanced_converter()