#!/usr/bin/env python3
"""
AI-Enhanced SVG Conversion Examples

Practical examples demonstrating the AI-enhanced SVG conversion system
with various use cases and workflows.
"""

import time
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.converters.ai_enhanced_converter import AIEnhancedSVGConverter
from backend.ai_modules.parameter_optimizer import VTracerParameterOptimizer
from backend.ai_modules.quality_validator import QualityValidator


def example_1_basic_conversion():
    """Example 1: Basic AI-enhanced conversion"""
    print("\n" + "="*60)
    print("Example 1: Basic AI-Enhanced Conversion")
    print("="*60)

    # Initialize the AI-enhanced converter
    converter = AIEnhancedSVGConverter()

    # Check if we have test images
    test_images = []
    logo_dirs = ["data/logos/simple_geometric", "data/logos/text_based", "data/logos/gradients"]

    for logo_dir in logo_dirs:
        logo_path = Path(logo_dir)
        if logo_path.exists():
            png_files = list(logo_path.glob("*.png"))
            if png_files:
                test_images.append(str(png_files[0]))  # Take first image

    if not test_images:
        print("⚠️ No test images found. Please ensure test dataset is available.")
        return

    # Convert each test image
    for image_path in test_images[:3]:  # Process first 3 images
        print(f"\n📁 Processing: {Path(image_path).name}")
        print("-" * 40)

        try:
            # Basic conversion with AI enhancement
            start_time = time.time()
            svg_content = converter.convert(image_path)
            conversion_time = time.time() - start_time

            print(f"✅ Conversion successful!")
            print(f"   Processing time: {conversion_time*1000:.1f}ms")
            print(f"   SVG size: {len(svg_content)} characters")

            # Check if AI metadata is present
            if "AI-Enhanced SVG Converter Metadata" in svg_content:
                print(f"   🤖 AI metadata embedded")

            # Save the result
            output_path = Path("output") / Path(image_path).with_suffix('.svg').name
            output_path.parent.mkdir(exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(svg_content)

            print(f"   💾 Saved to: {output_path}")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Show converter statistics
    stats = converter.get_ai_stats()
    print(f"\n📊 Converter Statistics:")
    print(f"   Total conversions: {stats['total_conversions']}")
    print(f"   AI enhanced: {stats['ai_enhanced_conversions']}")
    print(f"   Success rate: {stats['ai_success_rate']:.1f}%")


def example_2_detailed_analysis():
    """Example 2: Detailed AI analysis with quality validation"""
    print("\n" + "="*60)
    print("Example 2: Detailed AI Analysis with Quality Validation")
    print("="*60)

    # Initialize components
    converter = AIEnhancedSVGConverter()
    validator = QualityValidator(quality_threshold=0.85)

    # Find a test image
    test_image = None
    for logo_dir in ["data/logos/simple_geometric", "data/logos/text_based"]:
        logo_path = Path(logo_dir)
        if logo_path.exists():
            png_files = list(logo_path.glob("*.png"))
            if png_files:
                test_image = str(png_files[0])
                break

    if not test_image:
        print("⚠️ No test images found.")
        return

    print(f"📁 Analyzing: {Path(test_image).name}")
    print("-" * 40)

    try:
        # Perform detailed AI analysis
        result = converter.convert_with_ai_analysis(test_image)

        print(f"🤖 AI Analysis Results:")
        if result['ai_enhanced']:
            classification = result['classification']
            features = result['features']

            print(f"   Logo Type: {classification['logo_type']}")
            print(f"   Confidence: {classification['confidence']:.2%}")

            print(f"\n📊 Extracted Features:")
            for feature_name, value in features.items():
                print(f"   {feature_name}: {value:.3f}")

            print(f"\n⚙️ Optimized Parameters:")
            params = result['parameters_used']
            for param_name, value in params.items():
                print(f"   {param_name}: {value}")

            print(f"\n⏱️ Performance Metrics:")
            print(f"   AI analysis time: {result['ai_analysis_time']*1000:.1f}ms")
            print(f"   Conversion time: {result['conversion_time']*1000:.1f}ms")
            print(f"   Total time: {result['total_time']*1000:.1f}ms")

            # Validate quality
            print(f"\n🔍 Quality Validation:")
            try:
                quality_report = validator.validate_conversion(
                    test_image,
                    result['svg'],
                    result['parameters_used'],
                    result['features']
                )

                print(f"   SSIM Score: {quality_report.metrics.ssim_score:.3f}")
                print(f"   Quality Level: {quality_report.metrics.quality_level.value}")
                print(f"   Quality Passed: {'✅' if quality_report.quality_passed else '❌'}")
                print(f"   File Size Ratio: {quality_report.metrics.file_size_ratio:.2f}")

                if quality_report.recommendations:
                    print(f"\n💡 Quality Recommendations:")
                    for rec in quality_report.recommendations[:3]:
                        print(f"   - {rec}")

                if quality_report.parameter_suggestions:
                    print(f"\n🔧 Parameter Suggestions:")
                    for param, value in quality_report.parameter_suggestions.items():
                        print(f"   {param}: {value}")

            except Exception as e:
                print(f"   ⚠️ Quality validation failed: {e}")

        else:
            print(f"   AI enhancement not available, used standard conversion")

        # Save detailed result with metadata
        output_path = Path("output") / f"detailed_{Path(test_image).with_suffix('.svg').name}"
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(result['svg'])

        print(f"\n💾 Detailed result saved to: {output_path}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")


def example_3_parameter_optimization():
    """Example 3: Manual parameter optimization workflow"""
    print("\n" + "="*60)
    print("Example 3: Manual Parameter Optimization Workflow")
    print("="*60)

    # Initialize components
    try:
        from backend.ai_modules.feature_pipeline import FeaturePipeline
        feature_pipeline = FeaturePipeline()
        optimizer = VTracerParameterOptimizer()

        # Find test image
        test_image = None
        for logo_dir in ["data/logos/gradients", "data/logos/complex"]:
            logo_path = Path(logo_dir)
            if logo_path.exists():
                png_files = list(logo_path.glob("*.png"))
                if png_files:
                    test_image = str(png_files[0])
                    break

        if not test_image:
            print("⚠️ No test images found.")
            return

        print(f"📁 Optimizing parameters for: {Path(test_image).name}")
        print("-" * 40)

        # Step 1: Extract features and classify
        print(f"🔍 Step 1: Feature extraction and classification")
        pipeline_result = feature_pipeline.process_image(test_image)
        classification = pipeline_result['classification']
        features = pipeline_result['features']

        print(f"   Logo Type: {classification['logo_type']}")
        print(f"   Confidence: {classification['confidence']:.2%}")

        # Step 2: Get base parameters for logo type
        print(f"\n📋 Step 2: Base parameters for {classification['logo_type']} logos")
        base_params = optimizer.get_parameter_recommendations(classification['logo_type'])
        for param, value in base_params.items():
            print(f"   {param}: {value}")

        # Step 3: Optimize parameters
        print(f"\n⚙️ Step 3: AI-driven parameter optimization")
        optimization_result = optimizer.optimize_parameters(classification, features)

        print(f"   Optimization method: {optimization_result.optimization_method}")
        print(f"   Validation passed: {optimization_result.validation_passed}")
        print(f"   Adjustments applied: {', '.join(optimization_result.adjustments_applied) or 'None'}")
        print(f"   Optimization time: {optimization_result.optimization_time*1000:.1f}ms")

        print(f"\n🎯 Optimized Parameters:")
        for param, value in optimization_result.parameters.items():
            print(f"   {param}: {value}")

        # Step 4: Compare with base parameters
        print(f"\n📊 Parameter Comparison:")
        for param in ['color_precision', 'layer_difference', 'corner_threshold']:
            base_val = base_params.get(param, 'N/A')
            opt_val = optimization_result.parameters.get(param, 'N/A')
            if base_val != opt_val:
                print(f"   {param}: {base_val} → {opt_val}")

        # Step 5: Test conversion with optimized parameters
        print(f"\n🔄 Step 4: Test conversion with optimized parameters")
        try:
            from backend.converters.vtracer_converter import VTracerConverter
            vtracer = VTracerConverter()

            svg_content = vtracer.convert(test_image, **optimization_result.parameters)
            print(f"   ✅ Conversion successful!")
            print(f"   SVG size: {len(svg_content)} characters")

            # Save optimized result
            output_path = Path("output") / f"optimized_{Path(test_image).with_suffix('.svg').name}"
            output_path.parent.mkdir(exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(svg_content)

            print(f"   💾 Saved to: {output_path}")

        except Exception as e:
            print(f"   ❌ Conversion failed: {e}")

    except ImportError as e:
        print(f"⚠️ Feature pipeline not available: {e}")
        print("This example requires the Day 2 AI modules to be available.")


def example_4_batch_processing():
    """Example 4: Batch processing with statistics"""
    print("\n" + "="*60)
    print("Example 4: Batch Processing with Statistics")
    print("="*60)

    converter = AIEnhancedSVGConverter()

    # Find test images from multiple categories
    test_images = []
    logo_categories = ["simple_geometric", "text_based", "gradients", "complex"]

    for category in logo_categories:
        logo_dir = Path(f"data/logos/{category}")
        if logo_dir.exists():
            png_files = list(logo_dir.glob("*.png"))
            # Take up to 2 images from each category
            test_images.extend([(str(img), category) for img in png_files[:2]])

    if not test_images:
        print("⚠️ No test images found for batch processing.")
        return

    print(f"📦 Processing {len(test_images)} logos from {len(logo_categories)} categories")
    print("-" * 40)

    results = []
    total_start_time = time.time()

    # Process each image
    for image_path, category in test_images:
        try:
            result = converter.convert_with_ai_analysis(image_path)

            processing_result = {
                'file': Path(image_path).name,
                'category': category,
                'logo_type': result['classification'].get('logo_type', 'unknown'),
                'confidence': result['classification'].get('confidence', 0.0),
                'ai_enhanced': result['ai_enhanced'],
                'processing_time': result['total_time'],
                'svg_size': len(result['svg'])
            }
            results.append(processing_result)

            # Save SVG
            output_dir = Path("output/batch")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{category}_{Path(image_path).with_suffix('.svg').name}"

            with open(output_path, 'w') as f:
                f.write(result['svg'])

            status = "🤖" if result['ai_enhanced'] else "⚙️"
            print(f"{status} {processing_result['file']} ({category}) -> {processing_result['logo_type']} "
                  f"({processing_result['confidence']:.1%}) in {processing_result['processing_time']*1000:.0f}ms")

        except Exception as e:
            print(f"❌ {Path(image_path).name}: {e}")

    total_time = time.time() - total_start_time

    # Print comprehensive statistics
    print(f"\n📊 Batch Processing Results:")
    print(f"   Total processed: {len(results)}")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Average time per logo: {total_time/len(results)*1000:.0f}ms")

    # AI enhancement statistics
    ai_enhanced_count = sum(1 for r in results if r['ai_enhanced'])
    print(f"\n🤖 AI Enhancement Statistics:")
    print(f"   AI enhanced: {ai_enhanced_count}/{len(results)} ({ai_enhanced_count/len(results)*100:.1f}%)")

    # Get detailed converter stats
    converter_stats = converter.get_ai_stats()
    print(f"   AI success rate: {converter_stats['ai_success_rate']:.1f}%")
    print(f"   Average AI time: {converter_stats['average_ai_time']*1000:.1f}ms")

    # Logo type breakdown
    print(f"\n🏷️ Logo Type Classification:")
    type_counts = {}
    for result in results:
        logo_type = result['logo_type']
        type_counts[logo_type] = type_counts.get(logo_type, 0) + 1

    for logo_type, count in sorted(type_counts.items()):
        print(f"   {logo_type}: {count}")

    # Performance breakdown by category
    print(f"\n⚡ Performance by Category:")
    category_stats = {}
    for result in results:
        category = result['category']
        if category not in category_stats:
            category_stats[category] = []
        category_stats[category].append(result['processing_time'])

    for category, times in category_stats.items():
        avg_time = sum(times) / len(times)
        print(f"   {category}: {avg_time*1000:.0f}ms avg ({len(times)} images)")


def example_5_quality_comparison():
    """Example 5: Quality comparison between AI and standard conversion"""
    print("\n" + "="*60)
    print("Example 5: Quality Comparison (AI vs Standard)")
    print("="*60)

    # Initialize components
    ai_converter = AIEnhancedSVGConverter()
    validator = QualityValidator(quality_threshold=0.85)

    try:
        from backend.converters.vtracer_converter import VTracerConverter
        standard_converter = VTracerConverter()
    except ImportError:
        print("⚠️ VTracerConverter not available for comparison.")
        return

    # Find test image
    test_image = None
    for logo_dir in ["data/logos/simple_geometric", "data/logos/gradients"]:
        logo_path = Path(logo_dir)
        if logo_path.exists():
            png_files = list(logo_path.glob("*.png"))
            if png_files:
                test_image = str(png_files[0])
                break

    if not test_image:
        print("⚠️ No test images found.")
        return

    print(f"📁 Comparing conversions for: {Path(test_image).name}")
    print("-" * 40)

    try:
        # AI-enhanced conversion
        print(f"🤖 AI-Enhanced Conversion:")
        ai_start = time.time()
        ai_result = ai_converter.convert_with_ai_analysis(test_image)
        ai_time = time.time() - ai_start

        print(f"   Processing time: {ai_time*1000:.1f}ms")
        print(f"   SVG size: {len(ai_result['svg'])} characters")
        if ai_result['ai_enhanced']:
            print(f"   Logo type: {ai_result['classification']['logo_type']}")
            print(f"   Confidence: {ai_result['classification']['confidence']:.2%}")

        # Standard conversion
        print(f"\n⚙️ Standard Conversion:")
        standard_start = time.time()
        standard_svg = standard_converter.convert(test_image)
        standard_time = time.time() - standard_start

        print(f"   Processing time: {standard_time*1000:.1f}ms")
        print(f"   SVG size: {len(standard_svg)} characters")

        # Quality comparison
        print(f"\n🔍 Quality Comparison:")

        # Validate AI conversion quality
        try:
            ai_quality = validator.validate_conversion(
                test_image, ai_result['svg'],
                ai_result.get('parameters_used', {}),
                ai_result.get('features', {})
            )
            print(f"   AI SSIM: {ai_quality.metrics.ssim_score:.3f} ({ai_quality.metrics.quality_level.value})")
        except Exception as e:
            print(f"   AI quality validation failed: {e}")
            ai_quality = None

        # Validate standard conversion quality
        try:
            standard_quality = validator.validate_conversion(test_image, standard_svg)
            print(f"   Standard SSIM: {standard_quality.metrics.ssim_score:.3f} ({standard_quality.metrics.quality_level.value})")
        except Exception as e:
            print(f"   Standard quality validation failed: {e}")
            standard_quality = None

        # Comparison summary
        print(f"\n📊 Comparison Summary:")
        print(f"   Time difference: {(ai_time - standard_time)*1000:+.0f}ms")
        print(f"   Size difference: {len(ai_result['svg']) - len(standard_svg):+d} characters")

        if ai_quality and standard_quality:
            ssim_diff = ai_quality.metrics.ssim_score - standard_quality.metrics.ssim_score
            print(f"   SSIM difference: {ssim_diff:+.3f}")

            if ssim_diff > 0.01:
                print(f"   🏆 AI conversion achieved better quality!")
            elif ssim_diff < -0.01:
                print(f"   📈 Standard conversion achieved better quality!")
            else:
                print(f"   🤝 Quality difference negligible")

        # Save both results for comparison
        output_dir = Path("output/comparison")
        output_dir.mkdir(parents=True, exist_ok=True)

        ai_path = output_dir / f"ai_{Path(test_image).with_suffix('.svg').name}"
        standard_path = output_dir / f"standard_{Path(test_image).with_suffix('.svg').name}"

        with open(ai_path, 'w') as f:
            f.write(ai_result['svg'])

        with open(standard_path, 'w') as f:
            f.write(standard_svg)

        print(f"\n💾 Results saved:")
        print(f"   AI: {ai_path}")
        print(f"   Standard: {standard_path}")

    except Exception as e:
        print(f"❌ Comparison failed: {e}")


def main():
    """Run all examples"""
    print("🤖 AI-Enhanced SVG Conversion Examples")
    print("=" * 60)

    # Create output directory
    Path("output").mkdir(exist_ok=True)

    # Run examples
    example_1_basic_conversion()
    example_2_detailed_analysis()
    example_3_parameter_optimization()
    example_4_batch_processing()
    example_5_quality_comparison()

    print(f"\n✅ All examples completed!")
    print(f"📁 Results saved in the 'output' directory")


if __name__ == "__main__":
    main()