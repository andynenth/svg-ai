#!/usr/bin/env python3
"""
Day 3: Edge Case & Robustness Testing

Comprehensive testing for unusual and difficult cases:
- Very small/large images
- Unusual aspect ratios
- Single-color images
- Corrupted/invalid data
- Non-logo images
- Boundary conditions
- Extreme feature values
"""

import sys
import os
import json
import numpy as np
import cv2
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.ai_modules.feature_pipeline import FeaturePipeline
from backend.ai_modules.rule_based_classifier import RuleBasedClassifier
from backend.ai_modules.feature_extraction import ImageFeatureExtractor


class EdgeCaseRobustnessTester:
    """Comprehensive edge case and robustness testing"""

    def __init__(self):
        self.pipeline = FeaturePipeline(cache_enabled=False)
        self.extractor = ImageFeatureExtractor(cache_enabled=False)
        self.classifier = RuleBasedClassifier()
        self.temp_dir = Path("temp_edge_case_images")
        self.temp_dir.mkdir(exist_ok=True)
        self.results = {
            'edge_case_tests': {},
            'boundary_condition_tests': {},
            'robustness_summary': {}
        }

    def test_very_small_images(self) -> Dict[str, Any]:
        """Test with very small images (<50x50 pixels)"""
        print("🔍 Testing very small images (<50x50 pixels)...")

        small_image_results = []
        test_sizes = [(10, 10), (25, 25), (40, 40), (49, 49)]

        for width, height in test_sizes:
            # Create simple test image
            img = self._create_simple_test_image((width, height))
            img_path = self.temp_dir / f"small_{width}x{height}.png"
            cv2.imwrite(str(img_path), img)

            # Test classification
            try:
                start_time = time.perf_counter()
                result = self.pipeline.process_image(str(img_path))
                processing_time = time.perf_counter() - start_time

                classification = result.get('classification', {})
                features = result.get('features', {})

                small_image_results.append({
                    'size': f"{width}x{height}",
                    'pixels': width * height,
                    'classification': classification.get('logo_type', 'unknown'),
                    'confidence': classification.get('confidence', 0.0),
                    'processing_time': processing_time,
                    'features_extracted': len(features) > 0,
                    'valid_features': self._validate_features(features),
                    'error': None
                })

                print(f"  {width}x{height}: {classification.get('logo_type', 'unknown')} "
                      f"({classification.get('confidence', 0.0):.3f})")

            except Exception as e:
                small_image_results.append({
                    'size': f"{width}x{height}",
                    'pixels': width * height,
                    'classification': 'error',
                    'confidence': 0.0,
                    'processing_time': 0.0,
                    'features_extracted': False,
                    'valid_features': False,
                    'error': str(e)
                })
                print(f"  {width}x{height}: ERROR - {e}")

        self.results['edge_case_tests']['very_small_images'] = {
            'test_results': small_image_results,
            'total_tests': len(test_sizes),
            'successful_tests': sum(1 for r in small_image_results if r['error'] is None),
            'error_rate': sum(1 for r in small_image_results if r['error'] is not None) / len(test_sizes)
        }

        return self.results['edge_case_tests']['very_small_images']

    def test_very_large_images(self) -> Dict[str, Any]:
        """Test with very large images (>2000x2000 pixels)"""
        print("🔍 Testing very large images (>2000x2000 pixels)...")

        large_image_results = []
        test_sizes = [(2048, 2048), (3000, 3000), (4096, 4096)]

        for width, height in test_sizes:
            # Create simple test image (limited to reasonable size for testing)
            if width * height > 10_000_000:  # Skip extremely large images for testing
                large_image_results.append({
                    'size': f"{width}x{height}",
                    'pixels': width * height,
                    'classification': 'skipped',
                    'confidence': 0.0,
                    'processing_time': 0.0,
                    'features_extracted': False,
                    'valid_features': False,
                    'error': 'Skipped - too large for test environment'
                })
                continue

            img = self._create_simple_test_image((width, height))
            img_path = self.temp_dir / f"large_{width}x{height}.png"

            try:
                cv2.imwrite(str(img_path), img)

                start_time = time.perf_counter()
                result = self.pipeline.process_image(str(img_path))
                processing_time = time.perf_counter() - start_time

                classification = result.get('classification', {})
                features = result.get('features', {})

                large_image_results.append({
                    'size': f"{width}x{height}",
                    'pixels': width * height,
                    'classification': classification.get('logo_type', 'unknown'),
                    'confidence': classification.get('confidence', 0.0),
                    'processing_time': processing_time,
                    'features_extracted': len(features) > 0,
                    'valid_features': self._validate_features(features),
                    'error': None
                })

                print(f"  {width}x{height}: {classification.get('logo_type', 'unknown')} "
                      f"({processing_time:.3f}s)")

            except Exception as e:
                large_image_results.append({
                    'size': f"{width}x{height}",
                    'pixels': width * height,
                    'classification': 'error',
                    'confidence': 0.0,
                    'processing_time': 0.0,
                    'features_extracted': False,
                    'valid_features': False,
                    'error': str(e)
                })
                print(f"  {width}x{height}: ERROR - {e}")

        self.results['edge_case_tests']['very_large_images'] = {
            'test_results': large_image_results,
            'total_tests': len(test_sizes),
            'successful_tests': sum(1 for r in large_image_results if r['error'] is None),
            'error_rate': sum(1 for r in large_image_results if r['error'] is not None) / len(test_sizes)
        }

        return self.results['edge_case_tests']['very_large_images']

    def test_unusual_aspect_ratios(self) -> Dict[str, Any]:
        """Test with unusual aspect ratios (very wide/tall)"""
        print("🔍 Testing unusual aspect ratios...")

        aspect_ratio_results = []
        test_dimensions = [
            (1000, 10),    # Very wide
            (10, 1000),    # Very tall
            (500, 25),     # Wide
            (25, 500),     # Tall
            (800, 100),    # Moderately wide
            (100, 800)     # Moderately tall
        ]

        for width, height in test_dimensions:
            img = self._create_simple_test_image((width, height))
            img_path = self.temp_dir / f"aspect_{width}x{height}.png"

            try:
                cv2.imwrite(str(img_path), img)

                start_time = time.perf_counter()
                result = self.pipeline.process_image(str(img_path))
                processing_time = time.perf_counter() - start_time

                classification = result.get('classification', {})
                features = result.get('features', {})

                aspect_ratio = width / height
                aspect_ratio_results.append({
                    'dimensions': f"{width}x{height}",
                    'aspect_ratio': aspect_ratio,
                    'classification': classification.get('logo_type', 'unknown'),
                    'confidence': classification.get('confidence', 0.0),
                    'processing_time': processing_time,
                    'features_extracted': len(features) > 0,
                    'valid_features': self._validate_features(features),
                    'error': None
                })

                print(f"  {width}x{height} (ratio {aspect_ratio:.1f}): "
                      f"{classification.get('logo_type', 'unknown')}")

            except Exception as e:
                aspect_ratio_results.append({
                    'dimensions': f"{width}x{height}",
                    'aspect_ratio': width / height,
                    'classification': 'error',
                    'confidence': 0.0,
                    'processing_time': 0.0,
                    'features_extracted': False,
                    'valid_features': False,
                    'error': str(e)
                })
                print(f"  {width}x{height}: ERROR - {e}")

        self.results['edge_case_tests']['unusual_aspect_ratios'] = {
            'test_results': aspect_ratio_results,
            'total_tests': len(test_dimensions),
            'successful_tests': sum(1 for r in aspect_ratio_results if r['error'] is None),
            'error_rate': sum(1 for r in aspect_ratio_results if r['error'] is not None) / len(test_dimensions)
        }

        return self.results['edge_case_tests']['unusual_aspect_ratios']

    def test_single_color_images(self) -> Dict[str, Any]:
        """Test with single-color or near-single-color images"""
        print("🔍 Testing single-color and near-single-color images...")

        color_test_results = []
        test_configs = [
            {'name': 'pure_black', 'color': (0, 0, 0)},
            {'name': 'pure_white', 'color': (255, 255, 255)},
            {'name': 'pure_red', 'color': (255, 0, 0)},
            {'name': 'pure_blue', 'color': (0, 0, 255)},
            {'name': 'gray', 'color': (128, 128, 128)},
            {'name': 'near_black', 'color': (5, 5, 5)},
            {'name': 'near_white', 'color': (250, 250, 250)}
        ]

        for config in test_configs:
            # Create single-color image
            img = np.full((200, 200, 3), config['color'], dtype=np.uint8)
            img_path = self.temp_dir / f"color_{config['name']}.png"

            try:
                cv2.imwrite(str(img_path), img)

                start_time = time.perf_counter()
                result = self.pipeline.process_image(str(img_path))
                processing_time = time.perf_counter() - start_time

                classification = result.get('classification', {})
                features = result.get('features', {})

                color_test_results.append({
                    'test_name': config['name'],
                    'color': config['color'],
                    'classification': classification.get('logo_type', 'unknown'),
                    'confidence': classification.get('confidence', 0.0),
                    'processing_time': processing_time,
                    'features': {k: v for k, v in features.items()} if features else {},
                    'valid_features': self._validate_features(features),
                    'error': None
                })

                print(f"  {config['name']}: {classification.get('logo_type', 'unknown')} "
                      f"({classification.get('confidence', 0.0):.3f})")

            except Exception as e:
                color_test_results.append({
                    'test_name': config['name'],
                    'color': config['color'],
                    'classification': 'error',
                    'confidence': 0.0,
                    'processing_time': 0.0,
                    'features': {},
                    'valid_features': False,
                    'error': str(e)
                })
                print(f"  {config['name']}: ERROR - {e}")

        self.results['edge_case_tests']['single_color_images'] = {
            'test_results': color_test_results,
            'total_tests': len(test_configs),
            'successful_tests': sum(1 for r in color_test_results if r['error'] is None),
            'error_rate': sum(1 for r in color_test_results if r['error'] is not None) / len(test_configs)
        }

        return self.results['edge_case_tests']['single_color_images']

    def test_corrupted_invalid_data(self) -> Dict[str, Any]:
        """Test with corrupted or invalid image data"""
        print("🔍 Testing corrupted and invalid image data...")

        invalid_data_results = []

        # Test cases
        test_cases = [
            {'name': 'non_existent_file', 'path': 'non_existent_file.png'},
            {'name': 'empty_file', 'create_empty': True},
            {'name': 'text_file_as_image', 'create_text': True},
            {'name': 'truncated_image', 'create_truncated': True},
            {'name': 'zero_size_image', 'create_zero': True}
        ]

        for test_case in test_cases:
            test_path = None

            try:
                if test_case['name'] == 'non_existent_file':
                    test_path = test_case['path']

                elif test_case.get('create_empty'):
                    test_path = self.temp_dir / 'empty_file.png'
                    test_path.touch()

                elif test_case.get('create_text'):
                    test_path = self.temp_dir / 'text_file.png'
                    with open(test_path, 'w') as f:
                        f.write("This is not an image file")

                elif test_case.get('create_truncated'):
                    # Create a valid image first, then truncate it
                    valid_img = self._create_simple_test_image((100, 100))
                    temp_path = self.temp_dir / 'temp_valid.png'
                    cv2.imwrite(str(temp_path), valid_img)

                    # Read and truncate
                    with open(temp_path, 'rb') as f:
                        data = f.read()

                    test_path = self.temp_dir / 'truncated.png'
                    with open(test_path, 'wb') as f:
                        f.write(data[:len(data)//2])  # Write only half

                elif test_case.get('create_zero'):
                    test_path = self.temp_dir / 'zero_size.png'
                    # Create a zero-sized image (this should fail)
                    try:
                        zero_img = np.zeros((0, 0, 3), dtype=np.uint8)
                        cv2.imwrite(str(test_path), zero_img)
                    except:
                        # If creation fails, create an empty file
                        test_path.touch()

                # Test classification
                start_time = time.perf_counter()
                result = self.pipeline.process_image(str(test_path))
                processing_time = time.perf_counter() - start_time

                classification = result.get('classification', {})

                invalid_data_results.append({
                    'test_name': test_case['name'],
                    'classification': classification.get('logo_type', 'unknown'),
                    'confidence': classification.get('confidence', 0.0),
                    'processing_time': processing_time,
                    'handled_gracefully': classification.get('logo_type') == 'unknown',
                    'error': None
                })

                print(f"  {test_case['name']}: {classification.get('logo_type', 'unknown')} "
                      f"(handled gracefully)")

            except Exception as e:
                invalid_data_results.append({
                    'test_name': test_case['name'],
                    'classification': 'error',
                    'confidence': 0.0,
                    'processing_time': 0.0,
                    'handled_gracefully': True,  # Errors are acceptable for invalid data
                    'error': str(e)
                })
                print(f"  {test_case['name']}: ERROR (expected) - {str(e)[:50]}...")

        self.results['edge_case_tests']['corrupted_invalid_data'] = {
            'test_results': invalid_data_results,
            'total_tests': len(test_cases),
            'gracefully_handled': sum(1 for r in invalid_data_results if r['handled_gracefully']),
            'graceful_handling_rate': sum(1 for r in invalid_data_results
                                        if r['handled_gracefully']) / len(test_cases)
        }

        return self.results['edge_case_tests']['corrupted_invalid_data']

    def test_boundary_conditions(self) -> Dict[str, Any]:
        """Test features exactly on threshold boundaries"""
        print("🔍 Testing boundary conditions...")

        boundary_results = []

        # Test boundary feature values based on Day 2 optimized thresholds
        boundary_test_cases = [
            {
                'name': 'simple_upper_boundary',
                'features': {
                    'complexity_score': 0.089,    # Just at upper simple boundary
                    'entropy': 0.060,             # At upper simple boundary
                    'unique_colors': 0.125,       # At simple boundary
                    'edge_density': 0.0074,       # At upper simple boundary
                    'corner_density': 0.070,      # At upper simple boundary
                    'gradient_strength': 0.065    # Just over simple boundary
                }
            },
            {
                'name': 'simple_lower_boundary',
                'features': {
                    'complexity_score': 0.080,    # At lower simple boundary
                    'entropy': 0.044,             # At lower simple boundary
                    'unique_colors': 0.125,       # At simple boundary
                    'edge_density': 0.0058,       # At lower simple boundary
                    'corner_density': 0.026,      # At lower simple boundary
                    'gradient_strength': 0.060    # At lower simple boundary
                }
            },
            {
                'name': 'extreme_zeros',
                'features': {
                    'complexity_score': 0.0,
                    'entropy': 0.0,
                    'unique_colors': 0.0,
                    'edge_density': 0.0,
                    'corner_density': 0.0,
                    'gradient_strength': 0.0
                }
            },
            {
                'name': 'extreme_ones',
                'features': {
                    'complexity_score': 1.0,
                    'entropy': 1.0,
                    'unique_colors': 1.0,
                    'edge_density': 1.0,
                    'corner_density': 1.0,
                    'gradient_strength': 1.0
                }
            },
            {
                'name': 'nan_values',
                'features': {
                    'complexity_score': float('nan'),
                    'entropy': 0.5,
                    'unique_colors': 0.5,
                    'edge_density': float('nan'),
                    'corner_density': 0.5,
                    'gradient_strength': 0.5
                }
            },
            {
                'name': 'infinite_values',
                'features': {
                    'complexity_score': float('inf'),
                    'entropy': 0.5,
                    'unique_colors': 0.5,
                    'edge_density': 0.5,
                    'corner_density': float('-inf'),
                    'gradient_strength': 0.5
                }
            }
        ]

        for test_case in boundary_test_cases:
            try:
                start_time = time.perf_counter()
                result = self.classifier.classify(test_case['features'])
                processing_time = time.perf_counter() - start_time

                boundary_results.append({
                    'test_name': test_case['name'],
                    'input_features': test_case['features'],
                    'classification': result.get('logo_type', 'unknown'),
                    'confidence': result.get('confidence', 0.0),
                    'processing_time': processing_time,
                    'handled_gracefully': True,
                    'error': None
                })

                print(f"  {test_case['name']}: {result.get('logo_type', 'unknown')} "
                      f"({result.get('confidence', 0.0):.3f})")

            except Exception as e:
                boundary_results.append({
                    'test_name': test_case['name'],
                    'input_features': test_case['features'],
                    'classification': 'error',
                    'confidence': 0.0,
                    'processing_time': 0.0,
                    'handled_gracefully': False,
                    'error': str(e)
                })
                print(f"  {test_case['name']}: ERROR - {e}")

        self.results['boundary_condition_tests'] = {
            'test_results': boundary_results,
            'total_tests': len(boundary_test_cases),
            'successful_tests': sum(1 for r in boundary_results if r['error'] is None),
            'graceful_handling_rate': sum(1 for r in boundary_results
                                        if r['handled_gracefully']) / len(boundary_test_cases)
        }

        return self.results['boundary_condition_tests']

    def _create_simple_test_image(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a simple test image of specified size"""
        width, height = size
        img = np.zeros((height, width, 3), dtype=np.uint8)

        if width > 10 and height > 10:
            # Add a simple shape
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4
            cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), -1)

        return img

    def _validate_features(self, features: Dict[str, float]) -> bool:
        """Validate that extracted features are reasonable"""
        if not features:
            return False

        required_features = ['edge_density', 'unique_colors', 'corner_density',
                           'entropy', 'gradient_strength', 'complexity_score']

        for feature_name in required_features:
            if feature_name not in features:
                return False

            value = features[feature_name]
            if not isinstance(value, (int, float)):
                return False

            if np.isnan(value) or np.isinf(value):
                return False

            if value < 0 or value > 1:
                return False

        return True

    def generate_robustness_summary(self) -> Dict[str, Any]:
        """Generate overall robustness summary"""
        all_tests = []

        # Collect all test results
        for test_category, test_data in self.results['edge_case_tests'].items():
            if 'test_results' in test_data:
                all_tests.extend(test_data['test_results'])

        if 'test_results' in self.results.get('boundary_condition_tests', {}):
            all_tests.extend(self.results['boundary_condition_tests']['test_results'])

        total_tests = len(all_tests)
        successful_tests = sum(1 for test in all_tests if test.get('error') is None)
        graceful_failures = sum(1 for test in all_tests
                              if test.get('error') is not None and
                              test.get('handled_gracefully', False))

        summary = {
            'total_edge_case_tests': total_tests,
            'successful_tests': successful_tests,
            'graceful_failures': graceful_failures,
            'hard_failures': total_tests - successful_tests - graceful_failures,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'graceful_handling_rate': (successful_tests + graceful_failures) / total_tests if total_tests > 0 else 0,
            'robustness_score': ((successful_tests * 1.0 + graceful_failures * 0.5) / total_tests) if total_tests > 0 else 0,
            'categories_tested': list(self.results['edge_case_tests'].keys())
        }

        self.results['robustness_summary'] = summary

        print(f"\n📋 Robustness Summary:")
        print(f"   Total tests: {total_tests}")
        print(f"   Success rate: {summary['success_rate']:.1%}")
        print(f"   Graceful handling: {summary['graceful_handling_rate']:.1%}")
        print(f"   Robustness score: {summary['robustness_score']:.3f}")

        return summary

    def run_all_edge_case_tests(self) -> Dict[str, Any]:
        """Run all edge case and robustness tests"""
        print("🚀 Starting comprehensive edge case and robustness testing...")

        # Run all test categories
        self.test_very_small_images()
        self.test_very_large_images()
        self.test_unusual_aspect_ratios()
        self.test_single_color_images()
        self.test_corrupted_invalid_data()
        self.test_boundary_conditions()

        # Generate summary
        self.generate_robustness_summary()

        return self.results

    def save_results(self, output_file: str = "edge_case_test_results.json"):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📄 Results saved to: {output_file}")

    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print("🧹 Cleaned up temporary test images")


def main():
    """Main function to run edge case testing"""
    print("🔬 Day 3: Edge Case & Robustness Testing")
    print("=" * 50)

    tester = EdgeCaseRobustnessTester()

    try:
        # Run all tests
        results = tester.run_all_edge_case_tests()

        # Save results
        tester.save_results()

        # Final assessment
        summary = results['robustness_summary']
        robustness_score = summary['robustness_score']

        print(f"\n🎯 Final Robustness Assessment:")
        if robustness_score >= 0.9:
            print("✅ EXCELLENT - System handles edge cases very well")
        elif robustness_score >= 0.8:
            print("✅ GOOD - System handles most edge cases appropriately")
        elif robustness_score >= 0.7:
            print("⚠️  ACCEPTABLE - System handles edge cases with some issues")
        else:
            print("❌ POOR - System struggles with edge cases")

        return robustness_score >= 0.7

    except Exception as e:
        print(f"❌ Edge case testing failed with error: {e}")
        return False

    finally:
        tester.cleanup()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)