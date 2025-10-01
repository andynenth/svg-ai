#!/usr/bin/env python3
"""
End-to-End Pipeline Integration Test

Tests complete workflow: Image → Features → Classification → Result
Identifies exact point where pipeline fails and tests integration points.
"""

import sys
import os
import logging
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ai_modules.feature_extraction import ImageFeatureExtractor
from ai_modules.rule_based_classifier import RuleBasedClassifier


class EndToEndPipelineTester:
    """Test complete classification pipeline integration"""

    def __init__(self):
        self.logger = self._setup_logging()
        self.feature_extractor = ImageFeatureExtractor()
        self.classifier = RuleBasedClassifier()
        self.test_results = []

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def test_complete_pipeline(self, image_path: str) -> Dict[str, Any]:
        """
        Test complete pipeline: Image → Features → Classification → Result

        Args:
            image_path: Path to test image

        Returns:
            Test results with success/failure at each step
        """
        test_result = {
            'image_path': image_path,
            'pipeline_steps': {},
            'success': False,
            'failure_point': None,
            'final_result': None
        }

        self.logger.info(f"🧪 Testing complete pipeline for: {image_path}")

        try:
            # Step 1: Image Loading & Validation
            self.logger.info("Step 1: Image validation...")
            step1_result = self._test_image_validation(image_path)
            test_result['pipeline_steps']['image_validation'] = step1_result

            if not step1_result['success']:
                test_result['failure_point'] = 'image_validation'
                return test_result

            # Step 2: Feature Extraction
            self.logger.info("Step 2: Feature extraction...")
            step2_result = self._test_feature_extraction(image_path)
            test_result['pipeline_steps']['feature_extraction'] = step2_result

            if not step2_result['success']:
                test_result['failure_point'] = 'feature_extraction'
                return test_result

            # Step 3: Classification
            self.logger.info("Step 3: Classification...")
            features = step2_result['features']
            step3_result = self._test_classification(features)
            test_result['pipeline_steps']['classification'] = step3_result

            if not step3_result['success']:
                test_result['failure_point'] = 'classification'
                return test_result

            # Step 4: Result Validation
            self.logger.info("Step 4: Result validation...")
            classification_result = step3_result['result']
            step4_result = self._test_result_validation(classification_result)
            test_result['pipeline_steps']['result_validation'] = step4_result

            if not step4_result['success']:
                test_result['failure_point'] = 'result_validation'
                return test_result

            # Pipeline completed successfully
            test_result['success'] = True
            test_result['final_result'] = classification_result
            self.logger.info("✅ Complete pipeline test PASSED")

        except Exception as e:
            self.logger.error(f"❌ Pipeline test failed with exception: {e}")
            test_result['failure_point'] = 'exception'
            test_result['error'] = str(e)

        return test_result

    def _test_image_validation(self, image_path: str) -> Dict[str, Any]:
        """Test Step 1: Image file validation"""
        result = {'success': False, 'details': {}}

        try:
            path_obj = Path(image_path)
            result['details'] = {
                'exists': path_obj.exists(),
                'is_file': path_obj.is_file(),
                'readable': os.access(image_path, os.R_OK),
                'size_bytes': path_obj.stat().st_size if path_obj.exists() else 0
            }

            result['success'] = all(result['details'].values()) and result['details']['size_bytes'] > 0

            if result['success']:
                self.logger.info("   ✅ Image validation passed")
            else:
                self.logger.warning(f"   ⚠️ Image validation failed: {result['details']}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"   ❌ Image validation error: {e}")

        return result

    def _test_feature_extraction(self, image_path: str) -> Dict[str, Any]:
        """Test Step 2: Feature extraction"""
        result = {'success': False, 'features': {}, 'details': {}}

        try:
            # Extract features
            features = self.feature_extractor.extract_features(image_path)
            result['features'] = features

            # Validate extracted features
            expected_features = [
                'edge_density', 'unique_colors', 'entropy',
                'corner_density', 'gradient_strength', 'complexity_score'
            ]

            result['details'] = {
                'feature_count': len(features),
                'has_all_expected': all(f in features for f in expected_features),
                'missing_features': [f for f in expected_features if f not in features],
                'extra_features': [f for f in features if f not in expected_features],
                'valid_ranges': all(0.0 <= v <= 1.0 for v in features.values()),
                'no_nan_inf': all(not (np.isnan(v) or np.isinf(v)) for v in features.values())
            }

            result['success'] = (
                result['details']['has_all_expected'] and
                result['details']['valid_ranges'] and
                result['details']['no_nan_inf']
            )

            if result['success']:
                self.logger.info(f"   ✅ Feature extraction passed: {len(features)} features")
                for name, value in features.items():
                    self.logger.debug(f"      {name}: {value:.4f}")
            else:
                self.logger.warning(f"   ⚠️ Feature extraction issues: {result['details']}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"   ❌ Feature extraction error: {e}")

        return result

    def _test_classification(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Test Step 3: Classification"""
        result = {'success': False, 'result': None, 'details': {}}

        try:
            # Test classification
            classification_result = self.classifier.classify(features)
            result['result'] = classification_result

            # Analyze result
            result['details'] = {
                'result_type': type(classification_result).__name__,
                'result_length': len(classification_result) if hasattr(classification_result, '__len__') else 0,
                'is_not_none': classification_result is not None,
                'is_tuple': isinstance(classification_result, tuple),
                'is_dict': isinstance(classification_result, dict)
            }

            # Check if classification returned a valid result (even if wrong format)
            result['success'] = (
                classification_result is not None and
                len(str(classification_result)) > 0
            )

            if result['success']:
                self.logger.info(f"   ✅ Classification completed: {classification_result}")
            else:
                self.logger.warning(f"   ⚠️ Classification failed: {result['details']}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"   ❌ Classification error: {e}")

        return result

    def _test_result_validation(self, classification_result: Any) -> Dict[str, Any]:
        """Test Step 4: Result format validation"""
        result = {'success': False, 'details': {}}

        try:
            # Expected format: dict with logo_type, confidence, reasoning
            expected_fields = ['logo_type', 'confidence', 'reasoning']

            result['details'] = {
                'result_type': type(classification_result).__name__,
                'expected_type': 'dict',
                'is_dict': isinstance(classification_result, dict)
            }

            if isinstance(classification_result, dict):
                present_fields = list(classification_result.keys())
                missing_fields = [f for f in expected_fields if f not in present_fields]

                result['details']['present_fields'] = present_fields
                result['details']['missing_fields'] = missing_fields
                result['details']['has_all_required'] = len(missing_fields) == 0

                result['success'] = len(missing_fields) == 0

            elif isinstance(classification_result, tuple):
                result['details']['tuple_length'] = len(classification_result)
                result['details']['tuple_elements'] = [str(elem) for elem in classification_result]
                result['details']['format_issue'] = 'Result is tuple, expected dict'

            else:
                result['details']['format_issue'] = f'Unexpected type: {type(classification_result)}'

            if result['success']:
                self.logger.info("   ✅ Result validation passed")
            else:
                self.logger.warning(f"   ⚠️ Result validation failed: {result['details']}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"   ❌ Result validation error: {e}")

        return result

    def test_diverse_dataset(self, test_images: List[str]) -> Dict[str, Any]:
        """Test pipeline on diverse logo dataset"""
        self.logger.info(f"🧪 Testing pipeline on {len(test_images)} diverse images")

        results = {
            'total_tests': len(test_images),
            'successful_pipelines': 0,
            'failure_points': {},
            'individual_results': []
        }

        for image_path in test_images:
            test_result = self.test_complete_pipeline(image_path)
            results['individual_results'].append(test_result)

            if test_result['success']:
                results['successful_pipelines'] += 1
                self.logger.info(f"✅ {image_path}: PASSED")
            else:
                failure_point = test_result.get('failure_point', 'unknown')
                results['failure_points'][failure_point] = results['failure_points'].get(failure_point, 0) + 1
                self.logger.error(f"❌ {image_path}: FAILED at {failure_point}")

        # Calculate summary
        results['success_rate'] = results['successful_pipelines'] / results['total_tests']
        results['most_common_failure'] = max(results['failure_points'].items(), key=lambda x: x[1])[0] if results['failure_points'] else None

        return results

    def diagnose_integration_issues(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test results to identify integration issues"""
        issues = []
        recommendations = []

        # Analyze failure points
        failure_points = test_results.get('failure_points', {})

        if 'result_validation' in failure_points:
            issues.append("Result format validation failing - tuple vs dict format mismatch")
            recommendations.append("Fix classify() method to return dict format with logo_type, confidence, reasoning")

        if 'classification' in failure_points:
            issues.append("Classification step failing")
            recommendations.append("Debug classification logic and thresholds")

        if 'feature_extraction' in failure_points:
            issues.append("Feature extraction failing")
            recommendations.append("Check feature extraction pipeline dependencies and error handling")

        if 'image_validation' in failure_points:
            issues.append("Image validation failing")
            recommendations.append("Verify test image paths and file permissions")

        # Success rate analysis
        success_rate = test_results.get('success_rate', 0.0)
        if success_rate < 0.5:
            issues.append(f"Low pipeline success rate: {success_rate:.1%}")
            recommendations.append("Major integration issues need immediate attention")
        elif success_rate < 0.9:
            issues.append(f"Moderate pipeline success rate: {success_rate:.1%}")
            recommendations.append("Some integration issues need resolution")

        return {
            'issues_identified': issues,
            'recommendations': recommendations,
            'severity': 'critical' if success_rate < 0.5 else 'moderate' if success_rate < 0.9 else 'minor'
        }


def main():
    """Main test execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Test end-to-end classification pipeline")
    parser.add_argument('--image', type=str, help="Single image to test")
    parser.add_argument('--test-dir', type=str, help="Directory of test images")
    parser.add_argument('--max-images', type=int, default=10, help="Maximum images to test")

    args = parser.parse_args()

    tester = EndToEndPipelineTester()

    if args.image:
        # Test single image
        result = tester.test_complete_pipeline(args.image)
        print(f"\n🧪 Pipeline Test Results for {args.image}:")
        print(f"Success: {result['success']}")
        if not result['success']:
            print(f"Failure Point: {result['failure_point']}")
        print(f"Final Result: {result['final_result']}")

    elif args.test_dir:
        # Test directory of images
        test_dir = Path(args.test_dir)
        if test_dir.exists():
            test_images = list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg'))
            test_images = [str(img) for img in test_images[:args.max_images]]

            results = tester.test_diverse_dataset(test_images)
            diagnosis = tester.diagnose_integration_issues(results)

            print(f"\n🧪 Pipeline Test Summary:")
            print(f"Success Rate: {results['success_rate']:.1%}")
            print(f"Most Common Failure: {results['most_common_failure']}")
            print(f"Issues: {diagnosis['issues_identified']}")
            print(f"Recommendations: {diagnosis['recommendations']}")

        else:
            print(f"❌ Test directory not found: {test_dir}")
    else:
        # Default test
        default_images = [
            "data/logos/simple_geometric/circle_00.png",
            "data/logos/text_based/text_ai_04.png",
            "data/logos/gradients/gradient_radial_00.png"
        ]

        existing_images = [img for img in default_images if Path(img).exists()]

        if existing_images:
            results = tester.test_diverse_dataset(existing_images)
            diagnosis = tester.diagnose_integration_issues(results)

            print(f"\n🧪 Default Pipeline Test Summary:")
            print(f"Success Rate: {results['success_rate']:.1%}")
            print(f"Issues: {diagnosis['issues_identified']}")

        else:
            print("❌ No test images found. Please specify --image or --test-dir")


if __name__ == "__main__":
    main()