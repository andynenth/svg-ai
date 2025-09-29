#!/usr/bin/env python3
"""
Classification Debug Script - Day 1 Debugging

Step-by-step debugging of logo classification pipeline to identify
root causes of empty results and classification failures.
"""

import sys
import os
import logging
import traceback
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ai_modules.feature_extraction import ImageFeatureExtractor
from ai_modules.rule_based_classifier import RuleBasedClassifier


class ClassificationDebugger:
    """Comprehensive debugging tool for classification pipeline"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.feature_extractor = ImageFeatureExtractor(log_level="DEBUG")
        self.classifier = RuleBasedClassifier()
        self.debug_results = []

    def _setup_logging(self) -> logging.Logger:
        """Setup detailed logging for debugging"""
        logger = logging.getLogger('ClassificationDebugger')
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            # File handler for detailed logs
            log_file = Path(__file__).parent / 'debug_classification.log'
            file_handler = logging.FileHandler(log_file, mode='w')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        return logger

    def debug_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Debug classification for a single image with detailed step-by-step analysis

        Args:
            image_path: Path to image file

        Returns:
            Comprehensive debug results
        """
        debug_result = {
            'image_path': image_path,
            'success': False,
            'error': None,
            'steps': {},
            'issues_found': [],
            'recommendations': []
        }

        self.logger.info(f"🔍 Starting debug analysis for: {image_path}")

        try:
            # Step 1: Validate image file
            debug_result['steps']['file_validation'] = self._debug_file_validation(image_path)

            # Step 2: Feature extraction
            debug_result['steps']['feature_extraction'] = self._debug_feature_extraction(image_path)

            # Step 3: Feature validation
            features = debug_result['steps']['feature_extraction'].get('features', {})
            debug_result['steps']['feature_validation'] = self._debug_feature_validation(features)

            # Step 4: Classification process
            debug_result['steps']['classification'] = self._debug_classification_process(features)

            # Step 5: Result validation
            classification_result = debug_result['steps']['classification'].get('result')
            debug_result['steps']['result_validation'] = self._debug_result_validation(classification_result)

            # Analyze all steps for issues
            debug_result['issues_found'] = self._analyze_debug_results(debug_result['steps'])
            debug_result['recommendations'] = self._generate_recommendations(debug_result['issues_found'])

            debug_result['success'] = len(debug_result['issues_found']) == 0

        except Exception as e:
            self.logger.error(f"❌ Debug analysis failed: {e}")
            self.logger.error(traceback.format_exc())
            debug_result['error'] = str(e)
            debug_result['issues_found'].append(f"Critical error: {e}")

        return debug_result

    def _debug_file_validation(self, image_path: str) -> Dict[str, Any]:
        """Debug Step 1: Validate image file exists and is readable"""
        result = {
            'step': 'file_validation',
            'success': False,
            'details': {}
        }

        try:
            path_obj = Path(image_path)
            result['details']['exists'] = path_obj.exists()
            result['details']['is_file'] = path_obj.is_file()
            result['details']['readable'] = os.access(image_path, os.R_OK)
            result['details']['size_bytes'] = path_obj.stat().st_size if path_obj.exists() else 0

            result['success'] = all([
                result['details']['exists'],
                result['details']['is_file'],
                result['details']['readable'],
                result['details']['size_bytes'] > 0
            ])

            if result['success']:
                self.logger.debug(f"✅ File validation passed for {image_path}")
            else:
                self.logger.warning(f"⚠️ File validation issues: {result['details']}")

        except Exception as e:
            self.logger.error(f"❌ File validation failed: {e}")
            result['error'] = str(e)

        return result

    def _debug_feature_extraction(self, image_path: str) -> Dict[str, Any]:
        """Debug Step 2: Feature extraction with detailed analysis"""
        result = {
            'step': 'feature_extraction',
            'success': False,
            'features': {},
            'details': {}
        }

        try:
            self.logger.debug(f"🔧 Starting feature extraction for {image_path}")

            # Extract features
            features = self.feature_extractor.extract_features(image_path)
            result['features'] = features

            # Analyze extraction details
            result['details']['feature_count'] = len(features)
            result['details']['feature_names'] = list(features.keys())
            result['details']['has_all_expected'] = self._check_expected_features(features)

            result['success'] = (
                len(features) > 0 and
                result['details']['has_all_expected']
            )

            if result['success']:
                self.logger.debug(f"✅ Feature extraction successful: {len(features)} features")
                for name, value in features.items():
                    self.logger.debug(f"   {name}: {value:.4f}")
            else:
                self.logger.warning(f"⚠️ Feature extraction issues: {result['details']}")

        except Exception as e:
            self.logger.error(f"❌ Feature extraction failed: {e}")
            result['error'] = str(e)

        return result

    def _check_expected_features(self, features: Dict[str, float]) -> bool:
        """Check if all expected features are present"""
        expected_features = [
            'edge_density', 'unique_colors', 'entropy',
            'corner_density', 'gradient_strength', 'complexity_score'
        ]

        missing_features = [f for f in expected_features if f not in features]
        if missing_features:
            self.logger.warning(f"⚠️ Missing expected features: {missing_features}")
            return False

        return True

    def _debug_feature_validation(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Debug Step 3: Validate feature values are in expected ranges"""
        result = {
            'step': 'feature_validation',
            'success': False,
            'details': {}
        }

        try:
            issues = []

            for name, value in features.items():
                feature_issues = []

                # Check for invalid values
                if value is None:
                    feature_issues.append("None value")
                elif np.isnan(value):
                    feature_issues.append("NaN value")
                elif np.isinf(value):
                    feature_issues.append("Infinite value")
                elif not (0.0 <= value <= 1.0):
                    feature_issues.append(f"Out of range [0,1]: {value}")

                if feature_issues:
                    issues.append(f"{name}: {', '.join(feature_issues)}")

            result['details']['validation_issues'] = issues
            result['details']['total_features'] = len(features)
            result['details']['invalid_features'] = len(issues)

            result['success'] = len(issues) == 0

            if result['success']:
                self.logger.debug("✅ Feature validation passed - all values in [0,1] range")
            else:
                self.logger.warning(f"⚠️ Feature validation issues: {issues}")

        except Exception as e:
            self.logger.error(f"❌ Feature validation failed: {e}")
            result['error'] = str(e)

        return result

    def _debug_classification_process(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Debug Step 4: Classification process with intermediate results"""
        result = {
            'step': 'classification',
            'success': False,
            'result': None,
            'details': {}
        }

        try:
            self.logger.debug("🤖 Starting classification process")

            # Test both classification methods

            # Method 1: Basic classify()
            try:
                basic_result = self.classifier.classify(features)
                result['details']['basic_classify'] = {
                    'result': basic_result,
                    'type': type(basic_result).__name__,
                    'success': basic_result is not None
                }
                self.logger.debug(f"Basic classify() returned: {basic_result} (type: {type(basic_result)})")
            except Exception as e:
                result['details']['basic_classify'] = {'error': str(e)}
                self.logger.error(f"Basic classify() failed: {e}")

            # Method 2: Detailed classify_with_details()
            try:
                detailed_result = self.classifier.classify_with_details(features)
                result['details']['detailed_classify'] = {
                    'result': detailed_result,
                    'success': detailed_result is not None
                }
                self.logger.debug(f"Detailed classify() returned: {detailed_result}")
            except Exception as e:
                result['details']['detailed_classify'] = {'error': str(e)}
                self.logger.error(f"Detailed classify() failed: {e}")

            # Use basic result as primary (since that's what integration expects)
            basic_result = result['details']['basic_classify'].get('result')
            if basic_result:
                result['result'] = basic_result
                result['success'] = True

            # Analyze return format
            if basic_result:
                result['details']['return_format'] = {
                    'is_tuple': isinstance(basic_result, tuple),
                    'is_dict': isinstance(basic_result, dict),
                    'length': len(basic_result) if hasattr(basic_result, '__len__') else 0,
                    'expected_format': 'Should be dict with logo_type, confidence, reasoning'
                }

        except Exception as e:
            self.logger.error(f"❌ Classification process failed: {e}")
            result['error'] = str(e)

        return result

    def _debug_result_validation(self, classification_result: Any) -> Dict[str, Any]:
        """Debug Step 5: Validate classification result format"""
        result = {
            'step': 'result_validation',
            'success': False,
            'details': {}
        }

        try:
            if classification_result is None:
                result['details']['issue'] = "Classification result is None"
                return result

            # Check result format
            result['details']['actual_type'] = type(classification_result).__name__
            result['details']['expected_type'] = 'dict'

            if isinstance(classification_result, tuple):
                result['details']['tuple_analysis'] = {
                    'length': len(classification_result),
                    'elements': [str(elem) for elem in classification_result]
                }
                result['details']['issue'] = "Result is tuple, expected dict format"

            elif isinstance(classification_result, dict):
                # Check required fields
                required_fields = ['logo_type', 'confidence', 'reasoning']
                present_fields = list(classification_result.keys())
                missing_fields = [f for f in required_fields if f not in present_fields]

                result['details']['dict_analysis'] = {
                    'present_fields': present_fields,
                    'missing_fields': missing_fields,
                    'has_all_required': len(missing_fields) == 0
                }

                result['success'] = len(missing_fields) == 0

            else:
                result['details']['issue'] = f"Unexpected result type: {type(classification_result)}"

            if result['success']:
                self.logger.debug("✅ Result validation passed")
            else:
                self.logger.warning(f"⚠️ Result validation issues: {result['details']}")

        except Exception as e:
            self.logger.error(f"❌ Result validation failed: {e}")
            result['error'] = str(e)

        return result

    def _analyze_debug_results(self, steps: Dict[str, Any]) -> List[str]:
        """Analyze all debug steps to identify issues"""
        issues = []

        for step_name, step_result in steps.items():
            if not step_result.get('success', False):
                issues.append(f"{step_name}: Failed")

            if 'error' in step_result:
                issues.append(f"{step_name}: {step_result['error']}")

        # Specific issue checks
        classification_step = steps.get('classification', {})
        if classification_step.get('details', {}).get('return_format', {}).get('is_tuple'):
            issues.append("CRITICAL: classify() returns tuple instead of required dict format")

        return issues

    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """Generate specific recommendations based on identified issues"""
        recommendations = []

        for issue in issues:
            if "tuple instead of required dict" in issue:
                recommendations.append("Fix classify() method to return dict with logo_type, confidence, reasoning")
            elif "Feature validation" in issue:
                recommendations.append("Add input validation for feature values (NaN, infinity, range checks)")
            elif "Feature extraction" in issue:
                recommendations.append("Check feature extraction pipeline for missing or invalid features")
            elif "file_validation" in issue:
                recommendations.append("Verify input file exists and is readable")

        if not recommendations:
            recommendations.append("No specific issues found - check integration points")

        return recommendations

    def test_known_images(self, test_images: List[str]) -> Dict[str, Any]:
        """Test classification on known good images"""
        self.logger.info(f"🧪 Testing {len(test_images)} known images")

        test_results = {
            'total_tests': len(test_images),
            'successful_tests': 0,
            'failed_tests': 0,
            'results': [],
            'summary': {}
        }

        for image_path in test_images:
            self.logger.info(f"Testing: {image_path}")
            debug_result = self.debug_single_image(image_path)
            test_results['results'].append(debug_result)

            if debug_result['success']:
                test_results['successful_tests'] += 1
                self.logger.info("✅ PASSED")
            else:
                test_results['failed_tests'] += 1
                self.logger.error(f"❌ FAILED: {debug_result['issues_found']}")

        # Generate summary
        test_results['summary'] = {
            'success_rate': test_results['successful_tests'] / test_results['total_tests'],
            'common_issues': self._find_common_issues(test_results['results']),
            'overall_recommendations': self._generate_overall_recommendations(test_results['results'])
        }

        return test_results

    def _find_common_issues(self, results: List[Dict]) -> List[str]:
        """Find issues that appear across multiple test cases"""
        issue_counts = {}

        for result in results:
            for issue in result.get('issues_found', []):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        # Return issues that appear in more than 50% of cases
        threshold = len(results) * 0.5
        common_issues = [issue for issue, count in issue_counts.items() if count >= threshold]

        return common_issues

    def _generate_overall_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate overall recommendations based on all test results"""
        all_recommendations = []

        for result in results:
            all_recommendations.extend(result.get('recommendations', []))

        # Remove duplicates and prioritize
        unique_recommendations = list(set(all_recommendations))

        return unique_recommendations

    def save_debug_report(self, results: Dict[str, Any], output_file: str = None):
        """Save comprehensive debug report"""
        if output_file is None:
            output_file = Path(__file__).parent / 'classification_debug_report.json'

        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            self.logger.info(f"📄 Debug report saved to: {output_file}")

        except Exception as e:
            self.logger.error(f"Failed to save debug report: {e}")


def main():
    """Main debugging execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Debug logo classification pipeline")
    parser.add_argument('--image', type=str, help="Single image to debug")
    parser.add_argument('--test-dir', type=str, help="Directory of test images")
    parser.add_argument('--save-report', action='store_true', help="Save detailed report")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")

    args = parser.parse_args()

    debugger = ClassificationDebugger(verbose=args.verbose)

    if args.image:
        # Debug single image
        result = debugger.debug_single_image(args.image)
        print(f"\n🔍 Debug Results for {args.image}:")
        print(f"Success: {result['success']}")
        print(f"Issues: {result['issues_found']}")
        print(f"Recommendations: {result['recommendations']}")

        if args.save_report:
            debugger.save_debug_report({'single_image': result})

    elif args.test_dir:
        # Test directory of images
        test_dir = Path(args.test_dir)
        if test_dir.exists():
            test_images = list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg'))
            test_images = [str(img) for img in test_images[:10]]  # Limit to 10 for debugging

            results = debugger.test_known_images(test_images)
            print(f"\n🧪 Test Results:")
            print(f"Success Rate: {results['summary']['success_rate']:.1%}")
            print(f"Common Issues: {results['summary']['common_issues']}")

            if args.save_report:
                debugger.save_debug_report(results)
        else:
            print(f"❌ Test directory not found: {test_dir}")
    else:
        # Default test with sample images
        sample_images = [
            "data/logos/simple_geometric/circle_00.png",
            "data/logos/text_based/company_logo_01.png"
        ]

        # Filter to existing images
        existing_images = [img for img in sample_images if Path(img).exists()]

        if existing_images:
            results = debugger.test_known_images(existing_images)
            print(f"\n🧪 Default Test Results:")
            print(f"Success Rate: {results['summary']['success_rate']:.1%}")
            print(f"Common Issues: {results['summary']['common_issues']}")

            if args.save_report:
                debugger.save_debug_report(results)
        else:
            print("❌ No test images found. Please specify --image or --test-dir")


if __name__ == "__main__":
    main()