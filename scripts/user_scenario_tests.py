#!/usr/bin/env python3
"""
User Scenario Testing for Classification System
Tests user acceptance scenarios as specified in Day 9 plan
"""

import requests
import time
from pathlib import Path

# User Scenario Test Cases
USER_SCENARIOS = {
    'scenario_1_quick_classification': {
        'description': 'User wants quick logo type identification',
        'steps': [
            'Upload simple geometric logo',
            'Select "Rule-Based (Fast)" method',
            'Verify classification in <0.5s',
            'Check confidence score >0.8'
        ],
        'expected_outcome': 'Fast, accurate classification'
    },

    'scenario_2_detailed_analysis': {
        'description': 'User wants detailed logo analysis',
        'steps': [
            'Upload complex logo',
            'Select "Auto" method',
            'Enable "Show detailed features"',
            'Review classification and features'
        ],
        'expected_outcome': 'Comprehensive analysis with features'
    },

    'scenario_3_ai_enhanced_conversion': {
        'description': 'User wants best quality SVG conversion',
        'steps': [
            'Upload gradient logo',
            'Enable "Use AI-optimized conversion"',
            'Start conversion',
            'Compare result with standard conversion'
        ],
        'expected_outcome': 'Better quality SVG with AI optimization'
    },

    'scenario_4_batch_processing': {
        'description': 'User wants to process multiple logos',
        'steps': [
            'Select multiple logo files',
            'Use batch classification endpoint',
            'Review all results',
            'Check processing efficiency'
        ],
        'expected_outcome': 'Efficient batch processing'
    },

    'scenario_5_error_recovery': {
        'description': 'User uploads invalid file',
        'steps': [
            'Upload non-image file',
            'Attempt classification',
            'Observe error message',
            'Try with valid image'
        ],
        'expected_outcome': 'Clear error message, easy recovery'
    }
}

class UserScenarioTester:
    def __init__(self, base_url="http://localhost:8001/api"):
        self.base_url = base_url

    def test_scenario_1_quick_classification(self):
        """Test quick classification scenario"""
        print("Testing Scenario 1: Quick Classification")

        try:
            # Upload simple geometric logo
            image_path = 'data/test/simple_geometric_logo.png'
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {'method': 'rule_based'}  # Rule-Based (Fast) method

                start_time = time.time()
                response = requests.post(
                    f'{self.base_url}/classify-logo',
                    files=files,
                    data=data
                )
                processing_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                # Verify classification in <0.5s
                time_ok = processing_time < 0.5

                # Check confidence score >0.8
                confidence_ok = result.get('confidence', 0) > 0.8

                print(f"  ✅ Processing time: {processing_time:.3f}s (target: <0.5s) - {'PASS' if time_ok else 'FAIL'}")
                print(f"  ✅ Confidence: {result.get('confidence', 0):.2f} (target: >0.8) - {'PASS' if confidence_ok else 'FAIL'}")
                print(f"  ✅ Method used: {result.get('method_used', 'unknown')}")

                return time_ok and confidence_ok
            else:
                print(f"  ❌ Request failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False

    def test_scenario_2_detailed_analysis(self):
        """Test detailed analysis scenario"""
        print("Testing Scenario 2: Detailed Analysis")

        try:
            # Upload complex logo
            image_path = 'data/test/complex_logo.png'
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {
                    'method': 'auto',
                    'include_features': 'true'
                }

                response = requests.post(
                    f'{self.base_url}/classify-logo',
                    files=files,
                    data=data
                )

            if response.status_code == 200:
                result = response.json()

                # Check for comprehensive analysis
                has_classification = 'logo_type' in result and 'confidence' in result
                has_features = 'features' in result and result['features']
                has_method_info = 'method_used' in result

                print(f"  ✅ Classification: {result.get('logo_type', 'unknown')} (confidence: {result.get('confidence', 0):.2f})")
                print(f"  ✅ Features included: {'PASS' if has_features else 'FAIL'}")
                print(f"  ✅ Method used: {result.get('method_used', 'unknown')}")

                if has_features:
                    features = result['features']
                    print(f"  ✅ Feature count: {len(features)}")
                    for feature, value in list(features.items())[:3]:  # Show first 3
                        print(f"    {feature}: {value:.3f}")

                return has_classification and has_features and has_method_info
            else:
                print(f"  ❌ Request failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False

    def test_scenario_3_ai_enhanced_conversion(self):
        """Test AI-enhanced conversion scenario"""
        print("Testing Scenario 3: AI-Enhanced Conversion")

        try:
            # Upload gradient logo
            image_path = 'data/test/gradient_logo.png'

            # First upload to get file_id
            with open(image_path, 'rb') as f:
                upload_response = requests.post(
                    f'{self.base_url}/upload',
                    files={'file': f}
                )

            if upload_response.status_code != 200:
                print(f"  ❌ Upload failed: {upload_response.status_code}")
                return False

            file_id = upload_response.json()['file_id']

            # AI-enhanced conversion
            conversion_data = {
                'file_id': file_id,
                'use_ai': True,
                'ai_method': 'auto'
            }

            response = requests.post(
                f'{self.base_url}/convert',
                json=conversion_data,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                result = response.json()

                # Check AI enhancement
                ai_enhanced = result.get('ai_enhanced', False)
                has_ai_analysis = 'ai_analysis' in result
                has_svg = 'svg_content' in result and len(result['svg_content']) > 100

                print(f"  ✅ AI Enhanced: {'PASS' if ai_enhanced else 'FAIL'}")
                print(f"  ✅ AI Analysis: {'PASS' if has_ai_analysis else 'FAIL'}")
                print(f"  ✅ SVG Generated: {'PASS' if has_svg else 'FAIL'}")

                if has_ai_analysis:
                    ai_analysis = result['ai_analysis']
                    print(f"  ✅ Classified as: {ai_analysis.get('logo_type', 'unknown')}")
                    print(f"  ✅ Confidence: {ai_analysis.get('confidence', 0):.2f}")

                return ai_enhanced and has_ai_analysis and has_svg
            else:
                print(f"  ❌ Conversion failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False

    def test_scenario_4_batch_processing(self):
        """Test batch processing scenario"""
        print("Testing Scenario 4: Batch Processing")

        try:
            # Prepare multiple files
            image_paths = [
                'data/test/simple_geometric_logo.png',
                'data/test/text_based_logo.png',
                'data/test/gradient_logo.png'
            ]

            files = []
            for path in image_paths:
                files.append(('images', open(path, 'rb')))

            data = {'method': 'auto'}

            start_time = time.time()
            response = requests.post(
                f'{self.base_url}/classify-batch',
                files=files,
                data=data
            )
            processing_time = time.time() - start_time

            # Close all files
            for _, f in files:
                f.close()

            if response.status_code == 200:
                result = response.json()

                # Check batch processing
                total_images = result.get('total_images', 0)
                results_count = len(result.get('results', []))
                success = result.get('success', False)

                print(f"  ✅ Total images: {total_images}")
                print(f"  ✅ Results count: {results_count}")
                print(f"  ✅ Processing time: {processing_time:.3f}s")
                print(f"  ✅ Success: {'PASS' if success else 'FAIL'}")

                # Show individual results
                if 'results' in result:
                    for i, res in enumerate(result['results'][:3]):  # Show first 3
                        print(f"    Image {i+1}: {res.get('logo_type', 'unknown')} "
                              f"(confidence: {res.get('confidence', 0):.2f})")

                return success and total_images == results_count
            else:
                print(f"  ❌ Batch request failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False

    def test_scenario_5_error_recovery(self):
        """Test error recovery scenario"""
        print("Testing Scenario 5: Error Recovery")

        try:
            # Create a fake non-image file
            fake_file_content = b"This is not an image file"

            files = {'image': ('fake.txt', fake_file_content, 'text/plain')}
            data = {'method': 'auto'}

            response = requests.post(
                f'{self.base_url}/classify-logo',
                files=files,
                data=data
            )

            # Should fail with clear error message
            error_handled = response.status_code == 400

            if error_handled:
                error_response = response.json()
                has_error_message = 'error' in error_response

                print(f"  ✅ Error status: {response.status_code} - {'PASS' if error_handled else 'FAIL'}")
                print(f"  ✅ Error message: {'PASS' if has_error_message else 'FAIL'}")

                if has_error_message:
                    print(f"    Message: {error_response['error']}")

                # Now test with valid image for recovery
                image_path = 'data/test/simple_geometric_logo.png'
                with open(image_path, 'rb') as f:
                    files = {'image': f}

                    recovery_response = requests.post(
                        f'{self.base_url}/classify-logo',
                        files=files,
                        data=data
                    )

                recovery_success = recovery_response.status_code == 200
                print(f"  ✅ Recovery with valid image: {'PASS' if recovery_success else 'FAIL'}")

                return error_handled and has_error_message and recovery_success
            else:
                print(f"  ❌ Expected error not handled properly: {response.status_code}")
                return False

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False

def run_user_scenario_tests():
    """Execute all user scenario tests"""
    print("=" * 60)
    print("USER ACCEPTANCE TESTING - SCENARIO VALIDATION")
    print("=" * 60)

    tester = UserScenarioTester()

    scenarios = [
        ('scenario_1_quick_classification', tester.test_scenario_1_quick_classification),
        ('scenario_2_detailed_analysis', tester.test_scenario_2_detailed_analysis),
        ('scenario_3_ai_enhanced_conversion', tester.test_scenario_3_ai_enhanced_conversion),
        ('scenario_4_batch_processing', tester.test_scenario_4_batch_processing),
        ('scenario_5_error_recovery', tester.test_scenario_5_error_recovery)
    ]

    passed = 0
    total = len(scenarios)

    for scenario_id, test_func in scenarios:
        scenario_info = USER_SCENARIOS[scenario_id]
        print(f"\n--- {scenario_id.replace('_', ' ').title()} ---")
        print(f"Description: {scenario_info['description']}")
        print(f"Expected outcome: {scenario_info['expected_outcome']}")
        print()

        try:
            if test_func():
                print(f"✅ Scenario PASSED")
                passed += 1
            else:
                print(f"❌ Scenario FAILED")

        except Exception as e:
            print(f"❌ Scenario failed with error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("USER SCENARIO TESTING SUMMARY")
    print("=" * 60)
    print(f"Scenarios passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("🎉 All user scenarios PASSED! System ready for users.")
        return True
    else:
        print("⚠️  Some scenarios FAILED. User experience needs improvement.")
        return False

if __name__ == "__main__":
    import sys
    success = run_user_scenario_tests()
    sys.exit(0 if success else 1)