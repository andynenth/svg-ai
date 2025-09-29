#!/usr/bin/env python3
"""
Classification API Testing Suite
Tests the newly implemented classification endpoints for Day 8 integration
"""

import requests
import json
import time
import sys
from pathlib import Path
import tempfile
from PIL import Image
import numpy as np

class ClassificationAPITester:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.test_results = []

    def create_test_image(self):
        """Create a simple test image for testing"""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='blue')
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(temp_file.name)
        return temp_file.name

    def test_health_check(self):
        """Test basic health check endpoint"""
        print("Testing health check endpoint...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            success = response.status_code == 200
            self.test_results.append({
                'test': 'health_check',
                'success': success,
                'status_code': response.status_code,
                'response': response.json() if success else str(response.text)
            })
            print(f"✅ Health check: {'PASS' if success else 'FAIL'}")
            return success
        except Exception as e:
            print(f"❌ Health check: FAIL - {e}")
            self.test_results.append({
                'test': 'health_check',
                'success': False,
                'error': str(e)
            })
            return False

    def test_classification_status(self):
        """Test classification system status endpoint"""
        print("Testing classification status endpoint...")
        try:
            response = requests.get(f"{self.base_url}/api/classification-status", timeout=10)
            success = response.status_code == 200
            result = response.json() if success else None

            self.test_results.append({
                'test': 'classification_status',
                'success': success,
                'status_code': response.status_code,
                'response': result
            })

            if success:
                status = result.get('status', 'unknown')
                print(f"✅ Classification status: PASS - Status: {status}")
                if result.get('methods_available'):
                    methods = result['methods_available']
                    print(f"   Available methods: {methods}")
            else:
                print(f"❌ Classification status: FAIL - {response.status_code}")

            return success

        except Exception as e:
            print(f"❌ Classification status: FAIL - {e}")
            self.test_results.append({
                'test': 'classification_status',
                'success': False,
                'error': str(e)
            })
            return False

    def test_classify_logo(self):
        """Test logo classification endpoint"""
        print("Testing logo classification endpoint...")
        test_image_path = self.create_test_image()

        try:
            with open(test_image_path, 'rb') as f:
                files = {'image': f}
                data = {
                    'method': 'auto',
                    'include_features': 'true'
                }

                response = requests.post(
                    f"{self.base_url}/api/classify-logo",
                    files=files,
                    data=data,
                    timeout=30
                )

            success = response.status_code == 200
            result = response.json() if success else None

            self.test_results.append({
                'test': 'classify_logo',
                'success': success,
                'status_code': response.status_code,
                'response': result
            })

            if success:
                logo_type = result.get('logo_type', 'unknown')
                confidence = result.get('confidence', 0)
                method_used = result.get('method_used', 'unknown')
                processing_time = result.get('processing_time', 0)

                print(f"✅ Logo classification: PASS")
                print(f"   Logo type: {logo_type}")
                print(f"   Confidence: {confidence:.2f}")
                print(f"   Method: {method_used}")
                print(f"   Processing time: {processing_time:.3f}s")
            else:
                print(f"❌ Logo classification: FAIL - {response.status_code}")
                if result:
                    print(f"   Error: {result.get('error', 'Unknown error')}")

            return success

        except Exception as e:
            print(f"❌ Logo classification: FAIL - {e}")
            self.test_results.append({
                'test': 'classify_logo',
                'success': False,
                'error': str(e)
            })
            return False

        finally:
            # Cleanup test image
            try:
                Path(test_image_path).unlink()
            except:
                pass

    def test_analyze_features(self):
        """Test feature analysis endpoint"""
        print("Testing feature analysis endpoint...")
        test_image_path = self.create_test_image()

        try:
            with open(test_image_path, 'rb') as f:
                files = {'image': f}

                response = requests.post(
                    f"{self.base_url}/api/analyze-logo-features",
                    files=files,
                    timeout=30
                )

            success = response.status_code == 200
            result = response.json() if success else None

            self.test_results.append({
                'test': 'analyze_features',
                'success': success,
                'status_code': response.status_code,
                'response': result
            })

            if success:
                features = result.get('features', {})
                print(f"✅ Feature analysis: PASS")
                print(f"   Features extracted: {len(features)}")
                for feature, value in list(features.items())[:3]:  # Show first 3
                    print(f"   {feature}: {value:.3f}")
            else:
                print(f"❌ Feature analysis: FAIL - {response.status_code}")
                if result:
                    print(f"   Error: {result.get('error', 'Unknown error')}")

            return success

        except Exception as e:
            print(f"❌ Feature analysis: FAIL - {e}")
            self.test_results.append({
                'test': 'analyze_features',
                'success': False,
                'error': str(e)
            })
            return False

        finally:
            # Cleanup test image
            try:
                Path(test_image_path).unlink()
            except:
                pass

    def test_ai_conversion(self):
        """Test AI-enhanced conversion endpoint"""
        print("Testing AI-enhanced conversion...")

        # First upload a test image
        test_image_path = self.create_test_image()

        try:
            # Upload the image first
            with open(test_image_path, 'rb') as f:
                upload_response = requests.post(
                    f"{self.base_url}/api/upload",
                    files={'file': f},
                    timeout=30
                )

            if upload_response.status_code != 200:
                print(f"❌ AI conversion: FAIL - Upload failed: {upload_response.status_code}")
                return False

            file_id = upload_response.json().get('file_id')
            if not file_id:
                print(f"❌ AI conversion: FAIL - No file_id returned")
                return False

            # Test AI conversion
            conversion_data = {
                'file_id': file_id,
                'use_ai': True,
                'ai_method': 'auto'
            }

            response = requests.post(
                f"{self.base_url}/api/convert",
                json=conversion_data,
                headers={'Content-Type': 'application/json'},
                timeout=60
            )

            success = response.status_code == 200
            result = response.json() if success else None

            self.test_results.append({
                'test': 'ai_conversion',
                'success': success,
                'status_code': response.status_code,
                'response': result
            })

            if success:
                ai_enhanced = result.get('ai_enhanced', False)
                processing_time = result.get('processing_time', 0)
                ai_analysis = result.get('ai_analysis', {})

                print(f"✅ AI conversion: PASS")
                print(f"   AI enhanced: {ai_enhanced}")
                print(f"   Processing time: {processing_time:.3f}s")
                if ai_analysis:
                    logo_type = ai_analysis.get('logo_type', 'unknown')
                    confidence = ai_analysis.get('confidence', 0)
                    print(f"   AI analysis - Type: {logo_type}, Confidence: {confidence:.2f}")
            else:
                print(f"❌ AI conversion: FAIL - {response.status_code}")
                if result:
                    print(f"   Error: {result.get('error', 'Unknown error')}")

            return success

        except Exception as e:
            print(f"❌ AI conversion: FAIL - {e}")
            self.test_results.append({
                'test': 'ai_conversion',
                'success': False,
                'error': str(e)
            })
            return False

        finally:
            # Cleanup test image
            try:
                Path(test_image_path).unlink()
            except:
                pass

    def run_all_tests(self):
        """Run all API tests"""
        print("=" * 60)
        print("Classification API Testing Suite")
        print("=" * 60)

        start_time = time.time()

        # Run tests
        tests = [
            ('Basic Health Check', self.test_health_check),
            ('Classification Status', self.test_classification_status),
            ('Logo Classification', self.test_classify_logo),
            ('Feature Analysis', self.test_analyze_features),
            ('AI-Enhanced Conversion', self.test_ai_conversion)
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            print(f"\n--- {test_name} ---")
            if test_func():
                passed += 1

        # Summary
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Tests passed: {passed}/{total}")
        print(f"Success rate: {(passed/total)*100:.1f}%")
        print(f"Total time: {total_time:.2f}s")

        if passed == total:
            print("🎉 All tests PASSED! API is ready for production.")
            return True
        else:
            print("⚠️  Some tests FAILED. Check the results above.")
            return False

    def save_results(self, filename="classification_api_test_results.json"):
        """Save test results to file"""
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\nTest results saved to {filename}")

if __name__ == "__main__":
    # Check if custom URL provided
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"

    tester = ClassificationAPITester(base_url)
    success = tester.run_all_tests()
    tester.save_results()

    sys.exit(0 if success else 1)