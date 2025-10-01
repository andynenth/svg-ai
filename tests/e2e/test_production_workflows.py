import requests
import pytest
import time
import base64
from pathlib import Path
import concurrent.futures
import threading

class TestProductionWorkflows:
    def __init__(self):
        self.base_url = "http://localhost"
        self.test_images = self._load_test_images()

    def _load_test_images(self):
        """Load various test images for comprehensive testing"""
        images = {}
        test_dir = Path('data/test')

        image_types = ['simple_geometric', 'text_based', 'gradient', 'complex']
        for img_type in image_types:
            img_files = list(test_dir.glob(f'{img_type}*.png'))
            if img_files:
                with open(img_files[0], 'rb') as f:
                    images[img_type] = base64.b64encode(f.read()).decode('utf-8')

        return images

    def test_complete_conversion_workflow(self):
        """Test complete conversion workflow"""
        for img_type, img_data in self.test_images.items():
            print(f"Testing {img_type} workflow...")

            # Step 1: Upload and convert
            start_time = time.time()
            response = requests.post(f"{self.base_url}/api/convert", json={
                'image': img_data,
                'format': 'png',
                'options': {
                    'optimize': True,
                    'quality_target': 0.9
                }
            })

            assert response.status_code == 200, f"Conversion failed for {img_type}"
            conversion_time = time.time() - start_time

            result = response.json()
            assert 'svg' in result, f"No SVG content for {img_type}"
            assert 'quality' in result, f"No quality metrics for {img_type}"

            # Validate performance targets
            if img_type == 'simple_geometric':
                assert conversion_time < 2.0, f"Simple conversion too slow: {conversion_time:.2f}s"
            elif img_type in ['text_based', 'gradient']:
                assert conversion_time < 5.0, f"Medium conversion too slow: {conversion_time:.2f}s"
            else:  # complex
                assert conversion_time < 15.0, f"Complex conversion too slow: {conversion_time:.2f}s"

            # Validate quality
            quality_score = result['quality'].get('ssim', 0)
            assert quality_score > 0.7, f"Quality too low for {img_type}: {quality_score}"

            print(f"✅ {img_type}: {conversion_time:.2f}s, quality={quality_score:.3f}")

    def test_batch_processing_workflow(self):
        """Test batch processing capabilities"""
        batch_data = {
            'images': [
                {'name': 'test1.png', 'data': list(self.test_images.values())[0]},
                {'name': 'test2.png', 'data': list(self.test_images.values())[1]},
            ]
        }

        start_time = time.time()
        response = requests.post(f"{self.base_url}/api/batch-convert", json=batch_data)
        batch_time = time.time() - start_time

        assert response.status_code == 200, "Batch processing failed"
        results = response.json()
        assert 'results' in results, "No batch results returned"
        assert len(results['results']) == 2, "Incorrect number of results"

        print(f"✅ Batch processing: {batch_time:.2f}s for 2 images")

    def test_error_recovery_workflow(self):
        """Test error handling and recovery"""
        # Test with invalid data
        response = requests.post(f"{self.base_url}/api/convert", json={
            'image': 'invalid-base64-data'
        })

        assert response.status_code == 400, "Error handling failed"
        error_result = response.json()
        assert 'error' in error_result, "No error message returned"

        print("✅ Error handling working correctly")

    def test_concurrent_load(self):
        """Test system under concurrent load"""
        import concurrent.futures
        import threading

        def single_request():
            response = requests.post(f"{self.base_url}/api/convert", json={
                'image': list(self.test_images.values())[0],
                'format': 'png'
            })
            return response.status_code == 200

        # Test with 20 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(single_request) for _ in range(20)]
            results = [future.result() for future in futures]

        success_rate = sum(results) / len(results)
        assert success_rate >= 0.9, f"Concurrent load test failed: {success_rate:.1%} success rate"

        print(f"✅ Concurrent load test: {success_rate:.1%} success rate")

    def test_health_endpoints(self):
        """Test system health endpoints"""
        # Test main health endpoint
        response = requests.get(f"{self.base_url}/health")
        assert response.status_code == 200, "Main health check failed"

        health_data = response.json()
        assert health_data.get('status') == 'healthy', "System not healthy"

        # Test classification status
        response = requests.get(f"{self.base_url}/api/classification-status")
        assert response.status_code == 200, "Classification status check failed"

        print("✅ Health endpoints working correctly")

    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Make rapid requests to trigger rate limiting
        responses = []
        for i in range(15):  # Exceed the 10 per minute limit
            response = requests.post(f"{self.base_url}/api/convert", json={
                'image': list(self.test_images.values())[0],
                'format': 'png'
            })
            responses.append(response.status_code)

        # Should get some 429 (Too Many Requests) responses
        rate_limited = any(status == 429 for status in responses)

        print(f"✅ Rate limiting test: {'Working' if rate_limited else 'Not triggered'}")

    def test_security_validation(self):
        """Test security input validation"""
        # Test path traversal attempt
        response = requests.post(f"{self.base_url}/api/convert", json={
            'image': list(self.test_images.values())[0],
            'filename': '../../../etc/passwd'
        })

        # Should reject malicious filename
        assert response.status_code in [400, 422], "Security validation failed"

        # Test script injection attempt
        response = requests.post(f"{self.base_url}/api/convert", json={
            'image': list(self.test_images.values())[0],
            'filename': '<script>alert("xss")</script>'
        })

        assert response.status_code in [400, 422], "Script injection not blocked"

        print("✅ Security validation working correctly")

    def test_memory_management(self):
        """Test memory management under load"""
        # Process multiple large images to test memory handling
        large_requests = []
        for i in range(5):
            response = requests.post(f"{self.base_url}/api/convert", json={
                'image': list(self.test_images.values())[0],  # Use largest available image
                'format': 'png',
                'options': {'optimize': True}
            })
            large_requests.append(response.status_code == 200)

        success_rate = sum(large_requests) / len(large_requests)
        assert success_rate >= 0.8, f"Memory management test failed: {success_rate:.1%} success rate"

        print(f"✅ Memory management test: {success_rate:.1%} success rate")

    def run_all_tests(self):
        """Run complete production validation suite"""
        tests = [
            self.test_health_endpoints,
            self.test_complete_conversion_workflow,
            self.test_batch_processing_workflow,
            self.test_error_recovery_workflow,
            self.test_concurrent_load,
            self.test_rate_limiting,
            self.test_security_validation,
            self.test_memory_management
        ]

        passed_tests = 0
        total_tests = len(tests)

        print("🚀 Starting Production Validation Suite")
        print("=" * 50)

        for test in tests:
            try:
                print(f"\n📋 Running {test.__name__}...")
                test()
                passed_tests += 1
            except Exception as e:
                print(f"❌ Test {test.__name__} failed: {e}")
                continue

        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            print("🎉 All production validation tests PASSED!")
            return True
        else:
            print("⚠️  Some production validation tests FAILED!")
            return False

# Main execution
if __name__ == "__main__":
    validator = TestProductionWorkflows()
    success = validator.run_all_tests()

    if success:
        print("\n✅ Production system ready for launch!")
        exit(0)
    else:
        print("\n❌ Production system not ready - fix issues before launch!")
        exit(1)