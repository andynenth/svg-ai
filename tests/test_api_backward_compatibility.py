# tests/test_api_backward_compatibility.py
import pytest
import time
import io
import os
from PIL import Image
import numpy as np

class TestAPIBackwardCompatibility:
    """Ensure new AI endpoints don't break existing functionality"""

    def setup_method(self):
        """Setup test client with existing Flask app"""
        from backend.app import app
        self.client = app.test_client()

    def create_test_png(self):
        """Create a test PNG file for uploads"""
        # Create a simple test image (100x100 white square)
        img = Image.new('RGB', (100, 100), color='white')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr

    def upload_test_file(self):
        """Upload a test file and return file_id"""
        test_image = self.create_test_png()

        response = self.client.post('/api/upload',
                                  data={'file': (test_image, 'test.png')},
                                  content_type='multipart/form-data')

        assert response.status_code == 200
        result = response.get_json()
        return result['file_id']

    def test_existing_convert_endpoint_unchanged(self):
        """Test that /api/convert works exactly as before"""
        # First upload a test file
        file_id = self.upload_test_file()

        # Test data - should work exactly as before AI enhancement
        test_data = {
            'file_id': file_id,
            'converter': 'vtracer',
            'color_precision': 4,
            'corner_threshold': 30
        }

        response = self.client.post('/api/convert',
                                  json=test_data,
                                  content_type='application/json')

        # Should work exactly as before
        assert response.status_code == 200
        result = response.get_json()

        # Basic response structure should be unchanged
        assert 'success' in result
        assert 'svg' in result

        # Should NOT contain AI metadata in basic endpoint
        assert 'ai_metadata' not in result

    def test_existing_upload_endpoint_unchanged(self):
        """Test that /api/upload works exactly as before"""
        # Create test image file
        test_image = self.create_test_png()

        response = self.client.post('/api/upload',
                                  data={'file': (test_image, 'test.png')},
                                  content_type='multipart/form-data')

        assert response.status_code == 200
        result = response.get_json()

        # Response structure should be exactly the same
        assert 'file_id' in result
        assert 'filename' in result
        assert 'path' in result

        # Should NOT contain AI-related fields
        assert 'ai_analysis' not in result

    def test_new_ai_endpoints_isolated(self):
        """Test that new AI endpoints don't interfere with existing ones"""
        # Test both endpoints with same file
        file_id = self.upload_test_file()

        # Basic conversion
        basic_response = self.client.post('/api/convert',
                                        json={'file_id': file_id, 'converter': 'vtracer'},
                                        content_type='application/json')

        # AI conversion
        ai_response = self.client.post('/api/convert-ai',
                                     json={'file_id': file_id, 'tier': 1},
                                     content_type='application/json')

        # Both should succeed (AI might return 503 if unavailable)
        assert basic_response.status_code == 200
        assert ai_response.status_code in [200, 503]  # 503 if AI unavailable

        # Basic response should remain unchanged
        basic_result = basic_response.get_json()
        assert 'ai_metadata' not in basic_result

        # AI response should have additional metadata if successful
        if ai_response.status_code == 200:
            ai_result = ai_response.get_json()
            if ai_result.get('success'):
                assert 'ai_metadata' in ai_result

    def test_performance_regression(self):
        """Ensure existing endpoints don't become slower"""
        file_id = self.upload_test_file()

        # Time basic conversion
        start_time = time.time()
        response = self.client.post('/api/convert',
                                  json={'file_id': file_id, 'converter': 'vtracer'},
                                  content_type='application/json')
        basic_time = time.time() - start_time

        assert response.status_code == 200

        # Should complete in reasonable time (baseline + small overhead)
        assert basic_time < 5.0, f"Basic conversion took {basic_time:.2f}s, too slow"

    def test_health_endpoint_unchanged(self):
        """Test that health endpoint still works as expected"""
        response = self.client.get('/health')
        assert response.status_code == 200

        result = response.get_json()
        assert 'status' in result

        # Health endpoint may now include AI status, but original fields should remain
        assert result['status'] == 'ok'

    def test_existing_error_handling_unchanged(self):
        """Test that existing error handling behaviors are preserved"""
        # Test with non-existent file_id
        response = self.client.post('/api/convert',
                                  json={'file_id': 'nonexistent', 'converter': 'vtracer'},
                                  content_type='application/json')

        assert response.status_code == 404
        result = response.get_json()
        assert 'error' in result

        # Test with missing file_id
        response = self.client.post('/api/convert',
                                  json={'converter': 'vtracer'},
                                  content_type='application/json')

        assert response.status_code == 400
        result = response.get_json()
        assert 'error' in result

    def test_existing_parameter_validation_unchanged(self):
        """Test that parameter validation still works as before"""
        file_id = self.upload_test_file()

        # Test invalid parameters should still be rejected
        response = self.client.post('/api/convert',
                                  json={
                                      'file_id': file_id,
                                      'converter': 'vtracer',
                                      'color_precision': 15  # Invalid - max is 10
                                  },
                                  content_type='application/json')

        assert response.status_code == 400
        result = response.get_json()
        assert 'error' in result

    def test_ai_endpoints_graceful_fallback(self):
        """Test that AI endpoints fail gracefully when AI unavailable"""
        file_id = self.upload_test_file()

        # Test AI endpoints
        ai_response = self.client.post('/api/convert-ai',
                                     json={'file_id': file_id},
                                     content_type='application/json')

        # Should either work (200) or gracefully degrade (503)
        assert ai_response.status_code in [200, 503]

        if ai_response.status_code == 503:
            result = ai_response.get_json()
            assert 'fallback_suggestion' in result
            assert 'Use /api/convert' in result['fallback_suggestion']

    def test_ai_health_endpoints_exist(self):
        """Test that new AI health endpoints exist and work"""
        # Test AI health endpoint
        health_response = self.client.get('/api/ai-health')
        assert health_response.status_code == 200

        health_data = health_response.get_json()
        assert 'overall_status' in health_data
        assert 'components' in health_data

        # Test model status endpoint
        model_response = self.client.get('/api/model-status')
        assert model_response.status_code in [200, 503]  # 503 if AI unavailable

        if model_response.status_code == 200:
            model_data = model_response.get_json()
            assert 'models_available' in model_data

    def test_content_type_requirements_unchanged(self):
        """Test that content-type requirements remain the same"""
        file_id = self.upload_test_file()

        # Test that convert endpoint still requires application/json
        response = self.client.post('/api/convert',
                                  data={'file_id': file_id, 'converter': 'vtracer'})  # Wrong content-type

        assert response.status_code == 400

        # Test with correct content-type
        response = self.client.post('/api/convert',
                                  json={'file_id': file_id, 'converter': 'vtracer'},
                                  content_type='application/json')

        assert response.status_code == 200

    def test_cors_headers_unchanged(self):
        """Test that CORS headers are still present"""
        response = self.client.options('/api/convert')

        # Should have CORS headers (exact headers may vary based on Flask-CORS config)
        # Just check that some CORS-related headers are present
        headers = dict(response.headers)
        cors_headers = [h for h in headers.keys() if 'Access-Control' in h]

        # Should have at least some CORS headers
        assert len(cors_headers) > 0, "CORS headers missing"

    def test_security_headers_unchanged(self):
        """Test that security headers are still applied"""
        response = self.client.get('/health')

        headers = dict(response.headers)

        # Check for security headers that should be present
        expected_security_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection'
        ]

        for header in expected_security_headers:
            assert header in headers, f"Security header {header} missing"

    def test_route_registration_isolation(self):
        """Test that AI routes don't interfere with existing routes"""
        # Test that we can still access the root route
        response = self.client.get('/')
        # Should either serve frontend (200) or redirect, not crash
        assert response.status_code in [200, 301, 302, 404]

        # Test static file serving still works (if applicable)
        # This might 404 if no frontend files exist, which is fine
        response = self.client.get('/static/test.css')  # Non-existent file
        assert response.status_code in [404, 200]  # 404 expected for non-existent file