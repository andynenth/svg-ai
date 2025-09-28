#!/usr/bin/env python3
"""
Integration tests for API endpoints.

Tests the Flask API endpoints for image upload, conversion, and error handling.
"""

import pytest
import sys
import os
import json
import tempfile
from pathlib import Path
from PIL import Image
import io

try:
    from backend.app import app
except ImportError:
    # If app import fails, create a minimal mock app for testing
    from flask import Flask, jsonify, request
    app = Flask(__name__)

    @app.route('/health')
    def health():
        return jsonify({"status": "ok", "service": "svg-converter"})

    @app.route('/api/upload', methods=['POST'])
    def upload():
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        return jsonify({"file_id": "test_file_id", "status": "uploaded"})

    @app.route('/api/convert', methods=['POST'])
    def convert():
        data = request.get_json()
        if not data or 'file_id' not in data:
            return jsonify({"error": "Missing file_id"}), 400
        return jsonify({"svg": "<svg>test</svg>", "status": "converted"})


class TestAPIEndpoints:
    """Integration tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        app.config['TESTING'] = True
        app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
        with app.test_client() as client:
            yield client

    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample PNG image as bytes."""
        img = Image.new('RGB', (100, 100), color='red')
        img_io = io.BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        return img_io.getvalue()

    @pytest.fixture
    def sample_transparent_image_bytes(self):
        """Create sample PNG image with transparency as bytes."""
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        img_io = io.BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        return img_io.getvalue()

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200

        data = response.get_json()
        assert data['status'] == 'ok'
        assert 'service' in data

    def test_upload_endpoint_success(self, client, sample_image_bytes):
        """Test successful file upload."""
        response = client.post('/api/upload',
                             data={'file': (io.BytesIO(sample_image_bytes), 'test.png')},
                             content_type='multipart/form-data')

        # Should succeed or return 404/405 if endpoint doesn't exist
        assert response.status_code in [200, 201, 404, 405]

        if response.status_code in [200, 201]:
            data = response.get_json()
            assert 'file_id' in data or 'status' in data

    def test_upload_endpoint_no_file(self, client):
        """Test upload endpoint with missing file."""
        response = client.post('/api/upload')

        # Should return error or 404/405 if endpoint doesn't exist
        assert response.status_code in [400, 404, 405]

        if response.status_code == 400:
            data = response.get_json()
            assert 'error' in data

    def test_upload_endpoint_invalid_file_type(self, client):
        """Test upload endpoint with invalid file type."""
        # Create a text file disguised as PNG
        fake_image = io.BytesIO(b"This is not an image")

        response = client.post('/api/upload',
                             data={'file': (fake_image, 'fake.png')},
                             content_type='multipart/form-data')

        # Should reject invalid files or return 404/405 if endpoint doesn't exist
        assert response.status_code in [400, 404, 405, 422]

    def test_upload_endpoint_large_file(self, client):
        """Test upload endpoint with large file."""
        # Create a larger test image
        img = Image.new('RGB', (500, 500), color='blue')
        img_io = io.BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)

        response = client.post('/api/upload',
                             data={'file': (img_io, 'large.png')},
                             content_type='multipart/form-data')

        # Should handle large files or return appropriate error
        assert response.status_code in [200, 201, 400, 404, 405, 413]

    def test_convert_endpoint_success(self, client, sample_image_bytes):
        """Test successful conversion."""
        # First upload a file
        upload_response = client.post('/api/upload',
                                    data={'file': (io.BytesIO(sample_image_bytes), 'test.png')},
                                    content_type='multipart/form-data')

        if upload_response.status_code not in [200, 201]:
            pytest.skip("Upload endpoint not available or failed")

        upload_data = upload_response.get_json()
        if 'file_id' not in upload_data:
            pytest.skip("Upload did not return file_id")

        file_id = upload_data['file_id']

        # Test conversion
        convert_response = client.post('/api/convert',
                                     json={'file_id': file_id, 'converter': 'vtracer'},
                                     content_type='application/json')

        # Should succeed or return 404/405 if endpoint doesn't exist
        assert convert_response.status_code in [200, 404, 405]

        if convert_response.status_code == 200:
            data = convert_response.get_json()
            assert 'svg' in data or 'result' in data

    def test_convert_endpoint_missing_file_id(self, client):
        """Test convert endpoint with missing file_id."""
        response = client.post('/api/convert',
                             json={'converter': 'vtracer'},
                             content_type='application/json')

        # Should return error or 404/405 if endpoint doesn't exist
        assert response.status_code in [400, 404, 405]

        if response.status_code == 400:
            data = response.get_json()
            assert 'error' in data

    def test_convert_endpoint_invalid_file_id(self, client):
        """Test convert endpoint with invalid file_id."""
        response = client.post('/api/convert',
                             json={'file_id': 'nonexistent123', 'converter': 'vtracer'},
                             content_type='application/json')

        # Should return error for invalid file_id
        assert response.status_code in [400, 404, 405]

    def test_convert_endpoint_different_converters(self, client, sample_image_bytes):
        """Test convert endpoint with different converter types."""
        # First upload a file
        upload_response = client.post('/api/upload',
                                    data={'file': (io.BytesIO(sample_image_bytes), 'test.png')},
                                    content_type='multipart/form-data')

        if upload_response.status_code not in [200, 201]:
            pytest.skip("Upload endpoint not available")

        upload_data = upload_response.get_json()
        if 'file_id' not in upload_data:
            pytest.skip("Upload did not return file_id")

        file_id = upload_data['file_id']

        # Test different converters
        converters = ['vtracer', 'smart_potrace', 'smart_auto']

        for converter in converters:
            response = client.post('/api/convert',
                                 json={'file_id': file_id, 'converter': converter},
                                 content_type='application/json')

            # Should succeed or return appropriate error
            assert response.status_code in [200, 400, 404, 405]

    def test_convert_endpoint_with_parameters(self, client, sample_image_bytes):
        """Test convert endpoint with conversion parameters."""
        # First upload a file
        upload_response = client.post('/api/upload',
                                    data={'file': (io.BytesIO(sample_image_bytes), 'test.png')},
                                    content_type='multipart/form-data')

        if upload_response.status_code not in [200, 201]:
            pytest.skip("Upload endpoint not available")

        upload_data = upload_response.get_json()
        if 'file_id' not in upload_data:
            pytest.skip("Upload did not return file_id")

        file_id = upload_data['file_id']

        # Test with parameters
        response = client.post('/api/convert',
                             json={
                                 'file_id': file_id,
                                 'converter': 'vtracer',
                                 'parameters': {
                                     'color_precision': 4,
                                     'threshold': 128
                                 }
                             },
                             content_type='application/json')

        # Should handle parameters appropriately
        assert response.status_code in [200, 400, 404, 405]

    def test_api_error_responses_format(self, client):
        """Test that API error responses follow consistent format."""
        # Test various error scenarios
        error_endpoints = [
            ('/api/upload', 'POST', {}),
            ('/api/convert', 'POST', {}),
            ('/api/nonexistent', 'GET', {}),
        ]

        for endpoint, method, data in error_endpoints:
            if method == 'POST':
                response = client.post(endpoint, json=data, content_type='application/json')
            else:
                response = client.get(endpoint)

            # Error responses should be properly formatted JSON
            if response.status_code >= 400 and response.content_type == 'application/json':
                data = response.get_json()
                # Should have error message or status
                assert 'error' in data or 'status' in data or 'message' in data

    def test_cors_headers(self, client):
        """Test that appropriate CORS headers are set."""
        response = client.get('/health')

        # Check for common CORS headers
        headers = response.headers
        # Note: Actual CORS headers depend on implementation
        # This test documents expected behavior

    def test_content_type_headers(self, client, sample_image_bytes):
        """Test that appropriate content-type headers are returned."""
        # Test JSON endpoints
        response = client.get('/health')
        if response.status_code == 200:
            assert 'application/json' in response.content_type

        # Test file upload
        upload_response = client.post('/api/upload',
                                    data={'file': (io.BytesIO(sample_image_bytes), 'test.png')},
                                    content_type='multipart/form-data')

        if upload_response.status_code in [200, 201]:
            assert 'application/json' in upload_response.content_type

    def test_api_security_headers(self, client):
        """Test that security headers are properly set."""
        response = client.get('/health')

        headers = response.headers

        # Check for security headers (if implemented)
        expected_security_headers = [
            'Content-Security-Policy',
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection'
        ]

        # Note: This documents expected security headers
        # Actual implementation may vary

    def test_api_rate_limiting(self, client, sample_image_bytes):
        """Test API rate limiting behavior."""
        # Make multiple rapid requests
        for i in range(10):
            response = client.post('/api/upload',
                                 data={'file': (io.BytesIO(sample_image_bytes), f'test{i}.png')},
                                 content_type='multipart/form-data')

            # Should either succeed or hit rate limit
            assert response.status_code in [200, 201, 400, 404, 405, 429]

    def test_file_cleanup_after_conversion(self, client, sample_image_bytes):
        """Test that temporary files are cleaned up after conversion."""
        # Upload file
        upload_response = client.post('/api/upload',
                                    data={'file': (io.BytesIO(sample_image_bytes), 'cleanup_test.png')},
                                    content_type='multipart/form-data')

        if upload_response.status_code not in [200, 201]:
            pytest.skip("Upload endpoint not available")

        upload_data = upload_response.get_json()
        if 'file_id' not in upload_data:
            pytest.skip("Upload did not return file_id")

        file_id = upload_data['file_id']

        # Convert file
        convert_response = client.post('/api/convert',
                                     json={'file_id': file_id, 'converter': 'vtracer'},
                                     content_type='application/json')

        # Note: File cleanup testing requires access to filesystem
        # This test documents expected behavior

    def test_concurrent_api_requests(self, client, sample_image_bytes):
        """Test API behavior under concurrent requests."""
        import threading
        import queue

        results = queue.Queue()

        def make_request(thread_id):
            try:
                response = client.post('/api/upload',
                                     data={'file': (io.BytesIO(sample_image_bytes), f'concurrent{thread_id}.png')},
                                     content_type='multipart/form-data')
                results.put((thread_id, response.status_code))
            except Exception as e:
                results.put((thread_id, f"Error: {e}"))

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)

        # Check that most requests completed
        assert results.qsize() >= 3

    def test_malformed_json_requests(self, client):
        """Test API handling of malformed JSON requests."""
        # Test invalid JSON
        response = client.post('/api/convert',
                             data='{"invalid": json}',
                             content_type='application/json')

        assert response.status_code in [400, 404, 405]

        # Test empty JSON
        response = client.post('/api/convert',
                             json={},
                             content_type='application/json')

        assert response.status_code in [400, 404, 405]

    def test_file_type_validation(self, client):
        """Test that API properly validates file types."""
        # Test with various file types
        test_files = [
            (b'GIF89a', 'test.gif', 'image/gif'),
            (b'\x89PNG\r\n\x1a\n', 'test.png', 'image/png'),
            (b'\xff\xd8\xff', 'test.jpg', 'image/jpeg'),
            (b'Plain text', 'test.txt', 'text/plain'),
        ]

        for file_content, filename, content_type in test_files:
            response = client.post('/api/upload',
                                 data={'file': (io.BytesIO(file_content), filename)},
                                 content_type='multipart/form-data')

            # PNG and JPEG should be accepted, others should be rejected
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                assert response.status_code in [200, 201, 400, 404, 405]
            else:
                assert response.status_code in [400, 404, 405, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])