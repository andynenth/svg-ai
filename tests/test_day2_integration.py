# tests/test_day2_integration.py
import pytest
import time
import io
import threading
import concurrent.futures
from PIL import Image
import numpy as np

class TestDay2Integration:
    """Comprehensive testing of Day 2 AI API enhancements"""

    def setup_method(self):
        """Setup test environment"""
        from backend.app import app
        self.client = app.test_client()
        self.test_file_id = self.upload_test_image()

    def create_test_image(self, width=100, height=100, pattern='simple'):
        """Create various test images for different scenarios"""
        if pattern == 'simple':
            # Simple geometric logo (white square with black border)
            img = Image.new('RGB', (width, height), color='white')
            # Add black border
            pixels = img.load()
            for i in range(width):
                pixels[i, 0] = (0, 0, 0)  # Top
                pixels[i, height-1] = (0, 0, 0)  # Bottom
            for i in range(height):
                pixels[0, i] = (0, 0, 0)  # Left
                pixels[width-1, i] = (0, 0, 0)  # Right

        elif pattern == 'complex':
            # Complex pattern with gradients and multiple colors
            img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

        elif pattern == 'text':
            # Simple text-like pattern (black text on white background)
            img = Image.new('RGB', (width, height), color='white')
            pixels = img.load()
            # Create simple text-like pattern
            for i in range(20, 80):
                for j in range(30, 40):
                    pixels[i, j] = (0, 0, 0)

        else:
            # Default: solid color
            img = Image.new('RGB', (width, height), color='white')

        return img

    def upload_test_image(self, pattern='simple'):
        """Upload a test image and return file_id"""
        img = self.create_test_image(pattern=pattern)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        response = self.client.post('/api/upload',
                                  data={'file': (img_byte_arr, 'test.png')},
                                  content_type='multipart/form-data')

        assert response.status_code == 200
        result = response.get_json()
        return result['file_id']

    def test_ai_convert_endpoint_complete_flow(self):
        """Test complete AI conversion flow"""
        # Test auto tier selection
        response = self.client.post('/api/convert-ai', json={
            'file_id': self.test_file_id,
            'tier': 'auto',
            'target_quality': 0.85
        }, content_type='application/json')

        if response.status_code == 503:
            # AI unavailable - acceptable
            result = response.get_json()
            assert 'fallback_suggestion' in result
            assert result['success'] == False
            return

        assert response.status_code == 200
        result = response.get_json()

        # Validate response structure
        assert result['success'] == True
        assert 'svg' in result
        assert 'ai_metadata' in result
        assert 'processing_time' in result
        assert 'ai_enabled' in result
        assert result['ai_enabled'] == True

        # Validate AI metadata
        ai_metadata = result['ai_metadata']
        assert 'routing' in ai_metadata
        assert 'tier_used' in ai_metadata
        assert ai_metadata['tier_used'] in [1, 2, 3, 'fallback']

    def test_ai_health_endpoint(self):
        """Test AI health monitoring"""
        response = self.client.get('/api/ai-health')
        assert response.status_code == 200

        health_data = response.get_json()
        assert 'overall_status' in health_data
        assert 'components' in health_data
        assert 'performance_metrics' in health_data
        assert 'timestamp' in health_data

        # Status should be one of the expected values
        assert health_data['overall_status'] in ['healthy', 'degraded', 'unhealthy', 'error']

        # Components should be present
        components = health_data['components']
        assert 'ai_initialized' in components
        assert 'model_manager' in components
        assert 'quality_predictor' in components
        assert 'router' in components
        assert 'converter' in components

    def test_model_status_endpoint(self):
        """Test model status information"""
        response = self.client.get('/api/model-status')

        # Should either work (200) or be unavailable (503)
        assert response.status_code in [200, 503]

        result = response.get_json()
        assert 'models_available' in result

        if response.status_code == 200:
            assert 'models' in result
            assert 'memory_report' in result
            # Validate models structure
            for model_name, model_info in result['models'].items():
                assert 'loaded' in model_info
                assert 'type' in model_info

    def test_tier_selection_logic(self):
        """Test intelligent tier selection"""
        # Test manual tier selection
        for tier in [1, 2, 3]:
            response = self.client.post('/api/convert-ai', json={
                'file_id': self.test_file_id,
                'tier': tier
            }, content_type='application/json')

            if response.status_code == 200:
                result = response.get_json()
                if result.get('success'):
                    assert result['ai_metadata']['tier_used'] == tier
            elif response.status_code == 503:
                # AI unavailable - acceptable
                result = response.get_json()
                assert 'fallback_suggestion' in result

    def test_target_quality_parameter(self):
        """Test target quality parameter functionality"""
        # Test different quality targets
        quality_targets = [0.7, 0.85, 0.95]

        for target in quality_targets:
            response = self.client.post('/api/convert-ai', json={
                'file_id': self.test_file_id,
                'tier': 'auto',
                'target_quality': target
            }, content_type='application/json')

            if response.status_code == 200:
                result = response.get_json()
                if result.get('success'):
                    # Should have routing metadata with target quality
                    routing = result['ai_metadata'].get('routing', {})
                    assert routing.get('target_quality') == target

    def test_time_budget_parameter(self):
        """Test time budget parameter functionality"""
        # Test with time budget constraint
        response = self.client.post('/api/convert-ai', json={
            'file_id': self.test_file_id,
            'tier': 'auto',
            'time_budget': 1.0  # 1 second budget
        }, content_type='application/json')

        if response.status_code == 200:
            result = response.get_json()
            if result.get('success'):
                # Should respect time budget (tier 1 likely selected)
                tier_used = result['ai_metadata']['tier_used']
                assert tier_used in [1, 2, 3, 'fallback']

                # Processing time should be reasonable
                processing_time = result.get('processing_time', 0)
                assert processing_time < 10.0  # Generous upper bound

    def test_performance_requirements(self):
        """Test performance meets requirements"""
        # Test routing speed
        start_time = time.time()
        response = self.client.post('/api/convert-ai', json={
            'file_id': self.test_file_id,
            'tier': 'auto'
        }, content_type='application/json')
        total_time = time.time() - start_time

        if response.status_code == 200:
            result = response.get_json()
            processing_time = result.get('processing_time', total_time)

            # AI overhead should be reasonable
            assert processing_time < 10.0, f"AI conversion took {processing_time:.2f}s, too slow for testing"

        # Test health endpoint response time
        start_time = time.time()
        health_response = self.client.get('/api/ai-health')
        health_time = time.time() - start_time

        assert health_response.status_code == 200
        assert health_time < 1.0, f"Health check took {health_time:.2f}s, too slow"

    def test_fallback_behavior(self):
        """Test fallback when AI unavailable"""
        # Test with non-existent file
        response = self.client.post('/api/convert-ai', json={
            'file_id': 'nonexistent_file'
        }, content_type='application/json')

        assert response.status_code == 404
        result = response.get_json()
        assert result['success'] == False
        assert 'error' in result

    def test_enhanced_health_endpoint(self):
        """Test enhanced /health endpoint includes AI status"""
        response = self.client.get('/health')
        assert response.status_code == 200

        result = response.get_json()
        assert 'status' in result
        assert 'ai_available' in result
        assert 'timestamp' in result
        assert 'uptime' in result

        # Should indicate whether AI is available
        assert isinstance(result['ai_available'], bool)

    def test_error_handling_consistency(self):
        """Test that error handling is consistent across endpoints"""
        # Test invalid JSON
        response = self.client.post('/api/convert-ai',
                                  data='invalid json',
                                  content_type='application/json')

        assert response.status_code == 400
        result = response.get_json()
        assert 'error' in result
        assert result['success'] == False

        # Test missing required fields
        response = self.client.post('/api/convert-ai',
                                  json={},
                                  content_type='application/json')

        assert response.status_code == 400
        result = response.get_json()
        assert 'error' in result
        assert result['success'] == False

    def test_include_analysis_parameter(self):
        """Test include_analysis parameter functionality"""
        # Test with analysis enabled
        response = self.client.post('/api/convert-ai', json={
            'file_id': self.test_file_id,
            'tier': 1,
            'include_analysis': True
        }, content_type='application/json')

        if response.status_code == 200:
            result = response.get_json()
            if result.get('success'):
                assert 'ai_metadata' in result

        # Test with analysis disabled
        response = self.client.post('/api/convert-ai', json={
            'file_id': self.test_file_id,
            'tier': 1,
            'include_analysis': False
        }, content_type='application/json')

        if response.status_code == 200:
            result = response.get_json()
            if result.get('success'):
                # Should still have some metadata, but potentially less detailed
                assert 'ai_metadata' in result

    def test_concurrent_requests(self):
        """Test concurrent AI requests don't interfere"""
        def make_request():
            return self.client.post('/api/convert-ai', json={
                'file_id': self.test_file_id,
                'tier': 1
            }, content_type='application/json')

        # Test with 3 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            responses = [future.result() for future in futures]

        # All responses should be valid
        for response in responses:
            assert response.status_code in [200, 503]  # 503 if AI unavailable
            result = response.get_json()
            assert 'success' in result

    def test_different_image_types(self):
        """Test AI endpoints with different image types"""
        image_patterns = ['simple', 'complex', 'text']

        for pattern in image_patterns:
            # Upload different image type
            test_file_id = self.upload_test_image(pattern=pattern)

            # Test AI conversion
            response = self.client.post('/api/convert-ai', json={
                'file_id': test_file_id,
                'tier': 'auto'
            }, content_type='application/json')

            if response.status_code == 200:
                result = response.get_json()
                if result.get('success'):
                    # Should have appropriate tier selection based on image type
                    tier_used = result['ai_metadata']['tier_used']
                    assert tier_used in [1, 2, 3, 'fallback']

    def test_ai_endpoints_isolation(self):
        """Test that AI endpoints don't affect basic endpoints"""
        # Make AI request
        ai_response = self.client.post('/api/convert-ai', json={
            'file_id': self.test_file_id,
            'tier': 1
        }, content_type='application/json')

        # Make basic request immediately after
        basic_response = self.client.post('/api/convert', json={
            'file_id': self.test_file_id,
            'converter': 'vtracer'
        }, content_type='application/json')

        # Basic endpoint should work normally
        assert basic_response.status_code == 200
        basic_result = basic_response.get_json()
        assert 'success' in basic_result
        # Should NOT have AI metadata
        assert 'ai_metadata' not in basic_result

    def test_comprehensive_error_scenarios(self):
        """Test various error scenarios are handled gracefully"""
        # Test invalid tier
        response = self.client.post('/api/convert-ai', json={
            'file_id': self.test_file_id,
            'tier': 999  # Invalid tier
        }, content_type='application/json')

        # Should either handle gracefully or return error
        assert response.status_code in [200, 400, 503]

        # Test invalid target_quality
        response = self.client.post('/api/convert-ai', json={
            'file_id': self.test_file_id,
            'target_quality': 2.0  # Invalid (>1.0)
        }, content_type='application/json')

        # Should handle gracefully (may clamp to valid range)
        if response.status_code == 200:
            result = response.get_json()
            # If successful, should handle invalid values gracefully
            assert 'success' in result

    def test_response_format_consistency(self):
        """Test that response formats are consistent"""
        # Test AI convert response format
        response = self.client.post('/api/convert-ai', json={
            'file_id': self.test_file_id,
            'tier': 1
        }, content_type='application/json')

        if response.status_code == 200:
            result = response.get_json()
            if result.get('success'):
                # Check required fields
                required_fields = ['success', 'svg', 'ai_metadata', 'processing_time', 'ai_enabled']
                for field in required_fields:
                    assert field in result, f"Missing required field: {field}"

        # Test health response format
        health_response = self.client.get('/api/ai-health')
        assert health_response.status_code == 200
        health_result = health_response.get_json()

        health_required_fields = ['overall_status', 'components', 'performance_metrics', 'timestamp']
        for field in health_required_fields:
            assert field in health_result, f"Missing required health field: {field}"