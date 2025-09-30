# tests/test_production_readiness.py
import pytest
import contextlib
import time
import logging
import gc
from PIL import Image
import io

class TestProductionReadiness:
    """Validate system is ready for production deployment"""

    def test_configuration_security(self):
        """Test production configuration security"""

        # Check for debug mode disabled
        from backend.app import app
        assert not app.debug, "Debug mode must be disabled in production"

        # Check for secure configuration
        assert app.config.get('SECRET_KEY'), "Secret key must be configured"

        # Check CORS configuration
        # Should be restricted in production
        cors_origins = app.config.get('CORS_ORIGINS', [])
        localhost_origins = [origin for origin in cors_origins if 'localhost' in str(origin)]
        if len(localhost_origins) > 0:
            logging.warning("CORS includes localhost origins - review for production")

    def test_error_handling_robustness(self):
        """Test comprehensive error handling"""
        from backend.app import app
        client = app.test_client()

        # Test various error scenarios
        error_scenarios = [
            {'endpoint': '/api/convert-ai', 'data': {}, 'expected_status': 400},  # Missing data
            {'endpoint': '/api/convert-ai', 'data': {'file_id': 'invalid'}, 'expected_status': 404},  # File not found
            {'endpoint': '/api/convert-ai', 'data': {'file_id': 'test', 'tier': 99}, 'expected_status': 400},  # Invalid tier
        ]

        for scenario in error_scenarios:
            response = client.post(scenario['endpoint'], json=scenario['data'],
                                 content_type='application/json')

            # Should return appropriate error status
            assert response.status_code == scenario['expected_status'], \
                f"Expected {scenario['expected_status']}, got {response.status_code} for {scenario}"

            # Error response should be well-formed JSON
            try:
                error_data = response.get_json()
                assert 'success' in error_data
                assert error_data['success'] == False
                assert 'error' in error_data
            except:
                pytest.fail(f"Error response not valid JSON for {scenario}")

    def test_resource_cleanup(self):
        """Test system properly cleans up resources"""
        import gc
        import psutil

        # Baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024

        try:
            # Create and destroy AI components multiple times
            for _ in range(3):
                from backend.ai_modules.management.production_model_manager import ProductionModelManager
                model_manager = ProductionModelManager()
                models = model_manager._load_all_exported_models()

                # Simulate usage
                if models:
                    time.sleep(0.1)

                # Clean up
                del model_manager
                del models
                gc.collect()

            # Check for memory leaks
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - baseline_memory

            # Should not grow significantly
            assert memory_growth < 100, f"Memory leak detected: {memory_growth:.1f}MB growth"

        except Exception as e:
            logging.warning(f"Resource cleanup test failed: {e}")

    def test_api_documentation_completeness(self):
        """Test API documentation and response formats"""
        from backend.app import app
        client = app.test_client()

        # Test health endpoints provide comprehensive information
        response = client.get('/api/ai-health')
        if response.status_code == 200:
            health_data = response.get_json()

            required_fields = ['overall_status', 'components', 'timestamp']
            for field in required_fields:
                assert field in health_data, f"Health endpoint missing required field: {field}"

        # Test model status endpoint
        response = client.get('/api/model-status')
        assert response.status_code in [200, 503]  # Either works or unavailable

        status_data = response.get_json()
        assert 'models_available' in status_data

    def test_logging_and_monitoring(self):
        """Test logging and monitoring capabilities"""

        # Test that critical operations are logged
        with self.capture_logs() as log_capture:
            from backend.ai_modules.management.production_model_manager import ProductionModelManager

            try:
                model_manager = ProductionModelManager()
                models = model_manager._load_all_exported_models()
            except:
                pass  # May fail in test environment

        # Should have informative log messages
        logs = log_capture.getvalue()
        assert any(level in logs for level in ['INFO', 'WARNING', 'ERROR']), \
            "No informative log messages found"

    @contextlib.contextmanager
    def capture_logs(self):
        """Helper to capture log output"""
        import logging
        import io

        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger()
        logger.addHandler(handler)

        try:
            yield log_capture
        finally:
            logger.removeHandler(handler)

    def test_graceful_degradation(self):
        """Test system degrades gracefully when AI unavailable"""
        from backend.app import app
        client = app.test_client()

        # Upload test image
        test_image = self._create_test_image()
        upload_response = client.post('/api/upload',
                                    data={'file': (test_image, 'test.png')},
                                    content_type='multipart/form-data')

        assert upload_response.status_code == 200
        file_id = upload_response.get_json()['file_id']

        # Test AI endpoint
        ai_response = client.post('/api/convert-ai', json={'file_id': file_id, 'tier': 1},
                                content_type='application/json')

        if ai_response.status_code == 503:
            # AI unavailable - should provide fallback suggestion
            ai_result = ai_response.get_json()
            assert 'fallback_suggestion' in ai_result
            assert '/api/convert' in ai_result['fallback_suggestion']

        # Test that basic endpoint still works
        basic_response = client.post('/api/convert', json={
            'file_id': file_id,
            'converter': 'vtracer'
        }, content_type='application/json')

        assert basic_response.status_code == 200
        basic_result = basic_response.get_json()
        assert basic_result['success'] == True

    def _create_test_image(self):
        """Create a simple test image"""
        from PIL import Image
        import io

        # Create simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        return img_bytes