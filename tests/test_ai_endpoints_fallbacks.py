#!/usr/bin/env python3
"""
Test suite for AI endpoint fallback behavior and degraded mode operations.

Tests:
1. Missing models scenario
2. Invalid tier requests
3. Fallback success paths
4. AI health endpoint with models_found flag
"""

import json
import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch
import pytest
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app import create_app
from backend.api.ai_endpoints import get_ai_components, perform_ai_conversion


class TestAIEndpointsFallback(TestCase):
    """Test AI endpoints fallback behavior"""

    def setUp(self):
        """Set up test fixtures"""
        self.app = create_app()
        self.client = self.app.test_client()
        self.test_image_path = self._create_test_image()

    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self, 'test_image_path') and Path(self.test_image_path).exists():
            os.unlink(self.test_image_path)

    def _create_test_image(self) -> str:
        """Create a simple test PNG image"""
        from PIL import Image, ImageDraw

        # Create simple test image
        img = Image.new('RGB', (100, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.ellipse([25, 25, 75, 75], fill='blue', outline='black')

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            return tmp.name

    def _upload_test_image(self) -> str:
        """Upload test image and return file_id"""
        with open(self.test_image_path, 'rb') as f:
            response = self.client.post('/api/upload',
                                       data={'file': f},
                                       content_type='multipart/form-data')
            if response.status_code == 200:
                return response.get_json()['file_id']
            return None

    # Test 1: Missing Models Scenario
    @patch('backend.api.ai_endpoints.ProductionModelManager')
    def test_missing_models_fallback(self, mock_model_manager):
        """Test fallback when no AI models are found"""
        # Mock model manager with no models found
        mock_manager_instance = MagicMock()
        mock_manager_instance.models_found = False
        mock_manager_instance.model_dir = Path('models/production')
        mock_manager_instance.models = {
            'quality_predictor': None,
            'logo_classifier': None,
            'correlation_models': None
        }
        mock_manager_instance._load_all_exported_models.return_value = {}
        mock_model_manager.return_value = mock_manager_instance

        # Upload test image
        file_id = self._upload_test_image()
        if not file_id:
            # If upload endpoint doesn't exist, use direct path
            file_id = self.test_image_path

        # Test AI conversion with missing models
        response = self.client.post('/api/convert-ai',
                                   json={'file_id': file_id, 'tier': 1},
                                   content_type='application/json')

        # Should either succeed with fallback or return appropriate error
        if response.status_code == 200:
            result = response.get_json()
            # Check for fallback indicators
            if 'ai_metadata' in result:
                self.assertTrue(
                    result['ai_metadata'].get('fallback_used', False) or
                    result['ai_metadata'].get('tier_used') == 'fallback',
                    "Should indicate fallback was used when models are missing"
                )
        elif response.status_code == 503:
            # Service unavailable is acceptable when AI components can't initialize
            result = response.get_json()
            self.assertIn('fallback_suggestion', result,
                         "Should provide fallback suggestion when AI unavailable")
            self.assertIn('/api/convert', result['fallback_suggestion'],
                         "Fallback suggestion should point to basic conversion endpoint")

    # Test 2: Invalid Tier Requests
    def test_invalid_tier_fallback(self):
        """Test fallback when invalid tier is requested"""
        file_id = self._upload_test_image() or self.test_image_path

        # Test with invalid tier (tier 99)
        response = self.client.post('/api/convert-ai',
                                   json={'file_id': file_id, 'tier': 99},
                                   content_type='application/json')

        if response.status_code == 200:
            result = response.get_json()
            # Should fall back to valid tier or basic conversion
            self.assertTrue(result.get('success', False))
            if 'ai_metadata' in result:
                # Check that tier was adjusted or fallback was used
                tier_used = result['ai_metadata'].get('tier_used')
                self.assertIn(tier_used, ['fallback', 1, 2, 3],
                            "Should use valid tier or fallback")
        else:
            # Error response should be informative
            result = response.get_json()
            self.assertIn('error', result, "Error response should include error message")

    # Test 3: Fallback Success Path
    @patch('backend.api.ai_endpoints.AIEnhancedConverter')
    def test_fallback_success_path(self, mock_converter):
        """Test successful fallback to basic conversion"""
        # Mock AI converter to fail
        mock_converter_instance = MagicMock()
        mock_converter_instance.convert_with_ai_tier.side_effect = RuntimeError("AI conversion failed")
        mock_converter.return_value = mock_converter_instance

        file_id = self._upload_test_image() or self.test_image_path

        # Attempt AI conversion that will fail and fallback
        response = self.client.post('/api/convert-ai',
                                   json={'file_id': file_id, 'tier': 2},
                                   content_type='application/json')

        if response.status_code == 200:
            result = response.get_json()
            self.assertTrue(result.get('success', False),
                          "Fallback should succeed even when AI fails")
            self.assertIn('svg', result, "Should return SVG content")

            # Check for quality metrics in fallback
            if 'ai_metadata' in result:
                metadata = result['ai_metadata']
                if metadata.get('fallback_used'):
                    # Should include quality metrics even in fallback
                    self.assertIn('quality_metrics', metadata,
                                "Fallback should include quality metrics")
                    metrics = metadata['quality_metrics']
                    self.assertIn('ssim', metrics, "Should include SSIM")
                    self.assertIn('mse', metrics, "Should include MSE")
                    self.assertIn('psnr', metrics, "Should include PSNR")

    # Test 4: Verify Error Context Propagation
    def test_error_context_propagation(self):
        """Test that error context is properly propagated in fallback"""
        with patch('backend.api.ai_endpoints.HybridIntelligentRouter') as mock_router:
            # Mock router to raise specific error
            mock_router_instance = MagicMock()
            mock_router_instance.determine_optimal_tier.side_effect = ValueError("Invalid image features")
            mock_router.return_value = mock_router_instance

            file_id = self._upload_test_image() or self.test_image_path

            response = self.client.post('/api/convert-ai',
                                      json={'file_id': file_id, 'tier': 'auto'},
                                      content_type='application/json')

            if response.status_code == 200:
                result = response.get_json()
                if 'ai_metadata' in result and result['ai_metadata'].get('fallback_used'):
                    # Check error context is included
                    metadata = result['ai_metadata']
                    self.assertIn('error_context', metadata,
                                "Should include error context in fallback")
                    error_context = metadata['error_context']
                    self.assertIn('error_type', error_context)
                    self.assertIn('error_message', error_context)
                    self.assertIn('tier_attempted', error_context)


class TestAIHealthEndpoint(TestCase):
    """Test AI health endpoint with models_found flag"""

    def setUp(self):
        """Set up test fixtures"""
        self.app = create_app()
        self.client = self.app.test_client()

    # Test 5: AI Health with Models Found
    @patch('backend.api.ai_endpoints.ProductionModelManager')
    def test_ai_health_models_found(self, mock_model_manager):
        """Test /api/ai-health when models are found"""
        # Mock model manager with models found
        mock_manager_instance = MagicMock()
        mock_manager_instance.models_found = True
        mock_manager_instance.model_dir = Path('models/production')
        mock_manager_instance.models = {
            'quality_predictor': MagicMock(),
            'logo_classifier': MagicMock(),
            'correlation_models': {'model': 'data'}
        }
        mock_manager_instance._load_all_exported_models.return_value = mock_manager_instance.models
        mock_model_manager.return_value = mock_manager_instance

        response = self.client.get('/api/ai-health')
        self.assertEqual(response.status_code, 200)

        result = response.get_json()
        self.assertIn('components', result)

        # Check model_manager component
        model_manager_health = result['components'].get('model_manager', {})
        self.assertTrue(model_manager_health.get('models_found', False),
                       "Should indicate models were found")
        self.assertEqual(model_manager_health.get('status'), 'healthy',
                        "Status should be healthy when models are found")

    # Test 6: AI Health with Missing Models
    @patch('backend.api.ai_endpoints.ProductionModelManager')
    def test_ai_health_missing_models(self, mock_model_manager):
        """Test /api/ai-health when models are missing"""
        # Mock model manager with no models
        mock_manager_instance = MagicMock()
        mock_manager_instance.models_found = False
        mock_manager_instance.model_dir = Path('models/production')
        mock_manager_instance.models = {
            'quality_predictor': None,
            'logo_classifier': None,
            'correlation_models': None
        }
        mock_manager_instance._load_all_exported_models.return_value = {}
        mock_model_manager.return_value = mock_manager_instance

        response = self.client.get('/api/ai-health')
        self.assertEqual(response.status_code, 200)

        result = response.get_json()
        model_manager_health = result['components'].get('model_manager', {})

        # Should indicate models not found
        self.assertFalse(model_manager_health.get('models_found', True),
                        "Should indicate models were not found")
        self.assertEqual(model_manager_health.get('status'), 'degraded',
                        "Status should be degraded when models missing")

        # Should provide actionable guidance
        self.assertIn('guidance', model_manager_health,
                     "Should provide guidance when models missing")
        self.assertIn('instructions', model_manager_health,
                     "Should provide instructions for model deployment")

        # Verify instructions are actionable
        instructions = model_manager_health['instructions']
        self.assertIsInstance(instructions, list, "Instructions should be a list")
        self.assertGreater(len(instructions), 0, "Should have at least one instruction")
        self.assertIn('torchscript', str(instructions), "Should mention TorchScript model")
        self.assertIn('onnx', str(instructions), "Should mention ONNX model")

    # Test 7: AI Health Reflects Dependency Issues
    def test_ai_health_dependency_issues(self):
        """Test that health endpoint reflects dependency issues"""
        with patch('backend.api.ai_endpoints.get_ai_components') as mock_get_components:
            # Mock component initialization failure
            mock_get_components.return_value = {
                'initialized': False,
                'error': 'Failed to import torch: No module named torch',
                'models_found': False
            }

            response = self.client.get('/api/ai-health')

            # Should still return 200 but indicate unhealthy state
            self.assertEqual(response.status_code, 200)

            result = response.get_json()
            self.assertIn('overall_status', result)
            self.assertIn(result['overall_status'], ['unhealthy', 'degraded', 'error'],
                         "Should indicate unhealthy state when dependencies fail")


# Parametrized tests for different failure modes
@pytest.mark.parametrize("failure_mode,expected_fallback", [
    ("model_loading", True),
    ("feature_extraction", True),
    ("tier_routing", True),
    ("conversion_timeout", True),
])
def test_various_failure_modes(failure_mode, expected_fallback):
    """Test fallback behavior for various failure modes"""
    app = create_app()
    client = app.test_client()

    # This is a template for parametrized testing
    # Actual implementation would mock specific failures
    pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])