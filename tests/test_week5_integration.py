# tests/test_week5_integration.py
import pytest
import time
import concurrent.futures
import io
from pathlib import Path
from typing import Dict
from PIL import Image
import numpy as np

class TestWeek5Integration:
    """Comprehensive testing of Week 5 backend enhancement"""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment with real test images"""
        self.test_images = self._prepare_test_images()
        self.performance_metrics = {}

        # Initialize client
        from backend.app import app
        self.client = app.test_client()

    def _prepare_test_images(self) -> Dict[str, str]:
        """Prepare diverse test images for comprehensive testing"""
        test_data_dir = Path("data/test")

        return {
            'simple': str(test_data_dir / "simple_geometric.png"),
            'text': str(test_data_dir / "text_based.png"),
            'gradient': str(test_data_dir / "gradient_logo.png"),
            'complex': str(test_data_dir / "complex_design.png"),
            'corrupted': str(test_data_dir / "corrupted_image.png")
        }

    def test_complete_ai_pipeline_all_tiers(self):
        """Test complete AI pipeline with all tiers"""
        for image_type, image_path in self.test_images.items():
            if image_type == 'corrupted':
                continue  # Skip corrupted for positive flow test

            # Upload test image
            file_id = self._upload_test_image(image_path)

            for tier in [1, 2, 3]:
                with pytest.subtest(image_type=image_type, tier=tier):
                    start_time = time.time()

                    response = self.client.post('/api/convert-ai', json={
                        'file_id': file_id,
                        'tier': tier,
                        'include_analysis': True
                    }, content_type='application/json')

                    processing_time = time.time() - start_time

                    # Record performance metrics
                    self._record_performance(image_type, tier, processing_time, response)

                    if response.status_code == 503:
                        # AI unavailable - acceptable for testing
                        pytest.skip("AI components unavailable")
                        continue

                    # Validate successful response
                    assert response.status_code == 200, f"Failed for {image_type} tier {tier}"
                    result = response.get_json()

                    # Validate response structure
                    self._validate_ai_response_structure(result, tier)

                    # Validate tier-specific performance
                    self._validate_tier_performance(tier, processing_time, result)

    def _validate_ai_response_structure(self, result: Dict, tier: int):
        """Validate AI response has correct structure"""
        assert result['success'] == True
        assert 'svg' in result
        assert 'ai_metadata' in result
        assert 'processing_time' in result

        ai_metadata = result['ai_metadata']
        assert 'tier_used' in ai_metadata
        assert ai_metadata['tier_used'] == tier
        assert 'routing' in ai_metadata

        # Validate SVG content
        svg_content = result['svg']
        assert svg_content.startswith('<?xml') or svg_content.startswith('<svg')
        assert 'svg' in svg_content.lower()

    def _validate_tier_performance(self, tier: int, processing_time: float, result: Dict):
        """Validate tier meets performance requirements"""
        # Performance targets from requirements
        tier_limits = {
            1: 0.5,   # Tier 1: <500ms
            2: 1.5,   # Tier 2: <1.5s
            3: 5.0    # Tier 3: <5s
        }

        assert processing_time < tier_limits[tier], \
            f"Tier {tier} took {processing_time:.2f}s, exceeds {tier_limits[tier]}s limit"

        # Check if quality prediction available
        ai_metadata = result['ai_metadata']
        if 'quality_prediction' in ai_metadata:
            quality = ai_metadata['quality_prediction']
            assert 0.0 <= quality <= 1.0, f"Invalid quality prediction: {quality}"

    def test_intelligent_routing_accuracy(self):
        """Test intelligent routing selects appropriate tiers"""
        file_id = self._upload_test_image(self.test_images['simple'])

        test_scenarios = [
            {'target_quality': 0.7, 'expected_tier_range': [1, 2]},
            {'target_quality': 0.9, 'expected_tier_range': [2, 3]},
            {'target_quality': 0.95, 'expected_tier_range': [3]},
            {'time_budget': 0.3, 'expected_tier_range': [1]},
            {'time_budget': 2.0, 'expected_tier_range': [1, 2]},
        ]

        for scenario in test_scenarios:
            with pytest.subtest(scenario=scenario):
                response = self.client.post('/api/convert-ai', json={
                    'file_id': file_id,
                    'tier': 'auto',
                    **scenario
                }, content_type='application/json')

                if response.status_code == 503:
                    pytest.skip("AI routing unavailable")
                    continue

                assert response.status_code == 200
                result = response.get_json()

                selected_tier = result['ai_metadata']['tier_used']
                expected_range = scenario['expected_tier_range']

                assert selected_tier in expected_range, \
                    f"Selected tier {selected_tier} not in expected range {expected_range} for {scenario}"

    def test_concurrent_ai_requests(self):
        """Test system handles concurrent AI requests"""
        file_id = self._upload_test_image(self.test_images['simple'])

        def ai_conversion_task():
            start_time = time.time()
            response = self.client.post('/api/convert-ai', json={
                'file_id': file_id,
                'tier': 1
            }, content_type='application/json')
            processing_time = time.time() - start_time

            return {
                'status_code': response.status_code,
                'processing_time': processing_time,
                'success': response.status_code == 200
            }

        # Test with 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(ai_conversion_task) for _ in range(10)]
            results = [future.result() for future in futures]

        # Analyze results
        successful_requests = [r for r in results if r['success']]
        avg_processing_time = sum(r['processing_time'] for r in successful_requests) / len(successful_requests) if successful_requests else 0

        # Validate concurrent performance
        assert len(successful_requests) >= 8, f"Only {len(successful_requests)}/10 requests succeeded"
        assert avg_processing_time < 2.0, f"Average processing time {avg_processing_time:.2f}s too high under load"

    def test_error_handling_edge_cases(self):
        """Test comprehensive error handling"""

        # Test invalid file ID
        response = self.client.post('/api/convert-ai', json={
            'file_id': 'nonexistent_file',
            'tier': 1
        }, content_type='application/json')
        assert response.status_code == 404

        # Test corrupted image
        if 'corrupted' in self.test_images:
            corrupted_file_id = self._upload_test_image(self.test_images['corrupted'])
            response = self.client.post('/api/convert-ai', json={
                'file_id': corrupted_file_id,
                'tier': 1
            }, content_type='application/json')
            # Should either fail gracefully or use fallback
            assert response.status_code in [200, 400, 500]

            if response.status_code == 200:
                result = response.get_json()
                # If succeeded, should indicate fallback was used
                assert 'ai_metadata' in result

        # Test invalid tier
        file_id = self._upload_test_image(self.test_images['simple'])
        response = self.client.post('/api/convert-ai', json={
            'file_id': file_id,
            'tier': 5  # Invalid tier
        }, content_type='application/json')
        assert response.status_code == 400

        # Test invalid quality target
        response = self.client.post('/api/convert-ai', json={
            'file_id': file_id,
            'tier': 1,
            'target_quality': 1.5  # Invalid quality > 1.0
        }, content_type='application/json')
        assert response.status_code == 400

    def test_fallback_mechanisms(self):
        """Test fallback when AI components fail"""
        file_id = self._upload_test_image(self.test_images['simple'])

        # Test AI conversion
        response = self.client.post('/api/convert-ai', json={
            'file_id': file_id,
            'tier': 1
        }, content_type='application/json')

        if response.status_code == 503:
            # AI unavailable - test fallback suggestion
            result = response.get_json()
            assert 'fallback_suggestion' in result
            assert '/api/convert' in result['fallback_suggestion']

        elif response.status_code == 200:
            # AI available - test that fallback field exists in metadata
            result = response.get_json()
            ai_metadata = result['ai_metadata']

            # May contain fallback information if partial failure occurred
            if 'fallback_used' in ai_metadata:
                assert isinstance(ai_metadata['fallback_used'], bool)

    def test_backward_compatibility_comprehensive(self):
        """Comprehensive backward compatibility testing"""
        file_id = self._upload_test_image(self.test_images['simple'])

        # Test original API exactly as before
        original_response = self.client.post('/api/convert', json={
            'file_id': file_id,
            'converter': 'vtracer',
            'color_precision': 4,
            'corner_threshold': 30
        }, content_type='application/json')

        assert original_response.status_code == 200
        original_result = original_response.get_json()

        # Validate original response structure unchanged
        assert 'success' in original_result
        assert 'svg' in original_result
        assert 'ssim' in original_result

        # Should NOT contain AI fields
        assert 'ai_metadata' not in original_result
        assert 'tier_used' not in original_result

        # Test that AI enhancement doesn't affect original endpoint performance
        start_time = time.time()
        for _ in range(5):
            response = self.client.post('/api/convert', json={
                'file_id': file_id,
                'converter': 'vtracer'
            }, content_type='application/json')
            assert response.status_code == 200

        avg_time = (time.time() - start_time) / 5
        assert avg_time < 1.0, f"Original API performance degraded: {avg_time:.2f}s average"

    def _upload_test_image(self, image_path: str) -> str:
        """Upload test image and return file_id"""
        try:
            with open(image_path, 'rb') as f:
                response = self.client.post('/api/upload',
                                          data={'file': (f, 'test.png')},
                                          content_type='multipart/form-data')

            assert response.status_code == 200
            result = response.get_json()
            return result['file_id']

        except FileNotFoundError:
            # Create a simple test image if file doesn't exist
            img = Image.new('RGB', (100, 100), color='white')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            response = self.client.post('/api/upload',
                                      data={'file': (img_bytes, 'test.png')},
                                      content_type='multipart/form-data')

            assert response.status_code == 200
            result = response.get_json()
            return result['file_id']

    def _record_performance(self, image_type: str, tier: int, processing_time: float, response):
        """Record performance metrics for analysis"""
        if image_type not in self.performance_metrics:
            self.performance_metrics[image_type] = {}

        if tier not in self.performance_metrics[image_type]:
            self.performance_metrics[image_type][tier] = []

        metrics = {
            'processing_time': processing_time,
            'status_code': response.status_code,
            'success': response.status_code == 200
        }

        if response.status_code == 200:
            result = response.get_json()
            if 'ai_metadata' in result:
                ai_metadata = result['ai_metadata']
                metrics.update({
                    'routing_time': ai_metadata.get('routing', {}).get('routing_time', 0),
                    'quality_prediction': ai_metadata.get('quality_prediction'),
                    'actual_quality': ai_metadata.get('actual_quality')
                })

        self.performance_metrics[image_type][tier].append(metrics)

    def get_performance_summary(self) -> Dict:
        """Get summary of recorded performance metrics"""
        summary = {}

        for image_type, tier_data in self.performance_metrics.items():
            summary[image_type] = {}

            for tier, metrics_list in tier_data.items():
                if metrics_list:
                    successful_metrics = [m for m in metrics_list if m['success']]

                    if successful_metrics:
                        avg_time = sum(m['processing_time'] for m in successful_metrics) / len(successful_metrics)
                        avg_routing = sum(m.get('routing_time', 0) for m in successful_metrics) / len(successful_metrics)

                        summary[image_type][tier] = {
                            'average_processing_time': avg_time,
                            'average_routing_time': avg_routing,
                            'success_count': len(successful_metrics),
                            'total_tests': len(metrics_list)
                        }

        return summary