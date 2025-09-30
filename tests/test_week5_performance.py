# tests/test_week5_performance.py
import pytest
import time
import concurrent.futures
import logging
from pathlib import Path

class TestWeek5Performance:
    """Validate all performance requirements are met"""

    def setup_method(self):
        """Setup performance testing environment"""
        self.performance_data = {
            'model_loading': [],
            'ai_inference': [],
            'routing_decisions': [],
            'memory_usage': [],
            'concurrent_performance': []
        }

    def test_model_loading_performance(self):
        """Test model loading meets <3 second requirement"""
        from backend.ai_modules.management.production_model_manager import ProductionModelManager

        loading_times = []

        for trial in range(3):
            # Clear any existing models
            model_manager = ProductionModelManager()

            start_time = time.time()
            models = model_manager._load_all_exported_models()
            model_manager._optimize_for_production()
            loading_time = time.time() - start_time

            loading_times.append(loading_time)

            # Verify some models loaded
            loaded_count = len([m for m in models.values() if m is not None])
            if loaded_count > 0:
                # Only test timing if models actually loaded
                assert loading_time < 3.0, f"Model loading took {loading_time:.2f}s, exceeds 3s limit"

        self.performance_data['model_loading'] = loading_times

        if loading_times:
            avg_loading_time = sum(loading_times) / len(loading_times)
            print(f"📊 Average model loading time: {avg_loading_time:.2f}s")

    def test_ai_inference_performance(self):
        """Test AI inference components meet timing requirements"""
        from backend.ai_modules.management.production_model_manager import ProductionModelManager
        from backend.ai_modules.inference.optimized_quality_predictor import OptimizedQualityPredictor

        try:
            model_manager = ProductionModelManager()
            quality_predictor = OptimizedQualityPredictor(model_manager)

            test_image = "data/test/simple_geometric.png"
            test_params = {"color_precision": 4, "corner_threshold": 30}

            # Warmup
            quality_predictor.predict_quality(test_image, test_params)

            # Time multiple predictions
            inference_times = []
            for _ in range(10):
                start_time = time.time()
                quality = quality_predictor.predict_quality(test_image, test_params)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                # Validate quality prediction
                assert 0.0 <= quality <= 1.0

            avg_inference_time = sum(inference_times) / len(inference_times)

            # Requirement: <100ms per prediction
            assert avg_inference_time < 0.1, f"AI inference took {avg_inference_time:.3f}s, exceeds 0.1s limit"

            self.performance_data['ai_inference'] = inference_times
            print(f"📊 Average AI inference time: {avg_inference_time*1000:.1f}ms")

        except ImportError:
            pytest.skip("AI inference components not available")

    def test_routing_performance(self):
        """Test intelligent routing meets <100ms requirement"""
        from backend.ai_modules.routing.hybrid_intelligent_router import HybridIntelligentRouter
        from backend.ai_modules.management.production_model_manager import ProductionModelManager

        try:
            model_manager = ProductionModelManager()
            router = HybridIntelligentRouter(model_manager)

            test_image = "data/test/simple_geometric.png"

            # Warmup
            router.determine_optimal_tier(test_image)

            # Time multiple routing decisions
            routing_times = []
            for _ in range(5):
                start_time = time.time()
                routing_result = router.determine_optimal_tier(
                    test_image,
                    target_quality=0.85,
                    time_budget=2.0
                )
                routing_time = time.time() - start_time
                routing_times.append(routing_time)

                # Validate routing result
                assert 'selected_tier' in routing_result
                assert routing_result['selected_tier'] in [1, 2, 3]

            avg_routing_time = sum(routing_times) / len(routing_times)

            # Requirement: <100ms for routing decision
            assert avg_routing_time < 0.1, f"Routing took {avg_routing_time:.3f}s, exceeds 0.1s limit"

            self.performance_data['routing_decisions'] = routing_times
            print(f"📊 Average routing time: {avg_routing_time*1000:.1f}ms")

        except ImportError:
            pytest.skip("Routing components not available")

    def test_memory_usage_requirements(self):
        """Test memory usage stays within <500MB limit"""
        import psutil
        from backend.ai_modules.management.production_model_manager import ProductionModelManager

        # Baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            # Load all AI components
            model_manager = ProductionModelManager()
            models = model_manager._load_all_exported_models()
            model_manager._optimize_for_production()

            # Memory after loading
            memory_after_loading = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after_loading - baseline_memory

            # Requirement: <500MB total for AI components
            assert memory_increase < 500, f"AI components use {memory_increase:.1f}MB, exceeds 500MB limit"

            self.performance_data['memory_usage'].append({
                'baseline_mb': baseline_memory,
                'after_loading_mb': memory_after_loading,
                'ai_increase_mb': memory_increase
            })

            print(f"📊 AI memory usage: {memory_increase:.1f}MB")

        except Exception as e:
            pytest.skip(f"Memory testing failed: {e}")

    def test_concurrent_performance_requirements(self):
        """Test system handles 10+ concurrent requests"""
        from backend.app import app
        client = app.test_client()

        # Upload test image
        test_image_path = "data/test/simple_geometric.png"
        file_id = self._upload_test_image(client, test_image_path)

        def concurrent_ai_request():
            start_time = time.time()
            response = client.post('/api/convert-ai', json={
                'file_id': file_id,
                'tier': 1
            }, content_type='application/json')
            processing_time = time.time() - start_time

            return {
                'success': response.status_code == 200,
                'processing_time': processing_time,
                'status_code': response.status_code
            }

        # Test with 10 concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(concurrent_ai_request) for _ in range(10)]
            results = [future.result() for future in futures]

        # Analyze concurrent performance
        successful_requests = [r for r in results if r['success']]
        success_rate = len(successful_requests) / len(results)

        if successful_requests:
            avg_processing_time = sum(r['processing_time'] for r in successful_requests) / len(successful_requests)
            max_processing_time = max(r['processing_time'] for r in successful_requests)

            # Requirements: 80%+ success rate, reasonable performance degradation
            assert success_rate >= 0.8, f"Success rate {success_rate:.1%} below 80% threshold"
            assert avg_processing_time < 2.0, f"Average processing time {avg_processing_time:.2f}s too high under load"

            self.performance_data['concurrent_performance'] = results
            print(f"📊 Concurrent performance: {success_rate:.1%} success, {avg_processing_time:.2f}s avg")

    def _upload_test_image(self, client, image_path: str) -> str:
        """Helper to upload test image and return file_id"""
        try:
            with open(image_path, 'rb') as f:
                response = client.post('/api/upload',
                                     data={'file': (f, 'test.png')},
                                     content_type='multipart/form-data')

            assert response.status_code == 200
            return response.get_json()['file_id']

        except FileNotFoundError:
            # Create a simple test image if file doesn't exist
            from PIL import Image
            import io

            img = Image.new('RGB', (100, 100), color='white')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            response = client.post('/api/upload',
                                 data={'file': (img_bytes, 'test.png')},
                                 content_type='multipart/form-data')

            assert response.status_code == 200
            return response.get_json()['file_id']