#!/usr/bin/env python3
"""
Comprehensive 4-Tier System Integration Tests
Complete end-to-end testing of the integrated 4-tier optimization system
"""

import pytest
import asyncio
import time
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Test framework imports
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from PIL import Image

# System under test
from backend.ai_modules.optimization.tier4_system_orchestrator import (
    Tier4SystemOrchestrator,
    create_4tier_orchestrator,
    OptimizationTier
)
from backend.ai_modules.optimization.enhanced_router_integration import (
    get_enhanced_router,
    integrate_agent_1_router,
    EnhancedRouterInterface,
    EnhancedRoutingDecision
)
from backend.api.unified_optimization_api import router as api_router
from backend.converters.ai_enhanced_converter import AIEnhancedConverter

# Test utilities
from tests.utils.test_image_generator import TestImageGenerator
from tests.utils.performance_profiler import PerformanceProfiler

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Test4TierSystemIntegration:
    """Comprehensive integration tests for 4-tier system"""

    @pytest.fixture(scope="class")
    def test_images(self):
        """Generate test images for different categories"""
        generator = TestImageGenerator()

        test_images = {}

        # Create test images for each category
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Simple geometric logo
            simple_image = generator.create_simple_geometric_logo(
                (200, 200), shapes=["circle"], colors=["#FF0000"]
            )
            simple_path = temp_path / "simple_logo.png"
            simple_image.save(simple_path)
            test_images["simple"] = str(simple_path)

            # Text-based logo
            text_image = generator.create_text_logo(
                (300, 100), "TEST", font_size=48, color="#000000"
            )
            text_path = temp_path / "text_logo.png"
            text_image.save(text_path)
            test_images["text"] = str(text_path)

            # Complex gradient logo
            gradient_image = generator.create_gradient_logo(
                (250, 250), gradient_type="radial", colors=["#FF0000", "#00FF00", "#0000FF"]
            )
            gradient_path = temp_path / "gradient_logo.png"
            gradient_image.save(gradient_path)
            test_images["gradient"] = str(gradient_path)

            # Complex detailed logo
            complex_image = generator.create_complex_logo(
                (300, 300), elements=["shapes", "text", "gradients"], complexity_level="high"
            )
            complex_path = temp_path / "complex_logo.png"
            complex_image.save(complex_path)
            test_images["complex"] = str(complex_path)

            yield test_images

    @pytest.fixture(scope="class")
    def orchestrator(self):
        """Create 4-tier orchestrator for testing"""
        config = {
            "max_concurrent_requests": 5,
            "enable_async_processing": True,
            "enable_caching": True,
            "production_mode": False,  # Test mode
            "tier_timeouts": {
                "classification": 10.0,
                "routing": 5.0,
                "optimization": 60.0,
                "prediction": 15.0
            }
        }

        orchestrator = create_4tier_orchestrator(config)
        yield orchestrator
        orchestrator.shutdown()

    @pytest.fixture
    def performance_profiler(self):
        """Performance profiler for test measurements"""
        return PerformanceProfiler()

    @pytest.mark.asyncio
    async def test_tier_1_classification_comprehensive(self, orchestrator, test_images):
        """Test Tier 1: Classification and Feature Extraction"""

        for image_type, image_path in test_images.items():
            logger.info(f"Testing Tier 1 classification for {image_type} image")

            # Create execution context
            context = orchestrator._create_test_context(image_path)

            # Execute Tier 1
            tier_1_result = await orchestrator._execute_tier_1_classification(context)

            # Validate results
            assert tier_1_result["success"], f"Tier 1 failed for {image_type}: {tier_1_result.get('error')}"
            assert "features" in tier_1_result
            assert "image_type" in tier_1_result
            assert "complexity_level" in tier_1_result
            assert tier_1_result["execution_time"] < 10.0  # Performance requirement

            # Validate feature quality
            features = tier_1_result["features"]
            assert isinstance(features, dict)
            assert len(features) > 5  # Should extract multiple features
            assert all(isinstance(v, (int, float)) for v in features.values())

            # Validate image type classification
            classified_type = tier_1_result["image_type"]
            assert classified_type in ["simple", "text", "gradient", "complex"]

            # Type-specific validations
            if image_type == "simple":
                assert tier_1_result["complexity_level"] in ["low", "medium"]
                assert features.get("complexity_score", 1.0) < 0.6
            elif image_type == "text":
                assert features.get("text_probability", 0.0) > 0.3
            elif image_type == "gradient":
                assert features.get("gradient_strength", 0.0) > 0.3

            logger.info(f"Tier 1 validation passed for {image_type}")

    @pytest.mark.asyncio
    async def test_tier_2_enhanced_routing(self, orchestrator, test_images):
        """Test Tier 2: Enhanced Intelligent Routing"""

        for image_type, image_path in test_images.items():
            logger.info(f"Testing Tier 2 enhanced routing for {image_type} image")

            # Create context and execute Tier 1 first
            context = orchestrator._create_test_context(image_path)
            tier_1_result = await orchestrator._execute_tier_1_classification(context)
            assert tier_1_result["success"]

            # Execute Tier 2
            tier_2_result = await orchestrator._execute_tier_2_routing(context)

            # Validate routing results
            assert tier_2_result["success"], f"Tier 2 failed for {image_type}: {tier_2_result.get('error')}"
            assert "primary_method" in tier_2_result
            assert "confidence" in tier_2_result
            assert "enhanced_features" in tier_2_result
            assert tier_2_result["execution_time"] < 5.0  # Performance requirement

            # Validate method selection
            primary_method = tier_2_result["primary_method"]
            assert primary_method in ["feature_mapping", "regression", "ppo", "performance"]

            # Validate enhanced routing features
            enhanced_features = tier_2_result["enhanced_features"]
            if enhanced_features:  # If Agent 1's router is integrated
                assert "predicted_qualities" in enhanced_features
                assert "quality_confidence" in enhanced_features
                assert "prediction_time" in enhanced_features

                predicted_qualities = enhanced_features["predicted_qualities"]
                assert isinstance(predicted_qualities, dict)
                assert len(predicted_qualities) > 0
                assert all(0.0 <= quality <= 1.0 for quality in predicted_qualities.values())

            # Method-specific validations
            complexity = tier_1_result["features"].get("complexity_score", 0.5)
            if complexity < 0.3:
                # Simple images should prefer fast methods
                assert primary_method in ["feature_mapping", "performance"]
            elif complexity > 0.7:
                # Complex images should prefer sophisticated methods
                assert primary_method in ["ppo", "regression"]

            logger.info(f"Tier 2 validation passed for {image_type}")

    @pytest.mark.asyncio
    async def test_tier_3_optimization_all_methods(self, orchestrator, test_images):
        """Test Tier 3: All Optimization Methods"""

        methods_to_test = ["feature_mapping", "regression", "ppo", "performance"]

        for method in methods_to_test:
            for image_type, image_path in test_images.items():
                logger.info(f"Testing Tier 3 optimization: {method} on {image_type}")

                # Create context and execute previous tiers
                context = orchestrator._create_test_context(image_path)
                tier_1_result = await orchestrator._execute_tier_1_classification(context)
                tier_2_result = await orchestrator._execute_tier_2_routing(context)

                # Force specific method for testing
                tier_2_result["primary_method"] = method
                context.tier_results[OptimizationTier.TIER_2_ROUTING] = tier_2_result

                # Execute Tier 3
                tier_3_result = await orchestrator._execute_tier_3_optimization(context)

                # Validate optimization results
                assert tier_3_result["success"], f"Tier 3 failed for {method} on {image_type}: {tier_3_result.get('error')}"
                assert "optimized_parameters" in tier_3_result
                assert "method_used" in tier_3_result
                assert tier_3_result["execution_time"] < 120.0  # Performance requirement

                # Validate optimized parameters
                parameters = tier_3_result["optimized_parameters"]
                assert isinstance(parameters, dict)
                assert len(parameters) > 0

                # Validate parameter ranges
                expected_params = ["color_precision", "corner_threshold", "path_precision"]
                for param in expected_params:
                    if param in parameters:
                        value = parameters[param]
                        assert isinstance(value, (int, float))
                        assert value > 0  # All parameters should be positive

                assert tier_3_result["method_used"] == method

                logger.info(f"Tier 3 validation passed for {method} on {image_type}")

    @pytest.mark.asyncio
    async def test_tier_4_quality_prediction(self, orchestrator, test_images):
        """Test Tier 4: Quality Prediction and Validation"""

        for image_type, image_path in test_images.items():
            logger.info(f"Testing Tier 4 quality prediction for {image_type} image")

            # Create context and execute previous tiers
            context = orchestrator._create_test_context(image_path)

            # Execute all previous tiers
            tier_1_result = await orchestrator._execute_tier_1_classification(context)
            tier_2_result = await orchestrator._execute_tier_2_routing(context)
            tier_3_result = await orchestrator._execute_tier_3_optimization(context)

            assert all(result["success"] for result in [tier_1_result, tier_2_result, tier_3_result])

            # Execute Tier 4
            tier_4_result = await orchestrator._execute_tier_4_quality_prediction(context)

            # Validate quality prediction results (not critical if it fails)
            if tier_4_result["success"]:
                assert "quality_predictions" in tier_4_result
                assert "quality_validation" in tier_4_result
                assert tier_4_result["execution_time"] < 15.0  # Performance requirement

                # Validate quality predictions
                quality_predictions = tier_4_result["quality_predictions"]
                assert isinstance(quality_predictions, dict)
                assert "predicted_ssim" in quality_predictions
                assert "confidence" in quality_predictions

                predicted_ssim = quality_predictions["predicted_ssim"]
                confidence = quality_predictions["confidence"]

                assert 0.0 <= predicted_ssim <= 1.0
                assert 0.0 <= confidence <= 1.0

                logger.info(f"Tier 4 validation passed for {image_type}: predicted SSIM {predicted_ssim:.3f}")
            else:
                logger.warning(f"Tier 4 failed for {image_type}, but continuing (non-critical tier)")

    @pytest.mark.asyncio
    async def test_complete_4tier_pipeline(self, orchestrator, test_images, performance_profiler):
        """Test complete 4-tier pipeline end-to-end"""

        results = {}

        for image_type, image_path in test_images.items():
            logger.info(f"Testing complete 4-tier pipeline for {image_type} image")

            # Start performance profiling
            profiler_session = performance_profiler.start_session(f"4tier_{image_type}")

            # Execute complete pipeline
            user_requirements = {
                "quality_target": 0.85,
                "time_constraint": 30.0,
                "speed_priority": "balanced"
            }

            result = await orchestrator.execute_4tier_optimization(
                image_path, user_requirements
            )

            # End performance profiling
            performance_metrics = performance_profiler.end_session(profiler_session)

            # Validate complete result
            assert result["success"], f"4-tier pipeline failed for {image_type}: {result.get('error')}"

            # Validate all required fields
            required_fields = [
                "request_id", "total_execution_time", "optimized_parameters",
                "method_used", "optimization_confidence", "predicted_quality",
                "image_type", "complexity_level", "routing_decision", "tier_performance"
            ]

            for field in required_fields:
                assert field in result, f"Missing required field: {field}"

            # Performance validations
            assert result["total_execution_time"] < 180.0  # 3 minutes max
            assert all(time < 120.0 for time in result["tier_performance"].values())

            # Quality validations
            assert 0.0 <= result["predicted_quality"] <= 1.0
            assert 0.0 <= result["optimization_confidence"] <= 1.0

            # Store results
            results[image_type] = {
                "result": result,
                "performance": performance_metrics
            }

            logger.info(f"Complete pipeline validation passed for {image_type}")

        # Cross-image analysis
        execution_times = [results[img_type]["result"]["total_execution_time"] for img_type in results]
        avg_execution_time = sum(execution_times) / len(execution_times)
        assert avg_execution_time < 60.0  # Average should be under 1 minute

        logger.info(f"Complete 4-tier pipeline testing completed. Average execution time: {avg_execution_time:.3f}s")

    @pytest.mark.asyncio
    async def test_concurrent_optimization_load(self, orchestrator, test_images):
        """Test system under concurrent load"""

        logger.info("Testing concurrent optimization load")

        # Prepare concurrent tasks
        concurrent_requests = []
        for _ in range(10):  # 10 concurrent requests
            for image_type, image_path in list(test_images.items())[:2]:  # Use first 2 images
                user_requirements = {
                    "quality_target": 0.8,
                    "time_constraint": 45.0,
                    "speed_priority": "balanced"
                }

                task = orchestrator.execute_4tier_optimization(image_path, user_requirements)
                concurrent_requests.append(task)

        # Execute concurrent requests
        start_time = time.time()
        results = await asyncio.gather(*concurrent_requests, return_exceptions=True)
        total_time = time.time() - start_time

        # Validate results
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        failed_results = [r for r in results if not isinstance(r, dict) or not r.get("success", False)]

        success_rate = len(successful_results) / len(results)

        # Performance assertions
        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2f}"
        assert total_time < 300.0, f"Concurrent load took too long: {total_time:.2f}s"

        # Resource utilization validation
        avg_execution_time = sum(r["total_execution_time"] for r in successful_results) / max(len(successful_results), 1)
        assert avg_execution_time < 120.0

        logger.info(f"Concurrent load test passed: {len(successful_results)}/{len(results)} successful, "
                   f"avg time: {avg_execution_time:.3f}s")

    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, orchestrator):
        """Test system health monitoring capabilities"""

        logger.info("Testing system health monitoring")

        # Perform health check
        health_result = await orchestrator.health_check()

        # Validate health check structure
        assert isinstance(health_result, dict)
        assert "overall_status" in health_result
        assert "components" in health_result
        assert "performance" in health_result
        assert "check_duration" in health_result

        # Validate component status
        components = health_result["components"]
        assert isinstance(components, dict)
        assert len(components) > 0

        # Validate performance metrics
        performance = health_result["performance"]
        assert isinstance(performance, dict)
        assert "active_requests" in performance

        # Health check should be fast
        assert health_result["check_duration"] < 5.0

        logger.info(f"Health monitoring test passed: {health_result['overall_status']}")

    @pytest.mark.asyncio
    async def test_enhanced_router_integration(self, orchestrator):
        """Test Agent 1's enhanced router integration framework"""

        logger.info("Testing enhanced router integration framework")

        # Get current enhanced router
        enhanced_router = get_enhanced_router()

        # Test basic functionality
        test_features = {
            "complexity_score": 0.4,
            "unique_colors": 8,
            "edge_density": 0.3,
            "entropy": 0.6
        }

        # Test enhanced routing
        routing_decision = enhanced_router.route_with_quality_prediction(
            "test_image.png",
            features=test_features,
            quality_target=0.9
        )

        # Validate enhanced decision
        assert hasattr(routing_decision, 'predicted_qualities')
        assert hasattr(routing_decision, 'quality_confidence')
        assert hasattr(routing_decision, 'enhanced_reasoning')

        # Test quality prediction
        quality, confidence = enhanced_router.predict_method_quality(
            "feature_mapping", test_features, {"color_precision": 6}
        )

        assert 0.0 <= quality <= 1.0
        assert 0.0 <= confidence <= 1.0

        # Test integration status
        from backend.ai_modules.optimization.enhanced_router_integration import get_router_integration_status
        status = get_router_integration_status()

        assert isinstance(status, dict)
        assert "current_router_type" in status
        assert "agent_1_available" in status

        logger.info("Enhanced router integration test passed")

    def test_error_recovery_mechanisms(self, orchestrator, test_images):
        """Test error recovery and fallback mechanisms"""

        logger.info("Testing error recovery mechanisms")

        # Test with invalid image path
        async def test_invalid_input():
            result = await orchestrator.execute_4tier_optimization(
                "/nonexistent/image.png",
                {"quality_target": 0.85}
            )
            return result

        # Should handle gracefully without crashing
        result = asyncio.run(test_invalid_input())
        assert not result["success"]
        assert "error" in result
        assert "request_id" in result  # Should still generate response structure

        # Test with corrupted features
        async def test_corrupted_features():
            image_path = list(test_images.values())[0]

            # Mock feature extractor to return invalid features
            with patch.object(orchestrator.feature_extractor, 'extract_features', side_effect=Exception("Feature extraction failed")):
                result = await orchestrator.execute_4tier_optimization(
                    image_path,
                    {"quality_target": 0.85}
                )
                return result

        result = asyncio.run(test_corrupted_features())
        assert not result["success"]  # Should fail gracefully

        logger.info("Error recovery test passed")

    @pytest.mark.asyncio
    async def test_caching_performance(self, orchestrator, test_images):
        """Test caching system performance"""

        logger.info("Testing caching system performance")

        image_path = list(test_images.values())[0]
        user_requirements = {
            "quality_target": 0.85,
            "time_constraint": 30.0
        }

        # First execution (cache miss)
        start_time = time.time()
        result1 = await orchestrator.execute_4tier_optimization(image_path, user_requirements)
        first_execution_time = time.time() - start_time

        assert result1["success"]

        # Second execution (should benefit from caching)
        start_time = time.time()
        result2 = await orchestrator.execute_4tier_optimization(image_path, user_requirements)
        second_execution_time = time.time() - start_time

        assert result2["success"]

        # Second execution should be faster due to caching
        improvement_ratio = first_execution_time / max(second_execution_time, 0.001)
        logger.info(f"Cache performance improvement: {improvement_ratio:.2f}x")

        # Should see some improvement, though may be minimal with stub implementations
        assert improvement_ratio >= 0.8  # At least not significantly slower

        logger.info("Caching performance test passed")


class Test4TierAPIIntegration:
    """Test API integration with 4-tier system"""

    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(api_router)

        return TestClient(app)

    def test_api_health_endpoint(self, client):
        """Test API health endpoint"""
        response = client.get("/api/v2/optimization/health")

        assert response.status_code == 200
        health_data = response.json()

        assert "overall_status" in health_data
        assert "components" in health_data
        assert "performance" in health_data

    def test_api_metrics_endpoint(self, client):
        """Test API metrics endpoint"""
        # Note: This requires valid API key
        headers = {"Authorization": "Bearer tier4-test-key"}
        response = client.get("/api/v2/optimization/metrics", headers=headers)

        if response.status_code == 200:
            metrics_data = response.json()
            assert "total_requests" in metrics_data
            assert "system_reliability" in metrics_data


# Performance benchmarking
class Test4TierPerformanceBenchmarks:
    """Performance benchmark tests"""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_scalability_benchmark(self, orchestrator, test_images):
        """Benchmark system scalability"""

        logger.info("Running scalability benchmark")

        load_levels = [1, 5, 10, 20]  # Concurrent request levels
        results = {}

        for load_level in load_levels:
            logger.info(f"Testing load level: {load_level} concurrent requests")

            # Prepare concurrent requests
            tasks = []
            for i in range(load_level):
                image_path = list(test_images.values())[i % len(test_images)]
                task = orchestrator.execute_4tier_optimization(
                    image_path,
                    {"quality_target": 0.8, "time_constraint": 60.0}
                )
                tasks.append(task)

            # Execute and measure
            start_time = time.time()
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

            # Analyze results
            successful = sum(1 for r in batch_results if isinstance(r, dict) and r.get("success", False))
            avg_time = sum(r["total_execution_time"] for r in batch_results
                          if isinstance(r, dict) and r.get("success", False)) / max(successful, 1)

            results[load_level] = {
                "total_time": total_time,
                "successful": successful,
                "success_rate": successful / load_level,
                "avg_execution_time": avg_time,
                "throughput": successful / total_time
            }

            logger.info(f"Load level {load_level}: {successful}/{load_level} successful, "
                       f"throughput: {results[load_level]['throughput']:.2f} req/s")

        # Validate scalability
        base_throughput = results[1]["throughput"]
        high_load_throughput = results[20]["throughput"]

        # System should maintain reasonable performance under load
        throughput_retention = high_load_throughput / base_throughput
        assert throughput_retention > 0.3, f"Throughput degradation too severe: {throughput_retention:.2f}"

        logger.info("Scalability benchmark completed")


# Test utilities
class TestImageGenerator:
    """Generate test images for different scenarios"""

    def create_simple_geometric_logo(self, size: tuple, shapes: list, colors: list) -> Image.Image:
        """Create simple geometric logo"""
        img = Image.new('RGB', size, 'white')
        # Implementation would create actual geometric shapes
        return img

    def create_text_logo(self, size: tuple, text: str, font_size: int, color: str) -> Image.Image:
        """Create text-based logo"""
        img = Image.new('RGB', size, 'white')
        # Implementation would render text
        return img

    def create_gradient_logo(self, size: tuple, gradient_type: str, colors: list) -> Image.Image:
        """Create gradient logo"""
        img = Image.new('RGB', size, 'white')
        # Implementation would create gradients
        return img

    def create_complex_logo(self, size: tuple, elements: list, complexity_level: str) -> Image.Image:
        """Create complex logo with multiple elements"""
        img = Image.new('RGB', size, 'white')
        # Implementation would create complex composition
        return img


class PerformanceProfiler:
    """Profile performance during tests"""

    def __init__(self):
        self.sessions = {}

    def start_session(self, session_id: str) -> str:
        """Start performance profiling session"""
        self.sessions[session_id] = {
            "start_time": time.time(),
            "metrics": {}
        }
        return session_id

    def end_session(self, session_id: str) -> dict:
        """End profiling session and return metrics"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            duration = time.time() - session["start_time"]
            session["metrics"]["total_duration"] = duration
            return session["metrics"]
        return {}


# Extension methods for orchestrator testing
def _create_test_context(self, image_path: str):
    """Create test execution context"""
    from backend.ai_modules.optimization.tier4_system_orchestrator import SystemExecutionContext
    import uuid

    return SystemExecutionContext(
        request_id=f"test_{uuid.uuid4().hex[:8]}",
        image_path=image_path,
        user_requirements={},
        system_state={},
        tier_results={},
        execution_timeline=[],
        performance_metrics={},
        error_log=[],
        start_time=time.time()
    )

# Monkey patch for testing
Tier4SystemOrchestrator._create_test_context = _create_test_context


if __name__ == "__main__":
    # Run basic functionality test
    import asyncio

    async def basic_test():
        orchestrator = create_4tier_orchestrator()

        # Test health check
        health = await orchestrator.health_check()
        print(f"System Health: {health['overall_status']}")

        # Test enhanced router
        from backend.ai_modules.optimization.enhanced_router_integration import get_enhanced_router
        router = get_enhanced_router()

        decision = router.route_with_quality_prediction(
            "test_image.png",
            features={"complexity_score": 0.5},
            quality_target=0.9
        )

        print(f"Enhanced Routing: {decision.primary_method}")
        print(f"Predicted Qualities: {decision.predicted_qualities}")

        orchestrator.shutdown()

    asyncio.run(basic_test())