#!/usr/bin/env python3
"""
Production Load Testing Framework for 4-Tier SVG-AI System
Comprehensive load testing with production-level validation and metrics
"""

import asyncio
import time
import statistics
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import aiohttp
import requests
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoadTestRequest:
    """Load test request data"""
    request_id: str
    image_path: str
    parameters: Dict[str, Any]
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    response_time: float = 0.0
    quality_score: Optional[float] = None
    error_message: Optional[str] = None
    status_code: Optional[int] = None


@dataclass
class LoadTestMetrics:
    """Load test metrics and statistics"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    average_quality_score: float
    error_distribution: Dict[str, int]
    test_duration: float


class ProductionLoadTester:
    """Production-grade load testing framework"""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "tier4-test-key"):
        """Initialize load tester"""
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.test_dataset = self._load_production_test_dataset()
        self.session = None

    def _load_production_test_dataset(self) -> List[str]:
        """Load production test dataset"""
        test_data_dir = Path(__file__).parent.parent.parent / "data" / "logos"

        if test_data_dir.exists():
            # Load diverse test images
            test_images = []
            for category in ["simple_geometric", "text_based", "gradients", "complex"]:
                category_dir = test_data_dir / category
                if category_dir.exists():
                    images = list(category_dir.glob("*.png"))[:5]  # 5 images per category
                    test_images.extend([str(img) for img in images])

            if test_images:
                return test_images

        # Fallback to test image generation
        return [self._create_test_image() for _ in range(20)]

    def _create_test_image(self) -> str:
        """Create a test image for load testing"""
        try:
            from PIL import Image, ImageDraw
            import tempfile

            # Create a simple test image with random properties
            size = random.choice([(200, 200), (300, 300), (400, 400)])
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

            img = Image.new('RGB', size, 'white')
            draw = ImageDraw.Draw(img)

            # Draw random shapes
            for _ in range(random.randint(1, 5)):
                color = random.choice(colors)
                x1, y1 = random.randint(0, size[0]//2), random.randint(0, size[1]//2)
                x2, y2 = x1 + random.randint(50, 100), y1 + random.randint(50, 100)
                draw.rectangle([x1, y1, x2, y2], fill=color, outline='black')

            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w+b', suffix='.png', delete=False) as tmp:
                img.save(tmp.name, 'PNG')
                return tmp.name

        except Exception as e:
            logger.error(f"Failed to create test image: {e}")
            return "/tmp/claude/test_image.png"

    async def run_production_load_test(
        self,
        concurrent_users: int = 20,
        duration_minutes: int = 10,
        ramp_up_seconds: int = 30,
        target_rps: Optional[float] = None
    ) -> LoadTestMetrics:
        """Run comprehensive production load test"""

        logger.info(f"Starting production load test: {concurrent_users} concurrent users, {duration_minutes} minutes")

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        # Initialize metrics tracking
        requests: List[LoadTestRequest] = []
        active_requests = 0
        max_active_requests = 0

        # Create aiohttp session for concurrent requests
        connector = aiohttp.TCPConnector(limit=concurrent_users * 2)
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Authorization": f"Bearer {self.api_key}"}
        ) as session:

            # Ramp-up phase
            await self._ramp_up_users(session, concurrent_users, ramp_up_seconds, requests)

            # Main load testing phase
            tasks = []
            request_counter = 0

            while time.time() < end_time:
                # Control request rate if specified
                if target_rps:
                    await asyncio.sleep(1.0 / target_rps)

                # Maintain concurrent user load
                if len(tasks) < concurrent_users:
                    request_id = f"load_test_{request_counter}"
                    test_image = random.choice(self.test_dataset)
                    parameters = self._generate_test_parameters()

                    task = asyncio.create_task(
                        self._send_conversion_request(session, request_id, test_image, parameters)
                    )
                    tasks.append(task)
                    request_counter += 1

                # Collect completed requests
                done_tasks = [task for task in tasks if task.done()]
                for task in done_tasks:
                    try:
                        request_result = await task
                        requests.append(request_result)
                        active_requests = len(tasks) - len(done_tasks)
                        max_active_requests = max(max_active_requests, active_requests)
                    except Exception as e:
                        logger.error(f"Task failed: {e}")

                    tasks.remove(task)

                # Brief pause to prevent overwhelming
                await asyncio.sleep(0.1)

            # Wait for remaining tasks to complete
            if tasks:
                logger.info(f"Waiting for {len(tasks)} remaining requests to complete...")
                remaining_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in remaining_results:
                    if isinstance(result, LoadTestRequest):
                        requests.append(result)

        test_duration = time.time() - start_time

        # Calculate metrics
        metrics = self._calculate_load_test_metrics(requests, test_duration)

        # Log summary
        logger.info(f"Load test completed: {metrics.success_rate:.1%} success rate, "
                   f"{metrics.requests_per_second:.2f} RPS, "
                   f"P95: {metrics.p95_response_time:.2f}s")

        return metrics

    async def _ramp_up_users(
        self,
        session: aiohttp.ClientSession,
        target_users: int,
        ramp_up_seconds: int,
        requests: List[LoadTestRequest]
    ):
        """Gradually ramp up user load"""
        logger.info(f"Ramping up to {target_users} users over {ramp_up_seconds} seconds")

        interval = ramp_up_seconds / target_users
        tasks = []

        for i in range(target_users):
            request_id = f"rampup_{i}"
            test_image = random.choice(self.test_dataset)
            parameters = self._generate_test_parameters()

            task = asyncio.create_task(
                self._send_conversion_request(session, request_id, test_image, parameters)
            )
            tasks.append(task)

            # Wait before starting next user
            await asyncio.sleep(interval)

        # Collect ramp-up results
        rampup_results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in rampup_results:
            if isinstance(result, LoadTestRequest):
                requests.append(result)

    async def _send_conversion_request(
        self,
        session: aiohttp.ClientSession,
        request_id: str,
        image_path: str,
        parameters: Dict[str, Any]
    ) -> LoadTestRequest:
        """Send single conversion request"""

        request = LoadTestRequest(
            request_id=request_id,
            image_path=image_path,
            parameters=parameters,
            start_time=time.time()
        )

        try:
            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field('quality_target', str(parameters.get('quality_target', 0.85)))
            data.add_field('time_constraint', str(parameters.get('time_constraint', 60.0)))
            data.add_field('method_preference', parameters.get('method_preference', 'auto'))

            # Add image file
            with open(image_path, 'rb') as f:
                data.add_field(
                    'image',
                    f,
                    filename=Path(image_path).name,
                    content_type='image/png'
                )

                # Send request
                async with session.post(
                    f"{self.base_url}/api/v2/optimization/convert",
                    data=data
                ) as response:
                    request.status_code = response.status
                    request.end_time = time.time()
                    request.response_time = request.end_time - request.start_time

                    if response.status == 200:
                        result = await response.json()
                        request.success = result.get('success', False)
                        request.quality_score = result.get('actual_quality', 0.0)
                    else:
                        request.success = False
                        request.error_message = f"HTTP {response.status}: {await response.text()}"

        except Exception as e:
            request.end_time = time.time()
            request.response_time = request.end_time - request.start_time
            request.success = False
            request.error_message = str(e)

        return request

    def _generate_test_parameters(self) -> Dict[str, Any]:
        """Generate random test parameters"""
        return {
            'quality_target': random.uniform(0.7, 0.95),
            'time_constraint': random.choice([30.0, 60.0, 120.0, 180.0]),
            'method_preference': random.choice(['auto', 'vtracer', 'potrace', 'ai_enhanced'])
        }

    def _calculate_load_test_metrics(self, requests: List[LoadTestRequest], test_duration: float) -> LoadTestMetrics:
        """Calculate comprehensive load test metrics"""

        if not requests:
            return LoadTestMetrics(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                success_rate=0.0,
                average_response_time=0.0,
                median_response_time=0.0,
                p95_response_time=0.0,
                p99_response_time=0.0,
                min_response_time=0.0,
                max_response_time=0.0,
                requests_per_second=0.0,
                average_quality_score=0.0,
                error_distribution={},
                test_duration=test_duration
            )

        successful_requests = [r for r in requests if r.success]
        failed_requests = [r for r in requests if not r.success]

        response_times = [r.response_time for r in requests if r.response_time > 0]
        quality_scores = [r.quality_score for r in successful_requests if r.quality_score is not None]

        # Error distribution
        error_distribution = {}
        for request in failed_requests:
            error_key = request.error_message or "Unknown error"
            if request.status_code:
                error_key = f"HTTP {request.status_code}"
            error_distribution[error_key] = error_distribution.get(error_key, 0) + 1

        return LoadTestMetrics(
            total_requests=len(requests),
            successful_requests=len(successful_requests),
            failed_requests=len(failed_requests),
            success_rate=len(successful_requests) / len(requests),
            average_response_time=statistics.mean(response_times) if response_times else 0.0,
            median_response_time=statistics.median(response_times) if response_times else 0.0,
            p95_response_time=self._percentile(response_times, 95) if response_times else 0.0,
            p99_response_time=self._percentile(response_times, 99) if response_times else 0.0,
            min_response_time=min(response_times) if response_times else 0.0,
            max_response_time=max(response_times) if response_times else 0.0,
            requests_per_second=len(successful_requests) / test_duration if test_duration > 0 else 0.0,
            average_quality_score=statistics.mean(quality_scores) if quality_scores else 0.0,
            error_distribution=error_distribution,
            test_duration=test_duration
        )

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

    async def run_stress_test(
        self,
        max_users: int = 100,
        step_size: int = 10,
        step_duration: int = 60
    ) -> Dict[str, Any]:
        """Run stress test to find system breaking point"""

        logger.info(f"Starting stress test: scaling from {step_size} to {max_users} users")

        stress_results = {}
        current_users = step_size

        while current_users <= max_users:
            logger.info(f"Testing with {current_users} concurrent users...")

            # Run load test for this user level
            metrics = await self.run_production_load_test(
                concurrent_users=current_users,
                duration_minutes=step_duration // 60,
                ramp_up_seconds=min(30, step_duration // 4)
            )

            stress_results[current_users] = asdict(metrics)

            # Check if system is degrading
            if metrics.success_rate < 0.8 or metrics.p95_response_time > 30.0:
                logger.warning(f"System degradation detected at {current_users} users")
                stress_results['breaking_point'] = current_users
                break

            current_users += step_size

        return stress_results

    def generate_load_test_report(self, metrics: LoadTestMetrics, output_file: str = None) -> Dict[str, Any]:
        """Generate comprehensive load test report"""

        # SLA compliance check
        sla_targets = {
            'success_rate': 0.95,
            'p95_response_time': 15.0,
            'average_response_time': 8.0,
            'requests_per_second': 1.0,
            'average_quality_score': 0.8
        }

        sla_compliance = {
            'success_rate_sla': metrics.success_rate >= sla_targets['success_rate'],
            'p95_response_time_sla': metrics.p95_response_time <= sla_targets['p95_response_time'],
            'average_response_time_sla': metrics.average_response_time <= sla_targets['average_response_time'],
            'throughput_sla': metrics.requests_per_second >= sla_targets['requests_per_second'],
            'quality_sla': metrics.average_quality_score >= sla_targets['average_quality_score']
        }

        overall_sla_compliance = all(sla_compliance.values())

        report = {
            'test_summary': {
                'timestamp': datetime.now().isoformat(),
                'test_duration': metrics.test_duration,
                'total_requests': metrics.total_requests,
                'successful_requests': metrics.successful_requests,
                'success_rate': metrics.success_rate,
                'overall_sla_compliance': overall_sla_compliance
            },
            'performance_metrics': {
                'response_times': {
                    'average': metrics.average_response_time,
                    'median': metrics.median_response_time,
                    'p95': metrics.p95_response_time,
                    'p99': metrics.p99_response_time,
                    'min': metrics.min_response_time,
                    'max': metrics.max_response_time
                },
                'throughput': {
                    'requests_per_second': metrics.requests_per_second,
                    'quality_score': metrics.average_quality_score
                }
            },
            'sla_compliance': {
                'targets': sla_targets,
                'results': sla_compliance,
                'overall_compliance': overall_sla_compliance
            },
            'error_analysis': {
                'failed_requests': metrics.failed_requests,
                'error_rate': metrics.failed_requests / metrics.total_requests if metrics.total_requests > 0 else 0,
                'error_distribution': metrics.error_distribution
            },
            'recommendations': self._generate_performance_recommendations(metrics, sla_compliance)
        }

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Load test report saved to {output_file}")

        return report

    def _generate_performance_recommendations(
        self,
        metrics: LoadTestMetrics,
        sla_compliance: Dict[str, bool]
    ) -> List[str]:
        """Generate performance optimization recommendations"""

        recommendations = []

        if not sla_compliance['success_rate_sla']:
            recommendations.append(
                f"Success rate ({metrics.success_rate:.1%}) below target. "
                "Consider improving error handling and system stability."
            )

        if not sla_compliance['p95_response_time_sla']:
            recommendations.append(
                f"P95 response time ({metrics.p95_response_time:.2f}s) exceeds target. "
                "Consider optimizing processing pipeline or adding more resources."
            )

        if not sla_compliance['throughput_sla']:
            recommendations.append(
                f"Throughput ({metrics.requests_per_second:.2f} RPS) below target. "
                "Consider horizontal scaling or performance optimization."
            )

        if not sla_compliance['quality_sla']:
            recommendations.append(
                f"Average quality score ({metrics.average_quality_score:.2f}) below target. "
                "Review quality prediction models and optimization algorithms."
            )

        if metrics.error_distribution:
            top_error = max(metrics.error_distribution.items(), key=lambda x: x[1])
            recommendations.append(
                f"Most common error: '{top_error[0]}' ({top_error[1]} occurrences). "
                "Focus on resolving this error type."
            )

        if not recommendations:
            recommendations.append("System performance meets all SLA targets. Consider load testing with higher concurrent users.")

        return recommendations


async def main():
    """Main load testing function"""
    import argparse

    parser = argparse.ArgumentParser(description="Production Load Testing for 4-Tier SVG-AI System")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for API")
    parser.add_argument("--api-key", default="tier4-test-key", help="API key for authentication")
    parser.add_argument("--users", type=int, default=20, help="Number of concurrent users")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in minutes")
    parser.add_argument("--ramp-up", type=int, default=30, help="Ramp-up time in seconds")
    parser.add_argument("--stress-test", action="store_true", help="Run stress test instead of load test")
    parser.add_argument("--max-users", type=int, default=100, help="Maximum users for stress test")
    parser.add_argument("--output", default="load_test_report.json", help="Output report file")

    args = parser.parse_args()

    # Create load tester
    tester = ProductionLoadTester(args.url, args.api_key)

    try:
        if args.stress_test:
            # Run stress test
            logger.info("Running stress test...")
            results = await tester.run_stress_test(max_users=args.max_users)

            # Save stress test results
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\nStress test completed. Results saved to {args.output}")

        else:
            # Run load test
            logger.info("Running load test...")
            metrics = await tester.run_production_load_test(
                concurrent_users=args.users,
                duration_minutes=args.duration,
                ramp_up_seconds=args.ramp_up
            )

            # Generate report
            report = tester.generate_load_test_report(metrics, args.output)

            # Print summary
            print("\n" + "="*80)
            print("PRODUCTION LOAD TEST RESULTS")
            print("="*80)
            print(f"Test Duration: {metrics.test_duration:.2f} seconds")
            print(f"Total Requests: {metrics.total_requests}")
            print(f"Successful Requests: {metrics.successful_requests}")
            print(f"Success Rate: {metrics.success_rate:.1%}")
            print(f"Requests per Second: {metrics.requests_per_second:.2f}")
            print(f"Average Response Time: {metrics.average_response_time:.2f}s")
            print(f"P95 Response Time: {metrics.p95_response_time:.2f}s")
            print(f"Average Quality Score: {metrics.average_quality_score:.2f}")
            print(f"SLA Compliance: {'✅ PASS' if report['sla_compliance']['overall_compliance'] else '❌ FAIL'}")
            print(f"\nDetailed report: {args.output}")
            print("="*80)

    except Exception as e:
        logger.error(f"Load test failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))