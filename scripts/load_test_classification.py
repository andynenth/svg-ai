# scripts/load_test_classification.py
import concurrent.futures
import time
import requests
import statistics
from typing import List, Dict

class ClassificationLoadTest:
    def __init__(self, base_url: str = 'http://localhost:8001/api'):
        self.base_url = base_url
        self.results = []

    def single_classification_request(self, image_path: str, method: str = 'auto') -> Dict:
        """Single classification request for load testing"""
        start_time = time.time()

        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {'method': method}

                response = requests.post(
                    f'{self.base_url}/classify-logo',
                    files=files,
                    data=data,
                    timeout=30
                )

                end_time = time.time()

                return {
                    'success': response.status_code == 200,
                    'status_code': response.status_code,
                    'response_time': end_time - start_time,
                    'method': method,
                    'error': None if response.status_code == 200 else response.text
                }

        except Exception as e:
            return {
                'success': False,
                'status_code': -1,
                'response_time': time.time() - start_time,
                'method': method,
                'error': str(e)
            }

    def run_concurrent_load_test(self, num_requests: int = 50, max_workers: int = 10):
        """Run concurrent load test"""
        print(f"Starting load test: {num_requests} requests with {max_workers} concurrent workers")

        test_image = 'data/test/simple_geometric_logo.png'
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests
            futures = [
                executor.submit(self.single_classification_request, test_image)
                for _ in range(num_requests)
            ]

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                if len(results) % 10 == 0:
                    print(f"Completed {len(results)}/{num_requests} requests")

        return self.analyze_results(results)

    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze load test results"""
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]

        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            analysis = {
                'total_requests': len(results),
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'success_rate': len(successful_requests) / len(results) * 100,
                'response_times': {
                    'average': statistics.mean(response_times),
                    'median': statistics.median(response_times),
                    'min': min(response_times),
                    'max': max(response_times),
                    'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0
                },
                'percentiles': {
                    '95th': sorted(response_times)[int(0.95 * len(response_times))],
                    '99th': sorted(response_times)[int(0.99 * len(response_times))]
                }
            }
        else:
            analysis = {
                'total_requests': len(results),
                'successful_requests': 0,
                'failed_requests': len(failed_requests),
                'success_rate': 0,
                'errors': [r['error'] for r in failed_requests]
            }

        return analysis

    def run_sustained_load_test(self, duration_minutes: int = 10, requests_per_minute: int = 30):
        """Run sustained load test over time"""
        print(f"Starting sustained load test: {duration_minutes} minutes at {requests_per_minute} requests/minute")

        end_time = time.time() + (duration_minutes * 60)
        interval = 60.0 / requests_per_minute  # Time between requests
        results = []

        while time.time() < end_time:
            start_time = time.time()
            result = self.single_classification_request('data/test/simple_geometric_logo.png')
            results.append(result)

            # Wait for next interval
            elapsed = time.time() - start_time
            if elapsed < interval:
                time.sleep(interval - elapsed)

            if len(results) % 30 == 0:
                print(f"Completed {len(results)} requests...")

        return self.analyze_results(results)

# Run load tests
if __name__ == "__main__":
    load_tester = ClassificationLoadTest()

    # Test 1: Concurrent load test
    print("=" * 50)
    print("CONCURRENT LOAD TEST")
    print("=" * 50)
    concurrent_results = load_tester.run_concurrent_load_test(
        num_requests=50,
        max_workers=10
    )
    print(f"Success rate: {concurrent_results['success_rate']:.1f}%")
    print(f"Average response time: {concurrent_results['response_times']['average']:.3f}s")
    print(f"95th percentile: {concurrent_results['percentiles']['95th']:.3f}s")

    # Test 2: Sustained load test
    print("\n" + "=" * 50)
    print("SUSTAINED LOAD TEST")
    print("=" * 50)
    sustained_results = load_tester.run_sustained_load_test(
        duration_minutes=5,
        requests_per_minute=20
    )
    print(f"Success rate: {sustained_results['success_rate']:.1f}%")
    print(f"Average response time: {sustained_results['response_times']['average']:.3f}s")