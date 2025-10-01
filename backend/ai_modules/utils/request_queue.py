"""
Request Queuing System for AI SVG Converter

Queue Architecture:
- Priority-based queuing (1=highest, 10=lowest)
- Worker pool with configurable size
- Rate limiting with adaptive throttling
- Monitoring and statistics

Priorities:
- High (1): VIP users, urgent requests
- Normal (5): Regular conversions
- Low (10): Batch processing, background tasks

Rate Limiting:
- Requests per minute limits
- Burst handling
- Adaptive throttling based on system load
"""

import queue
import threading
import time
import uuid
import psutil
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Dict
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)


@dataclass(order=True)
class QueuedRequest:
    """Represents a queued request with priority ordering"""
    priority: int
    request_id: str = field(compare=False)
    timestamp: datetime = field(compare=False)
    data: Any = field(compare=False)
    callback: Optional[Callable] = field(compare=False)
    timeout: Optional[float] = field(compare=False, default=30.0)
    user_id: Optional[str] = field(compare=False, default=None)


class RequestQueue:
    """Priority-based request queue with worker pool"""

    def __init__(self, max_workers: int = 4, max_queue_size: int = 100):
        self.queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.workers = []
        self.max_workers = max_workers
        self.running = False
        self.stats = {
            'processed': 0,
            'failed': 0,
            'rejected': 0,
            'avg_wait_time': 0,
            'total_wait_time': 0
        }
        self._lock = threading.RLock()

    def start(self):
        """Start worker threads"""
        with self._lock:
            if self.running:
                return

            self.running = True
            for i in range(self.max_workers):
                worker = threading.Thread(
                    target=self._worker,
                    name=f"QueueWorker-{i}",
                    daemon=True
                )
                worker.start()
                self.workers.append(worker)

            logger.info(f"Started {self.max_workers} queue workers")

    def stop(self):
        """Stop worker threads"""
        with self._lock:
            if not self.running:
                return

            self.running = False

            # Add poison pills to wake up workers
            for _ in range(self.max_workers):
                try:
                    self.queue.put_nowait(QueuedRequest(
                        priority=999,
                        request_id='STOP',
                        timestamp=datetime.now(),
                        data=None
                    ))
                except queue.Full:
                    pass

            # Wait for workers to finish
            for worker in self.workers:
                worker.join(timeout=5)

            self.workers.clear()
            logger.info("Stopped all queue workers")

    def _worker(self):
        """Worker thread processing requests"""
        while self.running:
            try:
                request = self.queue.get(timeout=1)

                if request.request_id == 'STOP':
                    break

                # Calculate wait time
                wait_time = (datetime.now() - request.timestamp).total_seconds()
                self._update_avg_wait_time(wait_time)

                # Check timeout
                if request.timeout and wait_time > request.timeout:
                    logger.warning(f"Request {request.request_id} timed out after {wait_time:.2f}s")
                    if request.callback:
                        request.callback({'error': 'Request timeout', 'wait_time': wait_time})
                    with self._lock:
                        self.stats['failed'] += 1
                    continue

                # Process request
                try:
                    logger.debug(f"Processing request {request.request_id} (waited {wait_time:.2f}s)")
                    result = self._process_request(request)

                    if request.callback:
                        request.callback(result)

                    with self._lock:
                        self.stats['processed'] += 1

                except Exception as e:
                    logger.error(f"Request processing failed: {e}")
                    with self._lock:
                        self.stats['failed'] += 1

                    if request.callback:
                        request.callback({'error': str(e)})

                finally:
                    self.queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")

    def _process_request(self, request: QueuedRequest) -> Any:
        """Process a single request"""
        try:
            # Import here to avoid circular imports
            from ..pipeline.unified_pipeline import UnifiedAIPipeline

            pipeline = UnifiedAIPipeline()
            return pipeline.process(request.data)

        except ImportError:
            # Fallback for testing
            logger.warning("UnifiedAIPipeline not available, using mock processing")
            time.sleep(0.1)  # Simulate processing time
            return {
                'request_id': request.request_id,
                'processed_at': datetime.now().isoformat(),
                'mock': True
            }

    def add_request(self, data: Any, priority: int = 5,
                   callback: Optional[Callable] = None,
                   timeout: Optional[float] = 30.0,
                   user_id: Optional[str] = None) -> str:
        """Add request to queue"""
        request_id = str(uuid.uuid4())

        try:
            request = QueuedRequest(
                priority=priority,
                request_id=request_id,
                timestamp=datetime.now(),
                data=data,
                callback=callback,
                timeout=timeout,
                user_id=user_id
            )
            self.queue.put_nowait(request)
            logger.debug(f"Added request {request_id} with priority {priority}")
            return request_id

        except queue.Full:
            with self._lock:
                self.stats['rejected'] += 1
            logger.warning("Queue is full, rejecting request")
            raise Exception("Queue is full")

    def _update_avg_wait_time(self, wait_time: float):
        """Update average wait time statistics"""
        with self._lock:
            self.stats['total_wait_time'] += wait_time
            total_processed = self.stats['processed'] + self.stats['failed']
            if total_processed > 0:
                self.stats['avg_wait_time'] = self.stats['total_wait_time'] / total_processed

    def get_stats(self) -> Dict:
        """Get queue statistics"""
        with self._lock:
            total_requests = self.stats['processed'] + self.stats['failed'] + self.stats['rejected']
            success_rate = (self.stats['processed'] / max(1, total_requests)) * 100

            return {
                **self.stats,
                'queue_size': self.queue.qsize(),
                'workers': self.max_workers,
                'running': self.running,
                'success_rate': success_rate
            }

    def clear_queue(self):
        """Clear all pending requests"""
        with self._lock:
            try:
                while True:
                    self.queue.get_nowait()
                    self.queue.task_done()
            except queue.Empty:
                pass
            logger.info("Cleared all pending requests")


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self.lock = threading.Lock()

    def allow_request(self) -> bool:
        """Check if request is allowed"""
        with self.lock:
            now = time.time()

            # Remove old requests outside window
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()

            # Check if we can add new request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True

            return False

    def get_wait_time(self) -> float:
        """Get time until next request allowed"""
        with self.lock:
            if len(self.requests) < self.max_requests:
                return 0

            oldest = self.requests[0]
            wait = (oldest + self.window_seconds) - time.time()
            return max(0, wait)

    def get_stats(self) -> Dict:
        """Get rate limiter statistics"""
        with self.lock:
            return {
                'max_requests': self.max_requests,
                'window_seconds': self.window_seconds,
                'current_requests': len(self.requests),
                'remaining_capacity': self.max_requests - len(self.requests)
            }


class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter that adapts based on system load"""

    def __init__(self, base_rate: int, window_seconds: int):
        super().__init__(base_rate, window_seconds)
        self.base_rate = base_rate
        self.load_factor = 1.0
        self.last_update = time.time()

    def update_load_factor(self, cpu_percent: float, memory_percent: float):
        """Adjust rate based on system load"""
        now = time.time()

        # Only update every 10 seconds to avoid thrashing
        if now - self.last_update < 10:
            return

        with self.lock:
            # Reduce rate if system is under load
            if cpu_percent > 80 or memory_percent > 80:
                self.load_factor = 0.5
                logger.warning(f"High system load detected, throttling to 50%")
            elif cpu_percent > 60 or memory_percent > 60:
                self.load_factor = 0.75
                logger.info(f"Moderate system load detected, throttling to 75%")
            else:
                self.load_factor = 1.0

            # Update max requests
            new_max = int(self.base_rate * self.load_factor)
            if new_max != self.max_requests:
                logger.info(f"Adjusted rate limit: {self.max_requests} -> {new_max}")
                self.max_requests = new_max

            self.last_update = now

    def auto_adjust(self):
        """Automatically adjust based on current system stats"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            self.update_load_factor(cpu_percent, memory_percent)
        except Exception as e:
            logger.error(f"Failed to get system stats for rate limiting: {e}")


class PerUserRateLimiter:
    """Rate limiter with per-user tracking"""

    def __init__(self, default_rate: int, window_seconds: int):
        self.default_rate = default_rate
        self.window_seconds = window_seconds
        self.user_limiters = {}
        self.user_rates = {}  # Custom rates per user
        self.lock = threading.Lock()

    def set_user_rate(self, user_id: str, rate: int):
        """Set custom rate for specific user"""
        with self.lock:
            self.user_rates[user_id] = rate
            # Reset limiter if it exists
            if user_id in self.user_limiters:
                del self.user_limiters[user_id]

    def allow_request(self, user_id: str) -> bool:
        """Check if request is allowed for user"""
        with self.lock:
            if user_id not in self.user_limiters:
                rate = self.user_rates.get(user_id, self.default_rate)
                self.user_limiters[user_id] = RateLimiter(rate, self.window_seconds)

            return self.user_limiters[user_id].allow_request()

    def get_wait_time(self, user_id: str) -> float:
        """Get wait time for user"""
        with self.lock:
            if user_id in self.user_limiters:
                return self.user_limiters[user_id].get_wait_time()
            return 0

    def cleanup_inactive_users(self, max_age_seconds: int = 3600):
        """Remove rate limiters for inactive users"""
        with self.lock:
            # This is a simplified cleanup - in practice you'd track last access time
            if len(self.user_limiters) > 1000:  # Arbitrary threshold
                # Keep only recent users (simplified)
                self.user_limiters.clear()
                logger.info("Cleaned up inactive user rate limiters")


class QueueManager:
    """High-level queue manager combining queue and rate limiting"""

    def __init__(self,
                 max_workers: int = 4,
                 max_queue_size: int = 100,
                 rate_limit_per_minute: int = 60,
                 adaptive_rate_limiting: bool = True):

        self.request_queue = RequestQueue(max_workers, max_queue_size)

        # Set up rate limiting
        if adaptive_rate_limiting:
            self.rate_limiter = AdaptiveRateLimiter(rate_limit_per_minute, 60)
        else:
            self.rate_limiter = RateLimiter(rate_limit_per_minute, 60)

        self.per_user_limiter = PerUserRateLimiter(rate_limit_per_minute // 10, 60)

        # Background thread for adaptive rate limiting
        self.monitoring_thread = None
        self.monitoring_active = False

    def start(self):
        """Start the queue manager"""
        self.request_queue.start()

        # Start monitoring thread for adaptive rate limiting
        if isinstance(self.rate_limiter, AdaptiveRateLimiter):
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()

        logger.info("Queue manager started")

    def stop(self):
        """Stop the queue manager"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        self.request_queue.stop()
        logger.info("Queue manager stopped")

    def submit_request(self, data: Any,
                      priority: int = 5,
                      user_id: Optional[str] = None,
                      callback: Optional[Callable] = None) -> str:
        """Submit a request with rate limiting"""

        # Check global rate limit
        if not self.rate_limiter.allow_request():
            wait_time = self.rate_limiter.get_wait_time()
            raise Exception(f"Rate limit exceeded. Try again in {wait_time:.1f} seconds")

        # Check per-user rate limit
        if user_id and not self.per_user_limiter.allow_request(user_id):
            wait_time = self.per_user_limiter.get_wait_time(user_id)
            raise Exception(f"User rate limit exceeded. Try again in {wait_time:.1f} seconds")

        # Submit to queue
        return self.request_queue.add_request(
            data=data,
            priority=priority,
            user_id=user_id,
            callback=callback
        )

    def _monitoring_loop(self):
        """Background monitoring for adaptive rate limiting"""
        while self.monitoring_active:
            try:
                if isinstance(self.rate_limiter, AdaptiveRateLimiter):
                    self.rate_limiter.auto_adjust()

                # Cleanup inactive users every 10 minutes
                self.per_user_limiter.cleanup_inactive_users()

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")

            time.sleep(30)  # Check every 30 seconds

    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            'queue': self.request_queue.get_stats(),
            'rate_limiter': self.rate_limiter.get_stats(),
            'system_load': {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'load_factor': getattr(self.rate_limiter, 'load_factor', 1.0)
            }
        }


# Example usage and testing
def test_queue_system():
    """Test the queue system"""
    print("\n=== Testing Queue System ===")

    manager = QueueManager(max_workers=2, rate_limit_per_minute=10)
    manager.start()

    try:
        # Test request submission
        def result_callback(result):
            print(f"Request completed: {result.get('request_id', 'unknown')}")

        # Submit some test requests
        for i in range(5):
            try:
                request_id = manager.submit_request(
                    data=f"test_data_{i}",
                    priority=i % 3 + 1,  # Priorities 1, 2, 3
                    user_id=f"user_{i % 2}",  # Two users
                    callback=result_callback
                )
                print(f"Submitted request {i}: {request_id}")
            except Exception as e:
                print(f"Request {i} rejected: {e}")

        # Wait for processing
        time.sleep(2)

        # Get stats
        stats = manager.get_comprehensive_stats()
        print(f"Final stats: {stats}")

    finally:
        manager.stop()

    print("Queue system test complete!")


if __name__ == "__main__":
    test_queue_system()