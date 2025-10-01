"""
Adaptive Routing Based on Load - Task 3 Implementation
Load-aware routing system that adjusts tier selection based on system capacity.
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import queue
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Request:
    """Container for a routing request."""
    id: str
    image_path: str
    complexity: float
    target_quality: float
    tier: int
    priority: int = 0
    time_budget: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProcessingMetrics:
    """Metrics for a processing tier."""
    current_load: float
    queue_size: int
    avg_processing_time: float
    capacity_available: int
    estimated_wait_time: float


class AdaptiveRouter:
    """
    Adaptive routing system that considers system load and adjusts tier selection
    to maintain optimal performance and prevent overload.
    """

    def __init__(self,
                 max_concurrent_tier1: int = 10,
                 max_concurrent_tier2: int = 5,
                 max_concurrent_tier3: int = 2,
                 history_window: int = 100):
        """
        Initialize adaptive router.

        Args:
            max_concurrent_tier1: Max concurrent Tier 1 processes
            max_concurrent_tier2: Max concurrent Tier 2 processes
            max_concurrent_tier3: Max concurrent Tier 3 processes
            history_window: Size of history window for metrics
        """
        # Tier capacities
        self.tier_capabilities = {
            1: {
                'capacity': max_concurrent_tier1,
                'avg_time': 1.5,
                'max_time': 3.0,
                'quality': 0.75
            },
            2: {
                'capacity': max_concurrent_tier2,
                'avg_time': 4.0,
                'max_time': 8.0,
                'quality': 0.85
            },
            3: {
                'capacity': max_concurrent_tier3,
                'avg_time': 12.0,
                'max_time': 20.0,
                'quality': 0.95
            }
        }

        # Current processing state
        self.current_processing = {
            1: [],  # List of (request_id, start_time)
            2: [],
            3: []
        }

        # Request queues for each tier
        self.tier_queues = {
            1: queue.PriorityQueue(),
            2: queue.PriorityQueue(),
            3: queue.PriorityQueue()
        }

        # Processing time history
        self.processing_times = deque(maxlen=history_window)
        self.tier_processing_times = {
            1: deque(maxlen=history_window),
            2: deque(maxlen=history_window),
            3: deque(maxlen=history_window)
        }

        # Load metrics
        self.system_load = 0.0
        self.tier_loads = {1: 0.0, 2: 0.0, 3: 0.0}

        # Request tracking
        self.active_requests = {}
        self.completed_requests = deque(maxlen=1000)

        # Statistics
        self.routing_statistics = defaultdict(lambda: {
            'routed': 0,
            'completed': 0,
            'downgraded': 0,
            'upgraded': 0,
            'rejected': 0,
            'total_wait_time': 0,
            'total_processing_time': 0
        })

        # Thread safety
        self._lock = threading.RLock()

        # Load monitoring thread
        self._stop_monitoring = False
        self._monitor_thread = threading.Thread(target=self._monitor_load, daemon=True)
        self._monitor_thread.start()

        logger.info(f"AdaptiveRouter initialized with capacities: T1={max_concurrent_tier1}, "
                   f"T2={max_concurrent_tier2}, T3={max_concurrent_tier3}")

    def route_with_load_balancing(self, request: Request) -> int:
        """
        Route request considering current system load.

        Args:
            request: Request to route

        Returns:
            Selected tier (possibly adjusted for load)
        """
        with self._lock:
            original_tier = request.tier

            # Calculate current system load
            self._update_system_load()

            # Check if system is overloaded
            if self.system_load > 0.9:
                # System overloaded - try to downgrade
                adjusted_tier = self._handle_overload(request)
            elif self.system_load > 0.7:
                # High load - consider downgrading
                adjusted_tier = self._handle_high_load(request)
            else:
                # Normal load - check tier availability
                adjusted_tier = self._check_tier_availability(request)

            # Track routing decision
            if adjusted_tier < original_tier:
                self.routing_statistics[original_tier]['downgraded'] += 1
                logger.info(f"Downgraded request {request.id} from Tier {original_tier} to Tier {adjusted_tier}")
            elif adjusted_tier > original_tier:
                self.routing_statistics[original_tier]['upgraded'] += 1
                logger.info(f"Upgraded request {request.id} from Tier {original_tier} to Tier {adjusted_tier}")

            self.routing_statistics[adjusted_tier]['routed'] += 1

            # Add to appropriate queue
            self._enqueue_request(request, adjusted_tier)

            return adjusted_tier

    def _update_system_load(self):
        """Update current system load metrics."""
        total_capacity = sum(tier['capacity'] for tier in self.tier_capabilities.values())
        current_processing = sum(len(procs) for procs in self.current_processing.values())

        self.system_load = current_processing / total_capacity if total_capacity > 0 else 0

        # Update tier-specific loads
        for tier in [1, 2, 3]:
            capacity = self.tier_capabilities[tier]['capacity']
            processing = len(self.current_processing[tier])
            self.tier_loads[tier] = processing / capacity if capacity > 0 else 0

    def _handle_overload(self, request: Request) -> int:
        """
        Handle routing when system is overloaded.

        Args:
            request: Request to route

        Returns:
            Adjusted tier
        """
        # Try to downgrade to fastest tier
        if request.tier > 1:
            # Check if downgrading would still meet quality requirements
            min_quality_tier = self._find_min_quality_tier(request.target_quality)
            return max(1, min_quality_tier)
        return 1

    def _handle_high_load(self, request: Request) -> int:
        """
        Handle routing when system has high load.

        Args:
            request: Request to route

        Returns:
            Adjusted tier
        """
        original_tier = request.tier

        # Check if current tier is overloaded
        if self.tier_loads[original_tier] > 0.8:
            # Try to route to less loaded tier
            for tier in [1, 2, 3]:
                if tier != original_tier and self.tier_loads[tier] < 0.6:
                    # Check if tier meets minimum quality
                    if self.tier_capabilities[tier]['quality'] >= request.target_quality * 0.9:
                        return tier

            # If no better tier, consider downgrading
            if original_tier > 1:
                return original_tier - 1

        return original_tier

    def _check_tier_availability(self, request: Request) -> int:
        """
        Check tier availability and adjust if necessary.

        Args:
            request: Request to route

        Returns:
            Available tier
        """
        preferred_tier = request.tier

        # Check if preferred tier has capacity
        if self._has_capacity(preferred_tier):
            return preferred_tier

        # Check wait time for preferred tier
        wait_time = self._estimate_wait_time(preferred_tier)

        # If wait time is acceptable, keep original tier
        if request.time_budget and wait_time < request.time_budget * 0.3:
            return preferred_tier

        # Try adjacent tiers
        for tier_adjustment in [-1, 1]:
            alternative_tier = preferred_tier + tier_adjustment
            if 1 <= alternative_tier <= 3:
                if self._has_capacity(alternative_tier):
                    # Check if quality is acceptable
                    if self._is_quality_acceptable(alternative_tier, request.target_quality):
                        return alternative_tier

        # No good alternative, use original
        return preferred_tier

    def _has_capacity(self, tier: int) -> bool:
        """Check if tier has available capacity."""
        capacity = self.tier_capabilities[tier]['capacity']
        current = len(self.current_processing[tier])
        return current < capacity

    def _estimate_wait_time(self, tier: int) -> float:
        """
        Estimate wait time for a tier.

        Args:
            tier: Tier number

        Returns:
            Estimated wait time in seconds
        """
        # Get queue size
        queue_size = self.tier_queues[tier].qsize()

        # Get average processing time
        if len(self.tier_processing_times[tier]) > 0:
            avg_time = sum(self.tier_processing_times[tier]) / len(self.tier_processing_times[tier])
        else:
            avg_time = self.tier_capabilities[tier]['avg_time']

        # Estimate based on queue and capacity
        capacity = self.tier_capabilities[tier]['capacity']
        if capacity > 0:
            wait_time = (queue_size / capacity) * avg_time
        else:
            wait_time = queue_size * avg_time

        return wait_time

    def _is_quality_acceptable(self, tier: int, target_quality: float) -> bool:
        """Check if tier can meet quality requirements."""
        tier_quality = self.tier_capabilities[tier]['quality']
        # Allow 10% tolerance
        return tier_quality >= target_quality * 0.9

    def _find_min_quality_tier(self, target_quality: float) -> int:
        """Find minimum tier that meets quality requirement."""
        for tier in [1, 2, 3]:
            if self.tier_capabilities[tier]['quality'] >= target_quality:
                return tier
        return 3  # Default to highest tier

    def _enqueue_request(self, request: Request, tier: int):
        """Add request to tier queue."""
        # Priority queue uses (priority, timestamp, request) tuple
        # Lower priority value = higher priority
        priority_value = -request.priority  # Negate so higher priority goes first
        self.tier_queues[tier].put((priority_value, request.timestamp, request))
        self.active_requests[request.id] = request

    def process_next_request(self, tier: int) -> Optional[Request]:
        """
        Get next request to process for a tier.

        Args:
            tier: Tier to get request for

        Returns:
            Next request or None if queue is empty
        """
        with self._lock:
            if not self.tier_queues[tier].empty() and self._has_capacity(tier):
                try:
                    _, _, request = self.tier_queues[tier].get_nowait()

                    # Mark as processing
                    self.current_processing[tier].append((request.id, time.time()))

                    # Calculate wait time
                    wait_time = time.time() - request.timestamp
                    self.routing_statistics[tier]['total_wait_time'] += wait_time

                    logger.debug(f"Processing request {request.id} on Tier {tier} "
                               f"(waited {wait_time:.2f}s)")

                    return request
                except queue.Empty:
                    pass

        return None

    def complete_request(self, request_id: str, tier: int, processing_time: float, success: bool = True):
        """
        Mark request as completed.

        Args:
            request_id: Request ID
            tier: Tier that processed the request
            processing_time: Actual processing time
            success: Whether processing was successful
        """
        with self._lock:
            # Remove from current processing
            self.current_processing[tier] = [
                (rid, start_time) for rid, start_time in self.current_processing[tier]
                if rid != request_id
            ]

            # Update statistics
            self.routing_statistics[tier]['completed'] += 1
            self.routing_statistics[tier]['total_processing_time'] += processing_time

            # Add to processing time history
            self.processing_times.append(processing_time)
            self.tier_processing_times[tier].append(processing_time)

            # Move to completed
            if request_id in self.active_requests:
                request = self.active_requests.pop(request_id)
                self.completed_requests.append({
                    'request': request,
                    'tier': tier,
                    'processing_time': processing_time,
                    'success': success,
                    'completed_at': time.time()
                })

            logger.debug(f"Completed request {request_id} on Tier {tier} in {processing_time:.2f}s")

    def _monitor_load(self):
        """Background thread to monitor and log system load."""
        while not self._stop_monitoring:
            try:
                with self._lock:
                    self._update_system_load()

                    # Clean up stale processing entries (>max_time)
                    current_time = time.time()
                    for tier in [1, 2, 3]:
                        max_time = self.tier_capabilities[tier]['max_time']
                        self.current_processing[tier] = [
                            (rid, start_time) for rid, start_time in self.current_processing[tier]
                            if current_time - start_time < max_time
                        ]

                    # Log load metrics periodically
                    if len(self.processing_times) % 10 == 0 and len(self.processing_times) > 0:
                        logger.info(f"System load: {self.system_load:.2%}, "
                                  f"Tier loads: T1={self.tier_loads[1]:.2%}, "
                                  f"T2={self.tier_loads[2]:.2%}, T3={self.tier_loads[3]:.2%}")

                time.sleep(1)  # Monitor every second

            except Exception as e:
                logger.error(f"Error in load monitoring: {e}")

    def get_metrics(self, tier: Optional[int] = None) -> Dict[str, Any]:
        """
        Get current routing metrics.

        Args:
            tier: Specific tier to get metrics for (or None for all)

        Returns:
            Dictionary with metrics
        """
        with self._lock:
            if tier:
                return self._get_tier_metrics(tier)

            # System-wide metrics
            metrics = {
                'system_load': self.system_load,
                'tier_loads': dict(self.tier_loads),
                'total_active': len(self.active_requests),
                'total_completed': len(self.completed_requests),
                'tiers': {}
            }

            for t in [1, 2, 3]:
                metrics['tiers'][t] = self._get_tier_metrics(t)

            return metrics

    def _get_tier_metrics(self, tier: int) -> ProcessingMetrics:
        """Get metrics for a specific tier."""
        stats = self.routing_statistics[tier]

        # Calculate average times
        if stats['completed'] > 0:
            avg_wait = stats['total_wait_time'] / stats['completed']
            avg_processing = stats['total_processing_time'] / stats['completed']
        else:
            avg_wait = 0
            avg_processing = self.tier_capabilities[tier]['avg_time']

        return {
            'current_load': self.tier_loads[tier],
            'queue_size': self.tier_queues[tier].qsize(),
            'processing_count': len(self.current_processing[tier]),
            'capacity_available': self.tier_capabilities[tier]['capacity'] - len(self.current_processing[tier]),
            'estimated_wait_time': self._estimate_wait_time(tier),
            'avg_wait_time': avg_wait,
            'avg_processing_time': avg_processing,
            'routed_count': stats['routed'],
            'completed_count': stats['completed'],
            'downgraded_count': stats['downgraded'],
            'upgraded_count': stats['upgraded']
        }

    def simulate_load(self, duration: int = 30):
        """
        Simulate load for testing.

        Args:
            duration: Simulation duration in seconds
        """
        import random
        import uuid

        print(f"Simulating load for {duration} seconds...")
        start_time = time.time()
        request_count = 0

        while time.time() - start_time < duration:
            # Generate random requests
            if random.random() < 0.7:  # 70% chance of new request per iteration
                request = Request(
                    id=str(uuid.uuid4())[:8],
                    image_path=f"test_{request_count}.png",
                    complexity=random.random(),
                    target_quality=random.uniform(0.7, 0.95),
                    tier=random.randint(1, 3),
                    priority=random.randint(0, 2),
                    time_budget=random.uniform(2, 10) if random.random() < 0.5 else None
                )

                selected_tier = self.route_with_load_balancing(request)
                request_count += 1

                print(f"Request {request.id}: Tier {request.tier} -> {selected_tier} "
                     f"(Load: {self.system_load:.2%})")

            # Simulate processing completions
            for tier in [1, 2, 3]:
                if random.random() < 0.3:  # 30% chance of completion
                    req = self.process_next_request(tier)
                    if req:
                        processing_time = random.uniform(
                            self.tier_capabilities[tier]['avg_time'] * 0.5,
                            self.tier_capabilities[tier]['avg_time'] * 1.5
                        )
                        time.sleep(0.1)  # Simulate processing
                        self.complete_request(req.id, tier, processing_time)

            time.sleep(0.2)  # Small delay between iterations

        # Print final statistics
        print(f"\nSimulation complete. {request_count} requests processed.")
        metrics = self.get_metrics()
        print(f"Final system load: {metrics['system_load']:.2%}")
        print("Tier statistics:")
        for tier, tier_metrics in metrics['tiers'].items():
            print(f"  Tier {tier}: {tier_metrics['completed_count']} completed, "
                 f"{tier_metrics['downgraded_count']} downgraded, "
                 f"{tier_metrics['upgraded_count']} upgraded")

    def shutdown(self):
        """Shutdown the router and cleanup resources."""
        self._stop_monitoring = True
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2)
        logger.info("AdaptiveRouter shutdown complete")


def test_adaptive_router():
    """Test the adaptive router."""
    print("Testing Adaptive Router...")

    # Initialize router
    router = AdaptiveRouter(
        max_concurrent_tier1=5,
        max_concurrent_tier2=3,
        max_concurrent_tier3=1
    )

    # Test 1: Normal load routing
    print("\n✓ Testing normal load routing:")
    request1 = Request(
        id="req1",
        image_path="test1.png",
        complexity=0.3,
        target_quality=0.8,
        tier=2
    )
    tier = router.route_with_load_balancing(request1)
    print(f"  Request routed to Tier {tier}")
    assert tier == 2, "Should route to requested tier under normal load"

    # Test 2: Process request
    print("\n✓ Testing request processing:")
    next_req = router.process_next_request(2)
    assert next_req is not None, "Should get queued request"
    assert next_req.id == "req1", "Should get correct request"

    # Complete the request
    router.complete_request("req1", 2, 3.5)

    # Test 3: High load simulation
    print("\n✓ Testing high load handling:")
    # Fill up Tier 3
    for i in range(3):
        req = Request(
            id=f"high_{i}",
            image_path=f"high_{i}.png",
            complexity=0.8,
            target_quality=0.95,
            tier=3
        )
        router.route_with_load_balancing(req)
        router.process_next_request(3)

    # Try to route another Tier 3 request
    overload_req = Request(
        id="overload",
        image_path="overload.png",
        complexity=0.8,
        target_quality=0.95,
        tier=3
    )
    adjusted_tier = router.route_with_load_balancing(overload_req)
    print(f"  Overload request routed to Tier {adjusted_tier} (original: 3)")

    # Test 4: Get metrics
    print("\n✓ Testing metrics:")
    metrics = router.get_metrics()
    print(f"  System load: {metrics['system_load']:.2%}")
    print(f"  Active requests: {metrics['total_active']}")

    tier_metrics = router.get_metrics(tier=2)
    print(f"  Tier 2 metrics: Queue={tier_metrics['queue_size']}, "
         f"Completed={tier_metrics['completed_count']}")

    # Test 5: Priority handling
    print("\n✓ Testing priority handling:")
    high_priority = Request(
        id="high_pri",
        image_path="high_priority.png",
        complexity=0.5,
        target_quality=0.85,
        tier=2,
        priority=10  # High priority
    )
    low_priority = Request(
        id="low_pri",
        image_path="low_priority.png",
        complexity=0.5,
        target_quality=0.85,
        tier=2,
        priority=1  # Low priority
    )

    router.route_with_load_balancing(low_priority)
    router.route_with_load_balancing(high_priority)

    # High priority should be processed first
    next_req = router.process_next_request(2)
    assert next_req.id == "high_pri", "High priority request should be processed first"

    # Cleanup
    router.shutdown()

    print("\n✅ All adaptive router tests passed!")
    return router


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--simulate-load":
        # Run load simulation
        router = AdaptiveRouter()
        try:
            router.simulate_load(duration=20)
        finally:
            router.shutdown()
    else:
        # Run tests
        test_adaptive_router()