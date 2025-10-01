#!/usr/bin/env python3
"""
Test script for real-time monitoring infrastructure
Tests Task B7.1 Component 3 - Real-time Monitoring Integration

This script tests all components of the real-time monitoring system.
"""

import asyncio
import sys
import time
import json
import logging
import websockets
from pathlib import Path

# Add the backend modules to Python path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_modules.optimization.real_time_monitor import (
    RealTimeMonitor, TrainingMetrics, SystemMetrics, Alert
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonitoringTests:
    """Test suite for real-time monitoring"""

    def __init__(self):
        self.monitor = None
        self.test_results = {}

    async def run_all_tests(self):
        """Run all monitoring tests"""
        logger.info("🧪 Starting Real-time Monitoring Tests")

        tests = [
            ("Monitor Initialization", self.test_monitor_initialization),
            ("WebSocket Server", self.test_websocket_server),
            ("Training Callbacks", self.test_training_callbacks),
            ("Alert System", self.test_alert_system),
            ("System Monitoring", self.test_system_monitoring),
            ("Data Persistence", self.test_data_persistence),
            ("Dashboard Data", self.test_dashboard_data),
            ("Integration", self.test_integration)
        ]

        for test_name, test_func in tests:
            try:
                logger.info(f"Running test: {test_name}")
                result = await test_func()
                self.test_results[test_name] = {"status": "PASS", "details": result}
                logger.info(f"✅ {test_name} - PASSED")
            except Exception as e:
                self.test_results[test_name] = {"status": "FAIL", "error": str(e)}
                logger.error(f"❌ {test_name} - FAILED: {e}")

        self.print_test_summary()

    async def test_monitor_initialization(self):
        """Test monitor initialization"""
        self.monitor = RealTimeMonitor(
            websocket_port=8767,  # Different port for testing
            save_dir="logs/test_monitoring"
        )

        assert self.monitor is not None
        assert self.monitor.websocket_server is not None
        assert self.monitor.alert_system is not None
        assert self.monitor.performance_monitor is not None

        return "Monitor initialized successfully"

    async def test_websocket_server(self):
        """Test WebSocket server functionality"""
        if not self.monitor:
            raise Exception("Monitor not initialized")

        # Start the server
        await self.monitor.start_monitoring()

        # Give server time to start
        await asyncio.sleep(1)

        # Test connection
        try:
            uri = f"ws://localhost:8767"
            async with websockets.connect(uri) as websocket:
                # Send ping
                await websocket.send(json.dumps({"type": "ping"}))

                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)

                assert data.get("type") in ["pong", "connection"]

                return "WebSocket server working correctly"

        except Exception as e:
            raise Exception(f"WebSocket test failed: {e}")

    async def test_training_callbacks(self):
        """Test training callback functionality"""
        if not self.monitor:
            raise Exception("Monitor not initialized")

        # Create test callback
        callback = self.monitor.create_callback()
        assert callback is not None

        # Simulate training events
        self.monitor.on_training_start()
        assert self.monitor.training_active

        # Simulate episode completion
        test_metrics = TrainingMetrics(
            timestamp=time.time(),
            episode=1,
            reward=0.5,
            episode_length=25,
            quality=0.85,
            success=True
        )

        self.monitor.on_episode_complete(test_metrics)
        assert len(self.monitor.training_metrics) == 1

        self.monitor.on_training_end()
        assert not self.monitor.training_active

        return "Training callbacks working correctly"

    async def test_alert_system(self):
        """Test alert system functionality"""
        if not self.monitor:
            raise Exception("Monitor not initialized")

        alert_system = self.monitor.alert_system

        # Test alert creation
        alert = alert_system.create_alert(
            "warning", "test", "Test alert message", {"test": True}
        )

        assert alert.level == "warning"
        assert alert.category == "test"
        assert alert.message == "Test alert message"
        assert len(alert_system.alerts) > 0

        # Test system alerts
        test_system_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=95.0,  # High CPU to trigger alert
            memory_percent=85.0
        )

        alert_system.check_system_alerts(test_system_metrics)

        # Check if alert was created
        recent_alerts = alert_system.get_recent_alerts(1)
        assert len(recent_alerts) > 0

        return "Alert system working correctly"

    async def test_system_monitoring(self):
        """Test system monitoring functionality"""
        if not self.monitor:
            raise Exception("Monitor not initialized")

        perf_monitor = self.monitor.performance_monitor

        # Get system metrics
        metrics = perf_monitor.get_system_metrics()

        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.timestamp > 0

        return "System monitoring working correctly"

    async def test_data_persistence(self):
        """Test data persistence functionality"""
        if not self.monitor:
            raise Exception("Monitor not initialized")

        # Add some test data
        test_metrics = TrainingMetrics(
            timestamp=time.time(),
            episode=2,
            reward=0.7,
            episode_length=30,
            quality=0.9,
            success=True
        )

        self.monitor.on_episode_complete(test_metrics)

        # Save data
        self.monitor.save_monitoring_data()

        # Check if files were created
        save_dir = Path(self.monitor.save_dir)
        assert save_dir.exists()

        # Look for any JSON files
        json_files = list(save_dir.glob("*.json"))
        assert len(json_files) > 0

        return f"Data persistence working, {len(json_files)} files saved"

    async def test_dashboard_data(self):
        """Test dashboard data generation"""
        if not self.monitor:
            raise Exception("Monitor not initialized")

        # Generate dashboard data
        dashboard_data = self.monitor.get_dashboard_data()

        assert "training_metrics" in dashboard_data
        assert "system_metrics" in dashboard_data
        assert "alerts" in dashboard_data
        assert "training_active" in dashboard_data

        return "Dashboard data generation working correctly"

    async def test_integration(self):
        """Test full integration"""
        if not self.monitor:
            raise Exception("Monitor not initialized")

        # Simulate a complete training session
        self.monitor.on_training_start()

        # Simulate multiple episodes
        for i in range(5):
            metrics = TrainingMetrics(
                timestamp=time.time(),
                episode=i + 1,
                reward=0.5 + i * 0.1,
                episode_length=20 + i * 2,
                quality=0.8 + i * 0.02,
                success=True
            )
            self.monitor.on_episode_complete(metrics)
            await asyncio.sleep(0.1)

        self.monitor.on_training_end()

        # Generate report
        report = self.monitor.generate_monitoring_report()
        assert len(report) > 0
        assert "Training Performance" in report

        return "Full integration test passed"

    def print_test_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("🧪 REAL-TIME MONITORING TEST RESULTS")
        print("=" * 60)

        passed = 0
        failed = 0

        for test_name, result in self.test_results.items():
            status = result["status"]
            if status == "PASS":
                print(f"✅ {test_name}: PASSED")
                if "details" in result:
                    print(f"   {result['details']}")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
                print(f"   Error: {result['error']}")
                failed += 1

        print(f"\nSummary: {passed} passed, {failed} failed")

        if failed == 0:
            print("🎉 All tests passed! Real-time monitoring is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the errors above.")

        print("=" * 60)

    async def cleanup(self):
        """Clean up test resources"""
        if self.monitor:
            try:
                await self.monitor.stop_monitoring()
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")


async def main():
    """Main test function"""
    print("🧪 Real-time Monitoring System Tests")
    print("Task B7.1 Component 3 - Comprehensive Testing")
    print()

    tests = MonitoringTests()

    try:
        await tests.run_all_tests()
    finally:
        await tests.cleanup()


if __name__ == "__main__":
    asyncio.run(main())