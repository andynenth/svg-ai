#!/usr/bin/env python3
"""
Classification System Performance Monitor
Monitors and reports on the performance of the Day 8 integrated classification system
"""

import time
import json
import requests
import psutil
import threading
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import statistics

class PerformanceMonitor:
    def __init__(self, base_url="http://localhost:8001", monitoring_interval=10):
        self.base_url = base_url
        self.monitoring_interval = monitoring_interval
        self.metrics = {
            'api_response_times': deque(maxlen=100),
            'classification_times': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'cpu_usage': deque(maxlen=100),
            'error_count': 0,
            'total_requests': 0,
            'successful_requests': 0,
            'method_usage': defaultdict(int),
            'logo_type_distribution': defaultdict(int),
            'start_time': datetime.now()
        }
        self.running = False

    def test_api_endpoint(self, endpoint, timeout=5):
        """Test an API endpoint and measure response time"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}{endpoint}", timeout=timeout)
            response_time = time.time() - start_time

            success = response.status_code == 200
            return {
                'success': success,
                'response_time': response_time,
                'status_code': response.status_code,
                'data': response.json() if success else None
            }
        except Exception as e:
            return {
                'success': False,
                'response_time': timeout,
                'error': str(e)
            }

    def monitor_system_resources(self):
        """Monitor system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['cpu_usage'].append(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            self.metrics['memory_usage'].append(memory_mb)

            return {
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'memory_percent': memory.percent
            }
        except Exception as e:
            print(f"Error monitoring system resources: {e}")
            return None

    def monitor_classification_health(self):
        """Monitor classification system health and performance"""
        # Test classification status endpoint
        status_result = self.test_api_endpoint('/api/classification-status', timeout=10)
        self.metrics['total_requests'] += 1

        if status_result['success']:
            self.metrics['successful_requests'] += 1
            self.metrics['api_response_times'].append(status_result['response_time'])

            # Extract performance data from status
            status_data = status_result['data']
            if status_data and 'test_classification_time' in status_data:
                self.metrics['classification_times'].append(status_data['test_classification_time'])

            return status_data
        else:
            self.metrics['error_count'] += 1
            return None

    def calculate_statistics(self):
        """Calculate performance statistics"""
        stats = {}

        # API response times
        if self.metrics['api_response_times']:
            response_times = list(self.metrics['api_response_times'])
            stats['api_response'] = {
                'avg': statistics.mean(response_times),
                'min': min(response_times),
                'max': max(response_times),
                'median': statistics.median(response_times),
                'count': len(response_times)
            }

        # Classification times
        if self.metrics['classification_times']:
            class_times = list(self.metrics['classification_times'])
            stats['classification'] = {
                'avg': statistics.mean(class_times),
                'min': min(class_times),
                'max': max(class_times),
                'median': statistics.median(class_times),
                'count': len(class_times)
            }

        # System resources
        if self.metrics['memory_usage']:
            memory_usage = list(self.metrics['memory_usage'])
            stats['memory'] = {
                'avg_mb': statistics.mean(memory_usage),
                'min_mb': min(memory_usage),
                'max_mb': max(memory_usage),
                'current_mb': memory_usage[-1]
            }

        if self.metrics['cpu_usage']:
            cpu_usage = list(self.metrics['cpu_usage'])
            stats['cpu'] = {
                'avg_percent': statistics.mean(cpu_usage),
                'min_percent': min(cpu_usage),
                'max_percent': max(cpu_usage),
                'current_percent': cpu_usage[-1]
            }

        # Overall health
        if self.metrics['total_requests'] > 0:
            stats['health'] = {
                'success_rate': (self.metrics['successful_requests'] / self.metrics['total_requests']) * 100,
                'error_rate': (self.metrics['error_count'] / self.metrics['total_requests']) * 100,
                'total_requests': self.metrics['total_requests'],
                'uptime_minutes': (datetime.now() - self.metrics['start_time']).total_seconds() / 60
            }

        return stats

    def generate_report(self):
        """Generate a performance report"""
        stats = self.calculate_statistics()

        report = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_duration_minutes': (datetime.now() - self.metrics['start_time']).total_seconds() / 60,
            'statistics': stats,
            'raw_metrics': {
                'total_requests': self.metrics['total_requests'],
                'successful_requests': self.metrics['successful_requests'],
                'error_count': self.metrics['error_count'],
                'method_usage': dict(self.metrics['method_usage']),
                'logo_type_distribution': dict(self.metrics['logo_type_distribution'])
            }
        }

        return report

    def print_real_time_stats(self, stats):
        """Print real-time statistics to console"""
        print("\n" + "="*60)
        print(f"Performance Monitor - {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)

        # Health overview
        if 'health' in stats:
            health = stats['health']
            print(f"System Health:")
            print(f"  Success Rate: {health['success_rate']:.1f}%")
            print(f"  Total Requests: {health['total_requests']}")
            print(f"  Uptime: {health['uptime_minutes']:.1f} minutes")

        # API Performance
        if 'api_response' in stats:
            api = stats['api_response']
            print(f"API Response Times:")
            print(f"  Average: {api['avg']*1000:.1f}ms")
            print(f"  Median: {api['median']*1000:.1f}ms")
            print(f"  Range: {api['min']*1000:.1f}ms - {api['max']*1000:.1f}ms")

        # Classification Performance
        if 'classification' in stats:
            classification = stats['classification']
            print(f"Classification Performance:")
            print(f"  Average: {classification['avg']*1000:.1f}ms")
            print(f"  Median: {classification['median']*1000:.1f}ms")
            print(f"  Range: {classification['min']*1000:.1f}ms - {classification['max']*1000:.1f}ms")

        # System Resources
        if 'memory' in stats:
            memory = stats['memory']
            print(f"Memory Usage:")
            print(f"  Current: {memory['current_mb']:.1f}MB")
            print(f"  Average: {memory['avg_mb']:.1f}MB")

        if 'cpu' in stats:
            cpu = stats['cpu']
            print(f"CPU Usage:")
            print(f"  Current: {cpu['current_percent']:.1f}%")
            print(f"  Average: {cpu['avg_percent']:.1f}%")

        # Performance targets check
        print(f"Performance Targets:")
        if 'api_response' in stats:
            api_target_met = stats['api_response']['avg'] < 2.0  # 2 second target
            print(f"  API Response < 2s: {'✅ PASS' if api_target_met else '❌ FAIL'} ({stats['api_response']['avg']*1000:.1f}ms)")

        if 'classification' in stats:
            class_target_met = stats['classification']['avg'] < 1.5  # 1.5 second target
            print(f"  Classification < 1.5s: {'✅ PASS' if class_target_met else '❌ FAIL'} ({stats['classification']['avg']*1000:.1f}ms)")

    def start_monitoring(self, duration_minutes=None):
        """Start continuous monitoring"""
        print(f"Starting performance monitoring...")
        print(f"Base URL: {self.base_url}")
        print(f"Monitoring interval: {self.monitoring_interval}s")
        if duration_minutes:
            print(f"Duration: {duration_minutes} minutes")
        print("Press Ctrl+C to stop monitoring\n")

        self.running = True
        start_time = time.time()

        try:
            while self.running:
                # Monitor system resources
                sys_resources = self.monitor_system_resources()

                # Monitor classification health
                classification_health = self.monitor_classification_health()

                # Calculate and display stats
                stats = self.calculate_statistics()
                self.print_real_time_stats(stats)

                # Check if duration limit reached
                if duration_minutes:
                    elapsed_minutes = (time.time() - start_time) / 60
                    if elapsed_minutes >= duration_minutes:
                        print(f"\nMonitoring duration ({duration_minutes} minutes) completed.")
                        break

                # Wait for next monitoring cycle
                time.sleep(self.monitoring_interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")

        self.running = False

        # Generate final report
        final_report = self.generate_report()
        print("\n" + "="*60)
        print("FINAL MONITORING REPORT")
        print("="*60)

        # Save report to file
        report_filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(final_report, f, indent=2)

        print(f"Detailed report saved to: {report_filename}")

        return final_report

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Classification System Performance Monitor')
    parser.add_argument('--url', default='http://localhost:8001',
                       help='Base URL of the API server (default: http://localhost:8001)')
    parser.add_argument('--interval', type=int, default=10,
                       help='Monitoring interval in seconds (default: 10)')
    parser.add_argument('--duration', type=int,
                       help='Monitoring duration in minutes (default: continuous)')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate a single report and exit')

    args = parser.parse_args()

    monitor = PerformanceMonitor(args.url, args.interval)

    if args.report_only:
        # Just generate a single report
        print("Generating performance report...")
        sys_resources = monitor.monitor_system_resources()
        classification_health = monitor.monitor_classification_health()
        stats = monitor.calculate_statistics()
        monitor.print_real_time_stats(stats)
    else:
        # Start continuous monitoring
        monitor.start_monitoring(args.duration)

if __name__ == "__main__":
    main()