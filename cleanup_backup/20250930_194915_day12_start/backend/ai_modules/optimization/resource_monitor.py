# backend/ai_modules/optimization/resource_monitor.py
"""Comprehensive resource monitoring and management system for training optimization"""

import os
import psutil
import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

# Suppress some common warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Check for optional dependencies
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_temperature: Optional[float] = None
    process_cpu_percent: Optional[float] = None
    process_memory_mb: Optional[float] = None
    network_bytes_sent: Optional[int] = None
    network_bytes_recv: Optional[int] = None


@dataclass
class ResourceAlert:
    """Resource usage alert"""
    alert_id: str
    timestamp: float
    alert_type: str  # 'warning', 'critical'
    resource: str    # 'cpu', 'memory', 'disk', 'gpu'
    current_value: float
    threshold: float
    message: str
    resolved: bool = False
    resolved_timestamp: Optional[float] = None


@dataclass
class ResourceThresholds:
    """Resource usage thresholds for alerts"""
    cpu_warning: float = 80.0
    cpu_critical: float = 95.0
    memory_warning: float = 80.0
    memory_critical: float = 95.0
    disk_warning: float = 85.0
    disk_critical: float = 95.0
    gpu_warning: float = 90.0
    gpu_critical: float = 98.0
    gpu_memory_warning: float = 85.0
    gpu_memory_critical: float = 95.0
    gpu_temperature_warning: float = 80.0
    gpu_temperature_critical: float = 90.0


@dataclass
class OptimizationRecommendation:
    """Resource optimization recommendation"""
    recommendation_id: str
    timestamp: float
    category: str  # 'performance', 'memory', 'efficiency'
    priority: str  # 'high', 'medium', 'low'
    title: str
    description: str
    implementation_steps: List[str]
    expected_benefit: str


class GPUMonitor:
    """GPU monitoring utilities"""

    def __init__(self):
        self.gpu_available = self._check_gpu_availability()
        self.nvml_available = False

        if self.gpu_available and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_available = True
                self.pynvml = pynvml
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"GPU monitoring enabled for {self.gpu_count} GPU(s)")
            except Exception as e:
                logger.warning(f"GPU monitoring not available: {e}")
        elif self.gpu_available and not PYNVML_AVAILABLE:
            logger.warning("GPU detected but pynvml not available - install nvidia-ml-py for GPU monitoring")

    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def get_gpu_stats(self, gpu_id: int = 0) -> Dict[str, Optional[float]]:
        """Get GPU statistics"""
        if not self.nvml_available:
            return {
                'utilization': None,
                'memory_used_gb': None,
                'memory_total_gb': None,
                'temperature': None
            }

        try:
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

            # GPU utilization
            util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu

            # Memory info
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_gb = mem_info.used / 1024**3
            memory_total_gb = mem_info.total / 1024**3

            # Temperature
            try:
                temperature = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = None

            return {
                'utilization': float(gpu_util),
                'memory_used_gb': memory_used_gb,
                'memory_total_gb': memory_total_gb,
                'temperature': temperature
            }

        except Exception as e:
            logger.debug(f"Failed to get GPU stats: {e}")
            return {
                'utilization': None,
                'memory_used_gb': None,
                'memory_total_gb': None,
                'temperature': None
            }


class ProcessMonitor:
    """Process-specific monitoring"""

    def __init__(self, pid: Optional[int] = None):
        self.pid = pid or os.getpid()
        try:
            self.process = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            logger.warning(f"Process {self.pid} not found")
            self.process = None

    def get_process_stats(self) -> Dict[str, Optional[float]]:
        """Get process-specific statistics"""
        if not self.process:
            return {'cpu_percent': None, 'memory_mb': None}

        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            return {
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {'cpu_percent': None, 'memory_mb': None}


class ResourceMonitor:
    """Comprehensive resource monitoring system"""

    def __init__(self,
                 monitoring_interval: float = 5.0,
                 history_size: int = 1000,
                 thresholds: Optional[ResourceThresholds] = None,
                 save_dir: Optional[str] = None):
        """
        Initialize resource monitor

        Args:
            monitoring_interval: Interval between measurements in seconds
            history_size: Number of snapshots to keep in memory
            thresholds: Resource usage thresholds
            save_dir: Directory to save monitoring data
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.thresholds = thresholds or ResourceThresholds()
        self.save_dir = Path(save_dir) if save_dir else Path('resource_monitoring')
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize monitoring components
        self.gpu_monitor = GPUMonitor()
        self.process_monitor = ProcessMonitor()

        # Data storage
        self.resource_history: deque = deque(maxlen=history_size)
        self.alerts: List[ResourceAlert] = []
        self.optimization_recommendations: List[OptimizationRecommendation] = []

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.alert_callbacks: List[Callable] = []

        # Statistics
        self.monitoring_start_time = None
        self.total_measurements = 0

        logger.info("ResourceMonitor initialized")

    def start_monitoring(self) -> None:
        """Start resource monitoring"""
        if self.monitoring_active:
            logger.warning("Resource monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_start_time = time.time()

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info(f"Resource monitoring started (interval: {self.monitoring_interval}s)")

    def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        if not self.monitoring_active:
            return

        self.monitoring_active = False

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        logger.info("Resource monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect resource snapshot
                snapshot = self._collect_resource_snapshot()
                self.resource_history.append(snapshot)
                self.total_measurements += 1

                # Check for alerts
                self._check_alerts(snapshot)

                # Generate optimization recommendations periodically
                if self.total_measurements % 20 == 0:  # Every 20 measurements
                    self._generate_optimization_recommendations()

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_resource_snapshot(self) -> ResourceSnapshot:
        """Collect current resource snapshot"""
        timestamp = time.time()

        # System-wide stats
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Network stats
        try:
            network = psutil.net_io_counters()
            network_sent = network.bytes_sent
            network_recv = network.bytes_recv
        except:
            network_sent = None
            network_recv = None

        # GPU stats
        gpu_stats = self.gpu_monitor.get_gpu_stats()

        # Process stats
        process_stats = self.process_monitor.get_process_stats()

        return ResourceSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / 1024**3,
            memory_available_gb=memory.available / 1024**3,
            disk_usage_percent=disk.percent,
            disk_free_gb=disk.free / 1024**3,
            gpu_utilization=gpu_stats['utilization'],
            gpu_memory_used_gb=gpu_stats['memory_used_gb'],
            gpu_memory_total_gb=gpu_stats['memory_total_gb'],
            gpu_temperature=gpu_stats['temperature'],
            process_cpu_percent=process_stats['cpu_percent'],
            process_memory_mb=process_stats['memory_mb'],
            network_bytes_sent=network_sent,
            network_bytes_recv=network_recv
        )

    def _check_alerts(self, snapshot: ResourceSnapshot) -> None:
        """Check for resource usage alerts"""
        alerts_generated = []

        # CPU alerts
        if snapshot.cpu_percent >= self.thresholds.cpu_critical:
            alert = self._create_alert('critical', 'cpu', snapshot.cpu_percent,
                                     self.thresholds.cpu_critical,
                                     f"Critical CPU usage: {snapshot.cpu_percent:.1f}%")
            alerts_generated.append(alert)
        elif snapshot.cpu_percent >= self.thresholds.cpu_warning:
            alert = self._create_alert('warning', 'cpu', snapshot.cpu_percent,
                                     self.thresholds.cpu_warning,
                                     f"High CPU usage: {snapshot.cpu_percent:.1f}%")
            alerts_generated.append(alert)

        # Memory alerts
        if snapshot.memory_percent >= self.thresholds.memory_critical:
            alert = self._create_alert('critical', 'memory', snapshot.memory_percent,
                                     self.thresholds.memory_critical,
                                     f"Critical memory usage: {snapshot.memory_percent:.1f}%")
            alerts_generated.append(alert)
        elif snapshot.memory_percent >= self.thresholds.memory_warning:
            alert = self._create_alert('warning', 'memory', snapshot.memory_percent,
                                     self.thresholds.memory_warning,
                                     f"High memory usage: {snapshot.memory_percent:.1f}%")
            alerts_generated.append(alert)

        # Disk alerts
        if snapshot.disk_usage_percent >= self.thresholds.disk_critical:
            alert = self._create_alert('critical', 'disk', snapshot.disk_usage_percent,
                                     self.thresholds.disk_critical,
                                     f"Critical disk usage: {snapshot.disk_usage_percent:.1f}%")
            alerts_generated.append(alert)
        elif snapshot.disk_usage_percent >= self.thresholds.disk_warning:
            alert = self._create_alert('warning', 'disk', snapshot.disk_usage_percent,
                                     self.thresholds.disk_warning,
                                     f"High disk usage: {snapshot.disk_usage_percent:.1f}%")
            alerts_generated.append(alert)

        # GPU alerts
        if snapshot.gpu_utilization is not None:
            if snapshot.gpu_utilization >= self.thresholds.gpu_critical:
                alert = self._create_alert('critical', 'gpu', snapshot.gpu_utilization,
                                         self.thresholds.gpu_critical,
                                         f"Critical GPU utilization: {snapshot.gpu_utilization:.1f}%")
                alerts_generated.append(alert)
            elif snapshot.gpu_utilization >= self.thresholds.gpu_warning:
                alert = self._create_alert('warning', 'gpu', snapshot.gpu_utilization,
                                         self.thresholds.gpu_warning,
                                         f"High GPU utilization: {snapshot.gpu_utilization:.1f}%")
                alerts_generated.append(alert)

        # GPU memory alerts
        if (snapshot.gpu_memory_used_gb is not None and
            snapshot.gpu_memory_total_gb is not None and
            snapshot.gpu_memory_total_gb > 0):

            gpu_memory_percent = (snapshot.gpu_memory_used_gb / snapshot.gpu_memory_total_gb) * 100

            if gpu_memory_percent >= self.thresholds.gpu_memory_critical:
                alert = self._create_alert('critical', 'gpu_memory', gpu_memory_percent,
                                         self.thresholds.gpu_memory_critical,
                                         f"Critical GPU memory usage: {gpu_memory_percent:.1f}%")
                alerts_generated.append(alert)
            elif gpu_memory_percent >= self.thresholds.gpu_memory_warning:
                alert = self._create_alert('warning', 'gpu_memory', gpu_memory_percent,
                                         self.thresholds.gpu_memory_warning,
                                         f"High GPU memory usage: {gpu_memory_percent:.1f}%")
                alerts_generated.append(alert)

        # GPU temperature alerts
        if snapshot.gpu_temperature is not None:
            if snapshot.gpu_temperature >= self.thresholds.gpu_temperature_critical:
                alert = self._create_alert('critical', 'gpu_temperature', snapshot.gpu_temperature,
                                         self.thresholds.gpu_temperature_critical,
                                         f"Critical GPU temperature: {snapshot.gpu_temperature:.1f}°C")
                alerts_generated.append(alert)
            elif snapshot.gpu_temperature >= self.thresholds.gpu_temperature_warning:
                alert = self._create_alert('warning', 'gpu_temperature', snapshot.gpu_temperature,
                                         self.thresholds.gpu_temperature_warning,
                                         f"High GPU temperature: {snapshot.gpu_temperature:.1f}°C")
                alerts_generated.append(alert)

        # Process new alerts
        for alert in alerts_generated:
            self.alerts.append(alert)
            self._trigger_alert_callbacks(alert)
            logger.warning(f"Resource alert: {alert.message}")

    def _create_alert(self, alert_type: str, resource: str,
                     current_value: float, threshold: float, message: str) -> ResourceAlert:
        """Create resource alert"""
        alert_id = f"{resource}_{alert_type}_{int(time.time())}"

        return ResourceAlert(
            alert_id=alert_id,
            timestamp=time.time(),
            alert_type=alert_type,
            resource=resource,
            current_value=current_value,
            threshold=threshold,
            message=message
        )

    def _trigger_alert_callbacks(self, alert: ResourceAlert) -> None:
        """Trigger registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def register_alert_callback(self, callback: Callable[[ResourceAlert], None]) -> None:
        """Register callback for resource alerts"""
        self.alert_callbacks.append(callback)

    def _generate_optimization_recommendations(self) -> None:
        """Generate optimization recommendations based on resource usage patterns"""
        if len(self.resource_history) < 10:
            return

        recent_snapshots = list(self.resource_history)[-10:]

        # Analyze patterns
        avg_cpu = np.mean([s.cpu_percent for s in recent_snapshots])
        avg_memory = np.mean([s.memory_percent for s in recent_snapshots])
        avg_gpu_util = np.mean([s.gpu_utilization for s in recent_snapshots
                               if s.gpu_utilization is not None])

        recommendations = []

        # High CPU usage recommendations
        if avg_cpu > 80:
            rec = OptimizationRecommendation(
                recommendation_id=f"cpu_opt_{int(time.time())}",
                timestamp=time.time(),
                category='performance',
                priority='high',
                title='Optimize CPU Usage',
                description=f'Average CPU usage is high ({avg_cpu:.1f}%)',
                implementation_steps=[
                    'Reduce batch size in training configuration',
                    'Decrease number of parallel workers',
                    'Consider using smaller models',
                    'Implement gradient accumulation'
                ],
                expected_benefit='Reduced CPU load and improved system stability'
            )
            recommendations.append(rec)

        # High memory usage recommendations
        if avg_memory > 85:
            rec = OptimizationRecommendation(
                recommendation_id=f"memory_opt_{int(time.time())}",
                timestamp=time.time(),
                category='memory',
                priority='high',
                title='Optimize Memory Usage',
                description=f'Average memory usage is high ({avg_memory:.1f}%)',
                implementation_steps=[
                    'Reduce batch size',
                    'Clear unused variables and caches',
                    'Use gradient checkpointing',
                    'Consider model pruning'
                ],
                expected_benefit='Reduced memory pressure and OOM prevention'
            )
            recommendations.append(rec)

        # Low GPU utilization recommendations
        if avg_gpu_util is not None and not np.isnan(avg_gpu_util) and avg_gpu_util < 50:
            rec = OptimizationRecommendation(
                recommendation_id=f"gpu_opt_{int(time.time())}",
                timestamp=time.time(),
                category='efficiency',
                priority='medium',
                title='Improve GPU Utilization',
                description=f'GPU utilization is low ({avg_gpu_util:.1f}%)',
                implementation_steps=[
                    'Increase batch size if memory allows',
                    'Use mixed precision training',
                    'Optimize data loading pipeline',
                    'Consider increasing model complexity'
                ],
                expected_benefit='Better hardware utilization and faster training'
            )
            recommendations.append(rec)

        # Add new recommendations
        for rec in recommendations:
            # Check if similar recommendation already exists
            existing_similar = any(
                existing.category == rec.category and
                time.time() - existing.timestamp < 3600  # 1 hour
                for existing in self.optimization_recommendations
            )

            if not existing_similar:
                self.optimization_recommendations.append(rec)
                logger.info(f"New optimization recommendation: {rec.title}")

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current resource statistics"""
        if not self.resource_history:
            return {}

        latest = self.resource_history[-1]

        stats = {
            'timestamp': latest.timestamp,
            'cpu_percent': latest.cpu_percent,
            'memory_percent': latest.memory_percent,
            'memory_used_gb': latest.memory_used_gb,
            'disk_usage_percent': latest.disk_usage_percent,
            'disk_free_gb': latest.disk_free_gb,
            'monitoring_duration_minutes': (time.time() - self.monitoring_start_time) / 60
                                          if self.monitoring_start_time else 0,
            'total_measurements': self.total_measurements
        }

        # Add GPU stats if available
        if latest.gpu_utilization is not None:
            stats.update({
                'gpu_utilization': latest.gpu_utilization,
                'gpu_memory_used_gb': latest.gpu_memory_used_gb,
                'gpu_memory_total_gb': latest.gpu_memory_total_gb,
                'gpu_temperature': latest.gpu_temperature
            })

        # Add process stats if available
        if latest.process_cpu_percent is not None:
            stats.update({
                'process_cpu_percent': latest.process_cpu_percent,
                'process_memory_mb': latest.process_memory_mb
            })

        return stats

    def get_usage_trends(self, window_minutes: int = 30) -> Dict[str, List[float]]:
        """Get resource usage trends over specified time window"""
        if not self.resource_history:
            return {}

        current_time = time.time()
        cutoff_time = current_time - (window_minutes * 60)

        # Filter snapshots within time window
        recent_snapshots = [s for s in self.resource_history if s.timestamp >= cutoff_time]

        if not recent_snapshots:
            return {}

        trends = {
            'timestamps': [s.timestamp for s in recent_snapshots],
            'cpu_percent': [s.cpu_percent for s in recent_snapshots],
            'memory_percent': [s.memory_percent for s in recent_snapshots],
            'disk_usage_percent': [s.disk_usage_percent for s in recent_snapshots]
        }

        # Add GPU trends if available
        gpu_utils = [s.gpu_utilization for s in recent_snapshots if s.gpu_utilization is not None]
        if gpu_utils:
            trends['gpu_utilization'] = gpu_utils
            trends['gpu_memory_percent'] = [
                (s.gpu_memory_used_gb / s.gpu_memory_total_gb) * 100
                for s in recent_snapshots
                if s.gpu_memory_used_gb is not None and s.gpu_memory_total_gb is not None
            ]

        return trends

    def get_active_alerts(self) -> List[ResourceAlert]:
        """Get list of active (unresolved) alerts"""
        return [alert for alert in self.alerts if not alert.resolved]

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark alert as resolved"""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_timestamp = time.time()
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False

    def get_optimization_recommendations(self, category: Optional[str] = None) -> List[OptimizationRecommendation]:
        """Get optimization recommendations"""
        if category:
            return [rec for rec in self.optimization_recommendations if rec.category == category]
        return self.optimization_recommendations.copy()

    def save_monitoring_data(self, filename: Optional[str] = None) -> str:
        """Save monitoring data to file"""
        if filename is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resource_monitoring_{timestamp_str}.json"

        filepath = self.save_dir / filename

        # Prepare data for serialization
        monitoring_data = {
            'monitoring_info': {
                'start_time': self.monitoring_start_time,
                'total_measurements': self.total_measurements,
                'monitoring_interval': self.monitoring_interval,
                'thresholds': asdict(self.thresholds)
            },
            'resource_history': [asdict(snapshot) for snapshot in self.resource_history],
            'alerts': [asdict(alert) for alert in self.alerts],
            'optimization_recommendations': [asdict(rec) for rec in self.optimization_recommendations]
        }

        with open(filepath, 'w') as f:
            json.dump(monitoring_data, f, indent=2)

        logger.info(f"Monitoring data saved to: {filepath}")
        return str(filepath)

    def generate_monitoring_report(self) -> str:
        """Generate comprehensive monitoring report"""
        if not self.resource_history:
            return "No monitoring data available"

        report = []
        report.append("# Resource Monitoring Report")
        report.append("=" * 50)
        report.append("")

        # Monitoring summary
        duration_hours = (time.time() - self.monitoring_start_time) / 3600 if self.monitoring_start_time else 0
        report.append("## Monitoring Summary")
        report.append(f"- Duration: {duration_hours:.2f} hours")
        report.append(f"- Total measurements: {self.total_measurements}")
        report.append(f"- Measurement interval: {self.monitoring_interval}s")
        report.append("")

        # Current status
        current_stats = self.get_current_stats()
        report.append("## Current Resource Status")
        report.append(f"- CPU Usage: {current_stats.get('cpu_percent', 0):.1f}%")
        report.append(f"- Memory Usage: {current_stats.get('memory_percent', 0):.1f}%")
        report.append(f"- Disk Usage: {current_stats.get('disk_usage_percent', 0):.1f}%")

        if 'gpu_utilization' in current_stats:
            report.append(f"- GPU Utilization: {current_stats['gpu_utilization']:.1f}%")
            if 'gpu_temperature' in current_stats and current_stats['gpu_temperature']:
                report.append(f"- GPU Temperature: {current_stats['gpu_temperature']:.1f}°C")

        report.append("")

        # Alert summary
        active_alerts = self.get_active_alerts()
        report.append("## Alert Summary")
        report.append(f"- Total alerts: {len(self.alerts)}")
        report.append(f"- Active alerts: {len(active_alerts)}")

        if active_alerts:
            report.append("### Active Alerts:")
            for alert in active_alerts[-5:]:  # Show last 5
                report.append(f"- {alert.alert_type.upper()}: {alert.message}")

        report.append("")

        # Optimization recommendations
        recent_recommendations = [rec for rec in self.optimization_recommendations
                                if time.time() - rec.timestamp < 86400]  # Last 24 hours

        report.append("## Optimization Recommendations")
        report.append(f"- Total recommendations: {len(self.optimization_recommendations)}")
        report.append(f"- Recent recommendations (24h): {len(recent_recommendations)}")

        if recent_recommendations:
            report.append("### Recent Recommendations:")
            for rec in recent_recommendations:
                report.append(f"- {rec.priority.upper()}: {rec.title}")
                report.append(f"  {rec.description}")

        return "\n".join(report)

    def create_resource_plots(self, save_dir: Optional[str] = None) -> List[str]:
        """Create resource usage plots"""
        if len(self.resource_history) < 2:
            logger.warning("Insufficient data for plotting")
            return []

        plot_dir = Path(save_dir) if save_dir else self.save_dir / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)

        plots_created = []

        try:
            # Extract data
            timestamps = [s.timestamp for s in self.resource_history]
            cpu_usage = [s.cpu_percent for s in self.resource_history]
            memory_usage = [s.memory_percent for s in self.resource_history]

            # Convert timestamps to relative minutes
            start_time = timestamps[0]
            time_minutes = [(t - start_time) / 60 for t in timestamps]

            # Plot 1: CPU and Memory usage over time
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            ax1.plot(time_minutes, cpu_usage, 'b-', linewidth=1.5, label='CPU Usage')
            ax1.axhline(y=self.thresholds.cpu_warning, color='orange', linestyle='--', label='Warning')
            ax1.axhline(y=self.thresholds.cpu_critical, color='red', linestyle='--', label='Critical')
            ax1.set_title('CPU Usage Over Time', fontsize=14, fontweight='bold')
            ax1.set_ylabel('CPU Usage (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)

            ax2.plot(time_minutes, memory_usage, 'g-', linewidth=1.5, label='Memory Usage')
            ax2.axhline(y=self.thresholds.memory_warning, color='orange', linestyle='--', label='Warning')
            ax2.axhline(y=self.thresholds.memory_critical, color='red', linestyle='--', label='Critical')
            ax2.set_title('Memory Usage Over Time', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Time (minutes)')
            ax2.set_ylabel('Memory Usage (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)

            plt.tight_layout()
            cpu_memory_plot = plot_dir / 'cpu_memory_usage.png'
            plt.savefig(cpu_memory_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plots_created.append(str(cpu_memory_plot))

            # Plot 2: GPU usage if available
            gpu_data = [s.gpu_utilization for s in self.resource_history if s.gpu_utilization is not None]
            if gpu_data:
                fig, ax = plt.subplots(figsize=(12, 6))

                gpu_timestamps = [s.timestamp for s in self.resource_history if s.gpu_utilization is not None]
                gpu_time_minutes = [(t - start_time) / 60 for t in gpu_timestamps]

                ax.plot(gpu_time_minutes, gpu_data, 'r-', linewidth=1.5, label='GPU Utilization')
                ax.axhline(y=self.thresholds.gpu_warning, color='orange', linestyle='--', label='Warning')
                ax.axhline(y=self.thresholds.gpu_critical, color='red', linestyle='--', label='Critical')
                ax.set_title('GPU Utilization Over Time', fontsize=14, fontweight='bold')
                ax.set_xlabel('Time (minutes)')
                ax.set_ylabel('GPU Utilization (%)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 100)

                plt.tight_layout()
                gpu_plot = plot_dir / 'gpu_usage.png'
                plt.savefig(gpu_plot, dpi=300, bbox_inches='tight')
                plt.close()
                plots_created.append(str(gpu_plot))

            logger.info(f"Resource plots created: {len(plots_created)} plots")

        except Exception as e:
            logger.error(f"Failed to create resource plots: {e}")

        return plots_created


# Factory function for easy creation
def create_resource_monitor(monitoring_interval: float = 5.0,
                          save_dir: Optional[str] = None,
                          auto_start: bool = True) -> ResourceMonitor:
    """
    Factory function to create and optionally start resource monitor

    Args:
        monitoring_interval: Interval between measurements in seconds
        save_dir: Directory to save monitoring data
        auto_start: Whether to automatically start monitoring

    Returns:
        Configured ResourceMonitor
    """
    monitor = ResourceMonitor(
        monitoring_interval=monitoring_interval,
        save_dir=save_dir
    )

    if auto_start:
        monitor.start_monitoring()

    return monitor