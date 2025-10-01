"""
Pipeline Monitoring & Debugging - Task 5 Implementation
Comprehensive monitoring and debugging capabilities for the AI pipeline.
"""

import time
import json
import logging
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import traceback
from pathlib import Path
import sys


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    stage_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    average_duration: float = 0.0
    last_execution: Optional[str] = None
    error_count: Dict[str, int] = None

    def __post_init__(self):
        if self.error_count is None:
            self.error_count = defaultdict(int)


@dataclass
class PipelineExecution:
    """Record of a single pipeline execution."""
    execution_id: str
    image_path: str
    start_time: str
    end_time: Optional[str] = None
    total_duration: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    stage_durations: Dict[str, float] = None
    parameters_used: Dict[str, Any] = None
    quality_achieved: float = 0.0
    components_used: Dict[str, str] = None

    def __post_init__(self):
        if self.stage_durations is None:
            self.stage_durations = {}
        if self.parameters_used is None:
            self.parameters_used = {}
        if self.components_used is None:
            self.components_used = {}


class PipelineMonitor:
    """
    Comprehensive monitoring system for the AI pipeline.

    Tracks performance, errors, and debugging information across all pipeline stages.
    """

    def __init__(self,
                 max_history_size: int = 1000,
                 enable_detailed_logging: bool = True,
                 enable_profiling: bool = False,
                 export_path: Optional[str] = None):
        """
        Initialize pipeline monitor.

        Args:
            max_history_size: Maximum number of executions to keep in memory
            enable_detailed_logging: Whether to enable detailed debug logging
            enable_profiling: Whether to enable performance profiling
            export_path: Path to export metrics (optional)
        """
        self.max_history_size = max_history_size
        self.enable_detailed_logging = enable_detailed_logging
        self.enable_profiling = enable_profiling
        self.export_path = export_path

        # Metrics storage
        self.stage_metrics: Dict[str, StageMetrics] = {}
        self.execution_history: deque = deque(maxlen=max_history_size)
        self.active_executions: Dict[str, PipelineExecution] = {}

        # Threading safety
        self._lock = threading.RLock()

        # Performance tracking
        self.start_time = datetime.now()
        self.total_processed = 0
        self.total_successful = 0
        self.total_failed = 0

        # Error tracking
        self.error_patterns = defaultdict(int)
        self.recent_errors = deque(maxlen=100)

        # Component health tracking
        self.component_health = defaultdict(lambda: {'calls': 0, 'failures': 0, 'avg_time': 0.0})

        # Debug hooks
        self.debug_hooks: List[Callable] = []

        self.logger = logging.getLogger(__name__)

        if self.enable_detailed_logging:
            self.logger.setLevel(logging.DEBUG)

        self.logger.info(f"PipelineMonitor initialized (history_size={max_history_size}, "
                        f"detailed_logging={enable_detailed_logging}, profiling={enable_profiling})")

    def start_execution(self, execution_id: str, image_path: str) -> str:
        """
        Start monitoring a new pipeline execution.

        Args:
            execution_id: Unique identifier for this execution
            image_path: Path to the image being processed

        Returns:
            Execution ID for tracking
        """
        with self._lock:
            execution = PipelineExecution(
                execution_id=execution_id,
                image_path=image_path,
                start_time=datetime.now().isoformat()
            )

            self.active_executions[execution_id] = execution
            self.total_processed += 1

            if self.enable_detailed_logging:
                self.logger.debug(f"Started execution {execution_id} for {image_path}")

            return execution_id

    def end_execution(self,
                     execution_id: str,
                     success: bool,
                     error_message: Optional[str] = None,
                     quality_achieved: float = 0.0,
                     parameters_used: Optional[Dict[str, Any]] = None,
                     components_used: Optional[Dict[str, str]] = None):
        """
        End monitoring for a pipeline execution.

        Args:
            execution_id: Execution identifier
            success: Whether execution was successful
            error_message: Error message if failed
            quality_achieved: Quality score achieved
            parameters_used: Parameters used in the execution
            components_used: Components used in the execution
        """
        with self._lock:
            if execution_id not in self.active_executions:
                self.logger.warning(f"Execution {execution_id} not found in active executions")
                return

            execution = self.active_executions[execution_id]
            execution.end_time = datetime.now().isoformat()
            execution.success = success
            execution.error_message = error_message
            execution.quality_achieved = quality_achieved

            if parameters_used:
                execution.parameters_used = parameters_used
            if components_used:
                execution.components_used = components_used

            # Calculate total duration
            start_dt = datetime.fromisoformat(execution.start_time.replace('Z', '+00:00').replace('+00:00', ''))
            end_dt = datetime.fromisoformat(execution.end_time.replace('Z', '+00:00').replace('+00:00', ''))
            execution.total_duration = (end_dt - start_dt).total_seconds()

            # Update counters
            if success:
                self.total_successful += 1
            else:
                self.total_failed += 1
                if error_message:
                    self._track_error(error_message)

            # Move to history
            self.execution_history.append(execution)
            del self.active_executions[execution_id]

            if self.enable_detailed_logging:
                self.logger.debug(f"Ended execution {execution_id}: success={success}, "
                                f"duration={execution.total_duration:.3f}s")

    def record_stage(self,
                    execution_id: str,
                    stage: str,
                    duration: float,
                    success: bool,
                    error_message: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Record metrics for a specific pipeline stage.

        Args:
            execution_id: Execution identifier
            stage: Stage name
            duration: Stage duration in seconds
            success: Whether stage was successful
            error_message: Error message if failed
            metadata: Additional metadata
        """
        with self._lock:
            # Update stage metrics
            if stage not in self.stage_metrics:
                self.stage_metrics[stage] = StageMetrics(stage_name=stage)

            metrics = self.stage_metrics[stage]
            metrics.total_calls += 1
            metrics.total_duration += duration
            metrics.last_execution = datetime.now().isoformat()

            if success:
                metrics.successful_calls += 1
            else:
                metrics.failed_calls += 1
                if error_message:
                    error_type = error_message.split(':')[0] if ':' in error_message else error_message
                    metrics.error_count[error_type] += 1

            # Update duration statistics
            metrics.min_duration = min(metrics.min_duration, duration)
            metrics.max_duration = max(metrics.max_duration, duration)
            metrics.average_duration = metrics.total_duration / metrics.total_calls

            # Record in active execution
            if execution_id in self.active_executions:
                self.active_executions[execution_id].stage_durations[stage] = duration

            if self.enable_detailed_logging:
                self.logger.debug(f"Stage {stage} in {execution_id}: {duration:.3f}s, success={success}")

    def record_component_usage(self,
                              component_type: str,
                              component_name: str,
                              duration: float,
                              success: bool):
        """
        Record component usage metrics.

        Args:
            component_type: Type of component (e.g., 'classifier', 'optimizer')
            component_name: Specific component name
            duration: Processing duration
            success: Whether component call was successful
        """
        with self._lock:
            key = f"{component_type}:{component_name}"
            health = self.component_health[key]

            health['calls'] += 1
            if not success:
                health['failures'] += 1

            # Update average time (exponential moving average)
            alpha = 0.1  # Smoothing factor
            health['avg_time'] = alpha * duration + (1 - alpha) * health['avg_time']

    def _track_error(self, error_message: str):
        """Track error patterns and recent errors."""
        # Extract error pattern (first part before any specific details)
        error_pattern = error_message.split('\n')[0]  # First line only
        if ':' in error_pattern:
            error_pattern = error_pattern.split(':')[0]

        self.error_patterns[error_pattern] += 1
        self.recent_errors.append({
            'timestamp': datetime.now().isoformat(),
            'message': error_message,
            'pattern': error_pattern
        })

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline statistics.

        Returns:
            Dictionary containing all monitored metrics
        """
        with self._lock:
            # Calculate overall metrics
            uptime = (datetime.now() - self.start_time).total_seconds()
            success_rate = (self.total_successful / self.total_processed * 100
                          if self.total_processed > 0 else 0)

            # Stage statistics
            stage_stats = {}
            for stage_name, metrics in self.stage_metrics.items():
                stage_stats[stage_name] = {
                    'total_calls': metrics.total_calls,
                    'success_rate': (metrics.successful_calls / metrics.total_calls * 100
                                   if metrics.total_calls > 0 else 0),
                    'average_duration_ms': metrics.average_duration * 1000,
                    'min_duration_ms': metrics.min_duration * 1000 if metrics.min_duration != float('inf') else 0,
                    'max_duration_ms': metrics.max_duration * 1000,
                    'total_duration_sec': metrics.total_duration,
                    'error_count': dict(metrics.error_count),
                    'last_execution': metrics.last_execution
                }

            # Recent performance trends
            recent_executions = list(self.execution_history)[-50:]  # Last 50 executions
            if recent_executions:
                recent_durations = [e.total_duration for e in recent_executions if e.total_duration > 0]
                recent_success = [e.success for e in recent_executions]

                recent_stats = {
                    'count': len(recent_executions),
                    'average_duration_ms': statistics.mean(recent_durations) * 1000 if recent_durations else 0,
                    'median_duration_ms': statistics.median(recent_durations) * 1000 if recent_durations else 0,
                    'success_rate': sum(recent_success) / len(recent_success) * 100 if recent_success else 0
                }
            else:
                recent_stats = {'count': 0, 'average_duration_ms': 0, 'median_duration_ms': 0, 'success_rate': 0}

            # Component health
            component_stats = {}
            for key, health in self.component_health.items():
                failure_rate = health['failures'] / health['calls'] * 100 if health['calls'] > 0 else 0
                component_stats[key] = {
                    'calls': health['calls'],
                    'failure_rate': failure_rate,
                    'average_time_ms': health['avg_time'] * 1000,
                    'health_status': 'healthy' if failure_rate < 5 else ('degraded' if failure_rate < 20 else 'unhealthy')
                }

            return {
                'overview': {
                    'uptime_seconds': uptime,
                    'total_processed': self.total_processed,
                    'total_successful': self.total_successful,
                    'total_failed': self.total_failed,
                    'success_rate_percent': success_rate,
                    'active_executions': len(self.active_executions),
                    'history_size': len(self.execution_history)
                },
                'stage_statistics': stage_stats,
                'recent_performance': recent_stats,
                'component_health': component_stats,
                'error_patterns': dict(self.error_patterns),
                'recent_errors': list(self.recent_errors)[-10:],  # Last 10 errors
                'timestamp': datetime.now().isoformat()
            }

    def get_execution_details(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific execution.

        Args:
            execution_id: Execution identifier

        Returns:
            Execution details or None if not found
        """
        with self._lock:
            # Check active executions
            if execution_id in self.active_executions:
                return asdict(self.active_executions[execution_id])

            # Check history
            for execution in self.execution_history:
                if execution.execution_id == execution_id:
                    return asdict(execution)

            return None

    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks in the pipeline.

        Returns:
            List of identified bottlenecks with recommendations
        """
        bottlenecks = []

        with self._lock:
            # Stage-level bottlenecks
            for stage_name, metrics in self.stage_metrics.items():
                if metrics.total_calls > 5:  # Only analyze stages with sufficient data
                    avg_duration = metrics.average_duration
                    max_duration = metrics.max_duration
                    success_rate = metrics.successful_calls / metrics.total_calls * 100

                    # Slow stage detection
                    if avg_duration > 2.0:  # > 2 seconds average
                        bottlenecks.append({
                            'type': 'slow_stage',
                            'stage': stage_name,
                            'average_duration_sec': avg_duration,
                            'severity': 'high' if avg_duration > 5.0 else 'medium',
                            'recommendation': f"Optimize {stage_name} stage - consider caching or algorithm improvements"
                        })

                    # High variance detection
                    if max_duration > avg_duration * 3:  # High variance
                        bottlenecks.append({
                            'type': 'high_variance',
                            'stage': stage_name,
                            'variance_ratio': max_duration / avg_duration,
                            'severity': 'medium',
                            'recommendation': f"Investigate inconsistent performance in {stage_name}"
                        })

                    # Low success rate
                    if success_rate < 95:
                        bottlenecks.append({
                            'type': 'low_success_rate',
                            'stage': stage_name,
                            'success_rate': success_rate,
                            'severity': 'high' if success_rate < 80 else 'medium',
                            'recommendation': f"Improve error handling in {stage_name}"
                        })

            # Component-level bottlenecks
            for key, health in self.component_health.items():
                failure_rate = health['failures'] / health['calls'] * 100 if health['calls'] > 0 else 0

                if failure_rate > 10:
                    bottlenecks.append({
                        'type': 'unreliable_component',
                        'component': key,
                        'failure_rate': failure_rate,
                        'severity': 'high' if failure_rate > 25 else 'medium',
                        'recommendation': f"Investigate reliability issues with {key}"
                    })

                if health['avg_time'] > 3.0:
                    bottlenecks.append({
                        'type': 'slow_component',
                        'component': key,
                        'average_time_sec': health['avg_time'],
                        'severity': 'medium',
                        'recommendation': f"Optimize performance of {key}"
                    })

        return bottlenecks

    def export_metrics(self, file_path: Optional[str] = None) -> bool:
        """
        Export all metrics to a JSON file.

        Args:
            file_path: Path to export file (optional)

        Returns:
            True if export successful, False otherwise
        """
        export_path = file_path or self.export_path
        if not export_path:
            export_path = f"pipeline_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            # Get all statistics
            stats = self.get_statistics()

            # Add execution history
            with self._lock:
                execution_list = []
                for execution in self.execution_history:
                    execution_list.append(asdict(execution))

                stats['execution_history'] = execution_list

            # Export to file
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)

            self.logger.info(f"Metrics exported to {export_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False

    def generate_performance_report(self) -> str:
        """
        Generate a human-readable performance report.

        Returns:
            Formatted performance report string
        """
        stats = self.get_statistics()
        bottlenecks = self.identify_bottlenecks()

        # Build report
        report_lines = [
            "═══════════════════════════════════════════════",
            "           PIPELINE PERFORMANCE REPORT        ",
            "═══════════════════════════════════════════════",
            "",
            "📊 OVERVIEW:",
            f"   Uptime: {stats['overview']['uptime_seconds']:.1f} seconds",
            f"   Total Processed: {stats['overview']['total_processed']}",
            f"   Success Rate: {stats['overview']['success_rate_percent']:.1f}%",
            f"   Active Executions: {stats['overview']['active_executions']}",
            "",
            "⏱️  STAGE PERFORMANCE:",
        ]

        for stage_name, stage_stats in stats['stage_statistics'].items():
            report_lines.extend([
                f"   {stage_name.upper()}:",
                f"     Calls: {stage_stats['total_calls']}",
                f"     Success Rate: {stage_stats['success_rate']:.1f}%",
                f"     Avg Duration: {stage_stats['average_duration_ms']:.1f}ms",
                f"     Range: {stage_stats['min_duration_ms']:.1f}-{stage_stats['max_duration_ms']:.1f}ms",
                ""
            ])

        if stats['recent_performance']['count'] > 0:
            recent = stats['recent_performance']
            report_lines.extend([
                "📈 RECENT TREND (Last 50 executions):",
                f"   Average Duration: {recent['average_duration_ms']:.1f}ms",
                f"   Median Duration: {recent['median_duration_ms']:.1f}ms",
                f"   Success Rate: {recent['success_rate']:.1f}%",
                ""
            ])

        if bottlenecks:
            report_lines.extend([
                "⚠️  BOTTLENECKS IDENTIFIED:",
            ])
            for bottleneck in bottlenecks:
                severity_icon = "🔴" if bottleneck['severity'] == 'high' else "🟡"
                report_lines.extend([
                    f"   {severity_icon} {bottleneck['type'].upper()}:",
                    f"     {bottleneck['recommendation']}",
                    ""
                ])
        else:
            report_lines.extend([
                "✅ No significant bottlenecks identified",
                ""
            ])

        if stats['recent_errors']:
            report_lines.extend([
                "🐛 RECENT ERRORS:",
            ])
            for error in stats['recent_errors'][-5:]:  # Last 5 errors
                report_lines.extend([
                    f"   {error['timestamp']}: {error['pattern']}",
                ])
            report_lines.append("")

        report_lines.extend([
            "═══════════════════════════════════════════════",
            f"Report generated: {datetime.now().isoformat()}",
            "═══════════════════════════════════════════════"
        ])

        return "\n".join(report_lines)

    def add_debug_hook(self, hook: Callable[[str, Dict[str, Any]], None]):
        """
        Add a debug hook function.

        Args:
            hook: Function that takes (event_type, data) parameters
        """
        self.debug_hooks.append(hook)

    def trigger_debug_hook(self, event_type: str, data: Dict[str, Any]):
        """Trigger all debug hooks with event data."""
        for hook in self.debug_hooks:
            try:
                hook(event_type, data)
            except Exception as e:
                self.logger.error(f"Debug hook failed: {e}")

    def reset_metrics(self):
        """Reset all metrics and history."""
        with self._lock:
            self.stage_metrics.clear()
            self.execution_history.clear()
            self.active_executions.clear()
            self.error_patterns.clear()
            self.recent_errors.clear()
            self.component_health.clear()

            self.start_time = datetime.now()
            self.total_processed = 0
            self.total_successful = 0
            self.total_failed = 0

            self.logger.info("All metrics reset")


def test_pipeline_monitor():
    """Test the pipeline monitoring system."""
    print("Testing Pipeline Monitor...")

    # Initialize monitor
    monitor = PipelineMonitor(
        max_history_size=100,
        enable_detailed_logging=False,
        enable_profiling=False
    )

    print(f"✓ Monitor initialized")

    # Test execution tracking
    print("\n1. Testing execution tracking:")
    execution_id = "test_exec_001"
    monitor.start_execution(execution_id, "test_image.png")

    # Simulate some stages
    monitor.record_stage(execution_id, "feature_extraction", 0.5, True)
    monitor.record_stage(execution_id, "classification", 0.2, True)
    monitor.record_stage(execution_id, "optimization", 1.0, True)
    monitor.record_stage(execution_id, "conversion", 2.0, True)

    # End execution
    monitor.end_execution(
        execution_id,
        success=True,
        quality_achieved=0.85,
        parameters_used={"corner_threshold": 30},
        components_used={"classifier": "statistical", "optimizer": "learned"}
    )

    print(f"   ✓ Execution {execution_id} tracked successfully")

    # Test error tracking
    print("\n2. Testing error tracking:")
    error_exec = "test_exec_002"
    monitor.start_execution(error_exec, "error_image.png")
    monitor.record_stage(error_exec, "feature_extraction", 0.3, False, "Feature extraction failed: Invalid image")
    monitor.end_execution(error_exec, success=False, error_message="Pipeline failed at feature extraction")

    print(f"   ✓ Error execution tracked")

    # Test component health tracking
    print("\n3. Testing component health tracking:")
    monitor.record_component_usage("classifier", "statistical", 0.2, True)
    monitor.record_component_usage("classifier", "statistical", 0.3, True)
    monitor.record_component_usage("classifier", "rule_based", 0.1, False)

    print(f"   ✓ Component health tracked")

    # Test statistics
    print("\n4. Testing statistics generation:")
    stats = monitor.get_statistics()

    print(f"   ✓ Total processed: {stats['overview']['total_processed']}")
    print(f"   ✓ Success rate: {stats['overview']['success_rate_percent']:.1f}%")
    print(f"   ✓ Stages tracked: {len(stats['stage_statistics'])}")
    print(f"   ✓ Components tracked: {len(stats['component_health'])}")

    # Test bottleneck identification
    print("\n5. Testing bottleneck identification:")
    bottlenecks = monitor.identify_bottlenecks()
    print(f"   ✓ Bottlenecks identified: {len(bottlenecks)}")

    for bottleneck in bottlenecks:
        print(f"     - {bottleneck['type']}: {bottleneck.get('stage', bottleneck.get('component', 'unknown'))}")

    # Test report generation
    print("\n6. Testing report generation:")
    report = monitor.generate_performance_report()
    print(f"   ✓ Report generated ({len(report)} characters)")

    # Test export
    print("\n7. Testing metrics export:")
    export_success = monitor.export_metrics("/tmp/claude/test_metrics.json")
    print(f"   ✓ Export successful: {export_success}")

    if export_success and Path("/tmp/claude/test_metrics.json").exists():
        file_size = Path("/tmp/claude/test_metrics.json").stat().st_size
        print(f"   ✓ Export file size: {file_size} bytes")

    print("\n✓ Pipeline monitor tests completed successfully!")

    return monitor


if __name__ == "__main__":
    test_pipeline_monitor()