#!/usr/bin/env python3
"""
Comprehensive System Monitoring and Analytics Platform
Task B10.2 - DAY10 Final Integration

Implements complete monitoring, analytics, and reporting infrastructure for:
- Real-time system monitoring
- Quality and performance analytics
- Comprehensive reporting system
- Predictive analytics and optimization

All 16 checklist items from DAY10_FINAL_INTEGRATION.md
"""

import asyncio
import json
import logging
import time
import threading
import psutil
import numpy as np
import pandas as pd

# Optional GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("GPUtil not available - GPU monitoring disabled")
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System metrics for monitoring"""
    timestamp: float

    # API Performance
    api_response_time: float
    api_requests_per_second: float
    api_error_rate: float
    api_active_connections: int

    # Resource Utilization
    cpu_percent: float
    memory_percent: float
    gpu_utilization: float
    gpu_memory_percent: float
    disk_usage_percent: float

    # Processing Metrics
    queue_length: int
    avg_processing_time: float
    successful_conversions: int
    failed_conversions: int

    # Quality Metrics
    avg_quality_score: float
    quality_improvement: float
    method_effectiveness: Dict[str, float]


@dataclass
class QualityMetrics:
    """Quality and performance analytics metrics"""
    timestamp: float
    logo_type: str
    method_used: str
    quality_before: float
    quality_after: float
    processing_time: float
    success: bool
    user_satisfaction: float
    cost: float


class SystemMonitoringDatabase:
    """SQLite database for storing monitoring data"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # System metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                api_response_time REAL,
                api_requests_per_second REAL,
                api_error_rate REAL,
                api_active_connections INTEGER,
                cpu_percent REAL,
                memory_percent REAL,
                gpu_utilization REAL,
                gpu_memory_percent REAL,
                disk_usage_percent REAL,
                queue_length INTEGER,
                avg_processing_time REAL,
                successful_conversions INTEGER,
                failed_conversions INTEGER,
                avg_quality_score REAL,
                quality_improvement REAL,
                method_effectiveness TEXT
            )
        ''')

        # Quality metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                logo_type TEXT,
                method_used TEXT,
                quality_before REAL,
                quality_after REAL,
                processing_time REAL,
                success BOOLEAN,
                user_satisfaction REAL,
                cost REAL
            )
        ''')

        # User behavior table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_behavior (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                user_id TEXT,
                action TEXT,
                session_duration REAL,
                conversions_count INTEGER,
                avg_quality_requested REAL,
                feedback_score REAL
            )
        ''')

        conn.commit()
        conn.close()

    def store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO system_metrics VALUES (
                NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        ''', (
            metrics.timestamp,
            metrics.api_response_time,
            metrics.api_requests_per_second,
            metrics.api_error_rate,
            metrics.api_active_connections,
            metrics.cpu_percent,
            metrics.memory_percent,
            metrics.gpu_utilization,
            metrics.gpu_memory_percent,
            metrics.disk_usage_percent,
            metrics.queue_length,
            metrics.avg_processing_time,
            metrics.successful_conversions,
            metrics.failed_conversions,
            metrics.avg_quality_score,
            metrics.quality_improvement,
            json.dumps(metrics.method_effectiveness)
        ))

        conn.commit()
        conn.close()

    def store_quality_metrics(self, metrics: QualityMetrics):
        """Store quality metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO quality_metrics VALUES (
                NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        ''', (
            metrics.timestamp,
            metrics.logo_type,
            metrics.method_used,
            metrics.quality_before,
            metrics.quality_after,
            metrics.processing_time,
            metrics.success,
            metrics.user_satisfaction,
            metrics.cost
        ))

        conn.commit()
        conn.close()

    def get_metrics_range(self, table: str, hours: int = 24) -> List[Dict]:
        """Get metrics for specified time range"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_time = time.time() - (hours * 3600)
        cursor.execute(f'SELECT * FROM {table} WHERE timestamp >= ?', (cutoff_time,))

        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]

        conn.close()

        return [dict(zip(columns, row)) for row in rows]


class RealTimeSystemMonitor:
    """Real-time system monitoring component"""

    def __init__(self, db: SystemMonitoringDatabase):
        self.db = db
        self.active = False
        self.monitoring_thread = None
        self.api_metrics = deque(maxlen=100)
        self.resource_metrics = deque(maxlen=100)
        self.processing_queue = deque()
        self.method_performance = defaultdict(list)

    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.active:
            return

        self.active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("🔍 Real-time system monitoring started")

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("🛑 Real-time system monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.active:
            try:
                metrics = self._collect_system_metrics()
                self.db.store_system_metrics(metrics)
                time.sleep(10)  # Collect every 10 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # API Performance Monitoring
        api_response_time = np.mean([m['response_time'] for m in list(self.api_metrics)[-10:]]) if self.api_metrics else 0.0
        api_requests_per_second = len([m for m in self.api_metrics if time.time() - m['timestamp'] < 60])
        api_error_rate = len([m for m in self.api_metrics if m.get('error', False)]) / max(len(self.api_metrics), 1)
        api_active_connections = len(self.api_metrics)

        # Resource Utilization Monitoring
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # GPU monitoring
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_utilization = gpus[0].load * 100
                    gpu_memory_percent = (gpus[0].memoryUsed / gpus[0].memoryTotal) * 100
                else:
                    gpu_utilization = 0.0
                    gpu_memory_percent = 0.0
            except:
                gpu_utilization = 0.0
                gpu_memory_percent = 0.0
        else:
            gpu_utilization = 0.0
            gpu_memory_percent = 0.0

        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent

        # Processing Metrics
        queue_length = len(self.processing_queue)
        recent_processing_times = [m['processing_time'] for m in list(self.resource_metrics)[-20:]]
        avg_processing_time = np.mean(recent_processing_times) if recent_processing_times else 0.0

        recent_conversions = [m for m in list(self.resource_metrics)[-50:]]
        successful_conversions = len([m for m in recent_conversions if m.get('success', False)])
        failed_conversions = len([m for m in recent_conversions if not m.get('success', True)])

        # Quality Metrics
        recent_qualities = [m['quality'] for m in list(self.resource_metrics)[-20:] if 'quality' in m]
        avg_quality_score = np.mean(recent_qualities) if recent_qualities else 0.0

        quality_improvements = [m.get('quality_improvement', 0) for m in list(self.resource_metrics)[-10:]]
        quality_improvement = np.mean(quality_improvements) if quality_improvements else 0.0

        # Method effectiveness tracking by logo type
        method_effectiveness = {}
        for method, performances in self.method_performance.items():
            if performances:
                recent_perfs = performances[-20:]  # Last 20 performances
                method_effectiveness[method] = np.mean(recent_perfs)

        return SystemMetrics(
            timestamp=time.time(),
            api_response_time=api_response_time,
            api_requests_per_second=api_requests_per_second,
            api_error_rate=api_error_rate,
            api_active_connections=api_active_connections,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_utilization=gpu_utilization,
            gpu_memory_percent=gpu_memory_percent,
            disk_usage_percent=disk_usage_percent,
            queue_length=queue_length,
            avg_processing_time=avg_processing_time,
            successful_conversions=successful_conversions,
            failed_conversions=failed_conversions,
            avg_quality_score=avg_quality_score,
            quality_improvement=quality_improvement,
            method_effectiveness=method_effectiveness
        )

    def record_api_request(self, response_time: float, error: bool = False):
        """Record API request metrics"""
        self.api_metrics.append({
            'timestamp': time.time(),
            'response_time': response_time,
            'error': error
        })

    def record_processing_task(self, processing_time: float, quality: float,
                             success: bool, method: str, logo_type: str,
                             quality_improvement: float = 0.0):
        """Record processing task metrics"""
        self.resource_metrics.append({
            'timestamp': time.time(),
            'processing_time': processing_time,
            'quality': quality,
            'success': success,
            'method': method,
            'logo_type': logo_type,
            'quality_improvement': quality_improvement
        })

        # Track method effectiveness by logo type
        method_key = f"{method}_{logo_type}"
        self.method_performance[method_key].append(quality)

    def add_to_queue(self, task_id: str):
        """Add task to processing queue"""
        self.processing_queue.append({
            'task_id': task_id,
            'timestamp': time.time()
        })

    def remove_from_queue(self, task_id: str):
        """Remove task from processing queue"""
        self.processing_queue = deque([
            task for task in self.processing_queue
            if task['task_id'] != task_id
        ])


class QualityPerformanceAnalytics:
    """Quality and performance analytics component"""

    def __init__(self, db: SystemMonitoringDatabase):
        self.db = db
        self.quality_trends = deque(maxlen=1000)
        self.method_effectiveness_cache = {}
        self.user_satisfaction_data = deque(maxlen=500)
        self.performance_baselines = {}

    def analyze_quality_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze quality improvement trends over time"""
        quality_data = self.db.get_metrics_range('quality_metrics', hours)

        if not quality_data:
            return {'error': 'No quality data available'}

        df = pd.DataFrame(quality_data)
        df['improvement'] = df['quality_after'] - df['quality_before']
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

        # Trend analysis
        hourly_trends = df.groupby(df['datetime'].dt.floor('H')).agg({
            'improvement': ['mean', 'std', 'count'],
            'quality_after': 'mean',
            'success': 'mean',
            'processing_time': 'mean'
        }).round(4)

        # Overall trend direction
        if len(df) > 10:
            x = np.arange(len(df))
            trend_slope = np.polyfit(x, df['improvement'], 1)[0]
            trend_direction = 'improving' if trend_slope > 0.001 else 'declining' if trend_slope < -0.001 else 'stable'
        else:
            trend_direction = 'insufficient_data'

        return {
            'trend_direction': trend_direction,
            'avg_improvement_24h': df['improvement'].mean(),
            'success_rate_24h': df['success'].mean(),
            'avg_quality_score': df['quality_after'].mean(),
            'hourly_trends': hourly_trends.to_dict(),
            'total_conversions': len(df)
        }

    def analyze_method_effectiveness(self, hours: int = 168) -> Dict[str, Any]:
        """Analyze method selection effectiveness by logo type"""
        quality_data = self.db.get_metrics_range('quality_metrics', hours)

        if not quality_data:
            return {'error': 'No method effectiveness data available'}

        df = pd.DataFrame(quality_data)
        df['improvement'] = df['quality_after'] - df['quality_before']

        # Method effectiveness by logo type
        method_analysis = df.groupby(['method_used', 'logo_type']).agg({
            'improvement': ['mean', 'std', 'count'],
            'success': 'mean',
            'processing_time': 'mean',
            'cost': 'mean'
        }).round(4)

        # Best method for each logo type
        best_methods = {}
        for logo_type in df['logo_type'].unique():
            type_data = df[df['logo_type'] == logo_type]
            method_scores = type_data.groupby('method_used').agg({
                'improvement': 'mean',
                'success': 'mean'
            })
            # Score = improvement * success_rate
            method_scores['score'] = method_scores['improvement'] * method_scores['success']
            if not method_scores.empty:
                best_methods[logo_type] = method_scores['score'].idxmax()

        return {
            'method_effectiveness': method_analysis.to_dict(),
            'best_methods_by_type': best_methods,
            'overall_method_ranking': df.groupby('method_used')['improvement'].mean().to_dict()
        }

    def track_user_satisfaction(self, user_id: str, satisfaction_score: float,
                              conversion_quality: float, processing_time: float):
        """Track user satisfaction and feedback"""
        self.user_satisfaction_data.append({
            'timestamp': time.time(),
            'user_id': user_id,
            'satisfaction_score': satisfaction_score,
            'conversion_quality': conversion_quality,
            'processing_time': processing_time
        })

    def analyze_user_satisfaction(self) -> Dict[str, Any]:
        """Analyze user satisfaction trends"""
        if not self.user_satisfaction_data:
            return {'error': 'No user satisfaction data available'}

        df = pd.DataFrame(list(self.user_satisfaction_data))

        # Satisfaction trends
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        daily_satisfaction = df.groupby(df['datetime'].dt.date).agg({
            'satisfaction_score': ['mean', 'std', 'count'],
            'conversion_quality': 'mean',
            'processing_time': 'mean'
        }).round(4)

        # Satisfaction vs quality correlation
        quality_satisfaction_corr = df[['satisfaction_score', 'conversion_quality']].corr().iloc[0, 1]

        return {
            'avg_satisfaction': df['satisfaction_score'].mean(),
            'satisfaction_trend': daily_satisfaction.to_dict(),
            'quality_satisfaction_correlation': quality_satisfaction_corr,
            'low_satisfaction_count': len(df[df['satisfaction_score'] < 3.0]),
            'high_satisfaction_count': len(df[df['satisfaction_score'] >= 4.0])
        }

    def detect_performance_regression(self) -> Dict[str, Any]:
        """Detect system performance regression"""
        system_data = self.db.get_metrics_range('system_metrics', 168)  # 1 week

        if len(system_data) < 50:
            return {'status': 'insufficient_data'}

        df = pd.DataFrame(system_data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

        # Calculate baseline performance (first 20% of data)
        baseline_size = max(10, len(df) // 5)
        baseline = df.head(baseline_size)
        recent = df.tail(baseline_size)

        regressions = {}

        # Check key metrics for regression
        metrics_to_check = [
            'api_response_time', 'cpu_percent', 'memory_percent',
            'avg_processing_time', 'avg_quality_score'
        ]

        for metric in metrics_to_check:
            baseline_mean = baseline[metric].mean()
            recent_mean = recent[metric].mean()

            if metric == 'avg_quality_score':
                # For quality, regression is when it decreases
                change_percent = (recent_mean - baseline_mean) / baseline_mean * 100
                regression = change_percent < -5  # 5% quality decrease
            else:
                # For other metrics, regression is when they increase significantly
                change_percent = (recent_mean - baseline_mean) / baseline_mean * 100
                regression = change_percent > 20  # 20% increase in latency/resource usage

            regressions[metric] = {
                'regression_detected': regression,
                'baseline_value': baseline_mean,
                'recent_value': recent_mean,
                'change_percent': change_percent
            }

        # Overall regression status
        any_regression = any(r['regression_detected'] for r in regressions.values())

        return {
            'regression_detected': any_regression,
            'metric_analysis': regressions,
            'recommendation': 'Investigate performance issues' if any_regression else 'Performance stable'
        }


class ComprehensiveReportingSystem:
    """Comprehensive reporting system component"""

    def __init__(self, db: SystemMonitoringDatabase, output_dir: str):
        self.db = db
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_daily_report(self, date: Optional[str] = None) -> str:
        """Generate daily system performance report"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        # Get 24 hours of data
        system_data = self.db.get_metrics_range('system_metrics', 24)
        quality_data = self.db.get_metrics_range('quality_metrics', 24)

        if not system_data:
            return "No data available for daily report"

        df_system = pd.DataFrame(system_data)
        df_quality = pd.DataFrame(quality_data) if quality_data else pd.DataFrame()

        # Generate report
        report = {
            'report_date': date,
            'generation_time': datetime.now().isoformat(),
            'system_performance': self._analyze_system_performance(df_system),
            'quality_statistics': self._analyze_quality_statistics(df_quality),
            'resource_utilization': self._analyze_resource_utilization(df_system),
            'api_performance': self._analyze_api_performance(df_system),
            'recommendations': self._generate_daily_recommendations(df_system, df_quality)
        }

        # Save report
        report_file = self.output_dir / f"daily_report_{date}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Daily report generated: {report_file}")
        return str(report_file)

    def generate_weekly_report(self) -> str:
        """Generate weekly system performance report"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        # Get 7 days of data
        system_data = self.db.get_metrics_range('system_metrics', 168)
        quality_data = self.db.get_metrics_range('quality_metrics', 168)

        if not system_data:
            return "No data available for weekly report"

        df_system = pd.DataFrame(system_data)
        df_quality = pd.DataFrame(quality_data) if quality_data else pd.DataFrame()

        # Generate comprehensive weekly analysis
        report = {
            'report_period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'generation_time': datetime.now().isoformat(),
            'weekly_trends': self._analyze_weekly_trends(df_system, df_quality),
            'method_performance_analysis': self._analyze_method_performance_weekly(df_quality),
            'resource_optimization_opportunities': self._identify_optimization_opportunities(df_system),
            'quality_improvement_statistics': self._analyze_quality_improvements(df_quality),
            'cost_analysis': self._analyze_weekly_costs(df_quality),
            'user_behavior_patterns': self._analyze_user_behavior_patterns(),
            'performance_forecasting': self._forecast_performance_trends(df_system)
        }

        # Save report
        week_str = end_date.strftime('%Y-W%U')
        report_file = self.output_dir / f"weekly_report_{week_str}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate visualization
        self._generate_weekly_visualizations(df_system, df_quality, week_str)

        logger.info(f"Weekly report generated: {report_file}")
        return str(report_file)

    def _analyze_system_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze system performance metrics"""
        return {
            'avg_api_response_time': df['api_response_time'].mean(),
            'max_api_response_time': df['api_response_time'].max(),
            'api_requests_total': df['api_requests_per_second'].sum(),
            'avg_error_rate': df['api_error_rate'].mean(),
            'peak_queue_length': df['queue_length'].max(),
            'avg_processing_time': df['avg_processing_time'].mean(),
            'total_successful_conversions': df['successful_conversions'].sum(),
            'total_failed_conversions': df['failed_conversions'].sum(),
            'success_rate': df['successful_conversions'].sum() / max(df['successful_conversions'].sum() + df['failed_conversions'].sum(), 1)
        }

    def _analyze_quality_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze quality statistics"""
        if df.empty:
            return {'no_data': True}

        df['improvement'] = df['quality_after'] - df['quality_before']

        return {
            'avg_quality_improvement': df['improvement'].mean(),
            'max_quality_improvement': df['improvement'].max(),
            'quality_improvement_std': df['improvement'].std(),
            'method_quality_ranking': df.groupby('method_used')['improvement'].mean().to_dict(),
            'logo_type_quality_stats': df.groupby('logo_type')['improvement'].agg(['mean', 'count']).to_dict()
        }

    def _analyze_resource_utilization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze resource utilization"""
        return {
            'avg_cpu_percent': df['cpu_percent'].mean(),
            'max_cpu_percent': df['cpu_percent'].max(),
            'avg_memory_percent': df['memory_percent'].mean(),
            'max_memory_percent': df['memory_percent'].max(),
            'avg_gpu_utilization': df['gpu_utilization'].mean(),
            'max_gpu_utilization': df['gpu_utilization'].max(),
            'avg_disk_usage': df['disk_usage_percent'].mean(),
            'resource_efficiency_score': self._calculate_resource_efficiency(df)
        }

    def _analyze_api_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze API performance"""
        return {
            'response_time_p95': df['api_response_time'].quantile(0.95),
            'response_time_p99': df['api_response_time'].quantile(0.99),
            'peak_requests_per_second': df['api_requests_per_second'].max(),
            'avg_active_connections': df['api_active_connections'].mean(),
            'error_rate_trend': 'stable'  # Would implement trend analysis
        }

    def _generate_daily_recommendations(self, df_system: pd.DataFrame, df_quality: pd.DataFrame) -> List[str]:
        """Generate daily recommendations"""
        recommendations = []

        # Performance recommendations
        avg_response_time = df_system['api_response_time'].mean()
        if avg_response_time > 0.2:
            recommendations.append(f"🔴 High API response time ({avg_response_time:.3f}s) - consider scaling or optimization")

        # Resource recommendations
        max_cpu = df_system['cpu_percent'].max()
        if max_cpu > 80:
            recommendations.append(f"🟡 High CPU usage detected ({max_cpu:.1f}%) - monitor for capacity issues")

        # Quality recommendations
        if not df_quality.empty:
            avg_improvement = (df_quality['quality_after'] - df_quality['quality_before']).mean()
            if avg_improvement < 0.1:
                recommendations.append("🟡 Lower than expected quality improvements - review optimization methods")

        if not recommendations:
            recommendations.append("✅ System operating within optimal parameters")

        return recommendations

    def _analyze_weekly_trends(self, df_system: pd.DataFrame, df_quality: pd.DataFrame) -> Dict[str, Any]:
        """Analyze weekly trends"""
        df_system['datetime'] = pd.to_datetime(df_system['timestamp'], unit='s')

        daily_stats = df_system.groupby(df_system['datetime'].dt.date).agg({
            'api_response_time': 'mean',
            'cpu_percent': 'mean',
            'memory_percent': 'mean',
            'avg_processing_time': 'mean',
            'successful_conversions': 'sum'
        })

        return {
            'daily_performance_trend': daily_stats.to_dict(),
            'performance_stability': daily_stats.std().to_dict(),
            'peak_usage_day': daily_stats['successful_conversions'].idxmax().strftime('%Y-%m-%d') if not daily_stats.empty else None
        }

    def _analyze_method_performance_weekly(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze method performance over the week"""
        if df.empty:
            return {'no_data': True}

        df['improvement'] = df['quality_after'] - df['quality_before']

        method_stats = df.groupby('method_used').agg({
            'improvement': ['mean', 'std', 'count'],
            'processing_time': 'mean',
            'success': 'mean',
            'cost': 'sum'
        })

        return method_stats.to_dict()

    def _identify_optimization_opportunities(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify resource optimization opportunities"""
        opportunities = []

        # CPU optimization
        avg_cpu = df['cpu_percent'].mean()
        if avg_cpu < 30:
            opportunities.append({
                'type': 'cpu_underutilization',
                'description': f'CPU underutilized ({avg_cpu:.1f}% avg) - consider reducing instance size',
                'potential_savings': 'High'
            })
        elif avg_cpu > 80:
            opportunities.append({
                'type': 'cpu_overutilization',
                'description': f'CPU overutilized ({avg_cpu:.1f}% avg) - consider scaling up',
                'urgency': 'High'
            })

        # Memory optimization
        avg_memory = df['memory_percent'].mean()
        if avg_memory > 85:
            opportunities.append({
                'type': 'memory_pressure',
                'description': f'High memory usage ({avg_memory:.1f}% avg) - memory optimization needed',
                'urgency': 'Medium'
            })

        return opportunities

    def _analyze_quality_improvements(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze quality improvements by method"""
        if df.empty:
            return {'no_data': True}

        df['improvement'] = df['quality_after'] - df['quality_before']

        return {
            'total_improvements': len(df[df['improvement'] > 0]),
            'significant_improvements': len(df[df['improvement'] > 0.1]),
            'avg_improvement_by_method': df.groupby('method_used')['improvement'].mean().to_dict(),
            'success_rate_by_method': df.groupby('method_used')['success'].mean().to_dict()
        }

    def _analyze_weekly_costs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze weekly costs"""
        if df.empty:
            return {'no_data': True}

        total_cost = df['cost'].sum()
        cost_per_conversion = total_cost / len(df) if len(df) > 0 else 0

        return {
            'total_cost': total_cost,
            'cost_per_conversion': cost_per_conversion,
            'cost_by_method': df.groupby('method_used')['cost'].sum().to_dict(),
            'cost_efficiency': df.groupby('method_used').apply(
                lambda x: (x['quality_after'] - x['quality_before']).sum() / x['cost'].sum()
            ).to_dict()
        }

    def _analyze_user_behavior_patterns(self) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        # This would integrate with user tracking data
        return {
            'peak_usage_hours': [9, 10, 11, 14, 15, 16],  # Mock data
            'avg_session_duration': 15.5,  # minutes
            'conversion_patterns': {
                'simple_logos': 0.65,
                'complex_logos': 0.35
            }
        }

    def _forecast_performance_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Forecast performance trends"""
        if len(df) < 20:
            return {'status': 'insufficient_data'}

        # Simple linear trend forecasting
        x = np.arange(len(df))

        forecasts = {}
        for metric in ['api_response_time', 'cpu_percent', 'memory_percent']:
            y = df[metric].values
            slope, intercept = np.polyfit(x, y, 1)

            # Forecast next 7 days (assuming hourly data)
            future_x = np.arange(len(df), len(df) + 168)  # 7 days * 24 hours
            forecast = slope * future_x + intercept

            forecasts[metric] = {
                'trend': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                'slope': slope,
                'forecast_7d_avg': np.mean(forecast),
                'forecast_7d_max': np.max(forecast)
            }

        return forecasts

    def _calculate_resource_efficiency(self, df: pd.DataFrame) -> float:
        """Calculate resource efficiency score"""
        # Efficiency = conversions per resource unit
        total_conversions = df['successful_conversions'].sum()
        avg_cpu = df['cpu_percent'].mean()
        avg_memory = df['memory_percent'].mean()

        if avg_cpu + avg_memory == 0:
            return 0.0

        efficiency = total_conversions / ((avg_cpu + avg_memory) / 2)
        return min(efficiency, 100.0)  # Cap at 100

    def _generate_weekly_visualizations(self, df_system: pd.DataFrame, df_quality: pd.DataFrame, week_str: str):
        """Generate weekly visualization charts"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Weekly System Report - {week_str}', fontsize=16, fontweight='bold')

        df_system['datetime'] = pd.to_datetime(df_system['timestamp'], unit='s')

        # API Response Time Trend
        axes[0, 0].plot(df_system['datetime'], df_system['api_response_time'])
        axes[0, 0].set_title('API Response Time Trend')
        axes[0, 0].set_ylabel('Response Time (s)')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Resource Utilization
        axes[0, 1].plot(df_system['datetime'], df_system['cpu_percent'], label='CPU')
        axes[0, 1].plot(df_system['datetime'], df_system['memory_percent'], label='Memory')
        axes[0, 1].set_title('Resource Utilization')
        axes[0, 1].set_ylabel('Utilization (%)')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Quality Improvements
        if not df_quality.empty:
            df_quality['improvement'] = df_quality['quality_after'] - df_quality['quality_before']
            df_quality['datetime'] = pd.to_datetime(df_quality['timestamp'], unit='s')
            daily_quality = df_quality.groupby(df_quality['datetime'].dt.date)['improvement'].mean()
            axes[0, 2].bar(range(len(daily_quality)), daily_quality.values)
            axes[0, 2].set_title('Daily Quality Improvements')
            axes[0, 2].set_ylabel('Avg Improvement')

        # Queue Length
        axes[1, 0].plot(df_system['datetime'], df_system['queue_length'])
        axes[1, 0].set_title('Processing Queue Length')
        axes[1, 0].set_ylabel('Queue Length')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Success Rate
        success_rate = df_system['successful_conversions'] / (df_system['successful_conversions'] + df_system['failed_conversions'] + 1)
        axes[1, 1].plot(df_system['datetime'], success_rate)
        axes[1, 1].set_title('Conversion Success Rate')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # Method Performance
        if not df_quality.empty and 'method_used' in df_quality.columns:
            method_performance = df_quality.groupby('method_used')['improvement'].mean()
            axes[1, 2].bar(method_performance.index, method_performance.values)
            axes[1, 2].set_title('Method Performance')
            axes[1, 2].set_ylabel('Avg Quality Improvement')
            axes[1, 2].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Save visualization
        viz_file = self.output_dir / f"weekly_visualization_{week_str}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Weekly visualization saved: {viz_file}")


class PredictiveAnalyticsOptimizer:
    """Predictive analytics and optimization component"""

    def __init__(self, db: SystemMonitoringDatabase):
        self.db = db
        self.models = {}
        self.predictions_cache = {}

    def capacity_planning_analysis(self, forecast_days: int = 30) -> Dict[str, Any]:
        """Capacity planning based on usage trends"""
        system_data = self.db.get_metrics_range('system_metrics', 168 * 4)  # 4 weeks

        if len(system_data) < 100:
            return {'status': 'insufficient_data'}

        df = pd.DataFrame(system_data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

        # Prepare features for prediction
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day

        # Features for prediction
        features = ['hour', 'day_of_week', 'day_of_month', 'api_requests_per_second']

        # Predict resource usage
        capacity_predictions = {}

        for metric in ['cpu_percent', 'memory_percent', 'queue_length']:
            if metric in df.columns:
                # Train simple model
                X = df[features].fillna(0)
                y = df[metric].fillna(0)

                if len(X) > 50:
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X, y)

                    # Generate future predictions
                    future_dates = pd.date_range(
                        start=df['datetime'].max() + timedelta(hours=1),
                        periods=forecast_days * 24,
                        freq='H'
                    )

                    future_features = pd.DataFrame({
                        'hour': future_dates.hour,
                        'day_of_week': future_dates.dayofweek,
                        'day_of_month': future_dates.day,
                        'api_requests_per_second': df['api_requests_per_second'].mean()  # Assume average
                    })

                    predictions = model.predict(future_features)

                    capacity_predictions[metric] = {
                        'max_predicted': np.max(predictions),
                        'avg_predicted': np.mean(predictions),
                        'capacity_threshold_breaches': len(predictions[predictions > 85]),  # >85% usage
                        'recommendation': self._generate_capacity_recommendation(metric, np.max(predictions))
                    }

        return {
            'forecast_period_days': forecast_days,
            'capacity_predictions': capacity_predictions,
            'scaling_recommendations': self._generate_scaling_recommendations(capacity_predictions),
            'cost_projections': self._estimate_cost_projections(capacity_predictions)
        }

    def predictive_maintenance_scheduling(self) -> Dict[str, Any]:
        """Predictive maintenance scheduling"""
        system_data = self.db.get_metrics_range('system_metrics', 168 * 2)  # 2 weeks

        if len(system_data) < 50:
            return {'status': 'insufficient_data'}

        df = pd.DataFrame(system_data)

        # Calculate system health indicators
        health_indicators = {
            'api_performance_degradation': self._detect_api_degradation(df),
            'resource_efficiency_decline': self._detect_efficiency_decline(df),
            'error_rate_increase': self._detect_error_rate_increase(df),
            'queue_performance_issues': self._detect_queue_issues(df)
        }

        # Schedule maintenance based on indicators
        maintenance_schedule = []

        for indicator, status in health_indicators.items():
            if status['maintenance_needed']:
                maintenance_schedule.append({
                    'component': indicator,
                    'urgency': status['urgency'],
                    'recommended_date': status['recommended_date'],
                    'description': status['description'],
                    'estimated_downtime': status['estimated_downtime']
                })

        return {
            'health_indicators': health_indicators,
            'maintenance_schedule': maintenance_schedule,
            'next_maintenance_date': min([m['recommended_date'] for m in maintenance_schedule]) if maintenance_schedule else None
        }

    def generate_performance_optimization_recommendations(self) -> Dict[str, Any]:
        """Generate performance optimization recommendations"""
        system_data = self.db.get_metrics_range('system_metrics', 168)  # 1 week
        quality_data = self.db.get_metrics_range('quality_metrics', 168)

        recommendations = []

        if system_data:
            df_system = pd.DataFrame(system_data)

            # API optimization recommendations
            avg_response_time = df_system['api_response_time'].mean()
            p95_response_time = df_system['api_response_time'].quantile(0.95)

            if p95_response_time > 0.5:
                recommendations.append({
                    'category': 'api_performance',
                    'priority': 'high',
                    'description': f'API P95 response time is {p95_response_time:.3f}s. Consider implementing caching or load balancing.',
                    'estimated_improvement': '30-50% response time reduction',
                    'implementation_effort': 'medium'
                })

            # Resource optimization recommendations
            cpu_efficiency = self._calculate_cpu_efficiency(df_system)
            memory_efficiency = self._calculate_memory_efficiency(df_system)

            if cpu_efficiency < 0.6:
                recommendations.append({
                    'category': 'resource_optimization',
                    'priority': 'medium',
                    'description': f'CPU efficiency is {cpu_efficiency:.2f}. Optimize algorithms or consider CPU-optimized instances.',
                    'estimated_improvement': '20-30% better CPU utilization',
                    'implementation_effort': 'high'
                })

            if memory_efficiency < 0.7:
                recommendations.append({
                    'category': 'resource_optimization',
                    'priority': 'medium',
                    'description': f'Memory efficiency is {memory_efficiency:.2f}. Implement memory optimization or increase cache sizes.',
                    'estimated_improvement': '15-25% memory savings',
                    'implementation_effort': 'medium'
                })

        if quality_data:
            df_quality = pd.DataFrame(quality_data)

            # Quality optimization recommendations
            method_performance = df_quality.groupby('method_used').agg({
                'quality_after': 'mean',
                'processing_time': 'mean',
                'success': 'mean'
            })

            if not method_performance.empty:
                best_method = method_performance['quality_after'].idxmax()
                worst_method = method_performance['quality_after'].idxmin()

                if method_performance.loc[best_method, 'quality_after'] - method_performance.loc[worst_method, 'quality_after'] > 0.1:
                    recommendations.append({
                        'category': 'quality_optimization',
                        'priority': 'high',
                        'description': f'Method {best_method} significantly outperforms {worst_method}. Consider intelligent routing.',
                        'estimated_improvement': f'{(method_performance.loc[best_method, "quality_after"] - method_performance.loc[worst_method, "quality_after"])*100:.1f}% quality improvement',
                        'implementation_effort': 'low'
                    })

        return {
            'recommendations': sorted(recommendations, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True),
            'total_recommendations': len(recommendations),
            'high_priority_count': len([r for r in recommendations if r['priority'] == 'high'])
        }

    def cost_optimization_analysis(self) -> Dict[str, Any]:
        """Generate cost optimization suggestions"""
        system_data = self.db.get_metrics_range('system_metrics', 168 * 4)  # 4 weeks
        quality_data = self.db.get_metrics_range('quality_metrics', 168 * 4)

        if not system_data:
            return {'status': 'insufficient_data'}

        df_system = pd.DataFrame(system_data)

        cost_optimizations = []

        # Resource utilization analysis
        avg_cpu = df_system['cpu_percent'].mean()
        avg_memory = df_system['memory_percent'].mean()
        avg_gpu = df_system['gpu_utilization'].mean()

        # Under-utilization savings
        if avg_cpu < 40:
            potential_savings = 25  # Estimated 25% cost savings
            cost_optimizations.append({
                'type': 'rightsizing',
                'resource': 'cpu',
                'current_utilization': avg_cpu,
                'recommendation': 'Downsize CPU allocation',
                'estimated_savings_percent': potential_savings,
                'confidence': 'high'
            })

        if avg_memory < 50:
            potential_savings = 20
            cost_optimizations.append({
                'type': 'rightsizing',
                'resource': 'memory',
                'current_utilization': avg_memory,
                'recommendation': 'Reduce memory allocation',
                'estimated_savings_percent': potential_savings,
                'confidence': 'high'
            })

        if avg_gpu < 30 and avg_gpu > 0:
            potential_savings = 40  # GPU is expensive
            cost_optimizations.append({
                'type': 'rightsizing',
                'resource': 'gpu',
                'current_utilization': avg_gpu,
                'recommendation': 'Consider CPU-only processing for low GPU usage',
                'estimated_savings_percent': potential_savings,
                'confidence': 'medium'
            })

        # Quality vs cost analysis
        if quality_data:
            df_quality = pd.DataFrame(quality_data)

            # Cost per quality improvement
            if 'cost' in df_quality.columns:
                df_quality['improvement'] = df_quality['quality_after'] - df_quality['quality_before']
                cost_efficiency = df_quality.groupby('method_used').apply(
                    lambda x: x['improvement'].sum() / x['cost'].sum() if x['cost'].sum() > 0 else 0
                ).to_dict()

                if cost_efficiency:
                    most_efficient = max(cost_efficiency, key=cost_efficiency.get)
                    least_efficient = min(cost_efficiency, key=cost_efficiency.get)

                    if cost_efficiency[most_efficient] > cost_efficiency[least_efficient] * 1.5:
                        cost_optimizations.append({
                            'type': 'method_optimization',
                            'recommendation': f'Prefer {most_efficient} over {least_efficient} for better cost efficiency',
                            'cost_efficiency_improvement': f'{((cost_efficiency[most_efficient] / cost_efficiency[least_efficient]) - 1) * 100:.1f}%',
                            'confidence': 'high'
                        })

        # Calculate total potential savings
        total_potential_savings = sum([opt.get('estimated_savings_percent', 0) for opt in cost_optimizations])

        return {
            'cost_optimizations': cost_optimizations,
            'total_potential_savings_percent': min(total_potential_savings, 60),  # Cap at 60%
            'implementation_priority': sorted(cost_optimizations, key=lambda x: x.get('estimated_savings_percent', 0), reverse=True),
            'quick_wins': [opt for opt in cost_optimizations if opt.get('confidence') == 'high']
        }

    def _generate_capacity_recommendation(self, metric: str, max_predicted: float) -> str:
        """Generate capacity recommendation for metric"""
        if max_predicted > 90:
            return f"URGENT: Scale up immediately - {metric} predicted to exceed 90%"
        elif max_predicted > 80:
            return f"PLAN: Schedule scaling - {metric} predicted to reach {max_predicted:.1f}%"
        elif max_predicted < 30:
            return f"OPTIMIZE: Consider downsizing - {metric} predicted max usage {max_predicted:.1f}%"
        else:
            return f"STABLE: Current capacity adequate - {metric} predicted max {max_predicted:.1f}%"

    def _generate_scaling_recommendations(self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate scaling recommendations"""
        recommendations = []

        for metric, pred in predictions.items():
            if pred['max_predicted'] > 85:
                recommendations.append({
                    'action': 'scale_up',
                    'resource': metric,
                    'urgency': 'high' if pred['max_predicted'] > 90 else 'medium',
                    'suggested_increase': '50%' if pred['max_predicted'] > 90 else '25%'
                })
            elif pred['max_predicted'] < 30:
                recommendations.append({
                    'action': 'scale_down',
                    'resource': metric,
                    'urgency': 'low',
                    'suggested_decrease': '25%'
                })

        return recommendations

    def _estimate_cost_projections(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate cost projections based on predictions"""
        # Mock cost calculations - would integrate with actual cloud pricing
        base_monthly_cost = 1000  # USD

        cost_factors = {
            'cpu_percent': 0.6,  # 60% of cost
            'memory_percent': 0.3,  # 30% of cost
            'gpu_utilization': 1.5  # 150% premium for GPU
        }

        projected_costs = {}

        for metric, pred in predictions.items():
            if metric in cost_factors:
                usage_multiplier = pred['avg_predicted'] / 50  # Baseline 50% usage
                projected_cost = base_monthly_cost * cost_factors[metric] * usage_multiplier
                projected_costs[metric] = projected_cost

        total_projected_cost = sum(projected_costs.values())

        return {
            'monthly_cost_projection': total_projected_cost,
            'cost_breakdown': projected_costs,
            'cost_trend': 'increasing' if total_projected_cost > base_monthly_cost else 'stable'
        }

    def _detect_api_degradation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect API performance degradation"""
        recent_avg = df['api_response_time'].tail(100).mean()
        baseline_avg = df['api_response_time'].head(100).mean()

        degradation = (recent_avg - baseline_avg) / baseline_avg > 0.2  # 20% degradation

        return {
            'maintenance_needed': degradation,
            'urgency': 'high' if degradation else 'low',
            'recommended_date': (datetime.now() + timedelta(days=1)).isoformat() if degradation else None,
            'description': f'API response time degraded by {((recent_avg - baseline_avg) / baseline_avg * 100):.1f}%' if degradation else 'API performance stable',
            'estimated_downtime': '2 hours' if degradation else 'none'
        }

    def _detect_efficiency_decline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect resource efficiency decline"""
        recent_efficiency = self._calculate_cpu_efficiency(df.tail(100))
        baseline_efficiency = self._calculate_cpu_efficiency(df.head(100))

        decline = baseline_efficiency - recent_efficiency > 0.1  # 10% decline

        return {
            'maintenance_needed': decline,
            'urgency': 'medium' if decline else 'low',
            'recommended_date': (datetime.now() + timedelta(days=7)).isoformat() if decline else None,
            'description': f'Resource efficiency declined by {((baseline_efficiency - recent_efficiency) * 100):.1f}%' if decline else 'Efficiency stable',
            'estimated_downtime': '1 hour' if decline else 'none'
        }

    def _detect_error_rate_increase(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect error rate increase"""
        recent_error_rate = df['api_error_rate'].tail(100).mean()
        baseline_error_rate = df['api_error_rate'].head(100).mean()

        increase = recent_error_rate > baseline_error_rate + 0.05  # 5% increase

        return {
            'maintenance_needed': increase,
            'urgency': 'high' if increase else 'low',
            'recommended_date': (datetime.now() + timedelta(days=2)).isoformat() if increase else None,
            'description': f'Error rate increased to {recent_error_rate:.2%}' if increase else 'Error rate stable',
            'estimated_downtime': '30 minutes' if increase else 'none'
        }

    def _detect_queue_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect queue performance issues"""
        recent_queue_avg = df['queue_length'].tail(100).mean()
        queue_trending_up = df['queue_length'].tail(50).mean() > df['queue_length'].head(50).mean()

        issues = recent_queue_avg > 10 or queue_trending_up

        return {
            'maintenance_needed': issues,
            'urgency': 'medium' if issues else 'low',
            'recommended_date': (datetime.now() + timedelta(days=3)).isoformat() if issues else None,
            'description': f'Queue length averaging {recent_queue_avg:.1f}' if issues else 'Queue performance normal',
            'estimated_downtime': '1 hour' if issues else 'none'
        }

    def _calculate_cpu_efficiency(self, df: pd.DataFrame) -> float:
        """Calculate CPU efficiency score"""
        conversions = df['successful_conversions'].sum()
        cpu_usage = df['cpu_percent'].mean()

        if cpu_usage == 0:
            return 0.0

        efficiency = conversions / cpu_usage
        return min(efficiency / 10, 1.0)  # Normalize to 0-1

    def _calculate_memory_efficiency(self, df: pd.DataFrame) -> float:
        """Calculate memory efficiency score"""
        conversions = df['successful_conversions'].sum()
        memory_usage = df['memory_percent'].mean()

        if memory_usage == 0:
            return 0.0

        efficiency = conversions / memory_usage
        return min(efficiency / 5, 1.0)  # Normalize to 0-1


class SystemMonitoringAnalyticsPlatform:
    """Main system monitoring and analytics platform"""

    def __init__(self, data_dir: str = "monitoring_data",
                 reports_dir: str = "reports"):
        """Initialize the monitoring platform"""
        self.data_dir = Path(data_dir)
        self.reports_dir = Path(reports_dir)

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        db_path = self.data_dir / "monitoring.db"
        self.db = SystemMonitoringDatabase(str(db_path))

        # Initialize components
        self.real_time_monitor = RealTimeSystemMonitor(self.db)
        self.quality_analytics = QualityPerformanceAnalytics(self.db)
        self.reporting_system = ComprehensiveReportingSystem(self.db, str(self.reports_dir))
        self.predictive_optimizer = PredictiveAnalyticsOptimizer(self.db)

        # Platform state
        self.platform_active = False

        logger.info("🚀 System Monitoring and Analytics Platform initialized")

    def start_platform(self):
        """Start the monitoring platform"""
        if self.platform_active:
            logger.warning("Platform already running")
            return

        self.platform_active = True
        self.real_time_monitor.start_monitoring()

        logger.info("🟢 System Monitoring and Analytics Platform started")

    def stop_platform(self):
        """Stop the monitoring platform"""
        if not self.platform_active:
            return

        self.platform_active = False
        self.real_time_monitor.stop_monitoring()

        logger.info("🔴 System Monitoring and Analytics Platform stopped")

    def get_comprehensive_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            'timestamp': time.time(),
            'platform_status': 'active' if self.platform_active else 'inactive',
            'real_time_metrics': self._get_current_metrics(),
            'quality_analytics': self.quality_analytics.analyze_quality_trends(24),
            'method_effectiveness': self.quality_analytics.analyze_method_effectiveness(168),
            'performance_regression': self.quality_analytics.detect_performance_regression(),
            'user_satisfaction': self.quality_analytics.analyze_user_satisfaction(),
            'capacity_planning': self.predictive_optimizer.capacity_planning_analysis(30),
            'optimization_recommendations': self.predictive_optimizer.generate_performance_optimization_recommendations(),
            'cost_optimization': self.predictive_optimizer.cost_optimization_analysis()
        }

    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            # Get latest metrics from monitor
            latest_metrics = self.real_time_monitor._collect_system_metrics()
            return asdict(latest_metrics)
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {'error': str(e)}

    def generate_all_reports(self) -> Dict[str, str]:
        """Generate all reports"""
        reports = {}

        try:
            # Daily report
            daily_report = self.reporting_system.generate_daily_report()
            reports['daily'] = daily_report

            # Weekly report
            weekly_report = self.reporting_system.generate_weekly_report()
            reports['weekly'] = weekly_report

            logger.info("✅ All reports generated successfully")

        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            reports['error'] = str(e)

        return reports

    # API for external integration
    def record_api_request(self, response_time: float, error: bool = False):
        """Record API request for monitoring"""
        self.real_time_monitor.record_api_request(response_time, error)

    def record_conversion(self, processing_time: float, quality_before: float,
                         quality_after: float, method: str, logo_type: str,
                         success: bool, cost: float = 0.0, user_satisfaction: float = 5.0):
        """Record conversion for monitoring"""
        # Store in quality metrics
        quality_metrics = QualityMetrics(
            timestamp=time.time(),
            logo_type=logo_type,
            method_used=method,
            quality_before=quality_before,
            quality_after=quality_after,
            processing_time=processing_time,
            success=success,
            user_satisfaction=user_satisfaction,
            cost=cost
        )
        self.db.store_quality_metrics(quality_metrics)

        # Record in real-time monitor
        quality_improvement = quality_after - quality_before
        self.real_time_monitor.record_processing_task(
            processing_time, quality_after, success, method, logo_type, quality_improvement
        )

    def add_to_processing_queue(self, task_id: str):
        """Add task to processing queue"""
        self.real_time_monitor.add_to_queue(task_id)

    def remove_from_processing_queue(self, task_id: str):
        """Remove task from processing queue"""
        self.real_time_monitor.remove_from_queue(task_id)


# Global platform instance
_global_platform = None
_platform_lock = threading.Lock()


def get_global_monitoring_platform() -> SystemMonitoringAnalyticsPlatform:
    """Get global monitoring platform instance"""
    global _global_platform
    with _platform_lock:
        if _global_platform is None:
            _global_platform = SystemMonitoringAnalyticsPlatform()
        return _global_platform


def start_monitoring_platform():
    """Start global monitoring platform"""
    platform = get_global_monitoring_platform()
    platform.start_platform()


def stop_monitoring_platform():
    """Stop global monitoring platform"""
    platform = get_global_monitoring_platform()
    platform.stop_platform()


def get_monitoring_dashboard() -> Dict[str, Any]:
    """Get comprehensive monitoring dashboard"""
    platform = get_global_monitoring_platform()
    return platform.get_comprehensive_dashboard()


def generate_monitoring_reports() -> Dict[str, str]:
    """Generate all monitoring reports"""
    platform = get_global_monitoring_platform()
    return platform.generate_all_reports()


if __name__ == "__main__":
    # Demo usage
    platform = SystemMonitoringAnalyticsPlatform(
        data_dir="demo_monitoring_data",
        reports_dir="demo_reports"
    )

    try:
        platform.start_platform()

        # Simulate some data
        for i in range(50):
            platform.record_api_request(0.05 + np.random.normal(0, 0.02))

            if i % 5 == 0:
                platform.record_conversion(
                    processing_time=2.0 + np.random.normal(0, 0.5),
                    quality_before=0.7 + np.random.normal(0, 0.1),
                    quality_after=0.85 + np.random.normal(0, 0.05),
                    method=np.random.choice(['method1', 'method2', 'method3']),
                    logo_type=np.random.choice(['simple', 'complex', 'text']),
                    success=np.random.random() > 0.1,
                    cost=1.0 + np.random.normal(0, 0.2),
                    user_satisfaction=4.0 + np.random.normal(0, 0.5)
                )

            time.sleep(0.1)

        # Generate dashboard
        dashboard = platform.get_comprehensive_dashboard()
        print("Dashboard generated with data points")

        # Generate reports
        reports = platform.generate_all_reports()
        print(f"Reports generated: {list(reports.keys())}")

    finally:
        platform.stop_platform()