#!/usr/bin/env python3
"""
Capacity Planning and Resource Optimization System
Analyzes system usage patterns and provides recommendations for resource optimization
"""

import os
import sys
import json
import logging
import argparse
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ResourceUsage:
    """Resource usage data structure"""
    timestamp: datetime
    component: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    custom_metrics: Dict[str, float]

@dataclass
class CapacityRecommendation:
    """Capacity planning recommendation"""
    component: str
    current_resources: Dict[str, float]
    recommended_resources: Dict[str, float]
    reason: str
    priority: str  # low, medium, high, critical
    cost_impact: str
    implementation_timeline: str
    confidence_score: float

@dataclass
class ResourceForecast:
    """Resource usage forecast"""
    component: str
    metric: str
    current_usage: float
    predicted_usage_7d: float
    predicted_usage_30d: float
    predicted_usage_90d: float
    trend: str  # increasing, decreasing, stable
    seasonal_pattern: bool
    confidence_interval: Tuple[float, float]

class MetricsCollector:
    """Collects resource usage metrics from monitoring systems"""

    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url

    def collect_resource_metrics(self, time_range: str = "7d") -> List[ResourceUsage]:
        """Collect resource usage metrics over time"""
        metrics = []

        try:
            # Get time series data
            end_time = datetime.now()
            start_time = end_time - self._parse_time_range(time_range)

            # Collect CPU metrics
            cpu_data = self._query_prometheus_range(
                'avg(rate(node_cpu_seconds_total{mode!="idle"}[5m])) by (instance) * 100',
                start_time, end_time
            )

            # Collect memory metrics
            memory_data = self._query_prometheus_range(
                '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100',
                start_time, end_time
            )

            # Collect disk metrics
            disk_data = self._query_prometheus_range(
                '(1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100',
                start_time, end_time
            )

            # Collect network metrics
            network_data = self._query_prometheus_range(
                'rate(node_network_receive_bytes_total[5m]) + rate(node_network_transmit_bytes_total[5m])',
                start_time, end_time
            )

            # Collect application-specific metrics
            api_metrics = self._query_prometheus_range(
                'rate(http_requests_total{job="svg-ai-api"}[5m])',
                start_time, end_time
            )

            optimization_metrics = self._query_prometheus_range(
                'rate(optimization_requests_total[5m])',
                start_time, end_time
            )

            # Combine metrics
            metrics = self._combine_metrics(
                cpu_data, memory_data, disk_data, network_data,
                api_metrics, optimization_metrics
            )

        except Exception as e:
            logger.error(f"Error collecting resource metrics: {e}")

        return metrics

    def _parse_time_range(self, time_range: str) -> timedelta:
        """Parse time range string to timedelta"""
        if time_range.endswith('h'):
            return timedelta(hours=int(time_range[:-1]))
        elif time_range.endswith('d'):
            return timedelta(days=int(time_range[:-1]))
        elif time_range.endswith('w'):
            return timedelta(weeks=int(time_range[:-1]))
        else:
            return timedelta(hours=24)

    def _query_prometheus_range(self, query: str, start_time: datetime, end_time: datetime) -> Dict:
        """Query Prometheus for time series data"""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query_range",
                params={
                    'query': query,
                    'start': start_time.timestamp(),
                    'end': end_time.timestamp(),
                    'step': '300s'  # 5-minute intervals
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            if data['status'] == 'success':
                return data['data']['result']
            return []

        except Exception as e:
            logger.error(f"Prometheus query failed: {e}")
            return []

    def _combine_metrics(self, cpu_data, memory_data, disk_data, network_data,
                        api_metrics, optimization_metrics) -> List[ResourceUsage]:
        """Combine different metric types into ResourceUsage objects"""
        metrics = []

        # Process each data source and create ResourceUsage objects
        # This is simplified - in production, you'd need to align timestamps
        # and handle missing data points

        for cpu_series in cpu_data:
            instance = cpu_series.get('metric', {}).get('instance', 'unknown')
            values = cpu_series.get('values', [])

            for timestamp_str, cpu_value in values:
                timestamp = datetime.fromtimestamp(float(timestamp_str))

                # Find corresponding metrics for this timestamp/instance
                memory_value = self._find_metric_value(memory_data, timestamp, instance)
                disk_value = self._find_metric_value(disk_data, timestamp, instance)
                network_value = self._find_metric_value(network_data, timestamp, instance)

                # Application metrics
                api_rate = self._find_metric_value(api_metrics, timestamp)
                optimization_rate = self._find_metric_value(optimization_metrics, timestamp)

                metrics.append(ResourceUsage(
                    timestamp=timestamp,
                    component=f"node-{instance}",
                    cpu_usage=float(cpu_value),
                    memory_usage=memory_value,
                    disk_usage=disk_value,
                    network_io=network_value,
                    custom_metrics={
                        'api_request_rate': api_rate,
                        'optimization_rate': optimization_rate
                    }
                ))

        return metrics

    def _find_metric_value(self, data: List[Dict], timestamp: datetime,
                          instance: str = None) -> float:
        """Find metric value for specific timestamp and instance"""
        target_ts = timestamp.timestamp()

        for series in data:
            if instance and series.get('metric', {}).get('instance') != instance:
                continue

            values = series.get('values', [])
            for ts_str, value in values:
                if abs(float(ts_str) - target_ts) < 150:  # Within 2.5 minutes
                    return float(value)

        return 0.0

class CapacityAnalyzer:
    """Analyzes resource usage and generates capacity recommendations"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()

        # Resource thresholds for recommendations
        self.thresholds = {
            'cpu_usage': {'scale_up': 70, 'scale_down': 30, 'critical': 85},
            'memory_usage': {'scale_up': 75, 'scale_down': 40, 'critical': 90},
            'disk_usage': {'scale_up': 80, 'scale_down': 50, 'critical': 95},
            'api_request_rate': {'scale_up': 100, 'scale_down': 20},
            'optimization_rate': {'scale_up': 50, 'scale_down': 10}
        }

    def _init_database(self):
        """Initialize SQLite database for storing analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resource_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                component TEXT NOT NULL,
                cpu_usage REAL,
                memory_usage REAL,
                disk_usage REAL,
                network_io REAL,
                custom_metrics TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS capacity_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component TEXT NOT NULL,
                recommendation_date TEXT NOT NULL,
                current_resources TEXT NOT NULL,
                recommended_resources TEXT NOT NULL,
                reason TEXT,
                priority TEXT,
                cost_impact TEXT,
                implementation_timeline TEXT,
                confidence_score REAL,
                implemented BOOLEAN DEFAULT FALSE
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resource_forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component TEXT NOT NULL,
                metric TEXT NOT NULL,
                forecast_date TEXT NOT NULL,
                current_usage REAL,
                predicted_7d REAL,
                predicted_30d REAL,
                predicted_90d REAL,
                trend TEXT,
                seasonal_pattern BOOLEAN,
                confidence_interval TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def store_usage_data(self, usage_data: List[ResourceUsage]):
        """Store usage data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for usage in usage_data:
            cursor.execute('''
                INSERT INTO resource_usage
                (timestamp, component, cpu_usage, memory_usage, disk_usage, network_io, custom_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                usage.timestamp.isoformat(),
                usage.component,
                usage.cpu_usage,
                usage.memory_usage,
                usage.disk_usage,
                usage.network_io,
                json.dumps(usage.custom_metrics)
            ))

        conn.commit()
        conn.close()
        logger.info(f"Stored {len(usage_data)} usage data points")

    def analyze_capacity_needs(self, days_back: int = 7) -> List[CapacityRecommendation]:
        """Analyze capacity needs and generate recommendations"""
        recommendations = []

        conn = sqlite3.connect(self.db_path)

        # Get usage data for analysis
        df = pd.read_sql_query('''
            SELECT * FROM resource_usage
            WHERE timestamp > datetime('now', '-{} days')
            ORDER BY timestamp
        '''.format(days_back), conn)

        if df.empty:
            logger.warning("No usage data available for analysis")
            return recommendations

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Analyze each component
        for component in df['component'].unique():
            component_data = df[df['component'] == component]
            component_recommendations = self._analyze_component_capacity(component, component_data)
            recommendations.extend(component_recommendations)

        conn.close()
        return recommendations

    def _analyze_component_capacity(self, component: str, data: pd.DataFrame) -> List[CapacityRecommendation]:
        """Analyze capacity needs for a specific component"""
        recommendations = []

        # Calculate statistical metrics
        cpu_stats = self._calculate_usage_stats(data['cpu_usage'])
        memory_stats = self._calculate_usage_stats(data['memory_usage'])
        disk_stats = self._calculate_usage_stats(data['disk_usage'])

        # Current resource estimates (would come from actual deployment configs)
        current_resources = {
            'cpu_cores': 2,  # Default values - should be queried from Kubernetes
            'memory_gb': 4,
            'disk_gb': 20
        }

        # CPU recommendations
        cpu_rec = self._generate_cpu_recommendation(component, cpu_stats, current_resources)
        if cpu_rec:
            recommendations.append(cpu_rec)

        # Memory recommendations
        memory_rec = self._generate_memory_recommendation(component, memory_stats, current_resources)
        if memory_rec:
            recommendations.append(memory_rec)

        # Disk recommendations
        disk_rec = self._generate_disk_recommendation(component, disk_stats, current_resources)
        if disk_rec:
            recommendations.append(disk_rec)

        return recommendations

    def _calculate_usage_stats(self, usage_series: pd.Series) -> Dict[str, float]:
        """Calculate usage statistics"""
        return {
            'mean': usage_series.mean(),
            'median': usage_series.median(),
            'p95': usage_series.quantile(0.95),
            'p99': usage_series.quantile(0.99),
            'max': usage_series.max(),
            'std': usage_series.std(),
            'trend': self._calculate_trend(usage_series)
        }

    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend slope using linear regression"""
        if len(series) < 2:
            return 0.0

        x = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        model = LinearRegression().fit(x, y)
        return model.coef_[0]

    def _generate_cpu_recommendation(self, component: str, stats: Dict, current: Dict) -> Optional[CapacityRecommendation]:
        """Generate CPU-specific recommendations"""
        p95_usage = stats['p95']
        trend = stats['trend']
        current_cores = current['cpu_cores']

        # Scale up if consistently high usage
        if p95_usage > self.thresholds['cpu_usage']['scale_up']:
            recommended_cores = max(current_cores + 1, int(current_cores * 1.5))
            return CapacityRecommendation(
                component=component,
                current_resources={'cpu_cores': current_cores},
                recommended_resources={'cpu_cores': recommended_cores},
                reason=f"CPU usage P95: {p95_usage:.1f}% exceeds {self.thresholds['cpu_usage']['scale_up']}% threshold",
                priority='high' if p95_usage > self.thresholds['cpu_usage']['critical'] else 'medium',
                cost_impact='medium',
                implementation_timeline='1-2 days',
                confidence_score=0.8 if trend > 0 else 0.6
            )

        # Scale down if consistently low usage
        elif p95_usage < self.thresholds['cpu_usage']['scale_down'] and current_cores > 1:
            recommended_cores = max(1, current_cores - 1)
            return CapacityRecommendation(
                component=component,
                current_resources={'cpu_cores': current_cores},
                recommended_resources={'cpu_cores': recommended_cores},
                reason=f"CPU usage P95: {p95_usage:.1f}% below {self.thresholds['cpu_usage']['scale_down']}% threshold",
                priority='low',
                cost_impact='cost_savings',
                implementation_timeline='1 week',
                confidence_score=0.7
            )

        return None

    def _generate_memory_recommendation(self, component: str, stats: Dict, current: Dict) -> Optional[CapacityRecommendation]:
        """Generate memory-specific recommendations"""
        p95_usage = stats['p95']
        current_memory = current['memory_gb']

        # Scale up if high memory usage
        if p95_usage > self.thresholds['memory_usage']['scale_up']:
            recommended_memory = max(current_memory + 2, int(current_memory * 1.5))
            return CapacityRecommendation(
                component=component,
                current_resources={'memory_gb': current_memory},
                recommended_resources={'memory_gb': recommended_memory},
                reason=f"Memory usage P95: {p95_usage:.1f}% exceeds {self.thresholds['memory_usage']['scale_up']}% threshold",
                priority='high' if p95_usage > self.thresholds['memory_usage']['critical'] else 'medium',
                cost_impact='medium',
                implementation_timeline='1-2 days',
                confidence_score=0.9
            )

        # Scale down if consistently low usage
        elif p95_usage < self.thresholds['memory_usage']['scale_down'] and current_memory > 2:
            recommended_memory = max(2, current_memory - 1)
            return CapacityRecommendation(
                component=component,
                current_resources={'memory_gb': current_memory},
                recommended_resources={'memory_gb': recommended_memory},
                reason=f"Memory usage P95: {p95_usage:.1f}% below {self.thresholds['memory_usage']['scale_down']}% threshold",
                priority='low',
                cost_impact='cost_savings',
                implementation_timeline='1 week',
                confidence_score=0.8
            )

        return None

    def _generate_disk_recommendation(self, component: str, stats: Dict, current: Dict) -> Optional[CapacityRecommendation]:
        """Generate disk-specific recommendations"""
        p95_usage = stats['p95']
        trend = stats['trend']
        current_disk = current['disk_gb']

        # Scale up if high disk usage or growing trend
        if p95_usage > self.thresholds['disk_usage']['scale_up'] or (p95_usage > 60 and trend > 0.1):
            recommended_disk = max(current_disk + 10, int(current_disk * 1.5))
            return CapacityRecommendation(
                component=component,
                current_resources={'disk_gb': current_disk},
                recommended_resources={'disk_gb': recommended_disk},
                reason=f"Disk usage P95: {p95_usage:.1f}% with trend: {trend:.2f}",
                priority='high' if p95_usage > self.thresholds['disk_usage']['critical'] else 'medium',
                cost_impact='low',
                implementation_timeline='immediate',
                confidence_score=0.9
            )

        return None

    def generate_forecasts(self, days_back: int = 30) -> List[ResourceForecast]:
        """Generate resource usage forecasts"""
        forecasts = []

        conn = sqlite3.connect(self.db_path)

        # Get historical data
        df = pd.read_sql_query('''
            SELECT * FROM resource_usage
            WHERE timestamp > datetime('now', '-{} days')
            ORDER BY timestamp
        '''.format(days_back), conn)

        if df.empty:
            logger.warning("No usage data available for forecasting")
            return forecasts

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Generate forecasts for each component and metric
        for component in df['component'].unique():
            component_data = df[df['component'] == component]

            for metric in ['cpu_usage', 'memory_usage', 'disk_usage']:
                forecast = self._generate_metric_forecast(component, metric, component_data)
                if forecast:
                    forecasts.append(forecast)

        conn.close()
        return forecasts

    def _generate_metric_forecast(self, component: str, metric: str, data: pd.DataFrame) -> Optional[ResourceForecast]:
        """Generate forecast for specific metric"""
        if metric not in data.columns or data[metric].isnull().all():
            return None

        # Prepare time series data
        ts_data = data.set_index('timestamp')[metric].resample('1H').mean().dropna()

        if len(ts_data) < 24:  # Need at least 24 hours of data
            return None

        current_usage = ts_data.iloc[-1]

        # Simple linear regression forecast
        x = np.arange(len(ts_data)).reshape(-1, 1)
        y = ts_data.values

        model = LinearRegression().fit(x, y)

        # Predict future values
        future_7d = model.predict([[len(ts_data) + 24 * 7]])[0]
        future_30d = model.predict([[len(ts_data) + 24 * 30]])[0]
        future_90d = model.predict([[len(ts_data) + 24 * 90]])[0]

        # Determine trend
        slope = model.coef_[0]
        if slope > 0.1:
            trend = 'increasing'
        elif slope < -0.1:
            trend = 'decreasing'
        else:
            trend = 'stable'

        # Simple confidence interval (±20% of prediction)
        confidence_interval = (
            max(0, future_30d * 0.8),
            min(100, future_30d * 1.2)
        )

        # Detect seasonal patterns (simplified)
        seasonal_pattern = self._detect_seasonality(ts_data)

        return ResourceForecast(
            component=component,
            metric=metric,
            current_usage=current_usage,
            predicted_usage_7d=max(0, future_7d),
            predicted_usage_30d=max(0, future_30d),
            predicted_usage_90d=max(0, future_90d),
            trend=trend,
            seasonal_pattern=seasonal_pattern,
            confidence_interval=confidence_interval
        )

    def _detect_seasonality(self, ts_data: pd.Series) -> bool:
        """Detect if there's a seasonal pattern in the data"""
        if len(ts_data) < 48:  # Need at least 2 days
            return False

        # Simple approach: check if there's a daily pattern
        hourly_means = ts_data.groupby(ts_data.index.hour).mean()
        return hourly_means.std() > hourly_means.mean() * 0.2

    def store_recommendations(self, recommendations: List[CapacityRecommendation]):
        """Store capacity recommendations in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for rec in recommendations:
            cursor.execute('''
                INSERT INTO capacity_recommendations
                (component, recommendation_date, current_resources, recommended_resources,
                 reason, priority, cost_impact, implementation_timeline, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rec.component,
                datetime.now().isoformat(),
                json.dumps(rec.current_resources),
                json.dumps(rec.recommended_resources),
                rec.reason,
                rec.priority,
                rec.cost_impact,
                rec.implementation_timeline,
                rec.confidence_score
            ))

        conn.commit()
        conn.close()
        logger.info(f"Stored {len(recommendations)} recommendations")

    def store_forecasts(self, forecasts: List[ResourceForecast]):
        """Store resource forecasts in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for forecast in forecasts:
            cursor.execute('''
                INSERT INTO resource_forecasts
                (component, metric, forecast_date, current_usage, predicted_7d,
                 predicted_30d, predicted_90d, trend, seasonal_pattern, confidence_interval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                forecast.component,
                forecast.metric,
                datetime.now().isoformat(),
                forecast.current_usage,
                forecast.predicted_usage_7d,
                forecast.predicted_usage_30d,
                forecast.predicted_usage_90d,
                forecast.trend,
                forecast.seasonal_pattern,
                json.dumps(forecast.confidence_interval)
            ))

        conn.commit()
        conn.close()
        logger.info(f"Stored {len(forecasts)} forecasts")

class ReportGenerator:
    """Generates capacity planning reports"""

    def __init__(self, db_path: str, output_dir: str):
        self.db_path = db_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_capacity_report(self) -> str:
        """Generate comprehensive capacity planning report"""
        report_file = os.path.join(self.output_dir, f"capacity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")

        # Get data from database
        conn = sqlite3.connect(self.db_path)

        recommendations_df = pd.read_sql_query('''
            SELECT * FROM capacity_recommendations
            WHERE recommendation_date > datetime('now', '-7 days')
            ORDER BY priority DESC, confidence_score DESC
        ''', conn)

        forecasts_df = pd.read_sql_query('''
            SELECT * FROM resource_forecasts
            WHERE forecast_date > datetime('now', '-1 days')
            ORDER BY component, metric
        ''', conn)

        usage_df = pd.read_sql_query('''
            SELECT * FROM resource_usage
            WHERE timestamp > datetime('now', '-7 days')
            ORDER BY timestamp
        ''', conn)

        conn.close()

        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SVG-AI Capacity Planning Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .recommendation {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .high-priority {{ border-left: 5px solid #ff4444; }}
                .medium-priority {{ border-left: 5px solid #ffaa44; }}
                .low-priority {{ border-left: 5px solid #44ff44; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SVG-AI Capacity Planning Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Report Period: Last 7 days</p>
            </div>

            <div class="section">
                <h2>Executive Summary</h2>
                <p>Total Recommendations: {len(recommendations_df)}</p>
                <p>High Priority Items: {len(recommendations_df[recommendations_df['priority'] == 'high'])}</p>
                <p>Cost Optimization Opportunities: {len(recommendations_df[recommendations_df['cost_impact'] == 'cost_savings'])}</p>
            </div>

            <div class="section">
                <h2>Capacity Recommendations</h2>
        """

        # Add recommendations
        for _, rec in recommendations_df.iterrows():
            priority_class = f"{rec['priority']}-priority"
            html_content += f"""
                <div class="recommendation {priority_class}">
                    <h3>{rec['component']} - {rec['priority'].title()} Priority</h3>
                    <p><strong>Current Resources:</strong> {rec['current_resources']}</p>
                    <p><strong>Recommended Resources:</strong> {rec['recommended_resources']}</p>
                    <p><strong>Reason:</strong> {rec['reason']}</p>
                    <p><strong>Cost Impact:</strong> {rec['cost_impact']}</p>
                    <p><strong>Timeline:</strong> {rec['implementation_timeline']}</p>
                    <p><strong>Confidence:</strong> {rec['confidence_score']:.1%}</p>
                </div>
            """

        # Add forecasts table
        html_content += """
            </div>
            <div class="section">
                <h2>Resource Usage Forecasts</h2>
                <table>
                    <tr>
                        <th>Component</th>
                        <th>Metric</th>
                        <th>Current</th>
                        <th>7-Day Forecast</th>
                        <th>30-Day Forecast</th>
                        <th>Trend</th>
                    </tr>
        """

        for _, forecast in forecasts_df.iterrows():
            html_content += f"""
                <tr>
                    <td>{forecast['component']}</td>
                    <td>{forecast['metric']}</td>
                    <td>{forecast['current_usage']:.1f}%</td>
                    <td>{forecast['predicted_7d']:.1f}%</td>
                    <td>{forecast['predicted_30d']:.1f}%</td>
                    <td>{forecast['trend']}</td>
                </tr>
            """

        html_content += """
                </table>
            </div>
        </body>
        </html>
        """

        # Write report file
        with open(report_file, 'w') as f:
            f.write(html_content)

        logger.info(f"Capacity report generated: {report_file}")
        return report_file

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Capacity Planning and Resource Optimization")
    parser.add_argument('--prometheus-url', default='http://localhost:9090',
                        help='Prometheus server URL')
    parser.add_argument('--db-path', default='capacity_planning.db',
                        help='SQLite database path')
    parser.add_argument('--output-dir', default='capacity_reports',
                        help='Output directory for reports')
    parser.add_argument('--days-back', type=int, default=7,
                        help='Days of historical data to analyze')
    parser.add_argument('--mode', choices=['analyze', 'forecast', 'report'],
                        default='analyze', help='Operation mode')

    args = parser.parse_args()

    try:
        # Initialize components
        collector = MetricsCollector(args.prometheus_url)
        analyzer = CapacityAnalyzer(args.db_path)
        report_generator = ReportGenerator(args.db_path, args.output_dir)

        if args.mode == 'analyze':
            # Collect and analyze capacity
            logger.info("Collecting resource usage metrics...")
            usage_data = collector.collect_resource_metrics(f"{args.days_back}d")

            logger.info("Storing usage data...")
            analyzer.store_usage_data(usage_data)

            logger.info("Analyzing capacity needs...")
            recommendations = analyzer.analyze_capacity_needs(args.days_back)

            logger.info("Storing recommendations...")
            analyzer.store_recommendations(recommendations)

            # Print recommendations
            for rec in recommendations:
                print(f"\n{rec.component} ({rec.priority} priority):")
                print(f"  Current: {rec.current_resources}")
                print(f"  Recommended: {rec.recommended_resources}")
                print(f"  Reason: {rec.reason}")
                print(f"  Confidence: {rec.confidence_score:.1%}")

        elif args.mode == 'forecast':
            # Generate forecasts
            logger.info("Generating resource forecasts...")
            forecasts = analyzer.generate_forecasts(args.days_back)

            logger.info("Storing forecasts...")
            analyzer.store_forecasts(forecasts)

            # Print forecasts
            for forecast in forecasts:
                print(f"\n{forecast.component} - {forecast.metric}:")
                print(f"  Current: {forecast.current_usage:.1f}%")
                print(f"  30-day forecast: {forecast.predicted_usage_30d:.1f}%")
                print(f"  Trend: {forecast.trend}")

        elif args.mode == 'report':
            # Generate comprehensive report
            logger.info("Generating capacity planning report...")
            report_file = report_generator.generate_capacity_report()
            print(f"Report generated: {report_file}")

    except Exception as e:
        logger.error(f"Error in capacity planning: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()