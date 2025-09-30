"""Logging and analytics system for optimization results"""

import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
import hashlib
import gzip
import shutil
from threading import Lock


class OptimizationLogger:
    """Log and analyze optimization results"""

    def __init__(self, log_dir: str = "logs/optimization"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

        # File paths
        self.json_log = self.log_dir / f"optimization_{datetime.now():%Y%m%d}.jsonl"
        self.csv_log = self.log_dir / f"optimization_{datetime.now():%Y%m%d}.csv"
        self.summary_file = self.log_dir / "optimization_summary.json"

        # In-memory cache for analysis
        self.session_data = []
        self.statistics_cache = {}
        self.write_lock = Lock()

        # Initialize CSV if needed
        self._init_csv_log()

    def setup_logging(self):
        """Setup structured logging format"""
        handler = logging.FileHandler(self.log_dir / "optimization.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _init_csv_log(self):
        """Initialize CSV file with headers"""
        if not self.csv_log.exists():
            headers = [
                "timestamp", "image_path", "logo_type", "confidence",
                "ssim_before", "ssim_after", "ssim_improvement",
                "file_size_before", "file_size_after", "file_size_reduction",
                "conversion_time", "optimization_method"
            ]
            with open(self.csv_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_optimization(self,
                        image_path: str,
                        features: Dict[str, float],
                        params: Dict[str, Any],
                        quality_metrics: Dict[str, Any],
                        metadata: Optional[Dict] = None):
        """
        Log detailed optimization results.

        Args:
            image_path: Path to optimized image
            features: Extracted image features
            params: Optimized parameters
            quality_metrics: Quality measurement results
            metadata: Additional metadata
        """
        with self.write_lock:
            # Create comprehensive log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self._generate_session_id(),
                "image": {
                    "path": str(image_path),
                    "filename": Path(image_path).name,
                    "hash": self._hash_file(image_path) if Path(image_path).exists() else None
                },
                "features": features,
                "parameters": params,
                "quality": quality_metrics,
                "metadata": metadata or {}
            }

            # Add to session data
            self.session_data.append(log_entry)

            # Write to JSON lines file
            self._write_json_log(log_entry)

            # Write to CSV
            self._write_csv_log(log_entry)

            # Log summary
            self.logger.info(
                f"Optimization logged: {Path(image_path).name} - "
                f"SSIM improvement: {quality_metrics.get('improvements', {}).get('ssim_improvement', 0):.2f}%"
            )

    def _write_json_log(self, entry: Dict):
        """Write entry to JSON lines file"""
        try:
            with open(self.json_log, 'a') as f:
                f.write(json.dumps(entry, default=str) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write JSON log: {e}")

    def _write_csv_log(self, entry: Dict):
        """Write entry to CSV file"""
        try:
            # Extract relevant fields for CSV
            csv_row = [
                entry["timestamp"],
                entry["image"]["path"],
                entry.get("metadata", {}).get("logo_type", "unknown"),
                entry.get("metadata", {}).get("confidence", 0),
                entry.get("quality", {}).get("default_metrics", {}).get("ssim", 0),
                entry.get("quality", {}).get("optimized_metrics", {}).get("ssim", 0),
                entry.get("quality", {}).get("improvements", {}).get("ssim_improvement", 0),
                entry.get("quality", {}).get("default_metrics", {}).get("svg_size_bytes", 0),
                entry.get("quality", {}).get("optimized_metrics", {}).get("svg_size_bytes", 0),
                entry.get("quality", {}).get("improvements", {}).get("file_size_improvement", 0),
                entry.get("quality", {}).get("optimized_metrics", {}).get("conversion_time", 0),
                entry.get("metadata", {}).get("optimization_method", "Method1")
            ]

            with open(self.csv_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(csv_row)
        except Exception as e:
            self.logger.error(f"Failed to write CSV log: {e}")

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]

    def _hash_file(self, filepath: str) -> str:
        """Generate hash of file contents"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return "unknown"

    def export_to_csv(self, output_path: Optional[str] = None) -> str:
        """
        Export all logged data to CSV.

        Args:
            output_path: Optional output path

        Returns:
            Path to exported CSV file
        """
        if output_path is None:
            output_path = self.log_dir / f"export_{datetime.now():%Y%m%d_%H%M%S}.csv"
        else:
            output_path = Path(output_path)

        # Load all JSON logs
        all_data = self._load_all_logs()

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        # Flatten nested structures
        if not df.empty:
            # Flatten image info
            if 'image' in df.columns:
                df['image_path'] = df['image'].apply(lambda x: x.get('path') if isinstance(x, dict) else None)
                df['image_filename'] = df['image'].apply(lambda x: x.get('filename') if isinstance(x, dict) else None)

            # Flatten quality metrics
            if 'quality' in df.columns:
                df['ssim_improvement'] = df['quality'].apply(
                    lambda x: x.get('improvements', {}).get('ssim_improvement', 0) if isinstance(x, dict) else 0
                )
                df['file_size_improvement'] = df['quality'].apply(
                    lambda x: x.get('improvements', {}).get('file_size_improvement', 0) if isinstance(x, dict) else 0
                )

            # Select relevant columns
            export_columns = [
                'timestamp', 'image_path', 'image_filename',
                'ssim_improvement', 'file_size_improvement'
            ]
            export_columns = [col for col in export_columns if col in df.columns]
            df = df[export_columns]

        # Save to CSV
        df.to_csv(output_path, index=False)
        self.logger.info(f"Exported {len(df)} records to {output_path}")

        return str(output_path)

    def _load_all_logs(self) -> List[Dict]:
        """Load all JSON log entries"""
        all_data = []
        loaded_ids = set()

        # Load from JSON files first
        for json_file in self.log_dir.glob("optimization_*.jsonl"):
            try:
                with open(json_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            # Use timestamp + image path as unique identifier
                            entry_id = f"{entry.get('timestamp', '')}_{entry.get('image', {}).get('path', '')}"
                            if entry_id not in loaded_ids:
                                all_data.append(entry)
                                loaded_ids.add(entry_id)
            except Exception as e:
                self.logger.warning(f"Failed to load {json_file}: {e}")

        # Add session data that's not already loaded (avoid duplicates)
        for entry in self.session_data:
            entry_id = f"{entry.get('timestamp', '')}_{entry.get('image', {}).get('path', '')}"
            if entry_id not in loaded_ids:
                all_data.append(entry)
                loaded_ids.add(entry_id)

        return all_data

    def calculate_statistics(self,
                            logo_type: Optional[str] = None,
                            time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Calculate optimization statistics.

        Args:
            logo_type: Filter by logo type
            time_range: Filter by time range

        Returns:
            Dictionary of statistics
        """
        # Load all data
        data = self._load_all_logs()

        # Apply filters
        if logo_type:
            data = [d for d in data if d.get('metadata', {}).get('logo_type') == logo_type]

        if time_range:
            start, end = time_range
            data = [d for d in data
                   if start <= datetime.fromisoformat(d['timestamp']) <= end]

        if not data:
            return {"message": "No data found for the specified filters"}

        # Calculate statistics
        stats = {
            "total_optimizations": len(data),
            "time_range": {
                "start": min(d['timestamp'] for d in data),
                "end": max(d['timestamp'] for d in data)
            }
        }

        # Quality improvements
        ssim_improvements = []
        file_size_improvements = []
        speed_improvements = []

        for entry in data:
            quality = entry.get('quality', {})
            improvements = quality.get('improvements', {})

            if 'ssim_improvement' in improvements:
                ssim_improvements.append(improvements['ssim_improvement'])
            if 'file_size_improvement' in improvements:
                file_size_improvements.append(improvements['file_size_improvement'])
            if 'speed_improvement' in improvements:
                speed_improvements.append(improvements['speed_improvement'])

        # Calculate averages and ranges
        if ssim_improvements:
            stats['ssim_improvement'] = {
                'average': np.mean(ssim_improvements),
                'median': np.median(ssim_improvements),
                'std': np.std(ssim_improvements),
                'min': np.min(ssim_improvements),
                'max': np.max(ssim_improvements),
                'positive_rate': sum(1 for x in ssim_improvements if x > 0) / len(ssim_improvements)
            }

        if file_size_improvements:
            stats['file_size_improvement'] = {
                'average': np.mean(file_size_improvements),
                'median': np.median(file_size_improvements),
                'std': np.std(file_size_improvements),
                'min': np.min(file_size_improvements),
                'max': np.max(file_size_improvements)
            }

        if speed_improvements:
            stats['speed_improvement'] = {
                'average': np.mean(speed_improvements),
                'median': np.median(speed_improvements),
                'std': np.std(speed_improvements)
            }

        # Logo type distribution
        logo_types = defaultdict(int)
        for entry in data:
            lt = entry.get('metadata', {}).get('logo_type', 'unknown')
            logo_types[lt] += 1

        stats['logo_type_distribution'] = dict(logo_types)

        # Performance over time
        stats['performance_trend'] = self._calculate_trend(data)

        return stats

    def _calculate_trend(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate performance trend over time"""
        if not data:
            return {}

        # Sort by timestamp
        sorted_data = sorted(data, key=lambda x: x['timestamp'])

        # Group by day
        daily_stats = defaultdict(list)

        for entry in sorted_data:
            date = entry['timestamp'][:10]  # Extract date part
            quality = entry.get('quality', {})
            improvements = quality.get('improvements', {})

            if 'ssim_improvement' in improvements:
                daily_stats[date].append(improvements['ssim_improvement'])

        # Calculate daily averages
        trend = {}
        for date, values in daily_stats.items():
            if values:
                trend[date] = {
                    'average': np.mean(values),
                    'count': len(values)
                }

        return trend

    def identify_best_worst(self, metric: str = "ssim_improvement", n: int = 5) -> Dict[str, List]:
        """
        Identify best and worst performing optimizations.

        Args:
            metric: Metric to use for ranking
            n: Number of results to return

        Returns:
            Dict with 'best' and 'worst' lists
        """
        data = self._load_all_logs()

        # Extract metric values
        scored_data = []
        for entry in data:
            quality = entry.get('quality', {})
            improvements = quality.get('improvements', {})

            if metric in improvements:
                scored_data.append({
                    'image': entry['image']['filename'],
                    'timestamp': entry['timestamp'],
                    metric: improvements[metric],
                    'parameters': entry.get('parameters', {})
                })

        # Sort by metric
        scored_data.sort(key=lambda x: x[metric], reverse=True)

        return {
            'best': scored_data[:n],
            'worst': scored_data[-n:] if len(scored_data) > n else scored_data
        }

    def generate_correlation_analysis(self) -> Dict[str, Any]:
        """Generate correlation analysis between features and improvements"""
        data = self._load_all_logs()

        if not data:
            return {"error": "No data available for analysis"}

        # Extract features and improvements
        feature_names = set()
        records = []

        for entry in data:
            features = entry.get('features', {})
            improvements = entry.get('quality', {}).get('improvements', {})

            if features and improvements and 'ssim_improvement' in improvements:
                feature_names.update(features.keys())
                record = {**features, 'ssim_improvement': improvements['ssim_improvement']}
                records.append(record)

        if not records:
            return {"error": "Insufficient data for correlation analysis"}

        # Create DataFrame
        df = pd.DataFrame(records)

        # Calculate correlations
        correlations = {}
        for feature in feature_names:
            if feature in df.columns:
                corr = df[feature].corr(df['ssim_improvement'])
                if not np.isnan(corr):
                    correlations[feature] = corr

        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            'correlations': dict(sorted_corr),
            'most_influential': sorted_corr[0] if sorted_corr else None,
            'sample_size': len(records)
        }

    def create_dashboard_data(self) -> Dict[str, Any]:
        """
        Create data structure for visualization dashboard.

        Returns:
            Dashboard-ready data structure
        """
        stats = self.calculate_statistics()
        best_worst = self.identify_best_worst()
        correlations = self.generate_correlation_analysis()

        dashboard_data = {
            'summary': {
                'total_optimizations': stats.get('total_optimizations', 0),
                'average_ssim_improvement': stats.get('ssim_improvement', {}).get('average', 0),
                'average_file_size_reduction': stats.get('file_size_improvement', {}).get('average', 0)
            },
            'charts': {
                'performance_trend': stats.get('performance_trend', {}),
                'logo_type_distribution': stats.get('logo_type_distribution', {}),
                'correlations': correlations.get('correlations', {})
            },
            'tables': {
                'best_performers': best_worst.get('best', []),
                'worst_performers': best_worst.get('worst', [])
            },
            'metadata': {
                'last_updated': datetime.now().isoformat(),
                'data_points': stats.get('total_optimizations', 0)
            }
        }

        return dashboard_data

    def export_dashboard_html(self, output_path: Optional[str] = None) -> str:
        """
        Export dashboard as HTML file.

        Args:
            output_path: Optional output path

        Returns:
            Path to exported HTML file
        """
        if output_path is None:
            output_path = self.log_dir / f"dashboard_{datetime.now():%Y%m%d_%H%M%S}.html"
        else:
            output_path = Path(output_path)

        dashboard_data = self.create_dashboard_data()

        html_content = self._generate_dashboard_html(dashboard_data)

        with open(output_path, 'w') as f:
            f.write(html_content)

        self.logger.info(f"Dashboard exported to {output_path}")
        return str(output_path)

    def _generate_dashboard_html(self, data: Dict) -> str:
        """Generate HTML dashboard"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Optimization Analytics Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: 'Segoe UI', Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: #f5f5f5;
                }
                .header {
                    background: #2c3e50;
                    color: white;
                    padding: 20px;
                    margin: -20px -20px 20px -20px;
                }
                .summary-cards {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                .card {
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .metric {
                    font-size: 2em;
                    font-weight: bold;
                    color: #27ae60;
                }
                .label {
                    color: #7f8c8d;
                    margin-top: 5px;
                }
                .chart-container {
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    background: white;
                }
                th {
                    background: #34495e;
                    color: white;
                    padding: 10px;
                    text-align: left;
                }
                td {
                    padding: 10px;
                    border-bottom: 1px solid #ecf0f1;
                }
                tr:hover {
                    background: #f8f9fa;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Parameter Optimization Analytics</h1>
                <p>Last updated: """ + data['metadata']['last_updated'] + """</p>
            </div>

            <div class="summary-cards">
                <div class="card">
                    <div class="metric">""" + str(data['summary']['total_optimizations']) + """</div>
                    <div class="label">Total Optimizations</div>
                </div>
                <div class="card">
                    <div class="metric">""" + f"{data['summary']['average_ssim_improvement']:.1f}%" + """</div>
                    <div class="label">Avg SSIM Improvement</div>
                </div>
                <div class="card">
                    <div class="metric">""" + f"{data['summary']['average_file_size_reduction']:.1f}%" + """</div>
                    <div class="label">Avg File Size Reduction</div>
                </div>
            </div>

            <div class="chart-container">
                <h2>Performance Trend</h2>
                <div id="trend-chart"></div>
            </div>

            <div class="chart-container">
                <h2>Logo Type Distribution</h2>
                <div id="distribution-chart"></div>
            </div>

            <div class="chart-container">
                <h2>Feature Correlations</h2>
                <div id="correlation-chart"></div>
            </div>

            <div class="chart-container">
                <h2>Top Performers</h2>
                <table>
                    <tr>
                        <th>Image</th>
                        <th>SSIM Improvement</th>
                        <th>Timestamp</th>
                    </tr>
        """

        # Add best performers
        for item in data['tables']['best_performers'][:5]:
            html += f"""
                    <tr>
                        <td>{item.get('image', 'N/A')}</td>
                        <td>{item.get('ssim_improvement', 0):.2f}%</td>
                        <td>{item.get('timestamp', 'N/A')}</td>
                    </tr>
            """

        html += """
                </table>
            </div>

            <script>
                // Performance trend chart
                var trendData = """ + json.dumps(data['charts']['performance_trend']) + """;
                var dates = Object.keys(trendData);
                var values = dates.map(d => trendData[d].average);

                Plotly.newPlot('trend-chart', [{
                    x: dates,
                    y: values,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'SSIM Improvement'
                }], {
                    xaxis: { title: 'Date' },
                    yaxis: { title: 'Average Improvement (%)' }
                });

                // Logo type distribution
                var distData = """ + json.dumps(data['charts']['logo_type_distribution']) + """;
                Plotly.newPlot('distribution-chart', [{
                    labels: Object.keys(distData),
                    values: Object.values(distData),
                    type: 'pie'
                }]);

                // Correlation chart
                var corrData = """ + json.dumps(data['charts']['correlations']) + """;
                Plotly.newPlot('correlation-chart', [{
                    x: Object.keys(corrData),
                    y: Object.values(corrData),
                    type: 'bar'
                }], {
                    xaxis: { title: 'Feature' },
                    yaxis: { title: 'Correlation with SSIM Improvement' }
                });
            </script>
        </body>
        </html>
        """

        return html

    def rotate_logs(self, max_size_mb: int = 100, archive: bool = True):
        """
        Rotate log files when they get too large.

        Args:
            max_size_mb: Maximum size in MB before rotation
            archive: Whether to archive old logs
        """
        max_size_bytes = max_size_mb * 1024 * 1024

        for log_file in [self.json_log, self.csv_log]:
            if log_file.exists() and log_file.stat().st_size > max_size_bytes:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = log_file.with_suffix(f".{timestamp}{log_file.suffix}")

                if archive:
                    # Compress and archive
                    archive_name = new_name.with_suffix('.gz')
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(archive_name, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    log_file.unlink()
                    self.logger.info(f"Archived {log_file} to {archive_name}")
                else:
                    # Just rename
                    log_file.rename(new_name)
                    self.logger.info(f"Rotated {log_file} to {new_name}")

                # Reinitialize if CSV
                if log_file == self.csv_log:
                    self._init_csv_log()

    def cleanup_old_logs(self, days: int = 30):
        """
        Clean up logs older than specified days.

        Args:
            days: Number of days to retain logs
        """
        cutoff_date = datetime.now().timestamp() - (days * 86400)

        for log_file in self.log_dir.glob("optimization_*"):
            if log_file.stat().st_mtime < cutoff_date:
                log_file.unlink()
                self.logger.info(f"Deleted old log file: {log_file}")

    def get_summary(self) -> Dict[str, Any]:
        """Get quick summary of current session"""
        return {
            "session_optimizations": len(self.session_data),
            "log_directory": str(self.log_dir),
            "json_log": str(self.json_log),
            "csv_log": str(self.csv_log),
            "total_logged": len(self._load_all_logs())
        }