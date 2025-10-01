"""
Quality Tracking Database - Task 3 Implementation
SQLite database for tracking conversion results and quality metrics.
"""

import sqlite3
import json
import csv
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import statistics


@dataclass
class QualityRecord:
    """Quality record structure for conversion tracking."""
    image_id: str
    timestamp: datetime
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    model_version: str
    processing_time: float
    user_rating: Optional[int] = None
    id: Optional[int] = None


class QualityTracker:
    """Thread-safe SQLite database for tracking conversion quality."""

    def __init__(self, db_path: str = "data/quality_tracking.db"):
        """Initialize quality tracker with SQLite database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with quality tracking schema."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()

                # Create quality_tracking table with indexes
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quality_tracking (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_id TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        parameters TEXT NOT NULL,
                        metrics TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        processing_time REAL NOT NULL,
                        user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5)
                    )
                """)

                # Create indexes for performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_id ON quality_tracking (image_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON quality_tracking (timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_version ON quality_tracking (model_version)")

                conn.commit()
            finally:
                conn.close()

    def store_conversion_result(self, record: QualityRecord) -> int:
        """
        Store conversion result in database.

        Args:
            record: QualityRecord instance with conversion data

        Returns:
            int: ID of inserted record
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO quality_tracking
                    (image_id, timestamp, parameters, metrics, model_version, processing_time, user_rating)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.image_id,
                    record.timestamp.isoformat(),
                    json.dumps(record.parameters),
                    json.dumps(record.metrics),
                    record.model_version,
                    record.processing_time,
                    record.user_rating
                ))

                record_id = cursor.lastrowid
                conn.commit()
                return record_id
            finally:
                conn.close()

    def query_historical_quality(self,
                               image_id: Optional[str] = None,
                               days_back: int = 30,
                               limit: int = 1000) -> List[QualityRecord]:
        """
        Query historical quality records.

        Args:
            image_id: Specific image ID to filter by
            days_back: Number of days to look back
            limit: Maximum number of records to return

        Returns:
            List[QualityRecord]: Historical quality records
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()

                # Build query conditions
                conditions = []
                params = []

                if image_id:
                    conditions.append("image_id = ?")
                    params.append(image_id)

                if days_back > 0:
                    cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
                    conditions.append("timestamp >= ?")
                    params.append(cutoff_date)

                where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

                cursor.execute(f"""
                    SELECT id, image_id, timestamp, parameters, metrics,
                           model_version, processing_time, user_rating
                    FROM quality_tracking
                    {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, params + [limit])

                records = []
                for row in cursor.fetchall():
                    records.append(QualityRecord(
                        id=row[0],
                        image_id=row[1],
                        timestamp=datetime.fromisoformat(row[2]),
                        parameters=json.loads(row[3]),
                        metrics=json.loads(row[4]),
                        model_version=row[5],
                        processing_time=row[6],
                        user_rating=row[7]
                    ))

                return records
            finally:
                conn.close()

    def calculate_quality_trends(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Calculate quality trends over time.

        Args:
            days_back: Number of days to analyze

        Returns:
            Dict with trend analysis
        """
        records = self.query_historical_quality(days_back=days_back, limit=10000)

        if not records:
            return {
                'total_conversions': 0,
                'avg_composite_score': 0,
                'avg_processing_time': 0,
                'trend': 'no_data'
            }

        # Extract composite scores and processing times
        composite_scores = []
        processing_times = []
        daily_scores = {}

        for record in records:
            if 'composite_score' in record.metrics:
                composite_scores.append(record.metrics['composite_score'])
                processing_times.append(record.processing_time)

                # Group by day for trend analysis
                day_key = record.timestamp.date().isoformat()
                if day_key not in daily_scores:
                    daily_scores[day_key] = []
                daily_scores[day_key].append(record.metrics['composite_score'])

        if not composite_scores:
            return {
                'total_conversions': len(records),
                'avg_composite_score': 0,
                'avg_processing_time': statistics.mean(processing_times) if processing_times else 0,
                'trend': 'no_metrics'
            }

        # Calculate daily averages for trend detection
        daily_averages = []
        for day in sorted(daily_scores.keys()):
            daily_averages.append(statistics.mean(daily_scores[day]))

        # Simple trend detection (comparing first half vs second half)
        trend = 'stable'
        if len(daily_averages) >= 4:
            first_half = statistics.mean(daily_averages[:len(daily_averages)//2])
            second_half = statistics.mean(daily_averages[len(daily_averages)//2:])

            if second_half > first_half * 1.05:
                trend = 'improving'
            elif second_half < first_half * 0.95:
                trend = 'declining'

        return {
            'total_conversions': len(records),
            'avg_composite_score': statistics.mean(composite_scores),
            'median_composite_score': statistics.median(composite_scores),
            'std_composite_score': statistics.stdev(composite_scores) if len(composite_scores) > 1 else 0,
            'avg_processing_time': statistics.mean(processing_times),
            'median_processing_time': statistics.median(processing_times),
            'trend': trend,
            'daily_averages': daily_averages,
            'quality_distribution': {
                'excellent': len([s for s in composite_scores if s >= 0.9]),
                'good': len([s for s in composite_scores if 0.7 <= s < 0.9]),
                'fair': len([s for s in composite_scores if 0.5 <= s < 0.7]),
                'poor': len([s for s in composite_scores if s < 0.5])
            }
        }

    def find_best_parameters(self, image_type: str = None, min_score: float = 0.7) -> Dict[str, Any]:
        """
        Find best parameters for given image type or overall.

        Args:
            image_type: Type of image to analyze (optional)
            min_score: Minimum composite score threshold

        Returns:
            Dict with best parameter recommendations
        """
        records = self.query_historical_quality(days_back=90, limit=5000)

        # Filter by image type if specified
        if image_type:
            records = [r for r in records if r.parameters.get('image_type') == image_type]

        # Filter by minimum score
        good_records = []
        for record in records:
            if 'composite_score' in record.metrics and record.metrics['composite_score'] >= min_score:
                good_records.append(record)

        if not good_records:
            return {
                'parameter_recommendations': {},
                'best_score': 0,
                'sample_size': 0,
                'confidence': 'low'
            }

        # Analyze parameter frequency and scores
        param_scores = {}
        for record in good_records:
            score = record.metrics['composite_score']
            for param_name, param_value in record.parameters.items():
                if param_name not in param_scores:
                    param_scores[param_name] = {}

                param_key = str(param_value)
                if param_key not in param_scores[param_name]:
                    param_scores[param_name][param_key] = []

                param_scores[param_name][param_key].append(score)

        # Find best parameter values
        recommendations = {}
        for param_name, value_scores in param_scores.items():
            best_value = None
            best_avg_score = 0

            for value, scores in value_scores.items():
                if len(scores) >= 3:  # Require at least 3 samples
                    avg_score = statistics.mean(scores)
                    if avg_score > best_avg_score:
                        best_avg_score = avg_score
                        best_value = value

            if best_value is not None:
                recommendations[param_name] = {
                    'value': best_value,
                    'avg_score': best_avg_score,
                    'sample_count': len(param_scores[param_name][best_value])
                }

        # Calculate confidence based on sample size
        total_samples = len(good_records)
        if total_samples >= 50:
            confidence = 'high'
        elif total_samples >= 20:
            confidence = 'medium'
        else:
            confidence = 'low'

        return {
            'parameter_recommendations': recommendations,
            'best_score': max(r.metrics['composite_score'] for r in good_records),
            'sample_size': total_samples,
            'confidence': confidence,
            'image_type': image_type
        }

    def export_to_json(self, filename: str, days_back: int = 30) -> str:
        """
        Export quality data to JSON file.

        Args:
            filename: Output filename
            days_back: Number of days to export

        Returns:
            str: Path to exported file
        """
        records = self.query_historical_quality(days_back=days_back, limit=10000)

        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_records': len(records),
            'records': [asdict(record) for record in records]
        }

        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        return str(output_path)

    def export_to_csv(self, filename: str, days_back: int = 30) -> str:
        """
        Export quality data to CSV file.

        Args:
            filename: Output filename
            days_back: Number of days to export

        Returns:
            str: Path to exported file
        """
        records = self.query_historical_quality(days_back=days_back, limit=10000)

        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as csvfile:
            if not records:
                return str(output_path)

            # Flatten the first record to determine all possible fields
            sample_record = records[0]
            fieldnames = ['id', 'image_id', 'timestamp', 'model_version', 'processing_time', 'user_rating']

            # Add parameter fields
            for param in sample_record.parameters.keys():
                fieldnames.append(f'param_{param}')

            # Add metric fields
            for metric in sample_record.metrics.keys():
                fieldnames.append(f'metric_{metric}')

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for record in records:
                row = {
                    'id': record.id,
                    'image_id': record.image_id,
                    'timestamp': record.timestamp.isoformat(),
                    'model_version': record.model_version,
                    'processing_time': record.processing_time,
                    'user_rating': record.user_rating
                }

                # Add parameters
                for param, value in record.parameters.items():
                    row[f'param_{param}'] = value

                # Add metrics
                for metric, value in record.metrics.items():
                    row[f'metric_{metric}'] = value

                writer.writerow(row)

        return str(output_path)

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()

                # Total records
                cursor.execute("SELECT COUNT(*) FROM quality_tracking")
                total_records = cursor.fetchone()[0]

                # Records in last 24 hours
                yesterday = (datetime.now() - timedelta(days=1)).isoformat()
                cursor.execute("SELECT COUNT(*) FROM quality_tracking WHERE timestamp >= ?", (yesterday,))
                recent_records = cursor.fetchone()[0]

                # Unique images
                cursor.execute("SELECT COUNT(DISTINCT image_id) FROM quality_tracking")
                unique_images = cursor.fetchone()[0]

                # Database file size
                db_size_mb = self.db_path.stat().st_size / (1024 * 1024)

                return {
                    'total_records': total_records,
                    'recent_records_24h': recent_records,
                    'unique_images': unique_images,
                    'database_size_mb': round(db_size_mb, 2),
                    'database_path': str(self.db_path)
                }
            finally:
                conn.close()


def create_sample_record(image_id: str = "test_image.png") -> QualityRecord:
    """Create sample quality record for testing."""
    return QualityRecord(
        image_id=image_id,
        timestamp=datetime.now(),
        parameters={
            'color_precision': 4,
            'corner_threshold': 30,
            'path_precision': 8,
            'image_type': 'simple_geometric'
        },
        metrics={
            'ssim': 0.95,
            'mse': 0.02,
            'composite_score': 0.88
        },
        model_version='vtracer_v1.0',
        processing_time=0.45,
        user_rating=4
    )


if __name__ == "__main__":
    # Test the quality tracker
    tracker = QualityTracker("data/test_quality_tracking.db")

    # Store sample record
    sample = create_sample_record()
    record_id = tracker.store_conversion_result(sample)
    print(f"Stored sample record with ID: {record_id}")

    # Query records
    records = tracker.query_historical_quality(limit=10)
    print(f"Found {len(records)} records")

    # Calculate trends
    trends = tracker.calculate_quality_trends()
    print(f"Quality trends: {trends}")

    # Database stats
    stats = tracker.get_database_stats()
    print(f"Database stats: {stats}")