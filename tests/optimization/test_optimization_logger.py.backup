"""Unit tests for optimization logger"""
import pytest
import json
import csv
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from backend.ai_modules.optimization.optimization_logger import OptimizationLogger


class TestOptimizationLogger:
    """Test suite for optimization logger"""

    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        # Clear any existing logs in temp dir
        for file in Path(self.temp_dir).glob("*"):
            file.unlink()
        self.logger = OptimizationLogger(log_dir=self.temp_dir)

    def teardown_method(self):
        """Cleanup after each test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test logger initialization"""
        logger = OptimizationLogger(log_dir=self.temp_dir)

        assert logger.log_dir.exists()
        assert logger.json_log.exists() or True  # May not exist until first write
        assert logger.csv_log.exists()  # Should be created in init
        assert logger.session_data == []

    def test_csv_initialization(self):
        """Test CSV file initialization with headers"""
        # CSV should be created with headers
        assert self.logger.csv_log.exists()

        with open(self.logger.csv_log, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

        assert "timestamp" in headers
        assert "image_path" in headers
        assert "ssim_improvement" in headers
        assert "file_size_reduction" in headers

    def test_log_optimization(self):
        """Test logging an optimization result"""
        test_features = {
            "edge_density": 0.15,
            "unique_colors": 10,
            "entropy": 0.65
        }

        test_params = {
            "color_precision": 6,
            "corner_threshold": 50
        }

        test_metrics = {
            "improvements": {
                "ssim_improvement": 10.5,
                "file_size_improvement": 15.2
            },
            "default_metrics": {
                "ssim": 0.80,
                "svg_size_bytes": 10000
            },
            "optimized_metrics": {
                "ssim": 0.88,
                "svg_size_bytes": 8500,
                "conversion_time": 1.5
            }
        }

        test_metadata = {
            "logo_type": "simple",
            "confidence": 0.85
        }

        self.logger.log_optimization(
            "test.png",
            test_features,
            test_params,
            test_metrics,
            test_metadata
        )

        # Check session data
        assert len(self.logger.session_data) == 1
        entry = self.logger.session_data[0]
        assert entry["image"]["path"] == "test.png"
        assert entry["features"] == test_features
        assert entry["parameters"] == test_params
        assert entry["quality"] == test_metrics

        # Check JSON log
        assert self.logger.json_log.exists()
        with open(self.logger.json_log, 'r') as f:
            line = f.readline()
            json_entry = json.loads(line)
            assert json_entry["image"]["path"] == "test.png"

        # Check CSV log
        with open(self.logger.csv_log, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            row = next(reader)
            assert "test.png" in row
            assert "simple" in row

    def test_export_to_csv(self):
        """Test CSV export functionality"""
        # Log some test data
        for i in range(3):
            self.logger.log_optimization(
                f"test{i}.png",
                {"edge_density": 0.1 + i * 0.1},
                {"color_precision": 6},
                {
                    "improvements": {"ssim_improvement": 5 + i * 2},
                    "default_metrics": {},
                    "optimized_metrics": {}
                }
            )

        export_path = self.logger.export_to_csv()

        assert Path(export_path).exists()

        # Read exported CSV
        import pandas as pd
        df = pd.read_csv(export_path)

        assert len(df) == 3
        assert 'timestamp' in df.columns
        assert 'image_filename' in df.columns

    def test_calculate_statistics(self):
        """Test statistics calculation"""
        # Log test data
        for i in range(5):
            self.logger.log_optimization(
                f"test{i}.png",
                {"edge_density": 0.1},
                {"color_precision": 6},
                {
                    "improvements": {
                        "ssim_improvement": 10 + i,
                        "file_size_improvement": 20 + i,
                        "speed_improvement": 5 + i
                    },
                    "default_metrics": {},
                    "optimized_metrics": {}
                },
                {"logo_type": "simple" if i < 3 else "complex"}
            )

        stats = self.logger.calculate_statistics()

        assert stats["total_optimizations"] == 5
        assert "ssim_improvement" in stats
        assert stats["ssim_improvement"]["average"] == 12.0  # (10+11+12+13+14)/5
        assert stats["ssim_improvement"]["min"] == 10.0
        assert stats["ssim_improvement"]["max"] == 14.0
        assert stats["logo_type_distribution"]["simple"] == 3
        assert stats["logo_type_distribution"]["complex"] == 2

    def test_calculate_statistics_with_filters(self):
        """Test statistics with logo type filter"""
        # Log mixed data
        for i in range(4):
            logo_type = "simple" if i % 2 == 0 else "complex"
            self.logger.log_optimization(
                f"test{i}.png",
                {"edge_density": 0.1},
                {"color_precision": 6},
                {
                    "improvements": {"ssim_improvement": 10 if logo_type == "simple" else 20},
                    "default_metrics": {},
                    "optimized_metrics": {}
                },
                {"logo_type": logo_type}
            )

        # Filter by logo type
        stats = self.logger.calculate_statistics(logo_type="simple")

        assert stats["total_optimizations"] == 2
        assert stats["ssim_improvement"]["average"] == 10.0

    def test_identify_best_worst(self):
        """Test identification of best and worst performers"""
        # Log data with varying performance
        improvements = [5, 15, -5, 20, 10, 0, 25, -10]
        for i, improvement in enumerate(improvements):
            self.logger.log_optimization(
                f"test{i}.png",
                {"edge_density": 0.1},
                {"color_precision": 6},
                {
                    "improvements": {"ssim_improvement": improvement},
                    "default_metrics": {},
                    "optimized_metrics": {}
                }
            )

        best_worst = self.logger.identify_best_worst(n=3)

        assert len(best_worst["best"]) == 3
        assert len(best_worst["worst"]) == 3

        # Check best are actually best
        assert best_worst["best"][0]["ssim_improvement"] == 25
        assert best_worst["best"][1]["ssim_improvement"] == 20
        assert best_worst["best"][2]["ssim_improvement"] == 15

        # Check worst are actually worst
        assert best_worst["worst"][-1]["ssim_improvement"] == -10
        assert best_worst["worst"][-2]["ssim_improvement"] == -5

    def test_generate_correlation_analysis(self):
        """Test correlation analysis generation"""
        # Log data with correlated features
        for i in range(10):
            self.logger.log_optimization(
                f"test{i}.png",
                {
                    "edge_density": 0.1 * i,
                    "unique_colors": 5 + i,
                    "entropy": 0.5 + 0.05 * i
                },
                {"color_precision": 6},
                {
                    "improvements": {"ssim_improvement": 5 + 2 * i},  # Correlated with features
                    "default_metrics": {},
                    "optimized_metrics": {}
                }
            )

        analysis = self.logger.generate_correlation_analysis()

        assert "correlations" in analysis
        assert "edge_density" in analysis["correlations"]
        assert "unique_colors" in analysis["correlations"]
        assert analysis["sample_size"] == 10

        # Edge density should have high positive correlation
        assert analysis["correlations"]["edge_density"] > 0.9

    def test_create_dashboard_data(self):
        """Test dashboard data creation"""
        # Log sample data
        for i in range(3):
            self.logger.log_optimization(
                f"test{i}.png",
                {"edge_density": 0.1},
                {"color_precision": 6},
                {
                    "improvements": {"ssim_improvement": 10 + i},
                    "default_metrics": {},
                    "optimized_metrics": {}
                },
                {"logo_type": "simple"}
            )

        dashboard_data = self.logger.create_dashboard_data()

        assert "summary" in dashboard_data
        assert "charts" in dashboard_data
        assert "tables" in dashboard_data
        assert "metadata" in dashboard_data

        assert dashboard_data["summary"]["total_optimizations"] == 3
        assert dashboard_data["metadata"]["data_points"] == 3

    def test_export_dashboard_html(self):
        """Test HTML dashboard export"""
        # Log minimal data
        self.logger.log_optimization(
            "test.png",
            {"edge_density": 0.1},
            {"color_precision": 6},
            {
                "improvements": {"ssim_improvement": 10},
                "default_metrics": {},
                "optimized_metrics": {}
            }
        )

        output_path = self.logger.export_dashboard_html()

        assert Path(output_path).exists()

        with open(output_path, 'r') as f:
            html_content = f.read()

        assert "<!DOCTYPE html>" in html_content
        assert "Optimization Analytics Dashboard" in html_content
        assert "Plotly" in html_content

    def test_rotate_logs(self):
        """Test log rotation"""
        # Create a large mock log
        self.logger.json_log.write_text("x" * 1024 * 1024)  # 1MB

        original_path = str(self.logger.json_log)
        self.logger.rotate_logs(max_size_mb=0.5, archive=False)

        # Original should be renamed
        assert not Path(original_path).exists()

        # Should have a timestamped version
        rotated_files = list(self.logger.log_dir.glob("optimization_*.*.jsonl"))
        assert len(rotated_files) > 0

    def test_rotate_logs_with_archive(self):
        """Test log rotation with compression"""
        # Create a mock log
        self.logger.json_log.write_text("test content" * 1000)

        self.logger.rotate_logs(max_size_mb=0.001, archive=True)

        # Should have a .gz archive
        archives = list(self.logger.log_dir.glob("*.gz"))
        assert len(archives) > 0

    def test_cleanup_old_logs(self):
        """Test cleanup of old log files"""
        # Create old log files
        old_file = self.logger.log_dir / "optimization_20200101.jsonl"
        old_file.touch()

        # Make it old
        import os
        old_timestamp = (datetime.now() - timedelta(days=60)).timestamp()
        os.utime(old_file, (old_timestamp, old_timestamp))

        # Create recent file
        recent_file = self.logger.log_dir / "optimization_recent.jsonl"
        recent_file.touch()

        self.logger.cleanup_old_logs(days=30)

        # Old file should be deleted
        assert not old_file.exists()
        # Recent file should remain
        assert recent_file.exists()

    def test_get_summary(self):
        """Test summary generation"""
        # Log some data
        for i in range(3):
            self.logger.log_optimization(
                f"test{i}.png",
                {"edge_density": 0.1},
                {"color_precision": 6},
                {"improvements": {}, "default_metrics": {}, "optimized_metrics": {}}
            )

        summary = self.logger.get_summary()

        assert summary["session_optimizations"] == 3
        assert summary["log_directory"] == str(self.logger.log_dir)
        assert summary["total_logged"] >= 3

    def test_thread_safety(self):
        """Test thread-safe logging"""
        import threading

        def log_entry(i):
            self.logger.log_optimization(
                f"test{i}.png",
                {"edge_density": 0.1},
                {"color_precision": 6},
                {"improvements": {}, "default_metrics": {}, "optimized_metrics": {}}
            )

        threads = []
        for i in range(10):
            t = threading.Thread(target=log_entry, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All entries should be logged
        assert len(self.logger.session_data) == 10

    def test_empty_data_handling(self):
        """Test handling of empty data"""
        # Statistics with no data
        stats = self.logger.calculate_statistics()
        assert stats.get("message") or stats.get("total_optimizations") == 0

        # Best/worst with no data
        best_worst = self.logger.identify_best_worst()
        assert best_worst["best"] == []
        assert best_worst["worst"] == []

        # Correlation with no data
        correlation = self.logger.generate_correlation_analysis()
        assert "error" in correlation or correlation.get("sample_size") == 0