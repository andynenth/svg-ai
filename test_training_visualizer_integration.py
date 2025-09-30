#!/usr/bin/env python3
"""
Integration test for training visualizer
Tests the complete visualization pipeline with sample data
"""

import os
import tempfile
import shutil
import numpy as np
import json
from pathlib import Path
import sys

# Add the backend modules to path
sys.path.append('/Users/nrw/python/svg-ai')

from backend.ai_modules.optimization.training_visualizer import (
    TrainingVisualizer,
    create_training_visualizer
)

def generate_sample_training_data(num_episodes: int = 500) -> dict:
    """Generate realistic sample training data for testing"""

    # Simulate realistic training progression
    base_reward = 0.3
    learning_rate = 0.002
    noise_level = 0.1

    metrics_history = []
    parameter_history = []

    for episode in range(num_episodes):
        # Simulate learning curve with some noise and occasional setbacks
        progress = episode / num_episodes
        reward_trend = base_reward + progress * 0.6  # Learn from 0.3 to 0.9

        # Add realistic noise and occasional dips
        noise = np.random.normal(0, noise_level)
        if episode % 100 == 99:  # Occasional performance dips
            noise -= 0.1

        episode_reward = max(0.1, reward_trend + noise)

        # Quality improvements correlated with rewards
        quality_improvement = min(0.95, 0.6 + episode_reward * 0.4 + np.random.normal(0, 0.05))

        # Success rate improves with training
        success_rate = min(0.9, 0.2 + progress * 0.7 + np.random.normal(0, 0.1))

        # Training losses decrease over time
        training_loss = max(0.01, 2.0 * np.exp(-episode * 0.005) + np.random.normal(0, 0.1))
        value_loss = max(0.01, 1.5 * np.exp(-episode * 0.004) + np.random.normal(0, 0.08))
        policy_loss = max(0.01, 1.0 * np.exp(-episode * 0.003) + np.random.normal(0, 0.05))

        # Entropy decreases as agent becomes more confident
        entropy = max(0.01, 1.0 * np.exp(-episode * 0.002) + np.random.normal(0, 0.02))

        metrics_history.append({
            'episode': episode,
            'timestamp': 1234567890 + episode * 60,  # 1 minute per episode
            'episode_rewards': episode_reward,
            'quality_improvements': quality_improvement,
            'success_rates': success_rate,
            'training_loss': training_loss,
            'value_loss': value_loss,
            'policy_loss': policy_loss,
            'entropy': entropy,
            'episode_lengths': np.random.randint(20, 80)
        })

        # Parameter exploration history (less frequent)
        if episode % 10 == 0:
            parameter_history.append({
                'episode': episode,
                'learning_rate': 0.0003 + np.random.normal(0, 0.00005),
                'batch_size': 64,
                'clip_range': 0.2 + np.random.normal(0, 0.02),
                'entropy_coef': max(0.001, 0.01 + np.random.normal(0, 0.002)),
                'value_coef': 0.5 + np.random.normal(0, 0.05),
                'exploration_rate': max(0.01, 0.9 - episode * 0.0015)
            })

    return {
        'metrics_history': metrics_history,
        'parameter_history': parameter_history,
        'training_time': num_episodes * 60,  # 1 minute per episode
        'experiment_config': {
            'model_type': 'PPO',
            'environment': 'VTracerOptimization',
            'total_timesteps': num_episodes * 1000
        }
    }

def test_basic_visualization_creation():
    """Test basic visualizer creation and functionality"""
    print("🧪 Testing basic visualizer creation...")

    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = create_training_visualizer(temp_dir)

        assert visualizer is not None
        assert visualizer.output_dir.exists()
        assert visualizer.processor is not None
        assert visualizer.plotter is not None

        print("✅ Basic visualizer creation successful")

def test_comprehensive_report_generation():
    """Test comprehensive report generation with sample data"""
    print("🧪 Testing comprehensive report generation...")

    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = create_training_visualizer(temp_dir)

        # Generate sample training data
        training_data = generate_sample_training_data(300)

        # Generate comprehensive report
        try:
            report_files = visualizer.generate_comprehensive_report(
                training_data,
                experiment_name="test_experiment"
            )

            print(f"📊 Generated {len(report_files)} report components:")
            for component, file_path in report_files.items():
                print(f"  - {component}: {Path(file_path).name}")

                # Verify file exists
                assert Path(file_path).exists(), f"Report file not found: {file_path}"

                # Check file size (should not be empty)
                file_size = Path(file_path).stat().st_size
                assert file_size > 0, f"Report file is empty: {file_path}"

                print(f"    ✅ File exists and has content ({file_size} bytes)")

            # Verify specific expected components
            expected_components = [
                'learning_curves',
                'loss_curves',
                'reward_distribution',
                'parameter_exploration',
                'training_stability',
                'summary_statistics',
                'master_report'
            ]

            for component in expected_components:
                assert component in report_files, f"Missing expected component: {component}"

            print("✅ Comprehensive report generation successful")

        except Exception as e:
            print(f"❌ Report generation failed: {e}")
            raise

def test_experiment_comparison():
    """Test experiment comparison functionality"""
    print("🧪 Testing experiment comparison...")

    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = create_training_visualizer(temp_dir)

        # Generate sample data for multiple experiments
        experiment1 = generate_sample_training_data(200)
        experiment2 = generate_sample_training_data(250)

        # Modify experiment2 to have different performance
        for i, metrics in enumerate(experiment2['metrics_history']):
            metrics['episode_rewards'] *= 1.2  # Better performance
            metrics['quality_improvements'] += 0.05

        training_runs = {
            'experiment_1': experiment1,
            'experiment_2': experiment2
        }

        try:
            comparison_files = visualizer.compare_training_runs(
                training_runs,
                output_name="test_comparison"
            )

            print(f"📊 Generated {len(comparison_files)} comparison files:")
            for file_type, file_path in comparison_files.items():
                print(f"  - {file_type}: {Path(file_path).name}")
                assert Path(file_path).exists()

            print("✅ Experiment comparison successful")

        except Exception as e:
            print(f"❌ Experiment comparison failed: {e}")
            raise

def test_progress_monitoring():
    """Test training progress monitoring"""
    print("🧪 Testing progress monitoring...")

    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = create_training_visualizer(temp_dir)

        # Generate sample data
        training_data = generate_sample_training_data(100)

        try:
            progress_metrics = visualizer.monitor_training_progress(
                training_data['metrics_history']
            )

            print("📊 Progress metrics calculated:")
            for metric, value in progress_metrics.items():
                print(f"  - {metric}: {value}")

            # Verify expected metrics exist
            expected_metrics = ['latest_reward', 'average_reward', 'health_score']
            for metric in expected_metrics:
                assert metric in progress_metrics, f"Missing progress metric: {metric}"

            # Verify health score is reasonable
            health_score = progress_metrics['health_score']
            assert 0 <= health_score <= 100, f"Invalid health score: {health_score}"

            print("✅ Progress monitoring successful")

        except Exception as e:
            print(f"❌ Progress monitoring failed: {e}")
            raise

def test_anomaly_detection():
    """Test anomaly detection functionality"""
    print("🧪 Testing anomaly detection...")

    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = create_training_visualizer(temp_dir)

        # Generate data with intentional anomalies
        training_data = generate_sample_training_data(200)

        # Add some anomalies
        anomaly_episodes = [50, 100, 150]
        for episode_idx in anomaly_episodes:
            if episode_idx < len(training_data['metrics_history']):
                # Create reward anomaly
                training_data['metrics_history'][episode_idx]['episode_rewards'] = -0.5  # Negative reward

                # Create loss explosion
                training_data['metrics_history'][episode_idx]['training_loss'] = 10.0  # Very high loss

        try:
            # Process metrics
            metrics_df = visualizer.processor.process_metrics_history(
                training_data['metrics_history']
            )

            # Detect anomalies
            anomalies = visualizer.anomaly_detector.detect_training_anomalies(metrics_df)

            print(f"🔍 Detected anomalies:")
            total_anomalies = sum(len(anomaly_list) for anomaly_list in anomalies.values())
            print(f"  - Total anomalies: {total_anomalies}")

            for category, anomaly_list in anomalies.items():
                if anomaly_list:
                    print(f"  - {category}: {len(anomaly_list)} anomalies")

            # Generate anomaly report
            anomaly_report = visualizer.anomaly_detector.generate_anomaly_report(anomalies)
            assert len(anomaly_report) > 0, "Anomaly report should not be empty"

            print("✅ Anomaly detection successful")

        except Exception as e:
            print(f"❌ Anomaly detection failed: {e}")
            raise

def test_interactive_dashboard():
    """Test interactive dashboard creation (if plotly available)"""
    print("🧪 Testing interactive dashboard...")

    try:
        import plotly
        plotly_available = True
    except ImportError:
        plotly_available = False
        print("⚠️ Plotly not available, skipping interactive dashboard test")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = create_training_visualizer(temp_dir)

        if not visualizer.dashboard:
            print("⚠️ Dashboard not initialized, skipping test")
            return

        # Generate sample data
        training_data = generate_sample_training_data(150)
        metrics_df = visualizer.processor.process_metrics_history(
            training_data['metrics_history']
        )

        try:
            dashboard_html = visualizer.dashboard.create_interactive_training_dashboard(
                metrics_df,
                save_path=None  # Return HTML string
            )

            assert dashboard_html is not None
            assert len(dashboard_html) > 100  # Should be substantial HTML
            assert '<html>' in dashboard_html or 'plotly' in dashboard_html.lower()

            print("✅ Interactive dashboard creation successful")

        except Exception as e:
            print(f"❌ Interactive dashboard creation failed: {e}")
            raise

def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Starting Training Visualizer Integration Tests")
    print("=" * 60)

    tests = [
        test_basic_visualization_creation,
        test_comprehensive_report_generation,
        test_experiment_comparison,
        test_progress_monitoring,
        test_anomaly_detection,
        test_interactive_dashboard
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
            print("")
        except Exception as e:
            print(f"❌ Test failed: {test_func.__name__}")
            print(f"   Error: {e}")
            failed += 1
            print("")

    print("=" * 60)
    print(f"🏁 Integration Tests Complete")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")

    return failed == 0

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)