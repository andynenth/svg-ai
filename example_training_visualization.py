#!/usr/bin/env python3
"""
Example usage of the TrainingVisualizer for PPO agent training analysis

This script demonstrates how to use the training visualizer to analyze
RL training results from the PPO optimizer and curriculum training pipeline.
"""

import sys
import json
import logging
from pathlib import Path

# Add the backend modules to path
sys.path.append('/Users/nrw/python/svg-ai')

from backend.ai_modules.optimization.training_visualizer import create_training_visualizer
from backend.ai_modules.optimization.training_orchestrator import create_training_orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_single_experiment_analysis():
    """
    Example: Analyze results from a single training experiment
    """
    print("📊 Example: Single Experiment Analysis")
    print("=" * 50)

    # Create visualizer
    visualizer = create_training_visualizer("visualizations/single_experiment")

    # Load or simulate training data
    # In practice, this would come from your actual training run
    sample_data = {
        'metrics_history': [
            {
                'episode': i,
                'timestamp': 1234567890 + i * 60,
                'episode_rewards': 0.3 + i * 0.001 + (i % 10) * 0.02,
                'quality_improvements': 0.7 + i * 0.0005,
                'success_rates': min(0.9, 0.2 + i * 0.0015),
                'training_loss': max(0.01, 2.0 * (1 - i / 1000)),
                'value_loss': max(0.01, 1.5 * (1 - i / 1200)),
                'policy_loss': max(0.01, 1.0 * (1 - i / 800)),
                'entropy': max(0.01, 1.0 * (1 - i / 1500))
            }
            for i in range(1000)
        ],
        'parameter_history': [
            {
                'episode': i * 10,
                'learning_rate': 0.0003,
                'batch_size': 64,
                'clip_range': 0.2,
                'entropy_coef': max(0.001, 0.01 - i * 0.0001)
            }
            for i in range(100)
        ],
        'training_time': 3600,
        'experiment_config': {
            'model_type': 'PPO',
            'total_timesteps': 1000000,
            'learning_rate': 0.0003
        }
    }

    # Generate comprehensive analysis
    try:
        report_files = visualizer.generate_comprehensive_report(
            sample_data,
            experiment_name="vtracer_ppo_optimization"
        )

        print(f"✅ Generated {len(report_files)} visualization files:")
        for component, file_path in report_files.items():
            file_name = Path(file_path).name
            print(f"  📄 {component}: {file_name}")

        # Display key insights
        progress_metrics = visualizer.monitor_training_progress(
            sample_data['metrics_history']
        )

        print(f"\n🎯 Key Training Insights:")
        print(f"  • Final Reward: {progress_metrics['latest_reward']:.4f}")
        print(f"  • Average Reward: {progress_metrics['average_reward']:.4f}")
        print(f"  • Training Health Score: {progress_metrics['health_score']:.1f}/100")
        print(f"  • Anomalies Detected: {progress_metrics['anomalies_detected']}")

        if progress_metrics['health_score'] > 80:
            print("  🟢 Training appears healthy and stable")
        elif progress_metrics['health_score'] > 60:
            print("  🟡 Training shows some issues but is progressing")
        else:
            print("  🔴 Training may have significant problems")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")


def example_experiment_comparison():
    """
    Example: Compare multiple training experiments
    """
    print("\n📊 Example: Experiment Comparison")
    print("=" * 50)

    # Create visualizer
    visualizer = create_training_visualizer("visualizations/experiment_comparison")

    # Simulate results from different hyperparameter configurations
    experiments = {}

    # Experiment 1: Default settings
    experiments['default_lr_0.0003'] = {
        'metrics_history': [
            {
                'episode': i,
                'episode_rewards': 0.3 + i * 0.0008,
                'quality_improvements': 0.7 + i * 0.0003,
                'success_rates': min(0.85, 0.2 + i * 0.0012)
            }
            for i in range(500)
        ]
    }

    # Experiment 2: Higher learning rate
    experiments['high_lr_0.001'] = {
        'metrics_history': [
            {
                'episode': i,
                'episode_rewards': 0.3 + i * 0.0012 - (i % 50) * 0.01,  # More volatile
                'quality_improvements': 0.7 + i * 0.0005,
                'success_rates': min(0.90, 0.15 + i * 0.0018)
            }
            for i in range(500)
        ]
    }

    # Experiment 3: Lower learning rate
    experiments['low_lr_0.0001'] = {
        'metrics_history': [
            {
                'episode': i,
                'episode_rewards': 0.3 + i * 0.0005,  # Slower learning
                'quality_improvements': 0.7 + i * 0.0002,
                'success_rates': min(0.80, 0.25 + i * 0.0008)
            }
            for i in range(500)
        ]
    }

    try:
        # Generate comparison analysis
        comparison_files = visualizer.compare_training_runs(
            experiments,
            output_name="learning_rate_comparison"
        )

        print(f"✅ Generated {len(comparison_files)} comparison files:")
        for file_type, file_path in comparison_files.items():
            file_name = Path(file_path).name
            print(f"  📄 {file_type}: {file_name}")

        print(f"\n📈 Comparison Summary:")
        print(f"  • Compared {len(experiments)} different learning rate configurations")
        print(f"  • Generated statistical analysis of performance differences")
        print(f"  • Created side-by-side performance visualizations")

    except Exception as e:
        print(f"❌ Comparison failed: {e}")


def example_real_time_monitoring():
    """
    Example: Real-time training monitoring
    """
    print("\n📊 Example: Real-time Training Monitoring")
    print("=" * 50)

    # Create visualizer
    visualizer = create_training_visualizer("visualizations/realtime_monitoring")

    # Simulate incoming training data (like from actual training)
    print("🔄 Simulating real-time training data...")

    cumulative_metrics = []

    for episode in range(0, 201, 20):  # Monitor every 20 episodes
        # Add new metrics
        batch_metrics = [
            {
                'episode': episode + i,
                'episode_rewards': 0.3 + (episode + i) * 0.001,
                'quality_improvements': 0.7 + (episode + i) * 0.0003,
                'success_rates': min(0.9, 0.2 + (episode + i) * 0.002)
            }
            for i in range(20)
        ]

        cumulative_metrics.extend(batch_metrics)

        # Monitor progress
        try:
            progress_metrics = visualizer.monitor_training_progress(
                cumulative_metrics,
                real_time=True
            )

            print(f"Episode {episode + 20:3d}: "
                  f"Reward={progress_metrics['latest_reward']:.3f}, "
                  f"Health={progress_metrics['health_score']:.1f}, "
                  f"Anomalies={progress_metrics['anomalies_detected']}")

        except Exception as e:
            print(f"  ❌ Monitoring error at episode {episode}: {e}")

    print("✅ Real-time monitoring simulation complete")


def example_integration_with_training_pipeline():
    """
    Example: Integration with actual training pipeline
    """
    print("\n📊 Example: Integration with Training Pipeline")
    print("=" * 50)

    # This example shows how you would integrate the visualizer
    # with the actual training pipeline

    # Sample training data structure that would come from actual training
    training_data_path = "data/logos"  # This would be your actual data path
    output_dir = "training_results/example_run"

    print("🎯 Integration Steps:")
    print("1. Training Orchestrator would generate metrics during training")
    print("2. Metrics are collected in structured format")
    print("3. Visualizer processes metrics after training completes")
    print("4. Comprehensive reports are generated automatically")

    # Example of what the integration code would look like:
    code_example = '''
    # In your actual training script:
    from backend.ai_modules.optimization.training_orchestrator import create_training_orchestrator
    from backend.ai_modules.optimization.training_visualizer import create_training_visualizer

    # 1. Run training
    orchestrator = create_training_orchestrator(
        experiment_name="my_experiment",
        training_data_path="data/logos",
        output_dir="results/my_experiment"
    )

    training_results = orchestrator.run_training_experiment()

    # 2. Generate visualizations
    visualizer = create_training_visualizer("visualizations/my_experiment")

    report_files = visualizer.generate_comprehensive_report(
        training_results,
        experiment_name="my_experiment"
    )

    # 3. Monitor training health
    progress_metrics = visualizer.monitor_training_progress(
        training_results['training_results']['metrics_history']
    )
    '''

    print("\n💻 Example Integration Code:")
    print(code_example)

    print("✅ Integration example complete")


def main():
    """
    Run all examples to demonstrate training visualizer capabilities
    """
    print("🚀 Training Visualizer Examples")
    print("=" * 60)
    print("This script demonstrates comprehensive training analysis capabilities")
    print("for PPO agent training in the VTracer optimization environment.")
    print("")

    try:
        # Run examples
        example_single_experiment_analysis()
        example_experiment_comparison()
        example_real_time_monitoring()
        example_integration_with_training_pipeline()

        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("")
        print("📁 Generated visualization files can be found in:")
        print("  • visualizations/single_experiment/")
        print("  • visualizations/experiment_comparison/")
        print("  • visualizations/realtime_monitoring/")
        print("")
        print("🔧 Next steps:")
        print("  1. Install plotly for interactive visualizations: pip install plotly")
        print("  2. Integrate with your actual training pipeline")
        print("  3. Customize visualization parameters as needed")
        print("  4. Set up automated report generation")

    except Exception as e:
        print(f"❌ Example execution failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)