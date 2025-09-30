#!/usr/bin/env python3
"""
Example of how to integrate TrainingMonitor with PPO training pipeline
"""

import sys
from pathlib import Path

# Add backend modules to path
sys.path.append(str(Path(__file__).parent / "backend"))

def show_integration_example():
    """Show how to integrate TrainingMonitor with PPO training"""

    print("🔗 TrainingMonitor Integration Example")
    print("=" * 50)

    # Example integration code that would go in PPO training loop
    integration_code = '''
# In PPO training pipeline:

from backend.ai_modules.optimization.training_monitor import create_training_monitor

# Initialize monitor
monitor = create_training_monitor(
    log_dir="logs/ppo_training",
    session_name="vtracer_optimization_v1",
    use_tensorboard=True,
    use_wandb=False  # Enable if you have wandb configured
)

# During training episodes:
for episode in range(total_episodes):
    obs = env.reset()
    episode_reward = 0
    episode_length = 0
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1

        if done or truncated:
            # Extract quality metrics from environment
            quality_improvement = info.get('quality_improvement', 0)
            quality_final = info.get('quality_final', 0)
            quality_initial = info.get('quality_initial', 0)
            success = info.get('target_reached', False)

            # Log comprehensive episode metrics
            monitor.log_episode(
                episode=episode,
                reward=episode_reward,
                length=episode_length,
                quality_improvement=quality_improvement,
                quality_final=quality_final,
                quality_initial=quality_initial,
                termination_reason="target_reached" if success else "max_steps",
                success=success,
                additional_info={
                    'logo_type': info.get('logo_type'),
                    'difficulty': info.get('difficulty'),
                    'parameters_explored': info.get('current_params')
                }
            )

    # During PPO model updates:
    if episode % update_frequency == 0:
        # Get algorithm metrics from PPO training
        monitor.log_training_step(
            step=episode,
            policy_loss=policy_loss_value,
            value_loss=value_loss_value,
            entropy=entropy_value,
            kl_divergence=kl_div_value,
            learning_rate=current_lr,
            gradient_norm=grad_norm
        )

    # Periodic evaluation
    if episode % eval_frequency == 0:
        eval_results = evaluate_model(model, eval_env, n_episodes=5)
        monitor.log_evaluation(
            episode=episode,
            eval_reward=eval_results['mean_reward'],
            eval_quality=eval_results['mean_quality'],
            eval_success_rate=eval_results['success_rate']
        )

    # Real-time monitoring
    if episode % 10 == 0:
        dashboard_data = monitor.create_dashboard_data()
        # Send to web dashboard or log key metrics
        print(f"Episode {episode}: Health={dashboard_data['summary']['health_score']:.2%}")

# Export final results
final_export = monitor.export_metrics("json", "final_training_results.json")
monitor.close()
'''

    print("Integration code structure:")
    print(integration_code)

    print("\n📊 Key Integration Points:")
    print("1. Episode logging: log_episode() after each environment episode")
    print("2. Training step logging: log_training_step() during PPO updates")
    print("3. Evaluation logging: log_evaluation() during periodic evaluation")
    print("4. Real-time monitoring: create_dashboard_data() for live updates")
    print("5. Export functionality: export_metrics() for final results")

    print("\n📈 Benefits of Integration:")
    print("✅ Comprehensive training metrics collection")
    print("✅ Algorithm-specific PPO metrics (policy loss, value loss, entropy)")
    print("✅ Performance tracking (training time, memory usage)")
    print("✅ Real-time dashboard data for monitoring")
    print("✅ Export capabilities (JSON, CSV, TensorBoard)")
    print("✅ Training validation and health checks")
    print("✅ Convergence analysis and trend detection")

    print("\n🔧 Required Modifications to PPO Pipeline:")
    print("1. Add TrainingMonitor import and initialization")
    print("2. Extract metrics from PPO training loop")
    print("3. Call monitor.log_episode() after each episode")
    print("4. Call monitor.log_training_step() during model updates")
    print("5. Add periodic evaluation logging")
    print("6. Export results at end of training")


def show_api_reference():
    """Show key API methods for integration"""
    print("\n📚 TrainingMonitor API Reference")
    print("=" * 50)

    api_docs = '''
# Core logging methods:

log_episode(episode, reward, length, quality_improvement, quality_final,
           quality_initial, success=False, algorithm_metrics=None,
           performance_metrics=None, additional_info=None)
    # Log comprehensive episode results

log_training_step(step, policy_loss, value_loss, entropy,
                 kl_divergence=None, learning_rate=None, gradient_norm=None)
    # Log PPO algorithm training metrics

log_evaluation(episode, eval_reward, eval_quality, eval_success_rate,
              eval_episodes=1)
    # Log evaluation results

# Analysis methods:

get_training_statistics(window_size=None) -> Dict
    # Get comprehensive training statistics

get_convergence_analysis() -> Dict
    # Analyze training convergence and trends

validate_metrics() -> Dict
    # Validate metrics and check training health

# Export methods:

export_metrics(format="json", output_path=None, include_raw_data=True) -> str
    # Export training data (json, csv, pandas)

create_dashboard_data() -> Dict
    # Create real-time dashboard data

# Utility methods:

close()
    # Clean up resources and save final data
'''

    print(api_docs)


if __name__ == "__main__":
    show_integration_example()
    show_api_reference()

    print("\n🎉 TrainingMonitor Implementation Complete!")
    print("\nNext steps:")
    print("1. Integrate with PPO training pipeline")
    print("2. Test with actual training data")
    print("3. Configure TensorBoard/Wandb for enhanced monitoring")
    print("4. Setup real-time dashboard for live training monitoring")