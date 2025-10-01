"""
Day 12: Advanced Hyperparameter Optimization
Comprehensive hyperparameter optimization with Bayesian optimization and advanced techniques
Part of Task 12.2.1: Hyperparameter Optimization using GPU acceleration
"""

import torch
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import itertools
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# Import Day 12 components
from .gpu_training_pipeline import GPUTrainingPipeline, GPUDataLoader
from .gpu_model_architecture import ColabTrainingConfig, QualityPredictorGPU
from .day12_training_visualization import RealTimeTrainingMonitor

warnings.filterwarnings('ignore')


@dataclass
class HyperparameterSpace:
    """Define hyperparameter search space"""
    learning_rate: Tuple[float, float] = (1e-5, 1e-2)
    batch_size: List[int] = None
    hidden_dims: List[List[int]] = None
    dropout_rates: List[List[float]] = None
    weight_decay: Tuple[float, float] = (1e-6, 1e-3)
    optimizer_type: List[str] = None
    scheduler_type: List[str] = None

    def __post_init__(self):
        if self.batch_size is None:
            self.batch_size = [16, 32, 64, 128]
        if self.hidden_dims is None:
            self.hidden_dims = [
                [512, 256, 128],
                [1024, 512, 256],
                [1024, 512, 256, 128],
                [2048, 1024, 512],
                [1536, 768, 384],
                [512, 256],
                [1024, 256]
            ]
        if self.dropout_rates is None:
            self.dropout_rates = [
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.1],
                [0.3, 0.2, 0.1],
                [0.4, 0.3, 0.2],
                [0.5, 0.3, 0.1]
            ]
        if self.optimizer_type is None:
            self.optimizer_type = ['adamw', 'adam', 'sgd']
        if self.scheduler_type is None:
            self.scheduler_type = ['cosine_annealing', 'step', 'none']


@dataclass
class HyperparameterTrial:
    """Single hyperparameter optimization trial"""
    trial_id: int
    config: ColabTrainingConfig
    performance_score: float
    training_time: float
    converged: bool
    final_correlation: float
    best_correlation: float
    epochs_completed: int
    additional_metrics: Dict[str, float]


class BayesianOptimizer:
    """Bayesian optimization for efficient hyperparameter search"""

    def __init__(self, search_space: HyperparameterSpace, n_initial_points: int = 5):
        self.search_space = search_space
        self.n_initial_points = n_initial_points
        self.trials_history = []
        self.gp_model = None
        self.scaler = StandardScaler()
        self.best_trial = None

    def optimize(self, objective_function, n_trials: int = 20) -> Tuple[ColabTrainingConfig, List[HyperparameterTrial]]:
        """Execute Bayesian optimization"""

        print(f"🔬 Bayesian Hyperparameter Optimization ({n_trials} trials)")
        print("=" * 60)

        # Phase 1: Random exploration
        print(f"Phase 1: Random exploration ({self.n_initial_points} trials)")
        for i in range(self.n_initial_points):
            print(f"\n🎲 Random Trial {i+1}/{self.n_initial_points}")
            config = self._sample_random_config()
            trial = self._evaluate_config(config, objective_function, i+1)
            self.trials_history.append(trial)

        # Phase 2: Bayesian optimization
        if n_trials > self.n_initial_points:
            self._fit_gaussian_process()

            print(f"\nPhase 2: Bayesian optimization ({n_trials - self.n_initial_points} trials)")
            for i in range(self.n_initial_points, n_trials):
                print(f"\n🎯 Bayesian Trial {i+1}/{n_trials}")
                config = self._suggest_next_config()
                trial = self._evaluate_config(config, objective_function, i+1)
                self.trials_history.append(trial)
                self._update_gaussian_process()

        # Find best configuration
        self.best_trial = max(self.trials_history, key=lambda t: t.performance_score)

        print(f"\n✅ Optimization complete!")
        print(f"   Best performance: {self.best_trial.performance_score:.4f}")
        print(f"   Best correlation: {self.best_trial.best_correlation:.4f}")

        return self.best_trial.config, self.trials_history

    def _sample_random_config(self) -> ColabTrainingConfig:
        """Sample random configuration from search space"""
        return ColabTrainingConfig(
            learning_rate=np.random.uniform(*self.search_space.learning_rate),
            batch_size=np.random.choice(self.search_space.batch_size),
            hidden_dims=np.random.choice(self.search_space.hidden_dims, p=None).copy(),
            dropout_rates=np.random.choice(self.search_space.dropout_rates, p=None).copy(),
            weight_decay=np.random.uniform(*self.search_space.weight_decay),
            optimizer=np.random.choice(self.search_space.optimizer_type),
            scheduler=np.random.choice(self.search_space.scheduler_type),
            epochs=15,  # Quick evaluation
            device='cuda' if torch.cuda.is_available() else 'cpu',
            mixed_precision=True,
            early_stopping_patience=5
        )

    def _evaluate_config(self, config: ColabTrainingConfig, objective_function, trial_id: int) -> HyperparameterTrial:
        """Evaluate a hyperparameter configuration"""
        try:
            start_time = time.time()
            results = objective_function(config)
            training_time = time.time() - start_time

            # Calculate composite performance score
            performance_score = self._calculate_performance_score(results)

            trial = HyperparameterTrial(
                trial_id=trial_id,
                config=config,
                performance_score=performance_score,
                training_time=training_time,
                converged=results.get('converged', False),
                final_correlation=results.get('final_correlation', 0.0),
                best_correlation=results.get('best_correlation', 0.0),
                epochs_completed=results.get('epochs_completed', 0),
                additional_metrics=results.get('additional_metrics', {})
            )

            print(f"   Performance: {performance_score:.4f}")
            print(f"   Correlation: {trial.best_correlation:.4f}")
            print(f"   Time: {training_time:.1f}s")

            return trial

        except Exception as e:
            print(f"   ❌ Trial failed: {e}")
            # Return failed trial
            return HyperparameterTrial(
                trial_id=trial_id,
                config=config,
                performance_score=0.0,
                training_time=0.0,
                converged=False,
                final_correlation=0.0,
                best_correlation=0.0,
                epochs_completed=0,
                additional_metrics={}
            )

    def _calculate_performance_score(self, results: Dict[str, Any]) -> float:
        """Calculate composite performance score"""
        correlation = results.get('best_correlation', 0.0)
        converged = results.get('converged', False)
        training_time = results.get('training_time', float('inf'))

        # Base score from correlation
        score = correlation

        # Bonus for convergence
        if converged:
            score += 0.05

        # Penalty for long training time
        time_penalty = min(0.1, training_time / 1800)  # Penalty up to 0.1 for 30+ minutes
        score -= time_penalty

        # Bonus for efficiency (high correlation per time)
        if training_time > 0:
            efficiency_bonus = min(0.05, correlation / (training_time / 60))  # Correlation per minute
            score += efficiency_bonus

        return max(0.0, score)

    def _config_to_vector(self, config: ColabTrainingConfig) -> np.ndarray:
        """Convert configuration to numerical vector for GP"""
        vector = [
            np.log10(config.learning_rate),  # Log scale
            config.batch_size,
            len(config.hidden_dims),  # Architecture complexity
            np.mean(config.hidden_dims),  # Average layer size
            np.mean(config.dropout_rates),  # Average dropout
            np.log10(config.weight_decay),  # Log scale
            self.search_space.optimizer_type.index(config.optimizer),
            self.search_space.scheduler_type.index(config.scheduler) if config.scheduler != 'none' else -1
        ]
        return np.array(vector)

    def _fit_gaussian_process(self):
        """Fit Gaussian Process model to trial history"""
        if len(self.trials_history) < 2:
            return

        # Prepare data
        X = np.array([self._config_to_vector(trial.config) for trial in self.trials_history])
        y = np.array([trial.performance_score for trial in self.trials_history])

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Fit GP model
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=42
        )
        self.gp_model.fit(X_scaled, y)

    def _update_gaussian_process(self):
        """Update GP model with new trial data"""
        self._fit_gaussian_process()

    def _suggest_next_config(self) -> ColabTrainingConfig:
        """Suggest next configuration using acquisition function"""
        if self.gp_model is None:
            return self._sample_random_config()

        # Generate candidate configurations
        n_candidates = 100
        candidates = [self._sample_random_config() for _ in range(n_candidates)]

        # Evaluate acquisition function
        best_acquisition = -float('inf')
        best_config = None

        for config in candidates:
            acquisition = self._acquisition_function(config)
            if acquisition > best_acquisition:
                best_acquisition = acquisition
                best_config = config

        return best_config or self._sample_random_config()

    def _acquisition_function(self, config: ColabTrainingConfig) -> float:
        """Expected Improvement acquisition function"""
        if self.gp_model is None:
            return 0.0

        try:
            x = self._config_to_vector(config).reshape(1, -1)
            x_scaled = self.scaler.transform(x)

            mu, sigma = self.gp_model.predict(x_scaled, return_std=True)

            # Best observed value
            best_y = max(trial.performance_score for trial in self.trials_history)

            # Expected Improvement
            xi = 0.01  # Exploration parameter
            improvement = mu - best_y - xi

            if sigma > 0:
                z = improvement / sigma
                ei = improvement * self._normal_cdf(z) + sigma * self._normal_pdf(z)
            else:
                ei = 0.0

            return ei[0] if hasattr(ei, '__getitem__') else ei

        except:
            return 0.0

    def _normal_cdf(self, x):
        """Standard normal CDF approximation"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

    def _normal_pdf(self, x):
        """Standard normal PDF"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


class GridSearchOptimizer:
    """Grid search optimizer for systematic exploration"""

    def __init__(self, search_space: HyperparameterSpace):
        self.search_space = search_space
        self.trials_history = []

    def optimize(self, objective_function, max_trials: int = 50) -> Tuple[ColabTrainingConfig, List[HyperparameterTrial]]:
        """Execute grid search optimization"""

        print(f"📊 Grid Search Optimization (max {max_trials} trials)")
        print("=" * 60)

        # Generate grid configurations
        grid_configs = self._generate_grid_configurations()

        # Limit to max_trials
        if len(grid_configs) > max_trials:
            # Sample uniformly from grid
            indices = np.linspace(0, len(grid_configs)-1, max_trials, dtype=int)
            grid_configs = [grid_configs[i] for i in indices]

        print(f"Evaluating {len(grid_configs)} configurations...")

        # Evaluate each configuration
        for i, config in enumerate(grid_configs):
            print(f"\n🔄 Grid Trial {i+1}/{len(grid_configs)}")
            trial = self._evaluate_config(config, objective_function, i+1)
            self.trials_history.append(trial)

        # Find best configuration
        best_trial = max(self.trials_history, key=lambda t: t.performance_score)

        print(f"\n✅ Grid search complete!")
        print(f"   Best performance: {best_trial.performance_score:.4f}")

        return best_trial.config, self.trials_history

    def _generate_grid_configurations(self) -> List[ColabTrainingConfig]:
        """Generate grid of configurations"""
        configs = []

        # Create parameter combinations
        lr_values = np.logspace(np.log10(self.search_space.learning_rate[0]),
                               np.log10(self.search_space.learning_rate[1]), 5)
        wd_values = np.logspace(np.log10(self.search_space.weight_decay[0]),
                               np.log10(self.search_space.weight_decay[1]), 3)

        # Generate combinations
        for lr in lr_values:
            for batch_size in self.search_space.batch_size[:2]:  # Limit batch sizes
                for hidden_dims in self.search_space.hidden_dims[:3]:  # Limit architectures
                    for wd in wd_values:
                        config = ColabTrainingConfig(
                            learning_rate=lr,
                            batch_size=batch_size,
                            hidden_dims=hidden_dims.copy(),
                            weight_decay=wd,
                            epochs=10,  # Quick evaluation
                            device='cuda' if torch.cuda.is_available() else 'cpu',
                            mixed_precision=True,
                            early_stopping_patience=3
                        )
                        configs.append(config)

        return configs

    def _evaluate_config(self, config: ColabTrainingConfig, objective_function, trial_id: int) -> HyperparameterTrial:
        """Evaluate configuration (similar to Bayesian optimizer)"""
        try:
            start_time = time.time()
            results = objective_function(config)
            training_time = time.time() - start_time

            trial = HyperparameterTrial(
                trial_id=trial_id,
                config=config,
                performance_score=results.get('best_correlation', 0.0),
                training_time=training_time,
                converged=results.get('converged', False),
                final_correlation=results.get('final_correlation', 0.0),
                best_correlation=results.get('best_correlation', 0.0),
                epochs_completed=results.get('epochs_completed', 0),
                additional_metrics=results.get('additional_metrics', {})
            )

            print(f"   Correlation: {trial.best_correlation:.4f}")
            print(f"   Time: {training_time:.1f}s")

            return trial

        except Exception as e:
            print(f"   ❌ Trial failed: {e}")
            return HyperparameterTrial(
                trial_id=trial_id,
                config=config,
                performance_score=0.0,
                training_time=0.0,
                converged=False,
                final_correlation=0.0,
                best_correlation=0.0,
                epochs_completed=0,
                additional_metrics={}
            )


class AdvancedHyperparameterOptimizer:
    """Advanced hyperparameter optimizer with multiple strategies"""

    def __init__(self, save_dir: str = "/tmp/claude/hyperparameter_optimization"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.search_space = HyperparameterSpace()
        self.optimization_history = {}
        self.best_configs = {}

    def optimize_hyperparameters(
        self,
        train_loader,
        val_loader,
        optimization_strategy: str = 'bayesian',
        n_trials: int = 20
    ) -> Tuple[ColabTrainingConfig, Dict[str, Any]]:
        """Execute comprehensive hyperparameter optimization"""

        print(f"🎯 Advanced Hyperparameter Optimization")
        print(f"Strategy: {optimization_strategy}")
        print(f"Trials: {n_trials}")
        print("=" * 60)

        # Define objective function
        def objective_function(config: ColabTrainingConfig) -> Dict[str, Any]:
            return self._evaluate_configuration(config, train_loader, val_loader)

        # Execute optimization based on strategy
        if optimization_strategy == 'bayesian':
            optimizer = BayesianOptimizer(self.search_space)
            best_config, trials = optimizer.optimize(objective_function, n_trials)

        elif optimization_strategy == 'grid':
            optimizer = GridSearchOptimizer(self.search_space)
            best_config, trials = optimizer.optimize(objective_function, n_trials)

        elif optimization_strategy == 'random':
            best_config, trials = self._random_search(objective_function, n_trials)

        elif optimization_strategy == 'multi_stage':
            best_config, trials = self._multi_stage_optimization(objective_function, n_trials)

        else:
            raise ValueError(f"Unknown optimization strategy: {optimization_strategy}")

        # Store results
        self.optimization_history[optimization_strategy] = trials
        self.best_configs[optimization_strategy] = best_config

        # Generate optimization report
        optimization_report = self._generate_optimization_report(optimization_strategy, trials, best_config)

        # Visualize results
        self._visualize_optimization_results(optimization_strategy, trials)

        return best_config, optimization_report

    def _evaluate_configuration(
        self,
        config: ColabTrainingConfig,
        train_loader,
        val_loader
    ) -> Dict[str, Any]:
        """Evaluate a single hyperparameter configuration"""

        try:
            # Create training pipeline
            pipeline = GPUTrainingPipeline(config)

            # Execute training
            training_results = pipeline.train(train_loader, val_loader, save_checkpoints=False)

            # Extract metrics
            results = {
                'best_correlation': training_results['training_summary']['best_correlation'],
                'final_correlation': training_results['training_summary']['final_correlation'],
                'converged': training_results['performance_metrics']['target_achieved'],
                'epochs_completed': training_results['training_summary']['epochs_completed'],
                'training_time': training_results['performance_metrics']['total_training_time'],
                'additional_metrics': {
                    'final_train_loss': training_results['training_summary']['final_train_loss'],
                    'final_val_loss': training_results['training_summary']['final_val_loss'],
                    'average_epoch_time': training_results['performance_metrics']['average_epoch_time']
                }
            }

            # Clean up GPU memory
            del pipeline
            torch.cuda.empty_cache()

            return results

        except Exception as e:
            print(f"Configuration evaluation failed: {e}")
            return {
                'best_correlation': 0.0,
                'final_correlation': 0.0,
                'converged': False,
                'epochs_completed': 0,
                'training_time': 0.0,
                'additional_metrics': {}
            }

    def _random_search(self, objective_function, n_trials: int) -> Tuple[ColabTrainingConfig, List[HyperparameterTrial]]:
        """Execute random search optimization"""

        print(f"🎲 Random Search Optimization ({n_trials} trials)")
        trials = []

        for i in range(n_trials):
            print(f"\n🔄 Random Trial {i+1}/{n_trials}")

            # Sample random configuration
            config = ColabTrainingConfig(
                learning_rate=np.random.uniform(*self.search_space.learning_rate),
                batch_size=np.random.choice(self.search_space.batch_size),
                hidden_dims=np.random.choice(self.search_space.hidden_dims).copy(),
                dropout_rates=np.random.choice(self.search_space.dropout_rates).copy(),
                weight_decay=np.random.uniform(*self.search_space.weight_decay),
                optimizer=np.random.choice(self.search_space.optimizer_type),
                epochs=15,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                mixed_precision=True,
                early_stopping_patience=5
            )

            # Evaluate configuration
            results = objective_function(config)

            trial = HyperparameterTrial(
                trial_id=i+1,
                config=config,
                performance_score=results['best_correlation'],
                training_time=results['training_time'],
                converged=results['converged'],
                final_correlation=results['final_correlation'],
                best_correlation=results['best_correlation'],
                epochs_completed=results['epochs_completed'],
                additional_metrics=results['additional_metrics']
            )

            trials.append(trial)
            print(f"   Correlation: {trial.best_correlation:.4f}")

        best_trial = max(trials, key=lambda t: t.performance_score)
        return best_trial.config, trials

    def _multi_stage_optimization(self, objective_function, n_trials: int) -> Tuple[ColabTrainingConfig, List[HyperparameterTrial]]:
        """Multi-stage optimization: coarse → fine → refinement"""

        print(f"🎯 Multi-stage Optimization ({n_trials} trials)")

        all_trials = []

        # Stage 1: Coarse grid search (30% of trials)
        stage1_trials = max(1, n_trials // 3)
        print(f"\nStage 1: Coarse exploration ({stage1_trials} trials)")

        coarse_optimizer = GridSearchOptimizer(self.search_space)
        _, stage1_results = coarse_optimizer.optimize(objective_function, stage1_trials)
        all_trials.extend(stage1_results)

        # Stage 2: Focused random search around best results (40% of trials)
        stage2_trials = max(1, int(n_trials * 0.4))
        print(f"\nStage 2: Focused search ({stage2_trials} trials)")

        # Get top 3 configurations from stage 1
        top_configs = sorted(stage1_results, key=lambda t: t.performance_score, reverse=True)[:3]

        for i in range(stage2_trials):
            # Select base configuration
            base_config = np.random.choice(top_configs).config

            # Add noise to create variant
            config = self._create_config_variant(base_config)

            results = objective_function(config)
            trial = HyperparameterTrial(
                trial_id=len(all_trials) + 1,
                config=config,
                performance_score=results['best_correlation'],
                training_time=results['training_time'],
                converged=results['converged'],
                final_correlation=results['final_correlation'],
                best_correlation=results['best_correlation'],
                epochs_completed=results['epochs_completed'],
                additional_metrics=results['additional_metrics']
            )
            all_trials.append(trial)

        # Stage 3: Bayesian optimization refinement (30% of trials)
        stage3_trials = n_trials - len(all_trials)
        if stage3_trials > 0:
            print(f"\nStage 3: Bayesian refinement ({stage3_trials} trials)")

            bayesian_optimizer = BayesianOptimizer(self.search_space)
            # Initialize with existing trials
            bayesian_optimizer.trials_history = all_trials
            _, stage3_results = bayesian_optimizer.optimize(objective_function, len(all_trials) + stage3_trials)

            # Add only new trials
            all_trials.extend(stage3_results[len(all_trials):])

        best_trial = max(all_trials, key=lambda t: t.performance_score)
        return best_trial.config, all_trials

    def _create_config_variant(self, base_config: ColabTrainingConfig) -> ColabTrainingConfig:
        """Create a variant of a configuration with small modifications"""
        variant = ColabTrainingConfig(
            learning_rate=base_config.learning_rate * np.random.uniform(0.5, 2.0),
            batch_size=base_config.batch_size,
            hidden_dims=base_config.hidden_dims.copy(),
            dropout_rates=base_config.dropout_rates.copy(),
            weight_decay=base_config.weight_decay * np.random.uniform(0.1, 10.0),
            optimizer=base_config.optimizer,
            scheduler=base_config.scheduler,
            epochs=base_config.epochs,
            device=base_config.device,
            mixed_precision=base_config.mixed_precision
        )

        # Clamp learning rate
        variant.learning_rate = np.clip(variant.learning_rate, *self.search_space.learning_rate)
        variant.weight_decay = np.clip(variant.weight_decay, *self.search_space.weight_decay)

        return variant

    def _generate_optimization_report(
        self,
        strategy: str,
        trials: List[HyperparameterTrial],
        best_config: ColabTrainingConfig
    ) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""

        if not trials:
            return {}

        # Calculate statistics
        performance_scores = [t.performance_score for t in trials]
        correlations = [t.best_correlation for t in trials]
        training_times = [t.training_time for t in trials]

        report = {
            'optimization_strategy': strategy,
            'total_trials': len(trials),
            'best_performance': max(performance_scores),
            'average_performance': np.mean(performance_scores),
            'performance_std': np.std(performance_scores),
            'best_correlation': max(correlations),
            'average_correlation': np.mean(correlations),
            'correlation_std': np.std(correlations),
            'total_optimization_time': sum(training_times),
            'average_trial_time': np.mean(training_times),
            'convergence_rate': sum(1 for t in trials if t.converged) / len(trials),
            'best_configuration': asdict(best_config),
            'improvement_over_baseline': max(correlations) - min(correlations) if correlations else 0,
            'efficiency_score': max(correlations) / (sum(training_times) / 3600)  # Correlation per hour
        }

        # Parameter importance analysis
        report['parameter_analysis'] = self._analyze_parameter_importance(trials)

        # Save report
        report_path = self.save_dir / f"optimization_report_{strategy}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Optimization Report:")
        print(f"   Best correlation: {report['best_correlation']:.4f}")
        print(f"   Average correlation: {report['average_correlation']:.4f}")
        print(f"   Convergence rate: {report['convergence_rate']:.1%}")
        print(f"   Total time: {report['total_optimization_time']:.1f}s")

        return report

    def _analyze_parameter_importance(self, trials: List[HyperparameterTrial]) -> Dict[str, float]:
        """Analyze hyperparameter importance"""
        if len(trials) < 5:
            return {}

        # Analyze learning rate impact
        lr_correlation = []
        for trial in trials:
            lr_correlation.append((trial.config.learning_rate, trial.best_correlation))

        lr_impact = np.corrcoef([x[0] for x in lr_correlation], [x[1] for x in lr_correlation])[0, 1]

        # Analyze batch size impact
        batch_sizes = list(set(t.config.batch_size for t in trials))
        batch_performance = {}
        for batch_size in batch_sizes:
            batch_trials = [t for t in trials if t.config.batch_size == batch_size]
            if batch_trials:
                batch_performance[batch_size] = np.mean([t.best_correlation for t in batch_trials])

        return {
            'learning_rate_correlation': float(lr_impact) if not np.isnan(lr_impact) else 0.0,
            'batch_size_performance': batch_performance,
            'architecture_analysis': self._analyze_architecture_impact(trials)
        }

    def _analyze_architecture_impact(self, trials: List[HyperparameterTrial]) -> Dict[str, Any]:
        """Analyze neural network architecture impact"""
        arch_performance = {}

        for trial in trials:
            arch_key = str(trial.config.hidden_dims)
            if arch_key not in arch_performance:
                arch_performance[arch_key] = []
            arch_performance[arch_key].append(trial.best_correlation)

        # Calculate average performance per architecture
        arch_summary = {}
        for arch, performances in arch_performance.items():
            arch_summary[arch] = {
                'mean_correlation': np.mean(performances),
                'std_correlation': np.std(performances),
                'trial_count': len(performances)
            }

        return arch_summary

    def _visualize_optimization_results(self, strategy: str, trials: List[HyperparameterTrial]):
        """Visualize optimization results"""
        if len(trials) < 2:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Optimization progress
        trial_ids = [t.trial_id for t in trials]
        correlations = [t.best_correlation for t in trials]

        ax1.plot(trial_ids, correlations, 'b-o', alpha=0.7)
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Best Correlation')
        ax1.set_title(f'Optimization Progress - {strategy.title()}')
        ax1.grid(True, alpha=0.3)

        # Running best
        running_best = []
        best_so_far = 0
        for corr in correlations:
            best_so_far = max(best_so_far, corr)
            running_best.append(best_so_far)

        ax1.plot(trial_ids, running_best, 'r-', linewidth=2, alpha=0.8, label='Running Best')
        ax1.legend()

        # Performance distribution
        ax2.hist(correlations, bins=min(20, len(correlations)//2 + 1), alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Correlation')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Performance Distribution')
        ax2.axvline(np.mean(correlations), color='red', linestyle='--', label=f'Mean: {np.mean(correlations):.3f}')
        ax2.legend()

        # Parameter vs Performance (Learning Rate)
        learning_rates = [t.config.learning_rate for t in trials]
        ax3.scatter(learning_rates, correlations, alpha=0.6)
        ax3.set_xlabel('Learning Rate')
        ax3.set_ylabel('Best Correlation')
        ax3.set_title('Learning Rate vs Performance')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)

        # Training time vs Performance
        training_times = [t.training_time for t in trials]
        scatter = ax4.scatter(training_times, correlations, c=trial_ids, alpha=0.6, cmap='viridis')
        ax4.set_xlabel('Training Time (seconds)')
        ax4.set_ylabel('Best Correlation')
        ax4.set_title('Training Time vs Performance')
        plt.colorbar(scatter, ax=ax4, label='Trial Order')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save visualization
        viz_path = self.save_dir / f"optimization_visualization_{strategy}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📈 Optimization visualization saved: {viz_path}")


if __name__ == "__main__":
    # Demo the hyperparameter optimizer
    print("🧪 Testing Advanced Hyperparameter Optimizer")

    optimizer = AdvancedHyperparameterOptimizer()

    # Mock objective function for testing
    def mock_objective(config):
        # Simulate training results
        lr_factor = 1.0 - abs(np.log10(config.learning_rate) + 3.0) / 2.0  # Optimal around 1e-3
        batch_factor = 1.0 - abs(config.batch_size - 64) / 64.0  # Optimal around 64
        correlation = np.clip(0.7 + lr_factor * 0.2 + batch_factor * 0.1 + np.random.normal(0, 0.02), 0.5, 0.98)

        return {
            'best_correlation': correlation,
            'final_correlation': correlation - 0.01,
            'converged': correlation > 0.85,
            'epochs_completed': 15,
            'training_time': np.random.uniform(30, 120),
            'additional_metrics': {}
        }

    # Test different strategies
    for strategy in ['random', 'bayesian']:
        print(f"\n🔬 Testing {strategy} optimization...")
        config, report = optimizer.optimize_hyperparameters(
            None, None,  # Mock data loaders
            optimization_strategy=strategy,
            n_trials=10
        )
        print(f"✅ {strategy} optimization complete!")

    print(f"\n✅ All tests completed!")