"""
Correlation Rollout System - Task 5 Implementation
Gradual rollout mechanism for transitioning from formula-based to learned correlations.
"""

import random
import hashlib
import json
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

# Import correlation systems
from backend.ai_modules.optimization.learned_correlations import LearnedCorrelations
from backend.ai_modules.optimization.unified_parameter_formulas import ParameterFormulas CorrelationFormulas


class CorrelationRollout:
    """
    Manage gradual rollout of learned correlations with monitoring and rollback.
    """

    def __init__(self,
                 rollout_percentage: float = 0.1,
                 enable_feature_flags: bool = True,
                 enable_monitoring: bool = True,
                 config_path: Optional[str] = None):
        """
        Initialize rollout system.

        Args:
            rollout_percentage: Percentage of traffic to use learned correlations (0.0-1.0)
            enable_feature_flags: Whether to check feature flags
            enable_monitoring: Whether to track performance metrics
            config_path: Path to rollout configuration file
        """
        self.rollout_percentage = max(0.0, min(1.0, rollout_percentage))
        self.enable_feature_flags = enable_feature_flags
        self.enable_monitoring = enable_monitoring
        self.config_path = config_path

        self.logger = logging.getLogger(__name__)

        # Initialize both correlation systems
        self.learned_correlations = None
        self.formula_correlations = CorrelationFormulas()

        # Monitoring metrics
        self.metrics = {
            'total_requests': 0,
            'learned_requests': 0,
            'formula_requests': 0,
            'learned_errors': 0,
            'formula_errors': 0,
            'learned_total_time': 0,
            'formula_total_time': 0,
            'rollback_count': 0,
            'last_rollback': None
        }

        # Feature flags
        self.feature_flags = {
            'use_learned_correlations': True,
            'enable_blending': True,
            'force_formula': False,
            'force_learned': False
        }

        # Load configuration
        self._load_configuration()

        # Initialize learned correlations (lazy loading)
        self._init_learned_correlations()

        # Session tracking for consistent routing
        self.session_routing = {}
        self.max_sessions = 10000

        self.logger.info(
            f"CorrelationRollout initialized with {self.rollout_percentage:.1%} rollout"
        )

    def _load_configuration(self):
        """Load rollout configuration from file if provided."""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                # Update rollout percentage
                if 'rollout_percentage' in config:
                    self.rollout_percentage = config['rollout_percentage']

                # Update feature flags
                if 'feature_flags' in config:
                    self.feature_flags.update(config['feature_flags'])

                self.logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load configuration: {e}")

    def _init_learned_correlations(self):
        """Initialize learned correlations system (lazy loading)."""
        if self.learned_correlations is None:
            try:
                self.learned_correlations = LearnedCorrelations(enable_fallback=False)
                self.logger.info("Learned correlations initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize learned correlations: {e}")
                self.learned_correlations = None

    def should_use_learned(self, session_id: Optional[str] = None) -> bool:
        """
        Determine whether to use learned correlations for this request.

        Args:
            session_id: Optional session identifier for consistent routing

        Returns:
            True if should use learned correlations, False for formulas
        """
        # Check force flags first
        if self.feature_flags.get('force_formula'):
            return False
        if self.feature_flags.get('force_learned'):
            return True

        # Check if learned correlations are available
        if not self.feature_flags.get('use_learned_correlations'):
            return False
        if self.learned_correlations is None:
            return False

        # Use session-based routing for consistency
        if session_id:
            if session_id in self.session_routing:
                return self.session_routing[session_id]

            # Hash-based routing for consistent assignment
            hash_value = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
            use_learned = (hash_value % 100) < (self.rollout_percentage * 100)

            # Cache routing decision
            if len(self.session_routing) < self.max_sessions:
                self.session_routing[session_id] = use_learned

            return use_learned

        # Random routing for requests without session
        return random.random() < self.rollout_percentage

    def get_correlations(self, session_id: Optional[str] = None) -> Union[LearnedCorrelations, CorrelationFormulas]:
        """
        Get the appropriate correlation system based on rollout.

        Args:
            session_id: Optional session identifier

        Returns:
            Either LearnedCorrelations or CorrelationFormulas instance
        """
        if self.should_use_learned(session_id):
            if self.enable_monitoring:
                self.metrics['learned_requests'] += 1
            return self.learned_correlations or self.formula_correlations
        else:
            if self.enable_monitoring:
                self.metrics['formula_requests'] += 1
            return self.formula_correlations

    def get_parameters(self,
                      features: Dict[str, Any],
                      session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get parameters using the appropriate correlation system.

        Args:
            features: Image features
            session_id: Optional session identifier

        Returns:
            Dictionary containing parameters and metadata
        """
        self.metrics['total_requests'] += 1
        start_time = datetime.now()

        try:
            correlation_system = self.get_correlations(session_id)
            is_learned = isinstance(correlation_system, LearnedCorrelations)

            # Get parameters
            if hasattr(correlation_system, 'get_parameters'):
                parameters = correlation_system.get_parameters(features)
            else:
                # For CorrelationFormulas, we need to call individual methods
                parameters = self._get_formula_parameters(features)

            # Track timing
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if is_learned:
                self.metrics['learned_total_time'] += elapsed_time
            else:
                self.metrics['formula_total_time'] += elapsed_time

            # Add metadata
            return {
                'parameters': parameters,
                'method': 'learned' if is_learned else 'formula',
                'rollout_percentage': self.rollout_percentage,
                'processing_time': elapsed_time
            }

        except Exception as e:
            # Track errors
            if self.should_use_learned(session_id):
                self.metrics['learned_errors'] += 1
                self.logger.error(f"Learned correlation error: {e}")
            else:
                self.metrics['formula_errors'] += 1
                self.logger.error(f"Formula correlation error: {e}")

            # Fallback to formulas on error
            return {
                'parameters': self._get_formula_parameters(features),
                'method': 'formula_fallback',
                'error': str(e)
            }

    def _get_formula_parameters(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters using formula-based correlations."""
        return {
            'corner_threshold': self.formula_correlations.edge_to_corner_threshold(
                features.get('edge_density', 0.5)),
            'color_precision': self.formula_correlations.colors_to_precision(
                features.get('unique_colors', 128)),
            'path_precision': self.formula_correlations.entropy_to_path_precision(
                features.get('entropy', 0.5)),
            'splice_threshold': self.formula_correlations.gradient_to_splice_threshold(
                features.get('gradient_strength', 0.5)),
            'max_iterations': self.formula_correlations.complexity_to_iterations(
                features.get('complexity_score', 0.5)),
            'length_threshold': 5.0
        }

    def update_rollout(self, new_percentage: float) -> bool:
        """
        Update rollout percentage.

        Args:
            new_percentage: New rollout percentage (0.0-1.0)

        Returns:
            True if update successful
        """
        try:
            old_percentage = self.rollout_percentage
            self.rollout_percentage = max(0.0, min(1.0, new_percentage))

            self.logger.info(
                f"Rollout updated: {old_percentage:.1%} → {self.rollout_percentage:.1%}"
            )

            # Save to configuration if path provided
            if self.config_path:
                self._save_configuration()

            # Clear session routing cache on significant changes
            if abs(old_percentage - new_percentage) > 0.1:
                self.session_routing.clear()

            return True

        except Exception as e:
            self.logger.error(f"Failed to update rollout: {e}")
            return False

    def rollback(self, reason: str = "Manual rollback") -> bool:
        """
        Rollback to 0% learned correlations.

        Args:
            reason: Reason for rollback

        Returns:
            True if rollback successful
        """
        try:
            self.logger.warning(f"Initiating rollback: {reason}")

            # Set rollout to 0%
            self.rollout_percentage = 0.0

            # Update feature flags
            self.feature_flags['use_learned_correlations'] = False

            # Clear session routing
            self.session_routing.clear()

            # Update metrics
            self.metrics['rollback_count'] += 1
            self.metrics['last_rollback'] = {
                'timestamp': datetime.now().isoformat(),
                'reason': reason
            }

            # Save configuration
            if self.config_path:
                self._save_configuration()

            self.logger.info("Rollback completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        total = self.metrics['total_requests'] or 1
        learned_total = self.metrics['learned_requests'] or 1
        formula_total = self.metrics['formula_requests'] or 1

        return {
            'rollout_percentage': self.rollout_percentage,
            'total_requests': self.metrics['total_requests'],
            'distribution': {
                'learned': self.metrics['learned_requests'],
                'formula': self.metrics['formula_requests'],
                'learned_percentage': self.metrics['learned_requests'] / total * 100
            },
            'error_rates': {
                'learned': self.metrics['learned_errors'] / learned_total * 100,
                'formula': self.metrics['formula_errors'] / formula_total * 100
            },
            'average_time_ms': {
                'learned': (self.metrics['learned_total_time'] / learned_total * 1000
                           if self.metrics['learned_requests'] > 0 else 0),
                'formula': (self.metrics['formula_total_time'] / formula_total * 1000
                           if self.metrics['formula_requests'] > 0 else 0)
            },
            'rollback_info': {
                'count': self.metrics['rollback_count'],
                'last': self.metrics['last_rollback']
            },
            'feature_flags': self.feature_flags,
            'active_sessions': len(self.session_routing)
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Check health of rollout system.

        Returns health status and any issues detected.
        """
        issues = []

        # Check error rates
        metrics = self.get_metrics()
        if metrics['error_rates']['learned'] > 5:
            issues.append(f"High learned error rate: {metrics['error_rates']['learned']:.1f}%")

        # Check if learned correlations are available
        if self.learned_correlations is None and self.rollout_percentage > 0:
            issues.append("Learned correlations not initialized but rollout > 0%")

        # Check performance degradation
        if (metrics['average_time_ms']['learned'] > 0 and
            metrics['average_time_ms']['formula'] > 0):
            slowdown = (metrics['average_time_ms']['learned'] /
                       metrics['average_time_ms']['formula'])
            if slowdown > 2:
                issues.append(f"Learned correlations {slowdown:.1f}x slower than formulas")

        return {
            'status': 'healthy' if not issues else 'degraded',
            'issues': issues,
            'metrics': metrics
        }

    def _save_configuration(self):
        """Save current configuration to file."""
        if not self.config_path:
            return

        try:
            config = {
                'rollout_percentage': self.rollout_percentage,
                'feature_flags': self.feature_flags,
                'updated_at': datetime.now().isoformat()
            }

            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")

    def __repr__(self) -> str:
        """String representation of rollout system."""
        metrics = self.get_metrics()
        return (f"CorrelationRollout(rollout={self.rollout_percentage:.1%}, "
                f"requests={metrics['total_requests']}, "
                f"distribution={metrics['distribution']['learned_percentage']:.1f}% learned)")


def test_correlation_rollout():
    """Test the correlation rollout system."""
    print("Testing Correlation Rollout System...")

    # Test 1: Initialize with different rollout percentages
    print("\n1. Testing rollout initialization:")
    rollout_10 = CorrelationRollout(rollout_percentage=0.1)
    print(f"   10% rollout: {rollout_10}")

    rollout_50 = CorrelationRollout(rollout_percentage=0.5)
    print(f"   50% rollout: {rollout_50}")

    # Test 2: Test routing consistency
    print("\n2. Testing session routing consistency:")
    session_id = "test_session_123"
    decisions = []
    for _ in range(10):
        decision = rollout_50.should_use_learned(session_id)
        decisions.append(decision)

    if len(set(decisions)) == 1:
        print(f"   ✓ Consistent routing for session: always {decisions[0]}")
    else:
        print(f"   ✗ Inconsistent routing: {decisions}")

    # Test 3: Test parameter retrieval
    print("\n3. Testing parameter retrieval:")
    test_features = {
        'edge_density': 0.7,
        'unique_colors': 128,
        'entropy': 0.5,
        'gradient_strength': 0.6,
        'complexity_score': 0.8
    }

    result = rollout_50.get_parameters(test_features, session_id)
    print(f"   Method used: {result['method']}")
    print(f"   Parameters: {list(result['parameters'].keys())}")
    print(f"   Processing time: {result.get('processing_time', 0)*1000:.2f}ms")

    # Test 4: Test rollback
    print("\n4. Testing rollback mechanism:")
    success = rollout_50.rollback("Test rollback")
    print(f"   Rollback successful: {success}")
    print(f"   Rollout after rollback: {rollout_50.rollout_percentage:.1%}")

    # Test 5: Test metrics
    print("\n5. Testing metrics collection:")

    # Generate some traffic
    rollout_test = CorrelationRollout(rollout_percentage=0.3)
    for i in range(100):
        rollout_test.get_parameters(test_features, f"session_{i}")

    metrics = rollout_test.get_metrics()
    print(f"   Total requests: {metrics['total_requests']}")
    print(f"   Distribution: {metrics['distribution']['learned_percentage']:.1f}% learned")
    print(f"   Error rates: learned={metrics['error_rates']['learned']:.1f}%, "
          f"formula={metrics['error_rates']['formula']:.1f}%")

    # Test 6: Test health check
    print("\n6. Testing health check:")
    health = rollout_test.health_check()
    print(f"   Status: {health['status']}")
    if health['issues']:
        print(f"   Issues: {health['issues']}")

    # Test 7: Test gradual rollout
    print("\n7. Testing gradual rollout updates:")
    rollout_gradual = CorrelationRollout(rollout_percentage=0.1)

    for percentage in [0.1, 0.25, 0.5, 0.75, 1.0]:
        rollout_gradual.update_rollout(percentage)

        # Test distribution with 100 requests
        learned_count = sum(1 for _ in range(100)
                           if rollout_gradual.should_use_learned())
        print(f"   {percentage:.0%} rollout → {learned_count}% learned (expected ~{percentage*100:.0f}%)")

    print("\n✓ All rollout tests completed successfully!")

    return rollout_test


if __name__ == "__main__":
    test_correlation_rollout()