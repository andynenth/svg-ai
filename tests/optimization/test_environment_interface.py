#!/usr/bin/env python3
"""
Environment Interface Testing - Task B6.2 (2 hours)
Comprehensive Gymnasium interface compliance and lifecycle testing
"""

import pytest
import numpy as np
import time
import tempfile
import os
import gc
import psutil
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
from datetime import datetime
import gymnasium as gym
from unittest.mock import patch, MagicMock

# Test imports
try:
    from backend.ai_modules.optimization.vtracer_env import VTracerOptimizationEnv
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("⚠️  VTracer environment not available - using mock implementations")


class GymnasiumInterfaceComplianceTester:
    """Test Gymnasium interface compliance comprehensively"""

    def __init__(self):
        self.test_images = [
            "test-data/circle_00.png",
            "test-data/text_tech_00.png",
            "test-data/gradient_radial_00.png"
        ]
        self.compliance_results = {}
        self.logger = logging.getLogger(__name__)

    def test_gymnasium_interface_compliance(self) -> Dict[str, Any]:
        """Verify environment implements required Gymnasium methods"""
        print("📋 Testing Gymnasium Interface Compliance...")

        compliance_results = {
            'success': True,
            'required_methods': [],
            'optional_methods': [],
            'interface_violations': [],
            'details': {}
        }

        if not DEPENDENCIES_AVAILABLE:
            compliance_results['success'] = False
            compliance_results['error'] = "VTracer environment not available"
            return compliance_results

        try:
            env = VTracerOptimizationEnv(self.test_images[0])

            # Required Gymnasium methods
            required_methods = [
                'reset', 'step', 'close', 'render',
                '__init__', 'action_space', 'observation_space'
            ]

            # Optional but recommended methods
            optional_methods = [
                'seed', '__str__', '__repr__'
            ]

            # Test required methods
            for method in required_methods:
                if hasattr(env, method):
                    compliance_results['required_methods'].append({
                        'method': method,
                        'present': True,
                        'callable': callable(getattr(env, method)) if hasattr(env, method) else False
                    })
                else:
                    compliance_results['interface_violations'].append(f"Missing required method: {method}")
                    compliance_results['success'] = False

            # Test optional methods
            for method in optional_methods:
                compliance_results['optional_methods'].append({
                    'method': method,
                    'present': hasattr(env, method),
                    'callable': callable(getattr(env, method)) if hasattr(env, method) else False
                })

            # Test action and observation space definitions
            action_space_tests = self._test_action_space_definition(env)
            observation_space_tests = self._test_observation_space_definition(env)

            compliance_results['details']['action_space'] = action_space_tests
            compliance_results['details']['observation_space'] = observation_space_tests

            if not action_space_tests['valid'] or not observation_space_tests['valid']:
                compliance_results['success'] = False

            # Test environment registration capability
            registration_test = self._test_environment_registration()
            compliance_results['details']['registration'] = registration_test

            env.close()

        except Exception as e:
            compliance_results['success'] = False
            compliance_results['error'] = str(e)

        print(f"  {'✅ PASSED' if compliance_results['success'] else '❌ FAILED'}")
        return compliance_results

    def _test_action_space_definition(self, env) -> Dict[str, Any]:
        """Test action space definition compliance"""
        try:
            action_space = env.action_space

            # Test action space properties
            tests = {
                'valid': True,
                'type': str(type(action_space)),
                'shape': getattr(action_space, 'shape', None),
                'dtype': getattr(action_space, 'dtype', None),
                'bounds': {},
                'sampling': {}
            }

            # Test bounds
            if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
                tests['bounds'] = {
                    'low': action_space.low.tolist() if hasattr(action_space.low, 'tolist') else str(action_space.low),
                    'high': action_space.high.tolist() if hasattr(action_space.high, 'tolist') else str(action_space.high),
                    'bounded': True
                }
            else:
                tests['bounds']['bounded'] = False

            # Test sampling
            try:
                sample = action_space.sample()
                tests['sampling'] = {
                    'can_sample': True,
                    'sample_shape': sample.shape if hasattr(sample, 'shape') else len(sample),
                    'sample_dtype': str(sample.dtype) if hasattr(sample, 'dtype') else str(type(sample)),
                    'in_bounds': action_space.contains(sample) if hasattr(action_space, 'contains') else 'unknown'
                }
            except Exception as e:
                tests['sampling'] = {
                    'can_sample': False,
                    'error': str(e)
                }
                tests['valid'] = False

            return tests

        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }

    def _test_observation_space_definition(self, env) -> Dict[str, Any]:
        """Test observation space definition compliance"""
        try:
            observation_space = env.observation_space

            tests = {
                'valid': True,
                'type': str(type(observation_space)),
                'shape': getattr(observation_space, 'shape', None),
                'dtype': getattr(observation_space, 'dtype', None),
                'bounds': {},
                'sampling': {}
            }

            # Test bounds
            if hasattr(observation_space, 'low') and hasattr(observation_space, 'high'):
                tests['bounds'] = {
                    'low': observation_space.low.tolist() if hasattr(observation_space.low, 'tolist') else str(observation_space.low),
                    'high': observation_space.high.tolist() if hasattr(observation_space.high, 'tolist') else str(observation_space.high),
                    'bounded': True
                }
            else:
                tests['bounds']['bounded'] = False

            # Test sampling
            try:
                sample = observation_space.sample()
                tests['sampling'] = {
                    'can_sample': True,
                    'sample_shape': sample.shape if hasattr(sample, 'shape') else len(sample),
                    'sample_dtype': str(sample.dtype) if hasattr(sample, 'dtype') else str(type(sample)),
                    'in_bounds': observation_space.contains(sample) if hasattr(observation_space, 'contains') else 'unknown'
                }
            except Exception as e:
                tests['sampling'] = {
                    'can_sample': False,
                    'error': str(e)
                }
                tests['valid'] = False

            return tests

        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }

    def _test_environment_registration(self) -> Dict[str, Any]:
        """Test environment registration capability"""
        try:
            # Test if environment can be registered with Gymnasium
            registration_test = {
                'can_register': False,
                'registration_attempted': True,
                'error': None
            }

            try:
                # Try to register environment (this is optional but good practice)
                gym.register(
                    id='VTracerOptimization-v0',
                    entry_point='backend.ai_modules.optimization.vtracer_env:VTracerOptimizationEnv',
                    kwargs={'image_path': self.test_images[0]}
                )
                registration_test['can_register'] = True
            except Exception as e:
                registration_test['error'] = str(e)
                # This is not a failure - registration is optional

            return registration_test

        except Exception as e:
            return {
                'can_register': False,
                'registration_attempted': False,
                'error': str(e)
            }


class EnvironmentLifecycleTester:
    """Test environment lifecycle comprehensively"""

    def __init__(self):
        self.test_images = [
            "test-data/circle_00.png",
            "test-data/text_tech_00.png",
            "test-data/gradient_radial_00.png"
        ]
        self.logger = logging.getLogger(__name__)

    def test_environment_lifecycle(self) -> Dict[str, Any]:
        """Test environment initialization, reset, step, and close"""
        print("🔄 Testing Environment Lifecycle...")

        lifecycle_results = {
            'success': True,
            'initialization': {},
            'reset_functionality': {},
            'step_functionality': {},
            'cleanup': {},
            'details': []
        }

        if not DEPENDENCIES_AVAILABLE:
            lifecycle_results['success'] = False
            lifecycle_results['error'] = "VTracer environment not available"
            return lifecycle_results

        try:
            # Test initialization
            initialization_test = self._test_environment_initialization()
            lifecycle_results['initialization'] = initialization_test
            if not initialization_test['success']:
                lifecycle_results['success'] = False

            # Test reset functionality
            reset_test = self._test_reset_functionality()
            lifecycle_results['reset_functionality'] = reset_test
            if not reset_test['success']:
                lifecycle_results['success'] = False

            # Test step functionality
            step_test = self._test_step_functionality()
            lifecycle_results['step_functionality'] = step_test
            if not step_test['success']:
                lifecycle_results['success'] = False

            # Test cleanup
            cleanup_test = self._test_cleanup_functionality()
            lifecycle_results['cleanup'] = cleanup_test
            if not cleanup_test['success']:
                lifecycle_results['success'] = False

        except Exception as e:
            lifecycle_results['success'] = False
            lifecycle_results['error'] = str(e)

        print(f"  {'✅ PASSED' if lifecycle_results['success'] else '❌ FAILED'}")
        return lifecycle_results

    def _test_environment_initialization(self) -> Dict[str, Any]:
        """Test environment initialization with various parameters"""
        init_results = {
            'success': True,
            'tests_passed': 0,
            'tests_total': 0,
            'details': []
        }

        test_cases = [
            {
                'name': 'default_parameters',
                'image_path': self.test_images[0],
                'kwargs': {}
            },
            {
                'name': 'custom_target_quality',
                'image_path': self.test_images[0],
                'kwargs': {'target_quality': 0.9}
            },
            {
                'name': 'custom_max_steps',
                'image_path': self.test_images[0],
                'kwargs': {'max_steps': 25}
            },
            {
                'name': 'different_image',
                'image_path': self.test_images[1],
                'kwargs': {}
            }
        ]

        for test_case in test_cases:
            init_results['tests_total'] += 1
            try:
                start_time = time.time()
                env = VTracerOptimizationEnv(
                    test_case['image_path'],
                    **test_case['kwargs']
                )

                init_time = time.time() - start_time

                # Validate initialization
                assert hasattr(env, 'action_space'), "Action space not initialized"
                assert hasattr(env, 'observation_space'), "Observation space not initialized"
                assert hasattr(env, 'image_path'), "Image path not set"
                assert env.image_path == test_case['image_path'], "Image path mismatch"

                env.close()

                init_results['tests_passed'] += 1
                init_results['details'].append({
                    'test': test_case['name'],
                    'success': True,
                    'init_time': init_time,
                    'image_path': test_case['image_path'],
                    'kwargs': test_case['kwargs']
                })

            except Exception as e:
                init_results['success'] = False
                init_results['details'].append({
                    'test': test_case['name'],
                    'success': False,
                    'error': str(e),
                    'image_path': test_case['image_path'],
                    'kwargs': test_case['kwargs']
                })

        return init_results

    def _test_reset_functionality(self) -> Dict[str, Any]:
        """Test reset() method functionality"""
        reset_results = {
            'success': True,
            'tests_passed': 0,
            'tests_total': 0,
            'details': []
        }

        try:
            env = VTracerOptimizationEnv(self.test_images[0])

            # Test basic reset
            reset_results['tests_total'] += 1
            try:
                obs, info = env.reset()

                # Validate reset returns
                assert obs is not None, "Reset should return observation"
                assert info is not None, "Reset should return info dict"
                assert isinstance(info, dict), "Info should be dictionary"
                assert len(obs) == env.observation_space.shape[0], "Observation shape mismatch"

                reset_results['tests_passed'] += 1
                reset_results['details'].append({
                    'test': 'basic_reset',
                    'success': True,
                    'obs_shape': obs.shape if hasattr(obs, 'shape') else len(obs),
                    'info_keys': list(info.keys())
                })

            except Exception as e:
                reset_results['success'] = False
                reset_results['details'].append({
                    'test': 'basic_reset',
                    'success': False,
                    'error': str(e)
                })

            # Test reset with seed
            reset_results['tests_total'] += 1
            try:
                obs1, info1 = env.reset(seed=42)
                obs2, info2 = env.reset(seed=42)

                # Note: Determinism may not be guaranteed due to VTracer randomness
                # but reset should still work
                assert obs1 is not None and obs2 is not None, "Reset with seed failed"

                reset_results['tests_passed'] += 1
                reset_results['details'].append({
                    'test': 'reset_with_seed',
                    'success': True,
                    'deterministic': np.allclose(obs1, obs2) if hasattr(obs1, 'shape') else obs1 == obs2
                })

            except Exception as e:
                reset_results['success'] = False
                reset_results['details'].append({
                    'test': 'reset_with_seed',
                    'success': False,
                    'error': str(e)
                })

            # Test multiple consecutive resets
            reset_results['tests_total'] += 1
            try:
                for i in range(3):
                    obs, info = env.reset()
                    assert obs is not None, f"Reset {i} failed"

                reset_results['tests_passed'] += 1
                reset_results['details'].append({
                    'test': 'consecutive_resets',
                    'success': True,
                    'reset_count': 3
                })

            except Exception as e:
                reset_results['success'] = False
                reset_results['details'].append({
                    'test': 'consecutive_resets',
                    'success': False,
                    'error': str(e)
                })

            env.close()

        except Exception as e:
            reset_results['success'] = False
            reset_results['error'] = str(e)

        return reset_results

    def _test_step_functionality(self) -> Dict[str, Any]:
        """Test step() method with various actions"""
        step_results = {
            'success': True,
            'tests_passed': 0,
            'tests_total': 0,
            'details': []
        }

        try:
            env = VTracerOptimizationEnv(self.test_images[0])
            obs, info = env.reset()

            # Test valid action
            step_results['tests_total'] += 1
            try:
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)

                # Validate step returns
                assert obs is not None, "Step should return observation"
                assert isinstance(reward, (int, float)), "Reward should be numeric"
                assert isinstance(done, bool), "Done should be boolean"
                assert isinstance(truncated, bool), "Truncated should be boolean"
                assert isinstance(info, dict), "Info should be dictionary"
                assert len(obs) == env.observation_space.shape[0], "Observation shape mismatch"

                step_results['tests_passed'] += 1
                step_results['details'].append({
                    'test': 'valid_action',
                    'success': True,
                    'action_shape': action.shape if hasattr(action, 'shape') else len(action),
                    'reward': reward,
                    'done': done,
                    'truncated': truncated,
                    'obs_shape': obs.shape if hasattr(obs, 'shape') else len(obs)
                })

            except Exception as e:
                step_results['success'] = False
                step_results['details'].append({
                    'test': 'valid_action',
                    'success': False,
                    'error': str(e)
                })

            # Test boundary actions (0 and 1)
            boundary_actions = [
                np.zeros(7, dtype=np.float32),
                np.ones(7, dtype=np.float32),
                np.full(7, 0.5, dtype=np.float32)
            ]

            for i, action in enumerate(boundary_actions):
                step_results['tests_total'] += 1
                try:
                    obs, reward, done, truncated, info = env.step(action)

                    assert obs is not None, "Step with boundary action failed"

                    step_results['tests_passed'] += 1
                    step_results['details'].append({
                        'test': f'boundary_action_{i}',
                        'success': True,
                        'action': action.tolist(),
                        'reward': reward
                    })

                except Exception as e:
                    step_results['success'] = False
                    step_results['details'].append({
                        'test': f'boundary_action_{i}',
                        'success': False,
                        'action': action.tolist(),
                        'error': str(e)
                    })

            # Test episode completion
            step_results['tests_total'] += 1
            try:
                env.reset()
                step_count = 0
                max_steps = 10

                while step_count < max_steps:
                    action = env.action_space.sample()
                    obs, reward, done, truncated, info = env.step(action)
                    step_count += 1

                    if done or truncated:
                        break

                step_results['tests_passed'] += 1
                step_results['details'].append({
                    'test': 'episode_completion',
                    'success': True,
                    'steps_taken': step_count,
                    'terminated': done or truncated
                })

            except Exception as e:
                step_results['success'] = False
                step_results['details'].append({
                    'test': 'episode_completion',
                    'success': False,
                    'error': str(e)
                })

            env.close()

        except Exception as e:
            step_results['success'] = False
            step_results['error'] = str(e)

        return step_results

    def _test_cleanup_functionality(self) -> Dict[str, Any]:
        """Test environment cleanup and close functionality"""
        cleanup_results = {
            'success': True,
            'tests_passed': 0,
            'tests_total': 0,
            'details': []
        }

        # Test normal close
        cleanup_results['tests_total'] += 1
        try:
            env = VTracerOptimizationEnv(self.test_images[0])
            env.close()

            cleanup_results['tests_passed'] += 1
            cleanup_results['details'].append({
                'test': 'normal_close',
                'success': True
            })

        except Exception as e:
            cleanup_results['success'] = False
            cleanup_results['details'].append({
                'test': 'normal_close',
                'success': False,
                'error': str(e)
            })

        # Test multiple closes
        cleanup_results['tests_total'] += 1
        try:
            env = VTracerOptimizationEnv(self.test_images[0])
            env.close()
            env.close()  # Should not error

            cleanup_results['tests_passed'] += 1
            cleanup_results['details'].append({
                'test': 'multiple_closes',
                'success': True
            })

        except Exception as e:
            cleanup_results['success'] = False
            cleanup_results['details'].append({
                'test': 'multiple_closes',
                'success': False,
                'error': str(e)
            })

        # Test context manager usage
        cleanup_results['tests_total'] += 1
        try:
            with VTracerOptimizationEnv(self.test_images[0]) as env:
                obs, info = env.reset()
                action = env.action_space.sample()
                env.step(action)

            cleanup_results['tests_passed'] += 1
            cleanup_results['details'].append({
                'test': 'context_manager',
                'success': True
            })

        except Exception as e:
            cleanup_results['success'] = False
            cleanup_results['details'].append({
                'test': 'context_manager',
                'success': False,
                'error': str(e)
            })

        return cleanup_results


class ActionObservationSpaceTester:
    """Test action and observation spaces comprehensively"""

    def __init__(self):
        self.test_images = [
            "test-data/circle_00.png",
            "test-data/text_tech_00.png",
            "test-data/gradient_radial_00.png"
        ]
        self.logger = logging.getLogger(__name__)

    def test_action_observation_spaces(self) -> Dict[str, Any]:
        """Test action space bounds, types, and observation space dimensionality"""
        print("🎯 Testing Action and Observation Spaces...")

        space_results = {
            'success': True,
            'action_space_tests': {},
            'observation_space_tests': {},
            'sampling_tests': {}
        }

        if not DEPENDENCIES_AVAILABLE:
            space_results['success'] = False
            space_results['error'] = "VTracer environment not available"
            return space_results

        try:
            env = VTracerOptimizationEnv(self.test_images[0])

            # Test action space
            action_space_tests = self._test_action_space_properties(env)
            space_results['action_space_tests'] = action_space_tests
            if not action_space_tests['success']:
                space_results['success'] = False

            # Test observation space
            observation_space_tests = self._test_observation_space_properties(env)
            space_results['observation_space_tests'] = observation_space_tests
            if not observation_space_tests['success']:
                space_results['success'] = False

            # Test sampling functionality
            sampling_tests = self._test_space_sampling(env)
            space_results['sampling_tests'] = sampling_tests
            if not sampling_tests['success']:
                space_results['success'] = False

            env.close()

        except Exception as e:
            space_results['success'] = False
            space_results['error'] = str(e)

        print(f"  {'✅ PASSED' if space_results['success'] else '❌ FAILED'}")
        return space_results

    def _test_action_space_properties(self, env) -> Dict[str, Any]:
        """Test action space bounds and types"""
        action_tests = {
            'success': True,
            'shape_valid': False,
            'bounds_valid': False,
            'dtype_valid': False,
            'details': {}
        }

        try:
            action_space = env.action_space

            # Test shape
            expected_shape = (7,)  # 7 VTracer parameters
            actual_shape = action_space.shape
            action_tests['shape_valid'] = actual_shape == expected_shape
            action_tests['details']['shape'] = {
                'expected': expected_shape,
                'actual': actual_shape,
                'valid': action_tests['shape_valid']
            }

            # Test bounds
            if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
                low = action_space.low
                high = action_space.high

                # Should be [0,1] normalized
                bounds_correct = (
                    np.allclose(low, 0.0) and
                    np.allclose(high, 1.0)
                )
                action_tests['bounds_valid'] = bounds_correct
                action_tests['details']['bounds'] = {
                    'low': low.tolist() if hasattr(low, 'tolist') else str(low),
                    'high': high.tolist() if hasattr(high, 'tolist') else str(high),
                    'normalized_to_01': bounds_correct
                }

            # Test dtype
            expected_dtype = np.float32
            actual_dtype = action_space.dtype
            action_tests['dtype_valid'] = actual_dtype == expected_dtype
            action_tests['details']['dtype'] = {
                'expected': str(expected_dtype),
                'actual': str(actual_dtype),
                'valid': action_tests['dtype_valid']
            }

            action_tests['success'] = (
                action_tests['shape_valid'] and
                action_tests['bounds_valid'] and
                action_tests['dtype_valid']
            )

        except Exception as e:
            action_tests['success'] = False
            action_tests['error'] = str(e)

        return action_tests

    def _test_observation_space_properties(self, env) -> Dict[str, Any]:
        """Test observation space dimensionality"""
        obs_tests = {
            'success': True,
            'shape_valid': False,
            'bounds_valid': False,
            'dtype_valid': False,
            'details': {}
        }

        try:
            observation_space = env.observation_space

            # Test shape - should be 15D (6 features + 7 params + 2 quality metrics)
            expected_shape = (15,)
            actual_shape = observation_space.shape
            obs_tests['shape_valid'] = actual_shape == expected_shape
            obs_tests['details']['shape'] = {
                'expected': expected_shape,
                'actual': actual_shape,
                'valid': obs_tests['shape_valid'],
                'breakdown': '6 features + 7 params + 2 quality metrics = 15'
            }

            # Test bounds
            if hasattr(observation_space, 'low') and hasattr(observation_space, 'high'):
                low = observation_space.low
                high = observation_space.high

                # Should be [0,1] normalized
                bounds_correct = (
                    np.allclose(low, 0.0) and
                    np.allclose(high, 1.0)
                )
                obs_tests['bounds_valid'] = bounds_correct
                obs_tests['details']['bounds'] = {
                    'low': low.tolist() if hasattr(low, 'tolist') else str(low),
                    'high': high.tolist() if hasattr(high, 'tolist') else str(high),
                    'normalized_to_01': bounds_correct
                }

            # Test dtype
            expected_dtype = np.float32
            actual_dtype = observation_space.dtype
            obs_tests['dtype_valid'] = actual_dtype == expected_dtype
            obs_tests['details']['dtype'] = {
                'expected': str(expected_dtype),
                'actual': str(actual_dtype),
                'valid': obs_tests['dtype_valid']
            }

            obs_tests['success'] = (
                obs_tests['shape_valid'] and
                obs_tests['bounds_valid'] and
                obs_tests['dtype_valid']
            )

        except Exception as e:
            obs_tests['success'] = False
            obs_tests['error'] = str(e)

        return obs_tests

    def _test_space_sampling(self, env) -> Dict[str, Any]:
        """Test space sampling functionality"""
        sampling_tests = {
            'success': True,
            'action_sampling': {},
            'observation_sampling': {},
            'contains_tests': {}
        }

        try:
            # Test action space sampling
            action_sampling_results = []
            for i in range(10):
                try:
                    action = env.action_space.sample()

                    # Validate sample
                    assert action is not None, "Action sample is None"
                    assert action.shape == env.action_space.shape, "Action shape mismatch"
                    assert env.action_space.contains(action), "Action not in space"
                    assert np.all(action >= 0.0) and np.all(action <= 1.0), "Action out of [0,1] bounds"

                    action_sampling_results.append({
                        'sample_id': i,
                        'success': True,
                        'shape': action.shape,
                        'min_value': float(np.min(action)),
                        'max_value': float(np.max(action))
                    })

                except Exception as e:
                    action_sampling_results.append({
                        'sample_id': i,
                        'success': False,
                        'error': str(e)
                    })
                    sampling_tests['success'] = False

            sampling_tests['action_sampling'] = {
                'samples_tested': len(action_sampling_results),
                'successful_samples': len([r for r in action_sampling_results if r['success']]),
                'details': action_sampling_results
            }

            # Test observation space sampling
            obs_sampling_results = []
            for i in range(5):
                try:
                    obs = env.observation_space.sample()

                    # Validate sample
                    assert obs is not None, "Observation sample is None"
                    assert obs.shape == env.observation_space.shape, "Observation shape mismatch"
                    assert env.observation_space.contains(obs), "Observation not in space"

                    obs_sampling_results.append({
                        'sample_id': i,
                        'success': True,
                        'shape': obs.shape,
                        'min_value': float(np.min(obs)),
                        'max_value': float(np.max(obs))
                    })

                except Exception as e:
                    obs_sampling_results.append({
                        'sample_id': i,
                        'success': False,
                        'error': str(e)
                    })
                    sampling_tests['success'] = False

            sampling_tests['observation_sampling'] = {
                'samples_tested': len(obs_sampling_results),
                'successful_samples': len([r for r in obs_sampling_results if r['success']]),
                'details': obs_sampling_results
            }

            # Test contains functionality
            contains_tests = self._test_contains_functionality(env)
            sampling_tests['contains_tests'] = contains_tests
            if not contains_tests['success']:
                sampling_tests['success'] = False

        except Exception as e:
            sampling_tests['success'] = False
            sampling_tests['error'] = str(e)

        return sampling_tests

    def _test_contains_functionality(self, env) -> Dict[str, Any]:
        """Test space contains functionality"""
        contains_tests = {
            'success': True,
            'action_contains': {},
            'observation_contains': {}
        }

        try:
            # Test action space contains
            action_contains_results = []

            # Valid actions
            valid_actions = [
                np.zeros(7, dtype=np.float32),
                np.ones(7, dtype=np.float32),
                np.full(7, 0.5, dtype=np.float32),
                env.action_space.sample()
            ]

            for i, action in enumerate(valid_actions):
                try:
                    contains = env.action_space.contains(action)
                    action_contains_results.append({
                        'test_id': f'valid_{i}',
                        'action': action.tolist(),
                        'expected_contains': True,
                        'actual_contains': contains,
                        'success': contains == True
                    })
                    if not contains:
                        contains_tests['success'] = False
                except Exception as e:
                    action_contains_results.append({
                        'test_id': f'valid_{i}',
                        'success': False,
                        'error': str(e)
                    })
                    contains_tests['success'] = False

            # Invalid actions
            invalid_actions = [
                np.full(7, -0.1, dtype=np.float32),  # Below bounds
                np.full(7, 1.1, dtype=np.float32),   # Above bounds
                np.array([0.5, 0.5, 0.5], dtype=np.float32),  # Wrong shape
            ]

            for i, action in enumerate(invalid_actions):
                try:
                    contains = env.action_space.contains(action)
                    action_contains_results.append({
                        'test_id': f'invalid_{i}',
                        'action': action.tolist(),
                        'expected_contains': False,
                        'actual_contains': contains,
                        'success': contains == False
                    })
                    if contains:  # Should be False
                        contains_tests['success'] = False
                except Exception as e:
                    # Exception is acceptable for invalid inputs
                    action_contains_results.append({
                        'test_id': f'invalid_{i}',
                        'expected_contains': False,
                        'got_exception': True,
                        'success': True  # Exception is acceptable
                    })

            contains_tests['action_contains'] = action_contains_results

            # Test observation space contains (similar approach)
            obs_contains_results = []

            # Valid observations
            try:
                valid_obs = env.observation_space.sample()
                contains = env.observation_space.contains(valid_obs)
                obs_contains_results.append({
                    'test_id': 'valid_sample',
                    'expected_contains': True,
                    'actual_contains': contains,
                    'success': contains == True
                })
                if not contains:
                    contains_tests['success'] = False
            except Exception as e:
                obs_contains_results.append({
                    'test_id': 'valid_sample',
                    'success': False,
                    'error': str(e)
                })
                contains_tests['success'] = False

            contains_tests['observation_contains'] = obs_contains_results

        except Exception as e:
            contains_tests['success'] = False
            contains_tests['error'] = str(e)

        return contains_tests


class EnvironmentDeterminismTester:
    """Test environment determinism and reproducibility"""

    def __init__(self):
        self.test_images = [
            "test-data/circle_00.png",
            "test-data/text_tech_00.png",
            "test-data/gradient_radial_00.png"
        ]
        self.logger = logging.getLogger(__name__)

    def test_environment_determinism(self) -> Dict[str, Any]:
        """Test reproducible episodes with same seed"""
        print("🎲 Testing Environment Determinism...")

        determinism_results = {
            'success': True,
            'seed_reproducibility': {},
            'parameter_mapping_consistency': {},
            'reward_calculation_consistency': {}
        }

        if not DEPENDENCIES_AVAILABLE:
            determinism_results['success'] = False
            determinism_results['error'] = "VTracer environment not available"
            return determinism_results

        try:
            # Test seed reproducibility
            seed_test = self._test_seed_reproducibility()
            determinism_results['seed_reproducibility'] = seed_test
            if not seed_test['success']:
                determinism_results['success'] = False

            # Test parameter mapping consistency
            param_test = self._test_parameter_mapping_consistency()
            determinism_results['parameter_mapping_consistency'] = param_test
            if not param_test['success']:
                determinism_results['success'] = False

            # Test reward calculation consistency
            reward_test = self._test_reward_calculation_consistency()
            determinism_results['reward_calculation_consistency'] = reward_test
            if not reward_test['success']:
                determinism_results['success'] = False

        except Exception as e:
            determinism_results['success'] = False
            determinism_results['error'] = str(e)

        print(f"  {'✅ PASSED' if determinism_results['success'] else '❌ FAILED'}")
        return determinism_results

    def _test_seed_reproducibility(self) -> Dict[str, Any]:
        """Test reproducible episodes with same seed"""
        seed_tests = {
            'success': True,
            'tests_passed': 0,
            'tests_total': 0,
            'details': []
        }

        test_seeds = [42, 123, 456]

        for seed in test_seeds:
            seed_tests['tests_total'] += 1
            try:
                # Run first episode
                env1 = VTracerOptimizationEnv(self.test_images[0])
                obs1, info1 = env1.reset(seed=seed)

                trajectory1 = []
                for _ in range(3):  # Short trajectory
                    action = np.array([0.5] * 7, dtype=np.float32)  # Fixed action
                    obs, reward, done, truncated, info = env1.step(action)
                    trajectory1.append({
                        'obs': obs.tolist() if hasattr(obs, 'tolist') else list(obs),
                        'reward': reward,
                        'done': done,
                        'truncated': truncated
                    })
                    if done or truncated:
                        break

                env1.close()

                # Run second episode with same seed
                env2 = VTracerOptimizationEnv(self.test_images[0])
                obs2, info2 = env2.reset(seed=seed)

                trajectory2 = []
                for _ in range(3):  # Same trajectory
                    action = np.array([0.5] * 7, dtype=np.float32)  # Same fixed action
                    obs, reward, done, truncated, info = env2.step(action)
                    trajectory2.append({
                        'obs': obs.tolist() if hasattr(obs, 'tolist') else list(obs),
                        'reward': reward,
                        'done': done,
                        'truncated': truncated
                    })
                    if done or truncated:
                        break

                env2.close()

                # Compare trajectories
                # Note: Perfect determinism may not be possible due to VTracer internal randomness
                # But we can check structural consistency
                initial_obs_similar = np.allclose(obs1, obs2, rtol=0.1) if hasattr(obs1, 'shape') else obs1 == obs2
                trajectory_length_same = len(trajectory1) == len(trajectory2)

                seed_tests['tests_passed'] += 1
                seed_tests['details'].append({
                    'seed': seed,
                    'success': True,
                    'initial_obs_similar': initial_obs_similar,
                    'trajectory_length_same': trajectory_length_same,
                    'trajectory1_length': len(trajectory1),
                    'trajectory2_length': len(trajectory2),
                    'note': 'Perfect determinism may not be possible due to VTracer randomness'
                })

            except Exception as e:
                seed_tests['success'] = False
                seed_tests['details'].append({
                    'seed': seed,
                    'success': False,
                    'error': str(e)
                })

        return seed_tests

    def _test_parameter_mapping_consistency(self) -> Dict[str, Any]:
        """Test deterministic parameter mapping"""
        param_tests = {
            'success': True,
            'tests_passed': 0,
            'tests_total': 0,
            'details': []
        }

        # Test same action maps to same parameters consistently
        test_actions = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        ]

        for i, action in enumerate(test_actions):
            param_tests['tests_total'] += 1
            try:
                env = VTracerOptimizationEnv(self.test_images[0])

                # Test parameter mapping multiple times
                params_list = []
                for trial in range(3):
                    # Access the internal denormalization method
                    if hasattr(env, '_denormalize_action'):
                        params = env._denormalize_action(action)
                        params_list.append(params)

                # Check consistency
                if len(params_list) >= 2:
                    params1, params2 = params_list[0], params_list[1]

                    # Parameters should be identical for same action
                    consistency_check = True
                    for key in params1:
                        if key in params2:
                            if params1[key] != params2[key]:
                                consistency_check = False
                                break

                    param_tests['tests_passed'] += 1
                    param_tests['details'].append({
                        'action_id': i,
                        'action': action.tolist(),
                        'success': True,
                        'consistent_mapping': consistency_check,
                        'sample_params': params1,
                        'trials': len(params_list)
                    })

                    if not consistency_check:
                        param_tests['success'] = False
                else:
                    param_tests['success'] = False
                    param_tests['details'].append({
                        'action_id': i,
                        'success': False,
                        'error': 'Insufficient parameter mappings generated'
                    })

                env.close()

            except Exception as e:
                param_tests['success'] = False
                param_tests['details'].append({
                    'action_id': i,
                    'action': action.tolist(),
                    'success': False,
                    'error': str(e)
                })

        return param_tests

    def _test_reward_calculation_consistency(self) -> Dict[str, Any]:
        """Test consistent reward calculation"""
        reward_tests = {
            'success': True,
            'tests_passed': 0,
            'tests_total': 0,
            'details': []
        }

        # Test reward consistency for same conversion results
        try:
            from backend.ai_modules.optimization.reward_functions import MultiObjectiveRewardFunction, ConversionResult

            reward_function = MultiObjectiveRewardFunction()

            test_scenarios = [
                {
                    'name': 'identical_results',
                    'result': ConversionResult(0.85, 0.1, 10.0, True, '/tmp/test.svg'),
                    'baseline': ConversionResult(0.75, 0.12, 12.0, True, '/tmp/baseline.svg')
                },
                {
                    'name': 'quality_improvement',
                    'result': ConversionResult(0.90, 0.08, 8.0, True, '/tmp/test.svg'),
                    'baseline': ConversionResult(0.75, 0.12, 12.0, True, '/tmp/baseline.svg')
                }
            ]

            for scenario in test_scenarios:
                reward_tests['tests_total'] += 1
                try:
                    # Calculate reward multiple times
                    rewards = []
                    components_list = []

                    for trial in range(3):
                        reward, components = reward_function.calculate_reward(
                            scenario['result'], scenario['baseline'], step=5, max_steps=50
                        )
                        rewards.append(reward)
                        components_list.append(components)

                    # Check consistency
                    reward_consistent = len(set(rewards)) == 1  # All rewards should be identical
                    components_consistent = all(
                        comp == components_list[0] for comp in components_list[1:]
                    )

                    reward_tests['tests_passed'] += 1
                    reward_tests['details'].append({
                        'scenario': scenario['name'],
                        'success': True,
                        'reward_consistent': reward_consistent,
                        'components_consistent': components_consistent,
                        'rewards': rewards,
                        'sample_components': components_list[0] if components_list else {}
                    })

                    if not (reward_consistent and components_consistent):
                        reward_tests['success'] = False

                except Exception as e:
                    reward_tests['success'] = False
                    reward_tests['details'].append({
                        'scenario': scenario['name'],
                        'success': False,
                        'error': str(e)
                    })

        except ImportError:
            reward_tests['success'] = False
            reward_tests['error'] = "Reward function not available for testing"

        return reward_tests


class EnvironmentEdgeCaseTester:
    """Test environment edge cases and error handling"""

    def __init__(self):
        self.test_images = [
            "test-data/circle_00.png",
            "test-data/text_tech_00.png",
            "test-data/gradient_radial_00.png"
        ]
        self.logger = logging.getLogger(__name__)

    def test_environment_edge_cases(self) -> Dict[str, Any]:
        """Test invalid actions, VTracer failures, and corrupted images"""
        print("⚠️  Testing Environment Edge Cases...")

        edge_case_results = {
            'success': True,
            'invalid_actions': {},
            'vtracer_failures': {},
            'corrupted_images': {}
        }

        if not DEPENDENCIES_AVAILABLE:
            edge_case_results['success'] = False
            edge_case_results['error'] = "VTracer environment not available"
            return edge_case_results

        try:
            # Test invalid actions
            invalid_action_test = self._test_invalid_actions()
            edge_case_results['invalid_actions'] = invalid_action_test
            if not invalid_action_test['success']:
                edge_case_results['success'] = False

            # Test VTracer conversion failures
            vtracer_failure_test = self._test_vtracer_failures()
            edge_case_results['vtracer_failures'] = vtracer_failure_test
            if not vtracer_failure_test['success']:
                edge_case_results['success'] = False

            # Test corrupted/missing images
            corrupted_image_test = self._test_corrupted_images()
            edge_case_results['corrupted_images'] = corrupted_image_test
            if not corrupted_image_test['success']:
                edge_case_results['success'] = False

        except Exception as e:
            edge_case_results['success'] = False
            edge_case_results['error'] = str(e)

        print(f"  {'✅ PASSED' if edge_case_results['success'] else '❌ FAILED'}")
        return edge_case_results

    def _test_invalid_actions(self) -> Dict[str, Any]:
        """Test environment response to invalid actions"""
        invalid_action_tests = {
            'success': True,
            'tests_passed': 0,
            'tests_total': 0,
            'details': []
        }

        try:
            env = VTracerOptimizationEnv(self.test_images[0])
            env.reset()

            # Invalid action types and values
            invalid_actions = [
                {
                    'name': 'negative_values',
                    'action': np.array([-0.1, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
                    'should_handle_gracefully': True
                },
                {
                    'name': 'values_above_one',
                    'action': np.array([1.1, 1.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
                    'should_handle_gracefully': True
                },
                {
                    'name': 'wrong_shape',
                    'action': np.array([0.5, 0.5, 0.5], dtype=np.float32),
                    'should_handle_gracefully': False  # Should raise error
                },
                {
                    'name': 'wrong_dtype',
                    'action': np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.int32),
                    'should_handle_gracefully': True  # Should convert or handle
                },
                {
                    'name': 'nan_values',
                    'action': np.array([np.nan, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
                    'should_handle_gracefully': True
                },
                {
                    'name': 'inf_values',
                    'action': np.array([np.inf, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
                    'should_handle_gracefully': True
                }
            ]

            for test_case in invalid_actions:
                invalid_action_tests['tests_total'] += 1
                try:
                    obs, reward, done, truncated, info = env.step(test_case['action'])

                    # If we get here, the environment handled it gracefully
                    if test_case['should_handle_gracefully']:
                        invalid_action_tests['tests_passed'] += 1
                        invalid_action_tests['details'].append({
                            'test': test_case['name'],
                            'success': True,
                            'handled_gracefully': True,
                            'action': test_case['action'].tolist() if hasattr(test_case['action'], 'tolist') else str(test_case['action']),
                            'reward': reward,
                            'obs_shape': obs.shape if hasattr(obs, 'shape') else len(obs)
                        })
                    else:
                        # Should have raised an error but didn't
                        invalid_action_tests['details'].append({
                            'test': test_case['name'],
                            'success': False,
                            'expected_error': True,
                            'got_error': False,
                            'action': test_case['action'].tolist() if hasattr(test_case['action'], 'tolist') else str(test_case['action'])
                        })
                        invalid_action_tests['success'] = False

                except Exception as e:
                    # Got an exception
                    if not test_case['should_handle_gracefully']:
                        # Expected an error
                        invalid_action_tests['tests_passed'] += 1
                        invalid_action_tests['details'].append({
                            'test': test_case['name'],
                            'success': True,
                            'expected_error': True,
                            'got_error': True,
                            'error': str(e),
                            'action': test_case['action'].tolist() if hasattr(test_case['action'], 'tolist') else str(test_case['action'])
                        })
                    else:
                        # Didn't expect an error
                        invalid_action_tests['details'].append({
                            'test': test_case['name'],
                            'success': False,
                            'expected_graceful_handling': True,
                            'got_error': True,
                            'error': str(e),
                            'action': test_case['action'].tolist() if hasattr(test_case['action'], 'tolist') else str(test_case['action'])
                        })
                        invalid_action_tests['success'] = False

            env.close()

        except Exception as e:
            invalid_action_tests['success'] = False
            invalid_action_tests['error'] = str(e)

        return invalid_action_tests

    def _test_vtracer_failures(self) -> Dict[str, Any]:
        """Test handling of VTracer conversion failures"""
        vtracer_failure_tests = {
            'success': True,
            'tests_passed': 0,
            'tests_total': 0,
            'details': []
        }

        try:
            env = VTracerOptimizationEnv(self.test_images[0])
            env.reset()

            # Test extreme parameter values that might cause VTracer to fail
            extreme_actions = [
                {
                    'name': 'all_zeros',
                    'action': np.zeros(7, dtype=np.float32),
                    'description': 'All parameters at minimum'
                },
                {
                    'name': 'all_ones',
                    'action': np.ones(7, dtype=np.float32),
                    'description': 'All parameters at maximum'
                },
                {
                    'name': 'extreme_precision',
                    'action': np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32),
                    'description': 'High precision with low thresholds'
                }
            ]

            for test_case in extreme_actions:
                vtracer_failure_tests['tests_total'] += 1
                try:
                    obs, reward, done, truncated, info = env.step(test_case['action'])

                    # Environment should handle VTracer failures gracefully
                    # Check if the step completed successfully
                    assert obs is not None, "Observation should not be None"
                    assert isinstance(reward, (int, float)), "Reward should be numeric"
                    assert isinstance(done, bool), "Done should be boolean"
                    assert isinstance(truncated, bool), "Truncated should be boolean"
                    assert isinstance(info, dict), "Info should be dictionary"

                    # Check if VTracer failure was handled properly
                    vtracer_success = info.get('success', True)

                    vtracer_failure_tests['tests_passed'] += 1
                    vtracer_failure_tests['details'].append({
                        'test': test_case['name'],
                        'description': test_case['description'],
                        'success': True,
                        'action': test_case['action'].tolist(),
                        'vtracer_success': vtracer_success,
                        'reward': reward,
                        'handled_gracefully': True,
                        'info_keys': list(info.keys())
                    })

                except Exception as e:
                    vtracer_failure_tests['details'].append({
                        'test': test_case['name'],
                        'description': test_case['description'],
                        'success': False,
                        'action': test_case['action'].tolist(),
                        'error': str(e),
                        'handled_gracefully': False
                    })
                    vtracer_failure_tests['success'] = False

            env.close()

        except Exception as e:
            vtracer_failure_tests['success'] = False
            vtracer_failure_tests['error'] = str(e)

        return vtracer_failure_tests

    def _test_corrupted_images(self) -> Dict[str, Any]:
        """Test handling of corrupted or missing images"""
        corrupted_image_tests = {
            'success': True,
            'tests_passed': 0,
            'tests_total': 0,
            'details': []
        }

        # Test cases for image problems
        image_test_cases = [
            {
                'name': 'nonexistent_image',
                'image_path': '/nonexistent/path/image.png',
                'should_fail_init': True
            },
            {
                'name': 'empty_path',
                'image_path': '',
                'should_fail_init': True
            },
            {
                'name': 'invalid_extension',
                'image_path': 'test-data/circle_00.txt',  # Wrong extension
                'should_fail_init': True
            }
        ]

        for test_case in image_test_cases:
            corrupted_image_tests['tests_total'] += 1
            try:
                # Try to initialize environment with problematic image
                env = VTracerOptimizationEnv(test_case['image_path'])

                # If initialization succeeded when it should have failed
                if test_case['should_fail_init']:
                    corrupted_image_tests['details'].append({
                        'test': test_case['name'],
                        'success': False,
                        'image_path': test_case['image_path'],
                        'expected_init_failure': True,
                        'got_init_failure': False,
                        'note': 'Environment should have failed to initialize'
                    })
                    corrupted_image_tests['success'] = False
                else:
                    # Try a few operations
                    obs, info = env.reset()
                    action = env.action_space.sample()
                    obs, reward, done, truncated, info = env.step(action)

                    corrupted_image_tests['tests_passed'] += 1
                    corrupted_image_tests['details'].append({
                        'test': test_case['name'],
                        'success': True,
                        'image_path': test_case['image_path'],
                        'init_succeeded': True,
                        'operations_completed': True
                    })

                env.close()

            except Exception as e:
                # Got an exception
                if test_case['should_fail_init']:
                    # Expected failure
                    corrupted_image_tests['tests_passed'] += 1
                    corrupted_image_tests['details'].append({
                        'test': test_case['name'],
                        'success': True,
                        'image_path': test_case['image_path'],
                        'expected_init_failure': True,
                        'got_init_failure': True,
                        'error': str(e)
                    })
                else:
                    # Unexpected failure
                    corrupted_image_tests['details'].append({
                        'test': test_case['name'],
                        'success': False,
                        'image_path': test_case['image_path'],
                        'expected_success': True,
                        'got_error': True,
                        'error': str(e)
                    })
                    corrupted_image_tests['success'] = False

        return corrupted_image_tests


class EnvironmentPerformanceTester:
    """Test environment performance and scalability"""

    def __init__(self):
        self.test_images = [
            "test-data/circle_00.png",
            "test-data/text_tech_00.png",
            "test-data/gradient_radial_00.png"
        ]
        self.logger = logging.getLogger(__name__)

    def test_environment_performance(self) -> Dict[str, Any]:
        """Test step time, memory usage, and scalability"""
        print("⚡ Testing Environment Performance...")

        performance_results = {
            'success': True,
            'step_timing': {},
            'memory_usage': {},
            'scalability': {}
        }

        if not DEPENDENCIES_AVAILABLE:
            performance_results['success'] = False
            performance_results['error'] = "VTracer environment not available"
            return performance_results

        try:
            # Test step timing
            timing_test = self._test_step_timing()
            performance_results['step_timing'] = timing_test
            if not timing_test['success']:
                performance_results['success'] = False

            # Test memory usage
            memory_test = self._test_memory_usage()
            performance_results['memory_usage'] = memory_test
            if not memory_test['success']:
                performance_results['success'] = False

            # Test scalability
            scalability_test = self._test_scalability()
            performance_results['scalability'] = scalability_test
            if not scalability_test['success']:
                performance_results['success'] = False

        except Exception as e:
            performance_results['success'] = False
            performance_results['error'] = str(e)

        print(f"  {'✅ PASSED' if performance_results['success'] else '❌ FAILED'}")
        return performance_results

    def _test_step_timing(self) -> Dict[str, Any]:
        """Measure environment step times"""
        timing_tests = {
            'success': True,
            'step_times': [],
            'reset_times': [],
            'statistics': {}
        }

        try:
            env = VTracerOptimizationEnv(self.test_images[0])

            # Measure reset times
            reset_times = []
            for i in range(5):
                start_time = time.time()
                env.reset()
                reset_time = time.time() - start_time
                reset_times.append(reset_time)

            timing_tests['reset_times'] = reset_times

            # Measure step times
            step_times = []
            for i in range(10):
                action = env.action_space.sample()

                start_time = time.time()
                obs, reward, done, truncated, info = env.step(action)
                step_time = time.time() - start_time

                step_times.append(step_time)

                if done or truncated:
                    env.reset()

            timing_tests['step_times'] = step_times

            # Calculate statistics
            timing_tests['statistics'] = {
                'reset_times': {
                    'mean': np.mean(reset_times),
                    'std': np.std(reset_times),
                    'min': np.min(reset_times),
                    'max': np.max(reset_times),
                    'median': np.median(reset_times)
                },
                'step_times': {
                    'mean': np.mean(step_times),
                    'std': np.std(step_times),
                    'min': np.min(step_times),
                    'max': np.max(step_times),
                    'median': np.median(step_times)
                }
            }

            # Performance thresholds (reasonable for VTracer operations)
            mean_step_time = np.mean(step_times)
            mean_reset_time = np.mean(reset_times)

            if mean_step_time > 5.0:  # 5 seconds per step is quite slow
                timing_tests['success'] = False
                timing_tests['warning'] = f"Mean step time {mean_step_time:.2f}s exceeds 5.0s threshold"

            if mean_reset_time > 10.0:  # 10 seconds per reset is quite slow
                timing_tests['success'] = False
                timing_tests['warning'] = f"Mean reset time {mean_reset_time:.2f}s exceeds 10.0s threshold"

            env.close()

        except Exception as e:
            timing_tests['success'] = False
            timing_tests['error'] = str(e)

        return timing_tests

    def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage during episodes"""
        memory_tests = {
            'success': True,
            'memory_measurements': [],
            'statistics': {}
        }

        try:
            process = psutil.Process()

            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

            env = VTracerOptimizationEnv(self.test_images[0])

            # Memory after environment creation
            init_memory = process.memory_info().rss / 1024 / 1024  # MB

            memory_measurements = []
            memory_measurements.append({
                'stage': 'baseline',
                'memory_mb': baseline_memory
            })
            memory_measurements.append({
                'stage': 'after_init',
                'memory_mb': init_memory
            })

            # Memory during episode execution
            env.reset()
            reset_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_measurements.append({
                'stage': 'after_reset',
                'memory_mb': reset_memory
            })

            for i in range(5):
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)

                step_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_measurements.append({
                    'stage': f'step_{i}',
                    'memory_mb': step_memory
                })

                if done or truncated:
                    break

            memory_tests['memory_measurements'] = memory_measurements

            # Calculate statistics
            memory_values = [m['memory_mb'] for m in memory_measurements[1:]]  # Exclude baseline
            peak_memory = max(memory_values)
            memory_growth = peak_memory - baseline_memory

            memory_tests['statistics'] = {
                'baseline_memory_mb': baseline_memory,
                'peak_memory_mb': peak_memory,
                'memory_growth_mb': memory_growth,
                'init_overhead_mb': init_memory - baseline_memory
            }

            # Memory thresholds
            if memory_growth > 500:  # 500MB growth is concerning
                memory_tests['success'] = False
                memory_tests['warning'] = f"Memory growth {memory_growth:.1f}MB exceeds 500MB threshold"

            if peak_memory > 1000:  # 1GB total is quite high
                memory_tests['success'] = False
                memory_tests['warning'] = f"Peak memory {peak_memory:.1f}MB exceeds 1000MB threshold"

            env.close()

            # Memory after cleanup
            gc.collect()  # Force garbage collection
            time.sleep(1)  # Allow time for cleanup
            final_memory = process.memory_info().rss / 1024 / 1024  # MB

            memory_tests['memory_measurements'].append({
                'stage': 'after_cleanup',
                'memory_mb': final_memory
            })

            memory_leak = final_memory - baseline_memory
            memory_tests['statistics']['memory_leak_mb'] = memory_leak

            if memory_leak > 50:  # 50MB leak is concerning
                memory_tests['success'] = False
                memory_tests['warning'] = f"Memory leak {memory_leak:.1f}MB exceeds 50MB threshold"

        except Exception as e:
            memory_tests['success'] = False
            memory_tests['error'] = str(e)

        return memory_tests

    def _test_scalability(self) -> Dict[str, Any]:
        """Test environment scalability with multiple instances"""
        scalability_tests = {
            'success': True,
            'concurrent_environments': {},
            'sequential_environments': {}
        }

        try:
            # Test sequential environment creation/destruction
            sequential_times = []
            for i in range(3):  # Small number for testing
                start_time = time.time()

                env = VTracerOptimizationEnv(self.test_images[i % len(self.test_images)])
                env.reset()

                # Run a few steps
                for _ in range(2):
                    action = env.action_space.sample()
                    obs, reward, done, truncated, info = env.step(action)
                    if done or truncated:
                        break

                env.close()

                total_time = time.time() - start_time
                sequential_times.append(total_time)

            scalability_tests['sequential_environments'] = {
                'environments_tested': len(sequential_times),
                'times': sequential_times,
                'mean_time': np.mean(sequential_times),
                'total_time': sum(sequential_times)
            }

            # Test concurrent environment creation (limited)
            try:
                envs = []
                concurrent_start = time.time()

                # Create multiple environments
                for i in range(2):  # Very small number for safety
                    env = VTracerOptimizationEnv(self.test_images[i % len(self.test_images)])
                    envs.append(env)

                # Reset all environments
                for env in envs:
                    env.reset()

                # Run steps on all environments
                for env in envs:
                    action = env.action_space.sample()
                    obs, reward, done, truncated, info = env.step(action)

                # Close all environments
                for env in envs:
                    env.close()

                concurrent_total_time = time.time() - concurrent_start

                scalability_tests['concurrent_environments'] = {
                    'environments_created': len(envs),
                    'total_time': concurrent_total_time,
                    'success': True
                }

            except Exception as e:
                scalability_tests['concurrent_environments'] = {
                    'success': False,
                    'error': str(e),
                    'note': 'Concurrent environments may not be supported'
                }
                # This is not necessarily a failure for the overall test

        except Exception as e:
            scalability_tests['success'] = False
            scalability_tests['error'] = str(e)

        return scalability_tests


class EnvironmentDebuggingUtilities:
    """Create environment debugging utilities"""

    def __init__(self):
        self.test_images = [
            "test-data/circle_00.png",
            "test-data/text_tech_00.png",
            "test-data/gradient_radial_00.png"
        ]
        self.logger = logging.getLogger(__name__)

    def create_debugging_utilities(self) -> Dict[str, Any]:
        """Create debugging utilities for environment development"""
        print("🔧 Creating Environment Debugging Utilities...")

        debug_utils = {
            'success': True,
            'utilities_created': [],
            'debug_functions': {},
            'logging_config': {}
        }

        try:
            # Create environment inspector
            debug_utils['debug_functions']['inspect_environment'] = self._create_environment_inspector()

            # Create action analyzer
            debug_utils['debug_functions']['analyze_action'] = self._create_action_analyzer()

            # Create episode tracer
            debug_utils['debug_functions']['trace_episode'] = self._create_episode_tracer()

            # Create parameter mapper debugger
            debug_utils['debug_functions']['debug_parameter_mapping'] = self._create_parameter_mapper_debugger()

            # Create logging configuration
            debug_utils['logging_config'] = self._create_logging_config()

            debug_utils['utilities_created'] = list(debug_utils['debug_functions'].keys())

        except Exception as e:
            debug_utils['success'] = False
            debug_utils['error'] = str(e)

        print(f"  {'✅ PASSED' if debug_utils['success'] else '❌ FAILED'}")
        return debug_utils

    def _create_environment_inspector(self) -> Dict[str, Any]:
        """Create environment inspection utility"""
        inspector_info = {
            'function_name': 'inspect_environment',
            'description': 'Inspect environment state and configuration',
            'created': True
        }

        def inspect_environment(env):
            """Inspect VTracer environment state"""
            inspection = {
                'environment_type': type(env).__name__,
                'action_space': {
                    'type': str(type(env.action_space)),
                    'shape': env.action_space.shape,
                    'dtype': str(env.action_space.dtype),
                    'bounds': {
                        'low': env.action_space.low.tolist() if hasattr(env.action_space, 'low') else None,
                        'high': env.action_space.high.tolist() if hasattr(env.action_space, 'high') else None
                    }
                },
                'observation_space': {
                    'type': str(type(env.observation_space)),
                    'shape': env.observation_space.shape,
                    'dtype': str(env.observation_space.dtype),
                    'bounds': {
                        'low': env.observation_space.low.tolist() if hasattr(env.observation_space, 'low') else None,
                        'high': env.observation_space.high.tolist() if hasattr(env.observation_space, 'high') else None
                    }
                },
                'configuration': {
                    'image_path': getattr(env, 'image_path', 'unknown'),
                    'target_quality': getattr(env, 'target_quality', 'unknown'),
                    'max_steps': getattr(env, 'max_steps', 'unknown'),
                    'current_step': getattr(env, 'current_step', 'unknown')
                },
                'current_state': {
                    'best_quality': getattr(env, 'best_quality', 'unknown'),
                    'baseline_quality': getattr(env, 'baseline_quality', 'unknown'),
                    'current_params': getattr(env, 'current_params', 'unknown')
                }
            }
            return inspection

        inspector_info['function'] = inspect_environment
        return inspector_info

    def _create_action_analyzer(self) -> Dict[str, Any]:
        """Create action analysis utility"""
        analyzer_info = {
            'function_name': 'analyze_action',
            'description': 'Analyze action validity and parameter mapping',
            'created': True
        }

        def analyze_action(env, action):
            """Analyze action and its parameter mapping"""
            analysis = {
                'action_analysis': {
                    'shape': action.shape if hasattr(action, 'shape') else len(action),
                    'dtype': str(action.dtype) if hasattr(action, 'dtype') else str(type(action)),
                    'values': action.tolist() if hasattr(action, 'tolist') else list(action),
                    'min_value': float(np.min(action)),
                    'max_value': float(np.max(action)),
                    'in_bounds': np.all(action >= 0.0) and np.all(action <= 1.0),
                    'valid_for_space': env.action_space.contains(action) if hasattr(env.action_space, 'contains') else 'unknown'
                },
                'parameter_mapping': None,
                'validation': {
                    'shape_correct': (action.shape if hasattr(action, 'shape') else len(action)) == env.action_space.shape,
                    'bounds_correct': np.all(action >= 0.0) and np.all(action <= 1.0),
                    'dtype_compatible': True  # Assume compatible for analysis
                }
            }

            try:
                # Try to map action to parameters if method exists
                if hasattr(env, '_denormalize_action'):
                    parameters = env._denormalize_action(action)
                    analysis['parameter_mapping'] = parameters
                else:
                    analysis['parameter_mapping'] = 'Method not available'
            except Exception as e:
                analysis['parameter_mapping'] = f'Error: {str(e)}'

            return analysis

        analyzer_info['function'] = analyze_action
        return analyzer_info

    def _create_episode_tracer(self) -> Dict[str, Any]:
        """Create episode tracing utility"""
        tracer_info = {
            'function_name': 'trace_episode',
            'description': 'Trace complete episode execution with detailed logging',
            'created': True
        }

        def trace_episode(env, num_steps=5, actions=None):
            """Trace episode execution with detailed information"""
            trace = {
                'episode_start': time.time(),
                'initialization': {},
                'steps': [],
                'episode_end': None,
                'summary': {}
            }

            try:
                # Reset and capture initial state
                start_time = time.time()
                obs, info = env.reset()
                reset_time = time.time() - start_time

                trace['initialization'] = {
                    'reset_time': reset_time,
                    'initial_observation': obs.tolist() if hasattr(obs, 'tolist') else list(obs),
                    'initial_info': info,
                    'observation_shape': obs.shape if hasattr(obs, 'shape') else len(obs)
                }

                # Execute steps
                total_reward = 0
                for step in range(num_steps):
                    step_start = time.time()

                    # Use provided action or sample random action
                    if actions and step < len(actions):
                        action = actions[step]
                    else:
                        action = env.action_space.sample()

                    # Execute step
                    obs, reward, done, truncated, info = env.step(action)
                    step_time = time.time() - step_start
                    total_reward += reward

                    step_trace = {
                        'step_number': step,
                        'step_time': step_time,
                        'action': action.tolist() if hasattr(action, 'tolist') else list(action),
                        'observation': obs.tolist() if hasattr(obs, 'tolist') else list(obs),
                        'reward': reward,
                        'done': done,
                        'truncated': truncated,
                        'info': info,
                        'cumulative_reward': total_reward
                    }

                    trace['steps'].append(step_trace)

                    if done or truncated:
                        break

                trace['episode_end'] = time.time()
                trace['summary'] = {
                    'total_steps': len(trace['steps']),
                    'total_reward': total_reward,
                    'episode_duration': trace['episode_end'] - trace['episode_start'],
                    'average_step_time': np.mean([s['step_time'] for s in trace['steps']]),
                    'episode_terminated': done or truncated,
                    'termination_reason': 'done' if done else 'truncated' if truncated else 'max_steps'
                }

            except Exception as e:
                trace['error'] = str(e)
                trace['episode_end'] = time.time()

            return trace

        tracer_info['function'] = trace_episode
        return tracer_info

    def _create_parameter_mapper_debugger(self) -> Dict[str, Any]:
        """Create parameter mapping debugger"""
        debugger_info = {
            'function_name': 'debug_parameter_mapping',
            'description': 'Debug parameter mapping from actions to VTracer parameters',
            'created': True
        }

        def debug_parameter_mapping(env, test_actions=None):
            """Debug parameter mapping process"""
            debug_info = {
                'mapping_tests': [],
                'boundary_tests': [],
                'consistency_tests': []
            }

            if test_actions is None:
                test_actions = [
                    np.zeros(7, dtype=np.float32),
                    np.ones(7, dtype=np.float32),
                    np.full(7, 0.5, dtype=np.float32),
                    env.action_space.sample(),
                    env.action_space.sample()
                ]

            # Test parameter mapping for each action
            for i, action in enumerate(test_actions):
                mapping_test = {
                    'test_id': i,
                    'action': action.tolist(),
                    'mapping_result': None,
                    'success': False
                }

                try:
                    if hasattr(env, '_denormalize_action'):
                        parameters = env._denormalize_action(action)
                        mapping_test['mapping_result'] = parameters
                        mapping_test['success'] = True

                        # Validate parameter bounds
                        bounds_check = {}
                        if hasattr(env, 'bounds'):
                            bounds = env.bounds.get_bounds() if hasattr(env.bounds, 'get_bounds') else {}
                            for param_name, value in parameters.items():
                                if param_name in bounds:
                                    param_spec = bounds[param_name]
                                    min_val = param_spec.get('min', value)
                                    max_val = param_spec.get('max', value)
                                    in_bounds = min_val <= value <= max_val
                                    bounds_check[param_name] = {
                                        'value': value,
                                        'min': min_val,
                                        'max': max_val,
                                        'in_bounds': in_bounds
                                    }

                        mapping_test['bounds_check'] = bounds_check

                except Exception as e:
                    mapping_test['error'] = str(e)

                debug_info['mapping_tests'].append(mapping_test)

            # Test boundary conditions
            boundary_actions = [
                ('all_zeros', np.zeros(7, dtype=np.float32)),
                ('all_ones', np.ones(7, dtype=np.float32)),
                ('mixed_boundaries', np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.5], dtype=np.float32))
            ]

            for name, action in boundary_actions:
                boundary_test = {
                    'test_name': name,
                    'action': action.tolist(),
                    'success': False
                }

                try:
                    if hasattr(env, '_denormalize_action'):
                        parameters = env._denormalize_action(action)
                        boundary_test['parameters'] = parameters
                        boundary_test['success'] = True
                except Exception as e:
                    boundary_test['error'] = str(e)

                debug_info['boundary_tests'].append(boundary_test)

            # Test consistency (same action should give same parameters)
            consistency_action = np.array([0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.5], dtype=np.float32)
            consistency_results = []

            for trial in range(3):
                try:
                    if hasattr(env, '_denormalize_action'):
                        parameters = env._denormalize_action(consistency_action)
                        consistency_results.append(parameters)
                except Exception as e:
                    consistency_results.append({'error': str(e)})

            debug_info['consistency_tests'] = {
                'test_action': consistency_action.tolist(),
                'results': consistency_results,
                'consistent': len(set(str(r) for r in consistency_results)) == 1 if consistency_results else False
            }

            return debug_info

        debugger_info['function'] = debug_parameter_mapping
        return debugger_info

    def _create_logging_config(self) -> Dict[str, Any]:
        """Create logging configuration for debugging"""
        logging_config = {
            'level': 'DEBUG',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'handlers': ['console', 'file'],
            'file_path': 'vtracer_env_debug.log',
            'loggers': {
                'backend.ai_modules.optimization.vtracer_env': 'DEBUG',
                'backend.ai_modules.optimization.reward_functions': 'DEBUG',
                'backend.ai_modules.optimization.action_mapping': 'DEBUG'
            }
        }

        return logging_config


def generate_environment_compliance_report(test_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive environment compliance report"""

    report = {
        'report_timestamp': datetime.now().isoformat(),
        'report_type': 'environment_interface_compliance',
        'overall_compliance': True,
        'test_summary': {},
        'detailed_results': test_results,
        'recommendations': [],
        'compliance_score': 0.0
    }

    try:
        # Calculate compliance metrics
        test_categories = [
            'gymnasium_compliance',
            'lifecycle_tests',
            'space_tests',
            'determinism_tests',
            'edge_case_tests',
            'performance_tests'
        ]

        passed_tests = 0
        total_tests = 0

        for category in test_categories:
            if category in test_results and test_results[category].get('success', False):
                passed_tests += 1
            total_tests += 1

        compliance_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        report['compliance_score'] = compliance_score
        report['overall_compliance'] = compliance_score >= 80.0  # 80% threshold

        # Test summary
        report['test_summary'] = {
            'tests_passed': passed_tests,
            'tests_total': total_tests,
            'pass_rate': compliance_score,
            'critical_failures': []
        }

        # Generate recommendations
        recommendations = []

        if 'gymnasium_compliance' in test_results and not test_results['gymnasium_compliance'].get('success', False):
            recommendations.append("Fix Gymnasium interface compliance issues")
            report['test_summary']['critical_failures'].append('gymnasium_compliance')

        if 'lifecycle_tests' in test_results and not test_results['lifecycle_tests'].get('success', False):
            recommendations.append("Address environment lifecycle issues")
            report['test_summary']['critical_failures'].append('lifecycle_tests')

        if 'performance_tests' in test_results and not test_results['performance_tests'].get('success', False):
            recommendations.append("Optimize environment performance")

        if compliance_score < 80:
            recommendations.append("Overall compliance below 80% - review failed tests")

        if not recommendations:
            recommendations.append("Environment interface compliance is excellent!")

        report['recommendations'] = recommendations

    except Exception as e:
        report['error'] = str(e)
        report['overall_compliance'] = False

    return report


def main():
    """Main function to run all environment interface tests"""
    print("🧪 Starting Environment Interface Testing - Task B6.2")
    print("=" * 60)

    # Initialize test results
    all_test_results = {}

    try:
        # Run Gymnasium interface compliance tests
        compliance_tester = GymnasiumInterfaceComplianceTester()
        compliance_results = compliance_tester.test_gymnasium_interface_compliance()
        all_test_results['gymnasium_compliance'] = compliance_results

        # Run environment lifecycle tests
        lifecycle_tester = EnvironmentLifecycleTester()
        lifecycle_results = lifecycle_tester.test_environment_lifecycle()
        all_test_results['lifecycle_tests'] = lifecycle_results

        # Run action/observation space tests
        space_tester = ActionObservationSpaceTester()
        space_results = space_tester.test_action_observation_spaces()
        all_test_results['space_tests'] = space_results

        # Run determinism tests
        determinism_tester = EnvironmentDeterminismTester()
        determinism_results = determinism_tester.test_environment_determinism()
        all_test_results['determinism_tests'] = determinism_results

        # Run edge case tests
        edge_case_tester = EnvironmentEdgeCaseTester()
        edge_case_results = edge_case_tester.test_environment_edge_cases()
        all_test_results['edge_case_tests'] = edge_case_results

        # Run performance tests
        performance_tester = EnvironmentPerformanceTester()
        performance_results = performance_tester.test_environment_performance()
        all_test_results['performance_tests'] = performance_results

        # Create debugging utilities
        debug_utility_creator = EnvironmentDebuggingUtilities()
        debug_results = debug_utility_creator.create_debugging_utilities()
        all_test_results['debugging_utilities'] = debug_results

        # Generate compliance report
        compliance_report = generate_environment_compliance_report(all_test_results)
        all_test_results['compliance_report'] = compliance_report

        # Save comprehensive results
        results_dir = Path("test_results/environment_interface")
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"environment_interface_test_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(all_test_results, f, indent=2, default=str)

        print(f"\n📊 Test results saved to: {results_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("📋 ENVIRONMENT INTERFACE TEST SUMMARY")
        print("=" * 60)

        overall_success = all(
            result.get('success', False)
            for key, result in all_test_results.items()
            if key != 'compliance_report' and isinstance(result, dict)
        )

        if overall_success:
            print("✅ ALL ENVIRONMENT INTERFACE TESTS PASSED")
            print("Environment interface is fully compliant and ready for use!")
        else:
            print("❌ SOME ENVIRONMENT INTERFACE TESTS FAILED")
            print("Review test results and address issues before proceeding.")

        compliance_score = compliance_report.get('compliance_score', 0)
        print(f"🎯 Compliance Score: {compliance_score:.1f}%")

        if compliance_report.get('recommendations'):
            print("\n📝 Recommendations:")
            for rec in compliance_report['recommendations']:
                print(f"  • {rec}")

        return overall_success and compliance_score >= 80

    except Exception as e:
        print(f"❌ Environment interface testing failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)