#!/usr/bin/env python3
"""
Deployment Readiness Validator for Quality Prediction Integration
Final validation script to assess complete system readiness for production deployment
"""

import os
import sys
import time
import json
import logging
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import psutil
import numpy as np

# Import all validation components
from .unified_prediction_api import UnifiedPredictionAPI, UnifiedPredictionConfig
from .quality_prediction_integration import QualityPredictionIntegrator, QualityPredictionConfig
from .cpu_performance_optimizer import CPUPerformanceOptimizer, CPUOptimizationConfig
from .performance_testing_framework import PerformanceTestSuite, PerformanceTestConfig
from .production_deployment_framework import ProductionDeploymentManager, DeploymentConfig
from .end_to_end_validation import EndToEndValidator, ValidationConfig

logger = logging.getLogger(__name__)

@dataclass
class DeploymentReadinessResult:
    """Complete deployment readiness assessment result"""
    overall_ready: bool
    readiness_score: float  # 0.0 - 1.0
    critical_issues: List[str]
    warnings: List[str]
    performance_validation: Dict[str, Any]
    integration_validation: Dict[str, Any]
    system_validation: Dict[str, Any]
    deployment_validation: Dict[str, Any]
    recommendations: List[str]
    next_steps: List[str]
    timestamp: float

class DeploymentReadinessValidator:
    """Comprehensive deployment readiness validation system"""

    def __init__(self, target_performance_ms: float = 25.0):
        self.target_performance_ms = target_performance_ms
        self.validation_results = {}
        self.critical_issues = []
        self.warnings = []

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def validate_deployment_readiness(self, quick_mode: bool = False) -> DeploymentReadinessResult:
        """
        Run comprehensive deployment readiness validation

        Args:
            quick_mode: If True, run faster validation with fewer tests

        Returns:
            Complete deployment readiness assessment
        """
        logger.info("🚀 Starting Deployment Readiness Validation")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            # 1. System Environment Validation
            logger.info("📋 1. Validating System Environment...")
            system_validation = self._validate_system_environment()

            # 2. Model and Dependencies Validation
            logger.info("📦 2. Validating Models and Dependencies...")
            dependencies_validation = self._validate_dependencies()

            # 3. Performance Validation
            logger.info("⚡ 3. Validating Performance Targets...")
            performance_validation = self._validate_performance(quick_mode)

            # 4. Integration Validation
            logger.info("🔗 4. Validating System Integration...")
            integration_validation = self._validate_integration(quick_mode)

            # 5. Production Deployment Validation
            logger.info("🏭 5. Validating Production Deployment...")
            deployment_validation = self._validate_production_deployment(quick_mode)

            # 6. End-to-End Validation
            logger.info("🎯 6. Running End-to-End Validation...")
            e2e_validation = self._validate_end_to_end(quick_mode)

            # Generate final assessment
            readiness_result = self._generate_readiness_assessment({
                'system': system_validation,
                'dependencies': dependencies_validation,
                'performance': performance_validation,
                'integration': integration_validation,
                'deployment': deployment_validation,
                'end_to_end': e2e_validation
            })

            total_time = time.time() - start_time
            logger.info(f"✅ Validation completed in {total_time:.1f} seconds")

            return readiness_result

        except Exception as e:
            logger.error(f"❌ Deployment readiness validation failed: {e}")

            # Return failure result
            return DeploymentReadinessResult(
                overall_ready=False,
                readiness_score=0.0,
                critical_issues=[f"Validation failed: {str(e)}"],
                warnings=[],
                performance_validation={},
                integration_validation={},
                system_validation={},
                deployment_validation={},
                recommendations=["Fix validation errors and retry"],
                next_steps=["Investigate validation failure", "Check system requirements"],
                timestamp=time.time()
            )

    def _validate_system_environment(self) -> Dict[str, Any]:
        """Validate system environment and requirements"""
        validation = {
            'python_version': sys.version_info,
            'platform': os.uname()._asdict() if hasattr(os, 'uname') else {},
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'disk_free_gb': psutil.disk_usage('/').free / (1024**3),
            'requirements_met': True,
            'issues': []
        }

        # Check Python version
        if sys.version_info < (3, 8):
            validation['requirements_met'] = False
            validation['issues'].append(f"Python 3.8+ required, found {sys.version_info}")
            self.critical_issues.append("Python version too old")

        # Check memory
        if validation['memory_gb'] < 4.0:
            validation['requirements_met'] = False
            validation['issues'].append(f"Minimum 4GB RAM required, found {validation['memory_gb']:.1f}GB")
            self.critical_issues.append("Insufficient memory")

        # Check disk space
        if validation['disk_free_gb'] < 2.0:
            validation['requirements_met'] = False
            validation['issues'].append(f"Minimum 2GB free disk space required, found {validation['disk_free_gb']:.1f}GB")
            self.critical_issues.append("Insufficient disk space")

        # Check for required directories
        required_dirs = [
            "/tmp/claude",
            "backend/ai_modules/optimization",
            "backend/converters"
        ]

        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    validation['issues'].append(f"Cannot create required directory {dir_path}: {e}")
                    self.warnings.append(f"Directory creation issue: {dir_path}")

        return validation

    def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate required dependencies and models"""
        validation = {
            'required_packages': {},
            'optional_packages': {},
            'models_available': {},
            'dependencies_ready': True,
            'issues': []
        }

        # Check required packages
        required_packages = [
            'numpy',
            'torch',
            'psutil',
            'pathlib'
        ]

        for package in required_packages:
            try:
                __import__(package)
                validation['required_packages'][package] = True
            except ImportError:
                validation['required_packages'][package] = False
                validation['dependencies_ready'] = False
                validation['issues'].append(f"Required package missing: {package}")
                self.critical_issues.append(f"Missing dependency: {package}")

        # Check optional packages
        optional_packages = [
            'matplotlib',
            'seaborn',
            'pandas',
            'sklearn'
        ]

        for package in optional_packages:
            try:
                __import__(package)
                validation['optional_packages'][package] = True
            except ImportError:
                validation['optional_packages'][package] = False
                self.warnings.append(f"Optional package missing: {package}")

        # Check for model files (mock check)
        model_paths = [
            "backend/ai_modules/models/optimized/quality_predictor_distilled_optimized.pt",
            "backend/ai_modules/models/optimized/quality_predictor_quantized_optimized.pt",
            "models/quality_predictor_optimized.pt"
        ]

        models_found = 0
        for model_path in model_paths:
            if Path(model_path).exists():
                validation['models_available'][model_path] = True
                models_found += 1
            else:
                validation['models_available'][model_path] = False

        if models_found == 0:
            self.warnings.append("No optimized quality prediction models found - will use mock interface")

        return validation

    def _validate_performance(self, quick_mode: bool) -> Dict[str, Any]:
        """Validate performance requirements"""
        validation = {
            'target_ms': self.target_performance_ms,
            'performance_met': False,
            'avg_inference_time_ms': 0.0,
            'p95_inference_time_ms': 0.0,
            'target_achievement_rate': 0.0,
            'throughput_per_second': 0.0,
            'issues': []
        }

        try:
            # Create performance test configuration
            test_config = PerformanceTestConfig(
                target_inference_ms=self.target_performance_ms,
                test_iterations=50 if quick_mode else 200,
                enable_stress_testing=not quick_mode,
                stress_test_duration=30 if quick_mode else 120,
                save_detailed_results=False
            )

            # Run performance tests
            test_suite = PerformanceTestSuite(test_config)
            benchmark = test_suite.run_comprehensive_benchmark()

            # Extract metrics
            validation['avg_inference_time_ms'] = benchmark.avg_inference_time_ms
            validation['p95_inference_time_ms'] = benchmark.p95_inference_time_ms
            validation['target_achievement_rate'] = benchmark.target_achievement_rate
            validation['throughput_per_second'] = benchmark.throughput_per_second

            # Check if performance targets are met
            if benchmark.target_achievement_rate >= 0.8:
                validation['performance_met'] = True
            else:
                validation['issues'].append(f"Performance target achievement rate too low: {benchmark.target_achievement_rate:.1%}")
                self.critical_issues.append("Performance targets not met")

            if benchmark.avg_inference_time_ms > self.target_performance_ms * 1.5:
                validation['issues'].append(f"Average inference time too high: {benchmark.avg_inference_time_ms:.1f}ms")
                self.warnings.append("Average performance below target")

            # Cleanup
            test_suite.cleanup()

        except Exception as e:
            validation['issues'].append(f"Performance validation failed: {str(e)}")
            self.critical_issues.append("Performance validation error")

        return validation

    def _validate_integration(self, quick_mode: bool) -> Dict[str, Any]:
        """Validate system integration"""
        validation = {
            'api_initialization': False,
            'quality_integrator': False,
            'cpu_optimizer': False,
            'intelligent_router': False,
            'prediction_success': False,
            'batch_processing': False,
            'integration_ready': False,
            'issues': []
        }

        try:
            # Test API initialization
            api_config = UnifiedPredictionConfig(
                performance_target_ms=self.target_performance_ms,
                enable_quality_prediction=True,
                enable_intelligent_routing=True
            )

            api = UnifiedPredictionAPI(api_config)
            validation['api_initialization'] = True

            # Test quality integrator
            if hasattr(api, 'quality_integrator') and api.quality_integrator:
                validation['quality_integrator'] = True

            # Test CPU optimizer
            cpu_config = CPUOptimizationConfig(performance_target_ms=self.target_performance_ms)
            cpu_optimizer = CPUPerformanceOptimizer(cpu_config)
            validation['cpu_optimizer'] = True

            # Test prediction functionality
            test_params = {
                'color_precision': 3.0,
                'corner_threshold': 30.0,
                'path_precision': 8.0,
                'layer_difference': 5.0,
                'filter_speckle': 2.0,
                'splice_threshold': 45.0,
                'mode': 0.0,
                'hierarchical': 1.0
            }

            try:
                result = api.predict_quality("mock_image.png", test_params)
                if result and result.quality_score > 0:
                    validation['prediction_success'] = True
            except Exception as e:
                validation['issues'].append(f"Prediction test failed: {str(e)}")

            # Test batch processing
            if not quick_mode:
                try:
                    batch_results = api.predict_quality_batch(
                        ["mock1.png", "mock2.png"],
                        [test_params, test_params]
                    )
                    if batch_results and len(batch_results) == 2:
                        validation['batch_processing'] = True
                except Exception as e:
                    validation['issues'].append(f"Batch processing test failed: {str(e)}")

            # Overall integration assessment
            validation['integration_ready'] = (
                validation['api_initialization'] and
                validation['prediction_success'] and
                (validation['batch_processing'] or quick_mode)
            )

            # Cleanup
            api.cleanup()
            cpu_optimizer.cleanup()

        except Exception as e:
            validation['issues'].append(f"Integration validation failed: {str(e)}")
            self.critical_issues.append("Integration validation error")

        return validation

    def _validate_production_deployment(self, quick_mode: bool) -> Dict[str, Any]:
        """Validate production deployment capabilities"""
        validation = {
            'deployment_manager': False,
            'health_monitoring': False,
            'metrics_export': False,
            'error_handling': False,
            'graceful_shutdown': False,
            'deployment_ready': False,
            'issues': []
        }

        try:
            # Test deployment manager initialization
            deploy_config = DeploymentConfig(
                deployment_name="readiness_test",
                performance_target_ms=self.target_performance_ms,
                enable_health_checks=not quick_mode,  # Skip health checks in quick mode
                enable_auto_restart=False,
                enable_metrics_export=not quick_mode,
                log_level="ERROR"  # Reduce log noise
            )

            deployment = ProductionDeploymentManager(deploy_config)
            validation['deployment_manager'] = True

            # Test basic functionality
            test_params = {
                'color_precision': 3.0,
                'corner_threshold': 30.0,
                'path_precision': 8.0,
                'layer_difference': 5.0,
                'filter_speckle': 2.0,
                'splice_threshold': 45.0,
                'mode': 0.0,
                'hierarchical': 1.0
            }

            try:
                result = deployment.predict_quality("mock_image.png", test_params)
                if result and 'quality_score' in result:
                    validation['error_handling'] = True
            except Exception as e:
                validation['issues'].append(f"Production prediction test failed: {str(e)}")

            # Test metrics
            try:
                metrics = deployment.get_current_metrics()
                if metrics and 'total_requests' in metrics:
                    validation['metrics_export'] = True
            except Exception as e:
                validation['issues'].append(f"Metrics test failed: {str(e)}")

            # Test graceful shutdown
            try:
                deployment.shutdown()
                validation['graceful_shutdown'] = True
            except Exception as e:
                validation['issues'].append(f"Graceful shutdown test failed: {str(e)}")

            # Overall deployment assessment
            validation['deployment_ready'] = (
                validation['deployment_manager'] and
                validation['error_handling'] and
                validation['graceful_shutdown']
            )

        except Exception as e:
            validation['issues'].append(f"Production deployment validation failed: {str(e)}")
            self.critical_issues.append("Production deployment validation error")

        return validation

    def _validate_end_to_end(self, quick_mode: bool) -> Dict[str, Any]:
        """Validate end-to-end system functionality"""
        validation = {
            'e2e_validator': False,
            'integration_tests': False,
            'performance_tests': False,
            'stress_tests': False,
            'overall_success_rate': 0.0,
            'performance_achievement_rate': 0.0,
            'e2e_ready': False,
            'issues': []
        }

        try:
            # Configure end-to-end validation
            e2e_config = ValidationConfig(
                performance_target_ms=self.target_performance_ms,
                test_iterations=25 if quick_mode else 100,
                stress_test_duration=30 if quick_mode else 120,
                enable_stress_tests=not quick_mode,
                enable_visual_validation=False,  # Skip visuals for speed
                save_detailed_logs=False
            )

            validator = EndToEndValidator(e2e_config)
            validation['e2e_validator'] = True

            # Run validation
            summary = validator.run_comprehensive_validation()

            validation['overall_success_rate'] = summary.success_rate
            validation['performance_achievement_rate'] = summary.performance_target_achievement_rate
            validation['integration_tests'] = summary.success_rate > 0.8
            validation['performance_tests'] = summary.performance_target_achievement_rate > 0.7

            if not quick_mode:
                validation['stress_tests'] = 'stress' in summary.test_categories

            # Overall E2E assessment
            validation['e2e_ready'] = (
                validation['overall_success_rate'] >= 0.8 and
                validation['performance_achievement_rate'] >= 0.7
            )

            if not validation['e2e_ready']:
                self.critical_issues.append("End-to-end validation failed")

        except Exception as e:
            validation['issues'].append(f"End-to-end validation failed: {str(e)}")
            self.critical_issues.append("End-to-end validation error")

        return validation

    def _generate_readiness_assessment(self, validations: Dict[str, Dict[str, Any]]) -> DeploymentReadinessResult:
        """Generate final deployment readiness assessment"""

        # Calculate readiness score
        component_scores = {
            'system': 1.0 if validations['system'].get('requirements_met', False) else 0.0,
            'dependencies': 1.0 if validations['dependencies'].get('dependencies_ready', False) else 0.0,
            'performance': validations['performance'].get('target_achievement_rate', 0.0),
            'integration': 1.0 if validations['integration'].get('integration_ready', False) else 0.0,
            'deployment': 1.0 if validations['deployment'].get('deployment_ready', False) else 0.0,
            'end_to_end': validations['e2e'].get('overall_success_rate', 0.0)
        }

        # Weighted readiness score
        weights = {
            'system': 0.1,
            'dependencies': 0.1,
            'performance': 0.3,
            'integration': 0.2,
            'deployment': 0.15,
            'end_to_end': 0.15
        }

        readiness_score = sum(component_scores[comp] * weights[comp] for comp in component_scores)

        # Determine overall readiness
        overall_ready = (
            readiness_score >= 0.8 and
            len(self.critical_issues) == 0 and
            component_scores['performance'] >= 0.7 and
            component_scores['integration'] >= 0.8
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(component_scores, validations)

        # Generate next steps
        next_steps = self._generate_next_steps(overall_ready, component_scores)

        return DeploymentReadinessResult(
            overall_ready=overall_ready,
            readiness_score=readiness_score,
            critical_issues=self.critical_issues,
            warnings=self.warnings,
            performance_validation=validations['performance'],
            integration_validation=validations['integration'],
            system_validation=validations['system'],
            deployment_validation=validations['deployment'],
            recommendations=recommendations,
            next_steps=next_steps,
            timestamp=time.time()
        )

    def _generate_recommendations(self, scores: Dict[str, float], validations: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate deployment recommendations"""
        recommendations = []

        if scores['system'] < 1.0:
            recommendations.append("Upgrade system resources (memory, disk space, Python version)")

        if scores['dependencies'] < 1.0:
            recommendations.append("Install missing required dependencies")

        if scores['performance'] < 0.8:
            recommendations.append("Optimize model performance or upgrade hardware")

        if scores['integration'] < 0.8:
            recommendations.append("Fix integration issues and test API functionality")

        if scores['deployment'] < 0.8:
            recommendations.append("Resolve production deployment configuration issues")

        if scores['end_to_end'] < 0.8:
            recommendations.append("Address end-to-end validation failures")

        if len(recommendations) == 0:
            recommendations.append("System is ready for production deployment")

        return recommendations

    def _generate_next_steps(self, overall_ready: bool, scores: Dict[str, float]) -> List[str]:
        """Generate next steps based on readiness"""
        if overall_ready:
            return [
                "✅ Deploy to production environment",
                "✅ Setup production monitoring",
                "✅ Configure alerting and health checks",
                "✅ Plan rollout strategy",
                "✅ Document operational procedures"
            ]
        else:
            next_steps = []

            if scores['system'] < 1.0:
                next_steps.append("🔧 Fix system environment issues")

            if scores['dependencies'] < 1.0:
                next_steps.append("📦 Install missing dependencies")

            if scores['performance'] < 0.8:
                next_steps.append("⚡ Optimize performance")

            if scores['integration'] < 0.8:
                next_steps.append("🔗 Fix integration issues")

            if scores['deployment'] < 0.8:
                next_steps.append("🏭 Resolve deployment issues")

            next_steps.append("🔄 Re-run deployment readiness validation")

            return next_steps

def print_readiness_report(result: DeploymentReadinessResult):
    """Print comprehensive readiness report"""
    print("\n" + "="*80)
    print("🚀 DEPLOYMENT READINESS ASSESSMENT")
    print("="*80)

    # Overall status
    status_emoji = "✅" if result.overall_ready else "❌"
    print(f"\n{status_emoji} OVERALL STATUS: {'READY FOR DEPLOYMENT' if result.overall_ready else 'NOT READY'}")
    print(f"📊 Readiness Score: {result.readiness_score:.1%}")

    # Critical issues
    if result.critical_issues:
        print(f"\n🚨 CRITICAL ISSUES ({len(result.critical_issues)}):")
        for issue in result.critical_issues:
            print(f"   ❌ {issue}")

    # Warnings
    if result.warnings:
        print(f"\n⚠️  WARNINGS ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"   ⚠️  {warning}")

    # Performance summary
    perf = result.performance_validation
    print(f"\n⚡ PERFORMANCE:")
    print(f"   Target: {perf.get('target_ms', 0):.0f}ms")
    print(f"   Average: {perf.get('avg_inference_time_ms', 0):.1f}ms")
    print(f"   Achievement Rate: {perf.get('target_achievement_rate', 0):.1%}")

    # Integration summary
    integration = result.integration_validation
    print(f"\n🔗 INTEGRATION:")
    print(f"   API Ready: {'✅' if integration.get('api_initialization') else '❌'}")
    print(f"   Prediction: {'✅' if integration.get('prediction_success') else '❌'}")
    print(f"   Batch Processing: {'✅' if integration.get('batch_processing') else '❌'}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"   {i}. {rec}")

    # Next steps
    print(f"\n📋 NEXT STEPS:")
    for i, step in enumerate(result.next_steps, 1):
        print(f"   {i}. {step}")

    print("\n" + "="*80)

def main():
    """Main command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(description='Validate deployment readiness for quality prediction system')
    parser.add_argument('--target-ms', type=float, default=25.0, help='Performance target in milliseconds')
    parser.add_argument('--quick', action='store_true', help='Run quick validation (fewer tests)')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run validation
    validator = DeploymentReadinessValidator(args.target_ms)
    result = validator.validate_deployment_readiness(args.quick)

    # Print report
    print_readiness_report(result)

    # Save to file if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)
            print(f"\n💾 Results saved to: {args.output}")
        except Exception as e:
            print(f"\n❌ Failed to save results: {e}")

    # Return appropriate exit code
    return 0 if result.overall_ready else 1

if __name__ == "__main__":
    exit(main())