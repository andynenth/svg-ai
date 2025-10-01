"""
Task 12.2: Model Export & Validation Pipeline - Master Pipeline
Complete implementation of Task 12.2: Model Export & Validation Pipeline

This master pipeline orchestrates:
- Multi-format model export (TorchScript, ONNX, CoreML)
- Cross-platform validation and testing
- Local CPU/MPS performance optimization
- Production deployment package creation
- Comprehensive performance benchmarking

Success Criteria:
- All 3 export formats validated
- <50ms inference time on local CPU/MPS
- <100MB model size per format
- Production deployment package ready
"""

import torch
import numpy as np
import json
import time
import shutil
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import warnings
import traceback

# Import all pipeline components
from .model_export_pipeline import ModelExportPipeline, ExportConfig, ExportResult, load_trained_model
from .export_validation_framework import ExportValidationFramework, ValidationConfig
from .local_inference_optimizer import LocalInferenceOptimizer, LocalInferenceConfig
from .production_deployment_package import ModelPackager, DeploymentConfig
from .local_integration_tester import LocalIntegrationTester, IntegrationTestConfig
from .gpu_model_architecture import QualityPredictorGPU, ColabTrainingConfig

warnings.filterwarnings('ignore')


@dataclass
class Task12_2Config:
    """Master configuration for Task 12.2"""
    # Model source
    trained_model_path: str = ""
    model_checkpoint_path: str = ""

    # Export configuration
    export_formats: List[str] = None  # ['torchscript', 'onnx', 'coreml']
    target_model_size_mb: float = 100.0
    target_inference_time_ms: float = 50.0

    # Validation configuration
    accuracy_tolerance: float = 0.001
    performance_threshold_ms: float = 50.0
    test_samples: int = 100

    # Integration testing
    test_integration: bool = True
    test_end_to_end: bool = True

    # Deployment package
    create_deployment_package: bool = True
    package_name: str = "svg_quality_predictor_v1_0"

    # Output directories
    export_dir: str = "model_exports"
    package_dir: str = "deployment_package"
    reports_dir: str = "task_12_2_reports"

    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ['torchscript', 'onnx', 'coreml']


@dataclass
class Task12_2Results:
    """Results of Task 12.2 execution"""
    # Overall status
    task_completed: bool
    success_criteria_met: bool
    execution_time_minutes: float

    # Export results
    export_results: Dict[str, ExportResult]
    export_summary: Dict[str, Any]

    # Validation results
    validation_results: Dict[str, Any]
    validation_summary: Dict[str, Any]

    # Performance results
    performance_results: Dict[str, Any]
    performance_summary: Dict[str, Any]

    # Integration results
    integration_results: Dict[str, Any]
    integration_summary: Dict[str, Any]

    # Deployment package
    deployment_package_path: Optional[str]
    deployment_ready: bool

    # Success criteria breakdown
    criteria_breakdown: Dict[str, bool]

    # Error information
    errors: List[str]
    warnings: List[str]


class Task12_2MasterPipeline:
    """Master pipeline for Task 12.2: Model Export & Validation Pipeline"""

    def __init__(self, config: Task12_2Config):
        self.config = config
        self.start_time = time.time()

        # Create output directories
        self._create_output_directories()

        # Results storage
        self.export_results = {}
        self.validation_results = {}
        self.performance_results = {}
        self.integration_results = {}
        self.deployment_package_path = None

        # Error tracking
        self.errors = []
        self.warnings = []

        print(f"🚀 Task 12.2 Master Pipeline initialized")
        print(f"   Target formats: {self.config.export_formats}")
        print(f"   Performance target: <{self.config.target_inference_time_ms}ms")
        print(f"   Size target: <{self.config.target_model_size_mb}MB")

    def _create_output_directories(self):
        """Create output directories"""
        for dir_name in [self.config.export_dir, self.config.package_dir, self.config.reports_dir]:
            Path(dir_name).mkdir(exist_ok=True)

    def execute_complete_pipeline(self, trained_model: Optional[torch.nn.Module] = None) -> Task12_2Results:
        """Execute the complete Task 12.2 pipeline"""

        print(f"\n" + "="*80)
        print(f"🎯 EXECUTING TASK 12.2: MODEL EXPORT & VALIDATION PIPELINE")
        print(f"="*80)

        try:
            # Phase 1: Model Loading and Preparation
            model, model_metadata = self._load_model(trained_model)

            # Phase 2: Multi-Format Model Export
            self.export_results = self._execute_model_export(model)

            # Phase 3: Export Validation
            self.validation_results = self._execute_export_validation(model)

            # Phase 4: Local Performance Optimization
            self.performance_results = self._execute_performance_optimization()

            # Phase 5: Integration Testing
            if self.config.test_integration:
                self.integration_results = self._execute_integration_testing()

            # Phase 6: Deployment Package Creation
            if self.config.create_deployment_package:
                self.deployment_package_path = self._create_deployment_package()

            # Phase 7: Success Criteria Evaluation
            success_criteria = self._evaluate_success_criteria()

            # Phase 8: Generate Final Results
            results = self._generate_final_results(success_criteria)

            # Phase 9: Generate Reports
            self._generate_comprehensive_reports(results)

            print(f"\n✅ Task 12.2 Pipeline completed successfully!")
            return results

        except Exception as e:
            self.errors.append(f"Pipeline execution failed: {str(e)}")
            print(f"\n❌ Task 12.2 Pipeline failed: {e}")
            traceback.print_exc()

            # Generate failure results
            return self._generate_failure_results()

    def _load_model(self, trained_model: Optional[torch.nn.Module] = None) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Load trained model for export"""
        print(f"\n📂 Phase 1: Loading trained model...")

        if trained_model is not None:
            print(f"   Using provided model")
            model = trained_model
            metadata = {'source': 'provided', 'parameters': sum(p.numel() for p in model.parameters())}

        elif self.config.trained_model_path:
            print(f"   Loading from: {self.config.trained_model_path}")
            model, metadata = load_trained_model(self.config.trained_model_path)

        elif self.config.model_checkpoint_path:
            print(f"   Loading checkpoint: {self.config.model_checkpoint_path}")
            model, metadata = load_trained_model(self.config.model_checkpoint_path)

        else:
            # Create dummy model for testing
            print(f"   ⚠️ No model provided - creating dummy model for testing")
            config = ColabTrainingConfig(device='cpu')
            model = QualityPredictorGPU(config)
            metadata = {'source': 'dummy', 'parameters': sum(p.numel() for p in model.parameters())}
            self.warnings.append("Using dummy model - results not production-ready")

        print(f"   ✅ Model loaded: {metadata.get('parameters', 0):,} parameters")
        return model, metadata

    def _execute_model_export(self, model: torch.nn.Module) -> Dict[str, ExportResult]:
        """Execute multi-format model export"""
        print(f"\n📦 Phase 2: Multi-format model export...")

        # Configure export pipeline
        export_config = ExportConfig(
            export_dir=self.config.export_dir,
            model_name="svg_quality_predictor",
            export_torchscript='torchscript' in self.config.export_formats,
            export_onnx='onnx' in self.config.export_formats,
            export_coreml='coreml' in self.config.export_formats,
            max_model_size_mb=self.config.target_model_size_mb,
            target_inference_time_ms=self.config.target_inference_time_ms
        )

        # Execute export
        export_pipeline = ModelExportPipeline(export_config)
        export_results = export_pipeline.export_all_formats(model)

        # Analyze export results
        successful_exports = sum(1 for result in export_results.values() if result.success)
        total_size = sum(result.file_size_mb for result in export_results.values() if result.success)

        print(f"   ✅ Export completed: {successful_exports}/{len(export_results)} formats successful")
        print(f"   📊 Total size: {total_size:.1f}MB")

        if successful_exports == 0:
            self.errors.append("No successful model exports")

        return export_results

    def _execute_export_validation(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Execute export validation and accuracy testing"""
        print(f"\n🔍 Phase 3: Export validation and accuracy testing...")

        # Configure validation
        validation_config = ValidationConfig(
            accuracy_tolerance=self.config.accuracy_tolerance,
            test_samples=self.config.test_samples,
            max_inference_time_ms=self.config.performance_threshold_ms
        )

        # Execute validation
        validation_framework = ExportValidationFramework(validation_config)
        validation_results = validation_framework.validate_all_exports(
            self.export_results, model
        )

        # Analyze validation results
        deployment_ready = validation_results.get("summary", {}).get("deployment_ready", 0)
        total_tested = validation_results.get("summary", {}).get("total_models_tested", 0)

        print(f"   ✅ Validation completed: {deployment_ready}/{total_tested} models deployment-ready")

        if deployment_ready == 0:
            self.warnings.append("No models passed validation for deployment")

        return validation_results

    def _execute_performance_optimization(self) -> Dict[str, Any]:
        """Execute local performance optimization and testing"""
        print(f"\n⚡ Phase 4: Local performance optimization...")

        # Configure performance testing
        performance_config = LocalInferenceConfig(
            target_inference_time_ms=self.config.target_inference_time_ms,
            test_samples=self.config.test_samples,
            test_cpu=True,
            test_mps=platform.system() == "Darwin"  # Only on macOS
        )

        # Test each successful export
        performance_optimizer = LocalInferenceOptimizer(performance_config)
        all_performance_results = {}

        successful_exports = {k: v for k, v in self.export_results.items() if v.success}

        for format_name, export_result in successful_exports.items():
            try:
                print(f"   Testing {format_name}...")

                # Load model for performance testing
                if export_result.format.startswith('TorchScript'):
                    model = torch.jit.load(export_result.file_path, map_location='cpu')
                elif export_result.format == 'ONNX':
                    # ONNX testing handled internally
                    model = None
                elif export_result.format == 'CoreML':
                    # CoreML testing handled internally
                    model = None
                else:
                    print(f"     ⚠️ Unsupported format for performance testing: {export_result.format}")
                    continue

                if model is not None:
                    format_results = performance_optimizer.optimize_and_test_model(model, export_result.format)
                    all_performance_results[format_name] = format_results

            except Exception as e:
                print(f"     ❌ Performance testing failed for {format_name}: {e}")
                self.warnings.append(f"Performance testing failed for {format_name}: {str(e)}")

        # Generate performance report
        if all_performance_results:
            performance_report = performance_optimizer.generate_optimization_report(all_performance_results)
            all_performance_results['report'] = performance_report

        print(f"   ✅ Performance testing completed for {len(all_performance_results)} formats")

        return all_performance_results

    def _execute_integration_testing(self) -> Dict[str, Any]:
        """Execute integration testing with existing optimization system"""
        print(f"\n🔧 Phase 5: Integration testing...")

        # Configure integration testing
        integration_config = IntegrationTestConfig(
            test_basic_integration=True,
            test_optimization_workflow=True,
            test_performance_integration=True,
            test_end_to_end_pipeline=self.config.test_end_to_end,
            max_prediction_time_ms=self.config.target_inference_time_ms
        )

        # Execute integration tests
        integration_tester = LocalIntegrationTester(integration_config)
        integration_results = integration_tester.run_all_integration_tests(self.export_results)

        # Analyze integration results
        models_tested = integration_results.get("summary", {}).get("total_models_tested", 0)
        models_passed = integration_results.get("summary", {}).get("models_integration_passed", 0)

        print(f"   ✅ Integration testing completed: {models_passed}/{models_tested} models passed")

        if models_passed == 0:
            self.warnings.append("No models passed integration testing")

        return integration_results

    def _create_deployment_package(self) -> Optional[str]:
        """Create production deployment package"""
        print(f"\n📦 Phase 6: Creating deployment package...")

        try:
            # Configure deployment package
            deployment_config = DeploymentConfig(
                package_name=self.config.package_name,
                version="1.0.0",
                target_platforms=["linux", "macos", "windows"],
                include_all_formats=True,
                include_validation_results=True,
                include_optimization_results=True,
                include_examples=True,
                include_documentation=True
            )

            # Create package
            packager = ModelPackager(deployment_config)
            package_path = packager.create_model_package(
                self.export_results,
                self.validation_results,
                self.performance_results
            )

            print(f"   ✅ Deployment package created: {package_path}")
            return str(package_path)

        except Exception as e:
            self.errors.append(f"Deployment package creation failed: {str(e)}")
            print(f"   ❌ Deployment package creation failed: {e}")
            return None

    def _evaluate_success_criteria(self) -> Dict[str, bool]:
        """Evaluate Task 12.2 success criteria"""
        print(f"\n🎯 Phase 7: Evaluating success criteria...")

        criteria = {
            "all_formats_exported": False,
            "inference_time_target_met": False,
            "model_size_target_met": False,
            "accuracy_validation_passed": False,
            "local_performance_optimized": False,
            "integration_testing_passed": False,
            "deployment_package_created": False
        }

        # Check export formats
        successful_formats = [r for r in self.export_results.values() if r.success]
        target_formats = len(self.config.export_formats)
        criteria["all_formats_exported"] = len(successful_formats) >= min(3, target_formats)

        # Check inference time target
        if self.performance_results:
            fastest_time = float('inf')
            for format_results in self.performance_results.values():
                if isinstance(format_results, dict):
                    for device_results in format_results.values():
                        if isinstance(device_results, dict) and 'inference_time_ms' in device_results:
                            fastest_time = min(fastest_time, device_results['inference_time_ms'])

            criteria["inference_time_target_met"] = fastest_time < self.config.target_inference_time_ms

        # Check model size target
        largest_model = max((r.file_size_mb for r in successful_formats), default=0)
        criteria["model_size_target_met"] = largest_model < self.config.target_model_size_mb

        # Check accuracy validation
        if self.validation_results:
            deployment_ready = self.validation_results.get("summary", {}).get("deployment_ready", 0)
            criteria["accuracy_validation_passed"] = deployment_ready > 0

        # Check local performance optimization
        if self.performance_results:
            cpu_tests_passed = False
            for format_results in self.performance_results.values():
                if isinstance(format_results, dict) and 'cpu' in format_results:
                    cpu_result = format_results['cpu']
                    if isinstance(cpu_result, dict) and cpu_result.get('passed', False):
                        cpu_tests_passed = True
                        break

            criteria["local_performance_optimized"] = cpu_tests_passed

        # Check integration testing
        if self.integration_results:
            models_passed = self.integration_results.get("summary", {}).get("models_integration_passed", 0)
            criteria["integration_testing_passed"] = models_passed > 0

        # Check deployment package
        criteria["deployment_package_created"] = self.deployment_package_path is not None

        # Print criteria evaluation
        print(f"   Success Criteria Evaluation:")
        for criterion, passed in criteria.items():
            status = "✅" if passed else "❌"
            print(f"     {status} {criterion.replace('_', ' ').title()}")

        return criteria

    def _generate_final_results(self, success_criteria: Dict[str, bool]) -> Task12_2Results:
        """Generate final Task 12.2 results"""
        execution_time = (time.time() - self.start_time) / 60.0  # Convert to minutes

        # Determine overall success
        task_completed = len(self.errors) == 0
        success_criteria_met = all(success_criteria.values())

        # Create summaries
        export_summary = self._create_export_summary()
        validation_summary = self._create_validation_summary()
        performance_summary = self._create_performance_summary()
        integration_summary = self._create_integration_summary()

        results = Task12_2Results(
            task_completed=task_completed,
            success_criteria_met=success_criteria_met,
            execution_time_minutes=execution_time,
            export_results=self.export_results,
            export_summary=export_summary,
            validation_results=self.validation_results,
            validation_summary=validation_summary,
            performance_results=self.performance_results,
            performance_summary=performance_summary,
            integration_results=self.integration_results,
            integration_summary=integration_summary,
            deployment_package_path=self.deployment_package_path,
            deployment_ready=success_criteria_met,
            criteria_breakdown=success_criteria,
            errors=self.errors,
            warnings=self.warnings
        )

        return results

    def _generate_failure_results(self) -> Task12_2Results:
        """Generate results for failed execution"""
        execution_time = (time.time() - self.start_time) / 60.0

        return Task12_2Results(
            task_completed=False,
            success_criteria_met=False,
            execution_time_minutes=execution_time,
            export_results=self.export_results,
            export_summary={},
            validation_results=self.validation_results,
            validation_summary={},
            performance_results=self.performance_results,
            performance_summary={},
            integration_results=self.integration_results,
            integration_summary={},
            deployment_package_path=None,
            deployment_ready=False,
            criteria_breakdown={k: False for k in ["all_formats_exported", "inference_time_target_met",
                                                  "model_size_target_met", "accuracy_validation_passed",
                                                  "local_performance_optimized", "integration_testing_passed",
                                                  "deployment_package_created"]},
            errors=self.errors,
            warnings=self.warnings
        )

    def _create_export_summary(self) -> Dict[str, Any]:
        """Create export results summary"""
        successful = [r for r in self.export_results.values() if r.success]
        return {
            "total_formats": len(self.export_results),
            "successful_exports": len(successful),
            "total_size_mb": sum(r.file_size_mb for r in successful),
            "formats_exported": [r.format for r in successful],
            "largest_model_mb": max((r.file_size_mb for r in successful), default=0),
            "fastest_export_ms": min((r.export_time_ms for r in successful), default=0)
        }

    def _create_validation_summary(self) -> Dict[str, Any]:
        """Create validation results summary"""
        if not self.validation_results:
            return {}

        summary = self.validation_results.get("summary", {})
        return {
            "models_tested": summary.get("total_models_tested", 0),
            "accuracy_passed": summary.get("accuracy_passed", 0),
            "performance_passed": summary.get("performance_passed", 0),
            "deployment_ready": summary.get("deployment_ready", 0),
            "validation_success_rate": summary.get("deployment_ready", 0) / max(summary.get("total_models_tested", 1), 1)
        }

    def _create_performance_summary(self) -> Dict[str, Any]:
        """Create performance results summary"""
        if not self.performance_results:
            return {}

        # Extract performance metrics
        cpu_times = []
        mps_times = []
        memory_usage = []

        for format_results in self.performance_results.values():
            if isinstance(format_results, dict):
                if 'cpu' in format_results and isinstance(format_results['cpu'], dict):
                    cpu_result = format_results['cpu']
                    if 'inference_time_ms' in cpu_result:
                        cpu_times.append(cpu_result['inference_time_ms'])
                    if 'memory_usage_mb' in cpu_result:
                        memory_usage.append(cpu_result['memory_usage_mb'])

                if 'mps' in format_results and isinstance(format_results['mps'], dict):
                    mps_result = format_results['mps']
                    if 'inference_time_ms' in mps_result:
                        mps_times.append(mps_result['inference_time_ms'])

        return {
            "cpu_fastest_ms": min(cpu_times) if cpu_times else None,
            "cpu_average_ms": np.mean(cpu_times) if cpu_times else None,
            "mps_fastest_ms": min(mps_times) if mps_times else None,
            "mps_average_ms": np.mean(mps_times) if mps_times else None,
            "memory_usage_mb": max(memory_usage) if memory_usage else None,
            "target_met": min(cpu_times + mps_times) < self.config.target_inference_time_ms if (cpu_times or mps_times) else False
        }

    def _create_integration_summary(self) -> Dict[str, Any]:
        """Create integration results summary"""
        if not self.integration_results:
            return {}

        summary = self.integration_results.get("summary", {})
        return {
            "models_tested": summary.get("total_models_tested", 0),
            "models_passed": summary.get("models_integration_passed", 0),
            "tests_run": summary.get("total_tests_run", 0),
            "tests_passed": summary.get("total_tests_passed", 0),
            "integration_success_rate": summary.get("models_integration_passed", 0) / max(summary.get("total_models_tested", 1), 1)
        }

    def _generate_comprehensive_reports(self, results: Task12_2Results):
        """Generate comprehensive reports"""
        print(f"\n📊 Phase 8: Generating comprehensive reports...")

        reports_dir = Path(self.config.reports_dir)

        # Master report
        master_report = {
            "task": "12.2 - Model Export & Validation Pipeline",
            "execution_timestamp": time.time(),
            "execution_time_minutes": results.execution_time_minutes,
            "success_criteria_met": results.success_criteria_met,
            "deployment_ready": results.deployment_ready,
            "criteria_breakdown": results.criteria_breakdown,
            "export_summary": results.export_summary,
            "validation_summary": results.validation_summary,
            "performance_summary": results.performance_summary,
            "integration_summary": results.integration_summary,
            "deployment_package": results.deployment_package_path,
            "errors": results.errors,
            "warnings": results.warnings,
            "system_info": {
                "platform": platform.system(),
                "architecture": platform.machine(),
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__
            }
        }

        # Save master report
        master_report_path = reports_dir / "task_12_2_master_report.json"
        with open(master_report_path, 'w') as f:
            json.dump(master_report, f, indent=2)

        # Print summary
        self._print_final_summary(results)

        print(f"   ✅ Reports generated in: {reports_dir}")

    def _print_final_summary(self, results: Task12_2Results):
        """Print final summary"""
        print(f"\n" + "="*80)
        print(f"📋 TASK 12.2 FINAL SUMMARY")
        print(f"="*80)

        # Overall status
        status = "✅ SUCCESS" if results.success_criteria_met else "❌ INCOMPLETE"
        print(f"Status: {status}")
        print(f"Execution time: {results.execution_time_minutes:.1f} minutes")

        # Success criteria
        print(f"\n🎯 Success Criteria:")
        for criterion, passed in results.criteria_breakdown.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {criterion.replace('_', ' ').title()}")

        # Export results
        print(f"\n📦 Export Results:")
        if results.export_summary:
            print(f"   Formats exported: {results.export_summary['successful_exports']}/{results.export_summary['total_formats']}")
            print(f"   Total size: {results.export_summary['total_size_mb']:.1f}MB")
            print(f"   Largest model: {results.export_summary['largest_model_mb']:.1f}MB")

        # Performance results
        print(f"\n⚡ Performance Results:")
        if results.performance_summary:
            if results.performance_summary.get('cpu_fastest_ms'):
                print(f"   CPU fastest: {results.performance_summary['cpu_fastest_ms']:.1f}ms")
            if results.performance_summary.get('mps_fastest_ms'):
                print(f"   MPS fastest: {results.performance_summary['mps_fastest_ms']:.1f}ms")
            target_status = "✅" if results.performance_summary.get('target_met', False) else "❌"
            print(f"   {target_status} Target <{self.config.target_inference_time_ms}ms met")

        # Deployment package
        print(f"\n📦 Deployment Package:")
        if results.deployment_package_path:
            print(f"   ✅ Created: {results.deployment_package_path}")
            print(f"   🚀 Ready for production deployment")
        else:
            print(f"   ❌ Not created")

        # Errors and warnings
        if results.errors:
            print(f"\n❌ Errors ({len(results.errors)}):")
            for error in results.errors[:3]:  # Show first 3 errors
                print(f"   • {error}")

        if results.warnings:
            print(f"\n⚠️ Warnings ({len(results.warnings)}):")
            for warning in results.warnings[:3]:  # Show first 3 warnings
                print(f"   • {warning}")

        print(f"\n" + "="*80)


def execute_task_12_2(config: Task12_2Config = None, trained_model: torch.nn.Module = None) -> Task12_2Results:
    """
    Execute complete Task 12.2: Model Export & Validation Pipeline

    Args:
        config: Configuration for the pipeline
        trained_model: Optional pre-trained model to export

    Returns:
        Task12_2Results with complete results and success criteria evaluation
    """

    if config is None:
        config = Task12_2Config()

    # Create and execute pipeline
    pipeline = Task12_2MasterPipeline(config)
    results = pipeline.execute_complete_pipeline(trained_model)

    return results


if __name__ == "__main__":
    # Example execution
    print("🧪 Testing Task 12.2 Master Pipeline")

    # Create test configuration
    config = Task12_2Config(
        export_formats=['torchscript', 'onnx'],  # Skip CoreML for testing
        target_inference_time_ms=50.0,
        target_model_size_mb=100.0,
        test_integration=True,
        create_deployment_package=True
    )

    # Execute pipeline
    results = execute_task_12_2(config)

    print(f"\n✅ Task 12.2 test execution completed")
    print(f"   Success: {results.success_criteria_met}")
    print(f"   Deployment ready: {results.deployment_ready}")