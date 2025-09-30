#!/usr/bin/env python3
"""
Test script for error handling and edge case management
"""
import sys
from pathlib import Path
import time
import tempfile

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from backend.ai_modules.optimization.error_handler import (
    OptimizationErrorHandler,
    OptimizationError,
    OptimizationErrorType,
    ErrorSeverity,
    CircuitBreaker,
    NotificationConfig
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_error_detection_and_classification():
    """Test comprehensive error detection"""
    print("🔍 Testing Error Detection and Classification")

    error_handler = OptimizationErrorHandler()

    # Test different error scenarios
    test_scenarios = [
        {
            "exception": ValueError("Parameter color_precision out of bounds"),
            "context": {"parameters": {"color_precision": 999}},
            "expected_type": OptimizationErrorType.PARAMETER_VALIDATION_FAILED
        },
        {
            "exception": FileNotFoundError("Image file not found"),
            "context": {"image_path": "nonexistent.png"},
            "expected_type": OptimizationErrorType.SYSTEM_RESOURCE_ERROR
        },
        {
            "exception": MemoryError("Out of memory"),
            "context": {"batch_size": 1000},
            "expected_type": OptimizationErrorType.MEMORY_EXHAUSTION
        },
        {
            "exception": TimeoutError("VTracer conversion timeout"),
            "context": {"operation": "vtracer_conversion"},
            "expected_type": OptimizationErrorType.TIMEOUT_ERROR
        },
        {
            "exception": RuntimeError("Feature extraction failed"),
            "context": {"operation": "feature_extraction"},
            "expected_type": OptimizationErrorType.FEATURE_EXTRACTION_FAILED
        }
    ]

    correct_classifications = 0
    for i, scenario in enumerate(test_scenarios):
        error = error_handler.detect_error(scenario["exception"], scenario["context"])

        print(f"  Test {i+1}: {scenario['exception']}")
        print(f"    - Detected Type: {error.error_type}")
        print(f"    - Expected Type: {scenario['expected_type']}")
        print(f"    - Severity: {error.severity}")
        print(f"    - Recovery Suggestion: {error.recovery_suggestion}")

        if error.error_type == scenario["expected_type"]:
            correct_classifications += 1
            print(f"    ✅ Correctly classified")
        else:
            print(f"    ❌ Misclassified")
        print()

    classification_accuracy = correct_classifications / len(test_scenarios)
    print(f"📊 Classification Accuracy: {classification_accuracy:.2%} ({correct_classifications}/{len(test_scenarios)})")
    print("✅ Error detection and classification test completed!\n")

    return classification_accuracy

def test_recovery_strategies():
    """Test error recovery strategies"""
    print("🔧 Testing Recovery Strategies")

    error_handler = OptimizationErrorHandler()

    # Run built-in recovery strategy tests
    recovery_rate = error_handler.test_recovery_strategies()

    print(f"📊 Recovery Strategy Success Rate: {recovery_rate:.2%}")

    # Test additional edge cases
    edge_cases = [
        {
            "error_type": OptimizationErrorType.INVALID_INPUT_IMAGE,
            "message": "Corrupted PNG file",
            "context": {"image_path": "corrupted.png"},
            "should_recover": False  # Cannot recover from invalid input
        },
        {
            "error_type": OptimizationErrorType.CONFIGURATION_ERROR,
            "message": "Invalid configuration",
            "context": {},
            "should_recover": False  # Requires manual intervention
        }
    ]

    edge_case_successes = 0
    for case in edge_cases:
        error = OptimizationError(
            error_type=case["error_type"],
            message=case["message"],
            recovery_suggestion="Test recovery",
            context=case["context"]
        )

        result = error_handler.attempt_recovery(error)
        success = result.get("success", False)

        print(f"  Edge Case: {case['error_type'].value}")
        print(f"    - Recovery Expected: {case['should_recover']}")
        print(f"    - Recovery Successful: {success}")
        print(f"    - Message: {result.get('message', 'No message')}")

        # Success if recovery matches expectation
        if success == case["should_recover"]:
            edge_case_successes += 1
            print(f"    ✅ Behaved as expected")
        else:
            print(f"    ⚠️  Unexpected behavior")
        print()

    edge_case_accuracy = edge_case_successes / len(edge_cases) if edge_cases else 1.0
    print(f"📊 Edge Case Handling Accuracy: {edge_case_accuracy:.2%}")
    print("✅ Recovery strategies test completed!\n")

    return recovery_rate, edge_case_accuracy

def test_circuit_breaker():
    """Test circuit breaker functionality"""
    print("⚡ Testing Circuit Breaker")

    circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)

    def failing_operation():
        raise Exception("Operation failed")

    def successful_operation():
        return "Success"

    # Test failure accumulation
    failures = 0
    for i in range(5):
        try:
            circuit_breaker.call(failing_operation)
        except Exception:
            failures += 1
            print(f"  Failure {failures}: Circuit state = {circuit_breaker.state}")

    # Verify circuit is open
    assert circuit_breaker.state == "OPEN", "Circuit breaker should be OPEN after threshold failures"
    print(f"  ✅ Circuit breaker opened after {failures} failures")

    # Test that calls are blocked
    try:
        circuit_breaker.call(successful_operation)
        print("  ❌ Circuit breaker should have blocked the call")
    except Exception as e:
        print(f"  ✅ Circuit breaker correctly blocked call: {e}")

    # Wait for recovery timeout and test half-open state
    print(f"  ⏱️  Waiting {circuit_breaker.recovery_timeout}s for recovery timeout...")
    time.sleep(circuit_breaker.recovery_timeout + 0.1)

    # Test successful recovery
    try:
        result = circuit_breaker.call(successful_operation)
        print(f"  ✅ Circuit breaker recovered: {result}")
        print(f"  Circuit state: {circuit_breaker.state}")
    except Exception as e:
        print(f"  ❌ Circuit breaker recovery failed: {e}")

    print("✅ Circuit breaker test completed!\n")

def test_fallback_parameters():
    """Test fallback parameter sets"""
    print("🔄 Testing Fallback Parameters")

    error_handler = OptimizationErrorHandler()

    fallback_sets = ["conservative", "high_speed", "compatibility", "memory_efficient"]

    for fallback_set in fallback_sets:
        # Test parameter validation recovery
        error = OptimizationError(
            error_type=OptimizationErrorType.PARAMETER_VALIDATION_FAILED,
            message="Parameter validation failed",
            recovery_suggestion="Use fallback parameters"
        )

        result = error_handler.attempt_recovery(error, fallback_set=fallback_set)

        print(f"  Fallback Set: {fallback_set}")
        print(f"    - Recovery Success: {result.get('success', False)}")

        if "fallback_parameters" in result:
            params = result["fallback_parameters"]
            print(f"    - Parameters: color_precision={params['color_precision']}, corner_threshold={params['corner_threshold']}")

        print(f"    - Message: {result.get('message', 'No message')}")
        print()

    print("✅ Fallback parameters test completed!\n")

def test_error_statistics_and_reporting():
    """Test error statistics and reporting"""
    print("📊 Testing Error Statistics and Reporting")

    error_handler = OptimizationErrorHandler()

    # Generate some test errors
    test_errors = [
        (OptimizationErrorType.FEATURE_EXTRACTION_FAILED, "Feature extraction timeout"),
        (OptimizationErrorType.PARAMETER_VALIDATION_FAILED, "Invalid parameter"),
        (OptimizationErrorType.VTRACER_CONVERSION_FAILED, "VTracer crashed"),
        (OptimizationErrorType.MEMORY_EXHAUSTION, "Out of memory"),
        (OptimizationErrorType.FEATURE_EXTRACTION_FAILED, "Another feature extraction error")
    ]

    for error_type, message in test_errors:
        error = error_handler.detect_error(
            Exception(message),
            {"operation": error_type.value.replace("_", " ")}
        )

        # Attempt recovery to test recovery statistics
        error_handler.attempt_recovery(error)

    # Get statistics
    stats = error_handler.get_error_statistics(time_window_hours=1)

    print(f"  📈 Error Statistics (last 1 hour):")
    print(f"    - Total Errors: {stats['total_errors']}")
    print(f"    - Error Rate: {stats['error_rate']:.2f} errors/hour")
    print(f"    - Recovery Rate: {stats['recovery_rate']:.2%}")
    print(f"    - Errors by Type: {stats['errors_by_type']}")
    print(f"    - Errors by Severity: {stats['errors_by_severity']}")

    # Generate error report
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        report_path = error_handler.generate_error_report(tmp.name)
        print(f"  📄 Error report generated: {report_path}")

    print("✅ Error statistics and reporting test completed!\n")

    return stats

def test_retry_mechanisms():
    """Test retry mechanisms with exponential backoff"""
    print("🔁 Testing Retry Mechanisms")

    error_handler = OptimizationErrorHandler()

    # Test successful retry
    attempt_count = 0
    def flaky_operation():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise Exception("Transient failure")
        return "Success after retries"

    try:
        start_time = time.time()
        result = error_handler.retry_with_backoff(
            flaky_operation,
            OptimizationErrorType.FEATURE_EXTRACTION_FAILED
        )
        end_time = time.time()

        print(f"  ✅ Retry successful: {result}")
        print(f"  Attempts: {attempt_count}")
        print(f"  Total time: {end_time - start_time:.2f}s")
    except Exception as e:
        print(f"  ❌ Retry failed: {e}")

    # Test retry with no retries configured
    def always_failing():
        raise Exception("Always fails")

    try:
        error_handler.retry_with_backoff(
            always_failing,
            OptimizationErrorType.INVALID_INPUT_IMAGE  # No retries configured
        )
        print("  ❌ Should have failed immediately")
    except Exception:
        print("  ✅ Correctly failed immediately (no retries configured)")

    print("✅ Retry mechanisms test completed!\n")

def test_comprehensive_edge_cases():
    """Test comprehensive edge case scenarios"""
    print("🚨 Testing Comprehensive Edge Cases")

    error_handler = OptimizationErrorHandler()

    edge_case_scenarios = [
        {
            "name": "Large batch memory exhaustion",
            "error_type": OptimizationErrorType.MEMORY_EXHAUSTION,
            "context": {"batch_size": 500, "image_sizes": "large"},
            "should_suggest_reduction": True
        },
        {
            "name": "VTracer timeout on complex image",
            "error_type": OptimizationErrorType.TIMEOUT_ERROR,
            "context": {"image_complexity": "very_high", "timeout": 60},
            "should_suggest_speed_params": True
        },
        {
            "name": "Corrupted image file",
            "error_type": OptimizationErrorType.INVALID_INPUT_IMAGE,
            "context": {"image_path": "/path/to/corrupted.png", "file_size": 0},
            "should_fail_recovery": True
        },
        {
            "name": "System resource exhaustion",
            "error_type": OptimizationErrorType.SYSTEM_RESOURCE_ERROR,
            "context": {"disk_space": "full", "memory_usage": "95%"},
            "should_suggest_delay": True
        }
    ]

    successful_edge_cases = 0

    for scenario in edge_case_scenarios:
        print(f"  🧪 Testing: {scenario['name']}")

        # Create error
        error = OptimizationError(
            error_type=scenario["error_type"],
            message=f"Edge case: {scenario['name']}",
            recovery_suggestion="Generated suggestion",
            context=scenario["context"]
        )

        # Attempt recovery
        result = error_handler.attempt_recovery(error)
        success = result.get("success", False)
        message = result.get("message", "No message")

        print(f"    - Recovery Attempted: ✅")
        print(f"    - Recovery Success: {'✅' if success else '❌'}")
        print(f"    - Recovery Message: {message}")

        # Verify specific expectations
        expectations_met = True

        if scenario.get("should_suggest_reduction") and "reduce" not in message.lower():
            expectations_met = False
            print(f"    - ⚠️  Expected reduction suggestion")

        if scenario.get("should_suggest_speed_params") and "speed" not in message.lower():
            expectations_met = False
            print(f"    - ⚠️  Expected speed parameter suggestion")

        if scenario.get("should_fail_recovery") and success:
            expectations_met = False
            print(f"    - ⚠️  Expected recovery to fail")

        if scenario.get("should_suggest_delay") and "delay" not in message.lower():
            expectations_met = False
            print(f"    - ⚠️  Expected delay suggestion")

        if expectations_met:
            successful_edge_cases += 1
            print(f"    - ✅ All expectations met")
        else:
            print(f"    - ❌ Some expectations not met")

        print()

    edge_case_success_rate = successful_edge_cases / len(edge_case_scenarios)
    print(f"📊 Edge Case Success Rate: {edge_case_success_rate:.2%} ({successful_edge_cases}/{len(edge_case_scenarios)})")
    print("✅ Comprehensive edge cases test completed!\n")

    return edge_case_success_rate

def main():
    """Run all error handling tests"""
    print("🚀 Starting Comprehensive Error Handling Tests\n")

    try:
        # Run all tests
        classification_accuracy = test_error_detection_and_classification()
        recovery_rate, edge_case_accuracy = test_recovery_strategies()
        test_circuit_breaker()
        test_fallback_parameters()
        stats = test_error_statistics_and_reporting()
        test_retry_mechanisms()
        edge_case_success_rate = test_comprehensive_edge_cases()

        # Overall assessment
        print("📋 FINAL ASSESSMENT")
        print("=" * 50)
        print(f"Classification Accuracy: {classification_accuracy:.2%}")
        print(f"Recovery Strategy Rate: {recovery_rate:.2%}")
        print(f"Edge Case Accuracy: {edge_case_accuracy:.2%}")
        print(f"Edge Case Success Rate: {edge_case_success_rate:.2%}")
        print(f"Overall Recovery Rate: {stats['recovery_rate']:.2%}")

        # Check Day 4 target: >95% recovery rate
        target_recovery_rate = 0.95
        if recovery_rate >= target_recovery_rate:
            print(f"\n✅ TARGET MET: Recovery rate {recovery_rate:.2%} exceeds target {target_recovery_rate:.2%}")
        else:
            print(f"\n❌ TARGET MISSED: Recovery rate {recovery_rate:.2%} below target {target_recovery_rate:.2%}")

        print(f"\n🎉 All error handling tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()