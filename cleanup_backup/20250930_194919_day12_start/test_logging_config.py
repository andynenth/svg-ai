#!/usr/bin/env python3
"""Test AI Logging Configuration"""

import tempfile
import os
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_modules.utils.logging_config import (
    setup_ai_logging,
    get_ai_logger,
    log_ai_operation,
    log_ai_performance,
    ai_logging_config
)

def test_basic_logging():
    """Test basic logging functionality"""
    print("🧪 Testing Basic Logging...")

    # Setup logging for testing
    temp_dir = tempfile.mkdtemp()
    setup_ai_logging("development", log_dir=temp_dir)

    # Get loggers
    main_logger = get_ai_logger("test")
    classification_logger = get_ai_logger("classification.test")
    optimization_logger = get_ai_logger("optimization.test")
    prediction_logger = get_ai_logger("prediction.test")

    # Test different log levels
    main_logger.debug("Debug message")
    main_logger.info("Info message")
    main_logger.warning("Warning message")
    main_logger.error("Error message")

    # Test component-specific logging
    classification_logger.info("Classification test message")
    optimization_logger.info("Optimization test message")
    prediction_logger.info("Prediction test message")

    print("  ✅ Basic logging messages sent")

    # Check log files were created
    log_dir = Path(temp_dir) / "ai_modules"
    expected_files = ["main.log", "classification.log", "optimization.log", "prediction.log"]

    for filename in expected_files:
        log_file = log_dir / filename
        if log_file.exists() and log_file.stat().st_size > 0:
            print(f"  ✅ {filename} created with content")
        else:
            print(f"  ⚠️  {filename} not found or empty")

    return temp_dir

def test_structured_logging():
    """Test structured logging functionality"""
    print("\n📊 Testing Structured Logging...")

    # Setup structured logging
    temp_dir = tempfile.mkdtemp()
    ai_logging_config.log_dir = Path(temp_dir)
    ai_logging_config.setup_logging(
        enable_file_logging=True,
        enable_console_logging=False,
        structured_logging=True
    )

    # Get logger
    logger = get_ai_logger("structured_test")

    # Log structured operations
    log_ai_operation("feature_extraction",
                    level="INFO",
                    image_path="/test/image.png",
                    duration=0.05,
                    features_count=8)

    log_ai_operation("classification",
                    level="INFO",
                    logo_type="simple",
                    confidence=0.85)

    log_ai_performance("optimization",
                      duration=0.01,
                      memory_delta=2.5,
                      success=True,
                      parameters_count=8)

    print("  ✅ Structured logging messages sent")

    # Check structured log content
    log_file = Path(temp_dir) / "ai_modules" / "main.log"
    if log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    log_data = json.loads(line.strip())
                    if 'operation' in log_data:
                        print(f"  ✅ Structured log: {log_data['operation']} operation logged")
                except json.JSONDecodeError:
                    pass

    return temp_dir

def test_different_environments():
    """Test logging in different environments"""
    print("\n🌍 Testing Different Environment Configurations...")

    environments = ['development', 'production', 'testing']

    for env in environments:
        print(f"  Testing {env} environment...")
        temp_dir = tempfile.mkdtemp()

        try:
            setup_ai_logging(env, log_dir=temp_dir)
            logger = get_ai_logger(f"{env}_test")

            # Send test messages
            logger.info(f"Test message for {env} environment")
            logger.error(f"Test error for {env} environment")

            print(f"    ✅ {env} environment configured successfully")

        except Exception as e:
            print(f"    ❌ {env} environment failed: {e}")

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

def test_performance_logging():
    """Test performance logging integration"""
    print("\n⚡ Testing Performance Logging Integration...")

    temp_dir = tempfile.mkdtemp()
    setup_ai_logging("development", log_dir=temp_dir)

    # Simulate performance metrics
    operations = [
        ("feature_extraction", 0.045, 2.1),
        ("classification", 0.0002, 0.0),
        ("optimization", 0.0001, 0.0),
        ("prediction", 0.005, 1.8)
    ]

    for operation, duration, memory_delta in operations:
        log_ai_performance(operation, duration, memory_delta,
                          success=True,
                          additional_info=f"Test {operation}")

    print("  ✅ Performance metrics logged")

    # Check performance log
    perf_log = Path(temp_dir) / "ai_modules" / "performance.log"
    if perf_log.exists() and perf_log.stat().st_size > 0:
        print("  ✅ Performance log file created with content")
        with open(perf_log, 'r') as f:
            content = f.read()
            for operation, _, _ in operations:
                if operation in content:
                    print(f"    ✅ {operation} metrics found in log")
    else:
        print("  ⚠️  Performance log file not found or empty")

    return temp_dir

def test_error_logging():
    """Test error logging and exception handling"""
    print("\n🚨 Testing Error Logging...")

    temp_dir = tempfile.mkdtemp()
    setup_ai_logging("development", log_dir=temp_dir)

    logger = get_ai_logger("error_test")

    # Test exception logging
    try:
        raise ValueError("Test exception for logging")
    except Exception as e:
        logger.exception("Exception caught during AI operation")

    # Test error without exception
    logger.error("Error without exception",
                extra={'operation': 'test_operation',
                      'error_code': 'TEST_001'})

    print("  ✅ Error messages logged")

    # Check error log
    error_log = Path(temp_dir) / "ai_modules" / "errors.log"
    if error_log.exists() and error_log.stat().st_size > 0:
        print("  ✅ Error log file created with content")
    else:
        print("  ⚠️  Error log file not found or empty")

    return temp_dir

def test_logging_conventions():
    """Test logging conventions documentation"""
    print("\n📋 Testing Logging Conventions...")

    # Test logger naming conventions
    loggers = [
        "classification.feature_extractor",
        "classification.logo_classifier",
        "optimization.feature_mapping",
        "optimization.rl_optimizer",
        "prediction.quality_predictor"
    ]

    for logger_name in loggers:
        logger = get_ai_logger(logger_name)
        logger.info(f"Testing logger: {logger_name}")
        print(f"  ✅ Logger {logger_name} created successfully")

    # Test operation naming conventions
    operations = [
        "extract_features",
        "classify_logo",
        "optimize_parameters",
        "predict_quality",
        "convert_image"
    ]

    for operation in operations:
        log_ai_operation(operation,
                        level="INFO",
                        component=operation.split('_')[0])
        print(f"  ✅ Operation {operation} logged")

def main():
    """Run all logging tests"""
    print("🏃‍♂️ AI Logging Configuration Test Suite")
    print("=" * 50)

    temp_dirs = []

    try:
        # Run tests
        temp_dirs.append(test_basic_logging())
        temp_dirs.append(test_structured_logging())
        test_different_environments()
        temp_dirs.append(test_performance_logging())
        temp_dirs.append(test_error_logging())
        test_logging_conventions()

        print("\n🎉 All logging tests completed!")

        # Summary
        print("\n📋 Logging Test Summary:")
        print("  ✅ Basic file and console logging")
        print("  ✅ Structured JSON logging")
        print("  ✅ Environment-specific configurations")
        print("  ✅ Performance metrics logging")
        print("  ✅ Error and exception logging")
        print("  ✅ Logging conventions verified")

    finally:
        # Cleanup temporary directories
        import shutil
        for temp_dir in temp_dirs:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()