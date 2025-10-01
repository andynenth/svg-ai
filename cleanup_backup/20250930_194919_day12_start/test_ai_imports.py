#!/usr/bin/env python3
"""Test all AI module imports"""

def test_all_imports():
    """Test importing all AI modules"""

    print("🧪 Testing AI Module Imports...")
    print("=" * 40)

    try:
        # Classification modules
        from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
        from backend.ai_modules.classification.logo_classifier import LogoClassifier
        from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
        print("✅ Classification modules")

        # Optimization modules
        from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
        from backend.ai_modules.optimization.rl_optimizer import RLOptimizer
        from backend.ai_modules.optimization.adaptive_optimizer import AdaptiveOptimizer
        from backend.ai_modules.optimization.vtracer_environment import VTracerEnvironment
        print("✅ Optimization modules")

        # Prediction modules
        from backend.ai_modules.prediction.quality_predictor import QualityPredictor
        from backend.ai_modules.prediction.model_utils import ModelUtils
        print("✅ Prediction modules")

        # Base classes
        from backend.ai_modules.base_ai_converter import BaseAIConverter
        from backend.ai_modules.classification.base_feature_extractor import BaseFeatureExtractor
        from backend.ai_modules.optimization.base_optimizer import BaseOptimizer
        from backend.ai_modules.prediction.base_predictor import BasePredictor
        print("✅ Base AI classes")

        # Configuration and utilities
        from backend.ai_modules.config import MODEL_CONFIG, PERFORMANCE_TARGETS, get_config_summary
        from backend.ai_modules import check_dependencies
        print("✅ Configuration and utilities")

        # Test cross-module integration
        print("\n🔄 Testing cross-module integration...")

        # Test that classes can be instantiated together
        feature_extractor = ImageFeatureExtractor()
        classifier = RuleBasedClassifier()
        optimizer = FeatureMappingOptimizer()
        predictor = QualityPredictor()

        print("✅ All classes instantiate correctly")

        # Test dependency checking
        deps_ok = check_dependencies()
        print(f"✅ Dependencies check: {deps_ok}")

        # Test configuration
        config_summary = get_config_summary()
        print(f"✅ Configuration valid: {config_summary['config_valid']}")

        print("\n🎉 All imports successful!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_import_paths():
    """Test that import paths work from project root"""
    print("\n🔍 Testing import paths from project root...")

    import sys
    import os

    # Check that we're in the right directory
    current_dir = os.getcwd()
    if 'svg-ai' not in current_dir:
        print(f"❌ Not in project root. Current: {current_dir}")
        return False

    # Check that backend directory exists
    backend_path = os.path.join(current_dir, 'backend')
    if not os.path.exists(backend_path):
        print(f"❌ Backend directory not found: {backend_path}")
        return False

    # Check that ai_modules directory exists
    ai_modules_path = os.path.join(backend_path, 'ai_modules')
    if not os.path.exists(ai_modules_path):
        print(f"❌ AI modules directory not found: {ai_modules_path}")
        return False

    print("✅ Import paths correctly configured")
    return True

def test_circular_dependencies():
    """Test for circular dependency issues"""
    print("\n🔄 Testing for circular dependencies...")

    try:
        # These imports should not cause circular dependency issues
        from backend.ai_modules.base_ai_converter import BaseAIConverter
        from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
        from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
        from backend.ai_modules.prediction.quality_predictor import QualityPredictor

        # Test that we can create instances without issues (skip abstract BaseAIConverter)
        extractor = ImageFeatureExtractor()
        optimizer = FeatureMappingOptimizer()
        predictor = QualityPredictor()

        # Verify BaseAIConverter is importable (but don't instantiate abstract class)
        assert hasattr(BaseAIConverter, '__abstractmethods__')

        print("✅ No circular dependency issues detected")
        return True

    except Exception as e:
        print(f"❌ Circular dependency or other issue: {e}")
        return False

def main():
    """Run all import tests"""
    print("🚀 AI Module Import Validation")
    print("=" * 50)

    all_tests_passed = True

    # Test 1: Basic imports
    if not test_all_imports():
        all_tests_passed = False

    # Test 2: Import paths
    if not test_import_paths():
        all_tests_passed = False

    # Test 3: Circular dependencies
    if not test_circular_dependencies():
        all_tests_passed = False

    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All import tests PASSED!")
        return True
    else:
        print("❌ Some import tests FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)