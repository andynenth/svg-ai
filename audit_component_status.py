#!/usr/bin/env python3
"""
Component Status Audit for AI Pipeline
Systematically tests what's actually functional vs broken
"""

import sys
import importlib
import time
from pathlib import Path

def test_component_import(module_path, class_name=None):
    """Test if a component can be imported and optionally instantiated"""
    try:
        module = importlib.import_module(module_path)
        if class_name:
            cls = getattr(module, class_name)
            # Try basic instantiation with minimal args
            try:
                if class_name == "AIEnhancedConverter":
                    instance = cls()  # No args for this one
                elif class_name == "ImageFeatureExtractor":
                    instance = cls()
                elif class_name == "HybridClassifier":
                    instance = cls()
                elif class_name == "FeatureMappingOptimizer":
                    instance = cls()
                elif class_name == "PPOVTracerOptimizer":
                    instance = cls({'target_images': ['test.png']})
                elif class_name == "QualityPredictor":
                    instance = cls()
                elif class_name == "IntelligentRouter":
                    instance = cls()
                else:
                    instance = cls()
                return True, "✅ Import + Instantiate OK"
            except Exception as e:
                return True, f"⚠️  Import OK, Instantiate FAIL: {str(e)[:100]}"
        else:
            return True, "✅ Import OK"
    except Exception as e:
        return False, f"❌ Import FAIL: {str(e)[:100]}"

def audit_core_components():
    """Audit all core AI pipeline components"""

    print("🔍 AI Pipeline Component Status Audit")
    print("=" * 60)

    components = [
        # Core AI Components
        ("backend.ai_modules.feature_extraction", "ImageFeatureExtractor", "Feature Extraction"),
        ("backend.ai_modules.classification.hybrid_classifier", "HybridClassifier", "Logo Classification"),
        ("backend.ai_modules.optimization.feature_mapping_optimizer", "FeatureMappingOptimizer", "Method 1 Optimizer"),
        ("backend.ai_modules.optimization.ppo_optimizer", "PPOVTracerOptimizer", "Method 2 (PPO) Optimizer"),
        ("backend.ai_modules.optimization.adaptive_optimizer", "AdaptiveOptimizer", "Method 3 Optimizer"),
        ("backend.ai_modules.prediction.quality_predictor", "QualityPredictor", "Quality Predictor"),

        # Routing Components
        ("backend.ai_modules.optimization.intelligent_router", "IntelligentRouter", "Basic Router"),
        ("backend.ai_modules.optimization.enhanced_intelligent_router", "EnhancedIntelligentRouter", "Enhanced Router"),

        # Main Converter
        ("backend.converters.ai_enhanced_converter", "AIEnhancedConverter", "AI Enhanced Converter"),

        # Training Components
        ("backend.ai_modules.optimization.training_pipeline", "CurriculumTrainingPipeline", "Training Pipeline"),
        ("backend.ai_modules.optimization.training_monitor", "TrainingMonitor", "Training Monitor"),

        # Production Components
        ("backend.ai_modules.optimization.production_deployment_package", "ModelPackager", "Production Package"),
        ("backend.ai_modules.optimization.unified_prediction_api", "UnifiedPredictionAPI", "Unified API"),

        # Utility Components
        ("backend.ai_modules.optimization.performance_optimizer", "PerformanceOptimizer", "Performance Optimizer"),
        ("backend.ai_modules.optimization.quality_validator", "QualityValidator", "Quality Validator"),
    ]

    working_components = []
    broken_components = []
    partially_working = []

    for module_path, class_name, description in components:
        print(f"\n📦 Testing: {description}")
        print(f"   Module: {module_path}")
        if class_name:
            print(f"   Class: {class_name}")

        success, message = test_component_import(module_path, class_name)
        print(f"   Status: {message}")

        if success and "✅" in message:
            working_components.append((description, module_path, class_name))
        elif success and "⚠️" in message:
            partially_working.append((description, module_path, class_name, message))
        else:
            broken_components.append((description, module_path, class_name, message))

    return working_components, partially_working, broken_components

def test_basic_workflow():
    """Test if basic AI conversion workflow can run"""
    print("\n🔄 Testing Basic AI Conversion Workflow")
    print("=" * 50)

    try:
        # Test feature extraction
        from backend.ai_modules.feature_extraction import ImageFeatureExtractor
        extractor = ImageFeatureExtractor()
        print("✅ Feature extraction component ready")

        # Test classification
        from backend.ai_modules.classification.hybrid_classifier import HybridClassifier
        classifier = HybridClassifier()
        print("✅ Classification component ready")

        # Test Method 1 optimizer
        from backend.ai_modules.optimization.feature_mapping_optimizer import FeatureMappingOptimizer
        optimizer = FeatureMappingOptimizer()
        print("✅ Method 1 optimizer ready")

        # Test basic router
        from backend.ai_modules.optimization.intelligent_router import IntelligentRouter
        router = IntelligentRouter()
        print("✅ Basic router ready")

        # Test main converter
        try:
            from backend.converters.ai_enhanced_converter import AIEnhancedConverter
            converter = AIEnhancedConverter()
            print("✅ AI Enhanced Converter ready")
        except Exception as e:
            print(f"❌ AI Enhanced Converter failed: {e}")
            return False

        print("\n🎉 Basic workflow components are functional!")
        return True

    except Exception as e:
        print(f"❌ Basic workflow test failed: {e}")
        return False

def identify_priority_fixes():
    """Identify which broken components need immediate fixing"""
    print("\n🔧 Priority Fix Analysis")
    print("=" * 40)

    critical_components = [
        "AI Enhanced Converter",
        "Enhanced Router",
        "Feature Extraction",
        "Logo Classification",
        "Method 1 Optimizer"
    ]

    print("Critical Path Components (must work for MVP):")
    for component in critical_components:
        print(f"  - {component}")

    return critical_components

def generate_audit_report(working, partial, broken):
    """Generate comprehensive audit report"""

    print("\n" + "=" * 60)
    print("📊 COMPONENT STATUS AUDIT REPORT")
    print("=" * 60)

    print(f"\n✅ FULLY WORKING COMPONENTS ({len(working)}):")
    for desc, module, class_name in working:
        print(f"  ✅ {desc}")

    print(f"\n⚠️  PARTIALLY WORKING COMPONENTS ({len(partial)}):")
    for desc, module, class_name, message in partial:
        print(f"  ⚠️  {desc}")
        print(f"      Issue: {message.split('FAIL: ')[1] if 'FAIL: ' in message else 'Unknown'}")

    print(f"\n❌ BROKEN COMPONENTS ({len(broken)}):")
    for desc, module, class_name, message in broken:
        print(f"  ❌ {desc}")
        print(f"      Error: {message.split('FAIL: ')[1] if 'FAIL: ' in message else 'Unknown'}")

    # Calculate health score
    total = len(working) + len(partial) + len(broken)
    health_score = (len(working) + len(partial) * 0.5) / total * 100 if total > 0 else 0

    print(f"\n📈 SYSTEM HEALTH SCORE: {health_score:.1f}%")
    print(f"   - Working: {len(working)}/{total}")
    print(f"   - Partial: {len(partial)}/{total}")
    print(f"   - Broken: {len(broken)}/{total}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if health_score >= 70:
        print("  🟢 System is in good shape - focus on fixing partial components")
    elif health_score >= 50:
        print("  🟡 System needs significant fixes - prioritize critical path")
    else:
        print("  🔴 System needs major overhaul - focus on core components")

    return health_score

def main():
    """Run complete component audit"""
    start_time = time.time()

    print("🚀 Starting AI Pipeline Component Audit...")
    print(f"📅 Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Audit components
    working, partial, broken = audit_core_components()

    # Test basic workflow
    basic_workflow_ok = test_basic_workflow()

    # Identify priorities
    priority_fixes = identify_priority_fixes()

    # Generate report
    health_score = generate_audit_report(working, partial, broken)

    # Final recommendations
    print(f"\n🎯 NEXT STEPS:")
    print(f"  1. Fix critical broken components first")
    print(f"  2. Address partial component issues")
    print(f"  3. Test end-to-end workflow")
    print(f"  4. Validate with real images")

    print(f"\n⏱️  Audit completed in {time.time() - start_time:.2f} seconds")

    return {
        'working': working,
        'partial': partial,
        'broken': broken,
        'health_score': health_score,
        'basic_workflow_ok': basic_workflow_ok
    }

if __name__ == "__main__":
    result = main()

    # Exit with error code if health is poor
    if result['health_score'] < 50:
        sys.exit(1)
    else:
        sys.exit(0)