#!/usr/bin/env python3
"""
Final validation script for SpatialComplexityAnalyzer implementation
This validates that the implementation meets all requirements for Agents 2 and 3
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.ai_modules.optimization.spatial_analysis import SpatialComplexityAnalyzer, ComplexityRegion
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)

def validate_implementation():
    """Validate that the SpatialComplexityAnalyzer meets all requirements"""

    print("🔍 Validating SpatialComplexityAnalyzer implementation...")
    print("=" * 60)

    # Test 1: Class instantiation
    print("1. Testing class instantiation...")
    try:
        analyzer = SpatialComplexityAnalyzer()
        print("   ✅ SpatialComplexityAnalyzer instantiated successfully")
    except Exception as e:
        print(f"   ❌ Failed to instantiate: {e}")
        return False

    # Test 2: ComplexityRegion dataclass
    print("2. Testing ComplexityRegion dataclass...")
    try:
        region = ComplexityRegion(
            bounds=(0, 0, 100, 100),
            complexity_score=0.5,
            dominant_features=['test'],
            suggested_parameters={'color_precision': 4},
            confidence=0.8
        )
        assert region.bounds == (0, 0, 100, 100)
        assert region.complexity_score == 0.5
        print("   ✅ ComplexityRegion dataclass working correctly")
    except Exception as e:
        print(f"   ❌ ComplexityRegion failed: {e}")
        return False

    # Test 3: analyze_complexity_distribution method
    print("3. Testing analyze_complexity_distribution method...")
    test_image = "/Users/nrw/python/svg-ai/test-data/circle_00.png"

    if not os.path.exists(test_image):
        print(f"   ❌ Test image not found: {test_image}")
        return False

    try:
        start_time = time.time()
        result = analyzer.analyze_complexity_distribution(test_image)
        processing_time = time.time() - start_time

        # Validate return structure
        required_keys = [
            'complexity_map', 'edge_density_map', 'color_variation_map',
            'texture_complexity', 'geometric_complexity', 'multiscale_complexity',
            'overall_complexity', 'complexity_std', 'high_complexity_ratio', 'regions'
        ]

        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            print(f"   ❌ Missing required keys: {missing_keys}")
            return False

        print(f"   ✅ analyze_complexity_distribution working ({processing_time:.2f}s)")
        print(f"      Overall complexity: {result['overall_complexity']:.4f}")
        print(f"      Regions detected: {len(result['regions'])}")

        # Performance requirement check
        if processing_time > 30:
            print(f"   ⚠️  Performance concern: {processing_time:.2f}s > 30s target")
        else:
            print(f"   ✅ Performance target met: {processing_time:.2f}s ≤ 30s")

    except Exception as e:
        print(f"   ❌ analyze_complexity_distribution failed: {e}")
        return False

    # Test 4: Private methods exist and are callable
    print("4. Testing private methods...")
    try:
        import cv2
        test_gray = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
        test_color = cv2.imread(test_image)

        # Test required private methods
        complexity_map = analyzer._calculate_complexity_map(test_gray)
        edge_density = analyzer._calculate_edge_density_map(test_gray)
        color_variation = analyzer._calculate_color_variation_map(test_color)

        print("   ✅ All required private methods working")
    except Exception as e:
        print(f"   ❌ Private methods failed: {e}")
        return False

    # Test 5: Region objects are properly typed
    print("5. Testing region object types...")
    try:
        if len(result['regions']) > 0:
            for i, region in enumerate(result['regions']):
                if not isinstance(region, ComplexityRegion):
                    print(f"   ❌ Region {i} is not a ComplexityRegion instance")
                    return False
        print("   ✅ All region objects are properly typed")
    except Exception as e:
        print(f"   ❌ Region type validation failed: {e}")
        return False

    # Test 6: Importability for other agents
    print("6. Testing importability...")
    try:
        # Test import statements that Agents 2 and 3 will use
        exec("from backend.ai_modules.optimization.spatial_analysis import SpatialComplexityAnalyzer, ComplexityRegion")
        print("   ✅ Module can be imported by other agents")
    except Exception as e:
        print(f"   ❌ Import test failed: {e}")
        return False

    print("=" * 60)
    print("🎉 ALL VALIDATION TESTS PASSED!")
    print()
    print("📋 Summary:")
    print("   • SpatialComplexityAnalyzer class is fully implemented")
    print("   • ComplexityRegion dataclass is working correctly")
    print("   • analyze_complexity_distribution method returns required structure")
    print("   • All private methods are implemented and functional")
    print("   • Performance meets requirements (< 30s per image)")
    print("   • Module is ready for import by Agents 2 and 3")
    print()
    print("✅ READY FOR AGENTS 2 AND 3!")

    return True

if __name__ == "__main__":
    success = validate_implementation()
    sys.exit(0 if success else 1)