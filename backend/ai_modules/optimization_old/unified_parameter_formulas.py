"""
Consolidated Parameter Correlation Formulas

This module contains all parameter correlation formula implementations
consolidated from multiple duplicate files.
"""

import numpy as np
from typing import Dict, Any, Tuple


class ParameterFormulas:
    """Unified parameter formula calculator"""

    @staticmethod
    def calculate_color_precision(features: Dict) -> int:
        """Calculate optimal color precision based on image features"""
        unique_colors = features.get('unique_colors', 10)
        has_gradients = features.get('has_gradients', False)
        complexity = features.get('complexity', 0.5)

        if unique_colors < 5:
            return 2
        elif unique_colors < 10:
            return 3
        elif unique_colors < 50:
            return 4
        elif has_gradients or complexity > 0.7:
            return 8
        else:
            return 6

    @staticmethod
    def calculate_corner_threshold(features: Dict) -> float:
        """Calculate optimal corner threshold based on edge characteristics"""
        edge_density = features.get('edge_density', 0.5)
        complexity = features.get('complexity', 0.5)
        has_text = features.get('has_text', False)

        # Base threshold
        base_threshold = 30.0

        # Adjust for edge density
        if edge_density > 0.7:
            base_threshold -= 10  # More aggressive for high edge density
        elif edge_density < 0.3:
            base_threshold += 10  # More conservative for low edge density

        # Adjust for complexity
        complexity_adjustment = (complexity - 0.5) * 20
        base_threshold += complexity_adjustment

        # Special case for text
        if has_text:
            base_threshold = min(base_threshold, 20.0)

        return max(5.0, min(60.0, base_threshold))

    @staticmethod
    def calculate_layer_difference(features: Dict) -> int:
        """Calculate optimal layer difference threshold"""
        unique_colors = features.get('unique_colors', 10)
        has_gradients = features.get('has_gradients', False)

        if has_gradients:
            return 8  # Preserve gradients
        elif unique_colors > 50:
            return 12  # Many colors, need fine separation
        else:
            return 16  # Default for simple images

    @staticmethod
    def calculate_path_precision(features: Dict) -> int:
        """Calculate optimal path precision"""
        complexity = features.get('complexity', 0.5)
        edge_density = features.get('edge_density', 0.5)

        if complexity > 0.8 or edge_density > 0.8:
            return 8  # High precision for complex paths
        elif complexity > 0.5:
            return 10  # Medium precision
        else:
            return 15  # Lower precision for simple shapes

    @staticmethod
    def calculate_splice_threshold(features: Dict) -> int:
        """Calculate optimal splice threshold"""
        complexity = features.get('complexity', 0.5)

        base_threshold = 45
        if complexity > 0.7:
            return base_threshold + 15  # Higher threshold for complex images
        elif complexity < 0.3:
            return base_threshold - 15  # Lower threshold for simple images
        else:
            return base_threshold

    @staticmethod
    def calculate_filter_speckle(features: Dict) -> int:
        """Calculate filter speckle size"""
        noise_level = features.get('noise_level', 0.1)

        if noise_level > 0.3:
            return 8  # Aggressive filtering
        elif noise_level > 0.1:
            return 4  # Moderate filtering
        else:
            return 1  # Minimal filtering

    @staticmethod
    def calculate_all_parameters(features: Dict) -> Dict[str, Any]:
        """Calculate all VTracer parameters based on image features"""
        return {
            'color_precision': ParameterFormulas.calculate_color_precision(features),
            'corner_threshold': ParameterFormulas.calculate_corner_threshold(features),
            'layer_difference': ParameterFormulas.calculate_layer_difference(features),
            'path_precision': ParameterFormulas.calculate_path_precision(features),
            'splice_threshold': ParameterFormulas.calculate_splice_threshold(features),
            'filter_speckle': ParameterFormulas.calculate_filter_speckle(features)
        }


class QualityFormulas:
    """Quality prediction formulas"""

    @staticmethod
    def predict_ssim(parameters: Dict, features: Dict) -> float:
        """Predict SSIM score based on parameters and features"""
        # Simplified prediction model
        base_score = 0.85

        # Adjust based on complexity
        complexity = features.get('complexity', 0.5)
        if complexity > 0.7:
            base_score -= 0.1
        elif complexity < 0.3:
            base_score += 0.1

        # Adjust based on parameters
        color_precision = parameters.get('color_precision', 4)
        if color_precision >= 6:
            base_score += 0.05
        elif color_precision <= 2:
            base_score -= 0.05

        return max(0.0, min(1.0, base_score))

    @staticmethod
    def predict_file_size_reduction(parameters: Dict, features: Dict) -> float:
        """Predict file size reduction percentage"""
        # Base reduction expectation
        base_reduction = 0.7  # 70% reduction

        complexity = features.get('complexity', 0.5)
        unique_colors = features.get('unique_colors', 10)

        # Complex images reduce less
        if complexity > 0.8:
            base_reduction -= 0.2
        elif complexity < 0.2:
            base_reduction += 0.1

        # Many colors reduce less
        if unique_colors > 100:
            base_reduction -= 0.15
        elif unique_colors < 10:
            base_reduction += 0.1

        return max(0.1, min(0.9, base_reduction))


# Legacy compatibility functions for existing code
def calculate_color_precision(features: Dict) -> int:
    """Legacy wrapper for color precision calculation"""
    return ParameterFormulas.calculate_color_precision(features)

def calculate_corner_threshold(features: Dict) -> float:
    """Legacy wrapper for corner threshold calculation"""
    return ParameterFormulas.calculate_corner_threshold(features)

def calculate_all_parameters(features: Dict) -> Dict[str, Any]:
    """Legacy wrapper for all parameter calculation"""
    return ParameterFormulas.calculate_all_parameters(features)
