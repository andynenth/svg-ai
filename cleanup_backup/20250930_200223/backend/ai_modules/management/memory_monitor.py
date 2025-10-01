# backend/ai_modules/management/memory_monitor.py
import logging
from typing import Dict, Any

class ModelMemoryMonitor:
    def __init__(self):
        self.memory_stats = {}
        self.peak_usage = 0

    def track_model_memory(self, model_name: str, model) -> Dict[str, float]:
        """Track memory usage for a specific model"""
        import psutil
        import sys

        # Memory before model
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Estimate model size
        model_size = 0
        if hasattr(model, 'parameters'):
            model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        elif hasattr(model, 'get_session_config'):
            # ONNX model - estimate from file size
            model_size = 50  # Approximate for quality predictor

        # Memory after model loading
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = memory_after - memory_before

        self.memory_stats[model_name] = {
            'estimated_size_mb': model_size,
            'actual_memory_delta_mb': memory_delta,
            'total_memory_mb': memory_after
        }

        self.peak_usage = max(self.peak_usage, memory_after)

        logging.info(f"📊 {model_name}: {model_size:.1f}MB estimated, {memory_delta:.1f}MB actual")
        return self.memory_stats[model_name]

    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report"""
        import psutil

        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024

        return {
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_usage,
            'model_breakdown': self.memory_stats,
            'memory_limit_mb': 500,  # Target limit
            'within_limits': current_memory < 500
        }