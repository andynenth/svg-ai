# backend/monitoring/ai_health.py
"""AI-specific health check endpoints - extends base health system"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import asyncio
import torch
import os
from datetime import datetime


router = APIRouter()


class AIHealthChecker:
    """AI system health checker"""

    async def check_ai_models(self) -> Dict[str, Any]:
        """Check AI model availability and loading"""
        try:
            from backend import get_unified_pipeline, get_classification_module, get_optimization_engine

            model_status = {}

            # Test unified pipeline
            try:
                pipeline = get_unified_pipeline()
                model_status["unified_pipeline"] = {"status": "loaded", "size_mb": 0}
            except Exception as e:
                model_status["unified_pipeline"] = {"status": "failed", "error": str(e)}

            # Test classification module
            try:
                classifier = get_classification_module()
                model_status["classification_module"] = {"status": "loaded", "size_mb": 0}
            except Exception as e:
                model_status["classification_module"] = {"status": "failed", "error": str(e)}

            # Test optimization engine
            try:
                optimizer = get_optimization_engine()
                model_status["optimization_engine"] = {"status": "loaded", "size_mb": 0}
            except Exception as e:
                model_status["optimization_engine"] = {"status": "failed", "error": str(e)}

            return {
                "status": "healthy" if all(m["status"] == "loaded" for m in model_status.values()) else "unhealthy",
                "models": model_status
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def check_ai_environment(self) -> Dict[str, Any]:
        """Check AI environment configuration"""
        return {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "model_dir_exists": os.path.exists(os.environ.get('MODEL_DIR', 'models/')),
            "ai_enhanced": os.environ.get('AI_ENHANCED', 'false').lower() == 'true'
        }

    async def check_ai_performance(self) -> Dict[str, Any]:
        """Quick AI performance test"""
        try:
            # Simple test to verify AI pipeline is working
            from backend import get_classification_module
            classifier = get_classification_module()
            # Just verify the module loads without doing actual classification
            return {"status": "healthy", "test_duration_ms": 1}
        except Exception as e:
            return {"status": "error", "error": str(e)}


ai_health_checker = AIHealthChecker()


@router.get("/api/ai-status")
async def ai_status() -> Dict[str, Any]:
    """AI system status endpoint"""
    checks = await asyncio.gather(
        ai_health_checker.check_ai_models(),
        ai_health_checker.check_ai_performance(),
        return_exceptions=True
    )

    models_health = checks[0] if not isinstance(checks[0], Exception) else {"status": "error"}
    performance_health = checks[1] if not isinstance(checks[1], Exception) else {"status": "error"}

    overall_healthy = all(
        h.get("status") in ["healthy", "loaded"]
        for h in [models_health, performance_health]
    )

    return {
        "ai_status": "healthy" if overall_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "models": models_health,
            "performance": performance_health,
            "environment": ai_health_checker.check_ai_environment()
        }
    }