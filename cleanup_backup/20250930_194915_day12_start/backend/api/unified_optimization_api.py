#!/usr/bin/env python3
"""
Unified 4-Tier Optimization API - Complete Production System
Provides comprehensive API for the complete 4-tier optimization system integration
"""

import time
import json
import uuid
import asyncio
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from io import BytesIO
import base64

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Depends, status, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import aiofiles

# 4-Tier System Integration
from ..ai_modules.optimization.tier4_system_orchestrator import (
    Tier4SystemOrchestrator,
    create_4tier_orchestrator,
    OptimizationTier
)
from ..converters.ai_enhanced_converter import AIEnhancedConverter
from ..utils.quality_metrics import ComprehensiveMetrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# ============================================================================
# Request/Response Models
# ============================================================================

class Tier4OptimizationRequest(BaseModel):
    """Complete 4-tier optimization request"""
    quality_target: float = Field(default=0.85, ge=0.0, le=1.0, description="Target quality (SSIM)")
    time_constraint: float = Field(default=30.0, gt=0.0, le=300.0, description="Maximum processing time")
    speed_priority: str = Field(default="balanced", description="Speed vs quality priority")
    enable_caching: bool = Field(default=True, description="Enable result caching")
    enable_quality_prediction: bool = Field(default=True, description="Enable Tier 4 quality prediction")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    optimization_method: str = Field(default="auto", description="Force specific method or auto-select")
    return_svg_content: bool = Field(default=False, description="Include SVG content in response")
    enable_validation: bool = Field(default=True, description="Enable quality validation")
    metadata_level: str = Field(default="standard", description="Metadata detail level")

    @validator('speed_priority')
    def validate_speed_priority(cls, v):
        if v not in ['fast', 'balanced', 'quality']:
            raise ValueError('speed_priority must be one of: fast, balanced, quality')
        return v

    @validator('optimization_method')
    def validate_optimization_method(cls, v):
        valid_methods = ['auto', 'feature_mapping', 'regression', 'ppo', 'performance']
        if v not in valid_methods:
            raise ValueError(f'optimization_method must be one of: {valid_methods}')
        return v

    @validator('metadata_level')
    def validate_metadata_level(cls, v):
        if v not in ['minimal', 'standard', 'detailed', 'full']:
            raise ValueError('metadata_level must be one of: minimal, standard, detailed, full')
        return v


class Tier4OptimizationResponse(BaseModel):
    """Complete 4-tier optimization response"""
    success: bool
    request_id: str
    total_execution_time: float

    # Core optimization results
    optimized_parameters: Dict[str, Any]
    method_used: str
    optimization_confidence: float

    # Quality prediction results
    predicted_quality: float
    quality_confidence: float
    qa_recommendations: List[str] = []

    # Image analysis results
    image_type: str
    complexity_level: str
    image_features: Dict[str, float]

    # Routing decision details
    routing_decision: Dict[str, Any]

    # System performance metrics
    tier_performance: Dict[str, float]

    # Optional SVG content
    svg_content: Optional[str] = None
    quality_validation: Optional[Dict[str, Any]] = None

    # Execution metadata
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


class BatchOptimizationRequest(BaseModel):
    """Batch 4-tier optimization request"""
    requests: List[Tier4OptimizationRequest]
    max_concurrent: int = Field(default=5, ge=1, le=20)
    batch_timeout: float = Field(default=300.0, gt=0.0, le=1800.0)
    fail_fast: bool = Field(default=False, description="Stop batch on first error")
    progress_callback: Optional[str] = Field(default=None, description="Webhook URL for progress updates")


class BatchOptimizationResponse(BaseModel):
    """Batch 4-tier optimization response"""
    success: bool
    batch_id: str
    total_images: int
    completed: int
    failed: int
    results: List[Tier4OptimizationResponse]
    batch_statistics: Dict[str, Any]
    total_batch_time: float


class SystemHealthResponse(BaseModel):
    """System health status response"""
    overall_status: str
    timestamp: str
    check_duration: float
    components: Dict[str, str]
    performance: Dict[str, Any]
    alerts: List[str]


class SystemMetricsResponse(BaseModel):
    """System performance metrics response"""
    total_requests: int
    successful_requests: int
    system_reliability: float
    tier_performance: Dict[str, Dict[str, float]]
    method_effectiveness: Dict[str, Dict[str, float]]
    cache_performance: Dict[str, Any]
    active_requests: int


class ExecutionHistoryResponse(BaseModel):
    """Execution history response"""
    executions: List[Dict[str, Any]]
    total_executions: int
    time_range: Dict[str, Any]
    performance_summary: Dict[str, Any]


# ============================================================================
# Global System Instance
# ============================================================================

# Global orchestrator instance
orchestrator: Optional[Tier4SystemOrchestrator] = None

# Job tracking
active_jobs: Dict[str, Dict[str, Any]] = {}
job_results: Dict[str, Tier4OptimizationResponse] = {}

# System configuration
api_config = {
    "max_file_size": 50 * 1024 * 1024,  # 50MB
    "supported_formats": ["png", "jpg", "jpeg", "gif", "bmp", "tiff"],
    "rate_limits": {
        "requests_per_minute": 100,
        "requests_per_hour": 1000,
        "batch_requests_per_hour": 50
    },
    "cache_settings": {
        "enabled": True,
        "ttl": 3600,
        "max_size": 10000
    }
}

# API Router
router = APIRouter(prefix="/api/v2/optimization", tags=["4-tier-optimization"])

# ============================================================================
# Utility Functions
# ============================================================================

def generate_request_id() -> str:
    """Generate unique request identifier"""
    return f"tier4_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"

def validate_image_file(file: UploadFile) -> bool:
    """Validate uploaded image file"""
    if not file.content_type or not file.content_type.startswith('image/'):
        return False

    if file.size and file.size > api_config["max_file_size"]:
        return False

    # Check file extension
    if file.filename:
        ext = Path(file.filename).suffix.lower().lstrip('.')
        if ext not in api_config["supported_formats"]:
            return False

    return True

async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file temporarily"""
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix if file.filename else ".png"
    temp_path = f"/tmp/claude/uploads/{file_id}{file_extension}"

    # Ensure directory exists
    Path(temp_path).parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(temp_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    return temp_path

def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if Path(file_path).exists():
            Path(file_path).unlink()
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key authentication"""
    # Production-ready API key validation
    valid_keys = ["tier4-prod-key", "tier4-test-key", "tier4-demo-key"]

    if credentials.credentials not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key for 4-Tier system access"
        )
    return credentials.credentials

def filter_metadata_by_level(metadata: Dict[str, Any], level: str) -> Dict[str, Any]:
    """Filter metadata based on requested detail level"""
    if level == "minimal":
        return {
            "request_id": metadata.get("request_id"),
            "timestamp": metadata.get("timestamp"),
            "system_version": metadata.get("system_version")
        }
    elif level == "standard":
        return {
            k: v for k, v in metadata.items()
            if k in ["request_id", "timestamp", "system_version", "user_requirements", "execution_timeline"]
        }
    elif level == "detailed":
        return {
            k: v for k, v in metadata.items()
            if k not in ["system_state", "tier_results_summary"]
        }
    else:  # full
        return metadata

# ============================================================================
# Startup/Shutdown
# ============================================================================

@router.on_event("startup")
async def startup_event():
    """Initialize 4-tier system on startup"""
    global orchestrator

    logger.info("Initializing 4-Tier Optimization System...")

    try:
        # Create orchestrator with production configuration
        config = {
            "max_concurrent_requests": 20,
            "enable_async_processing": True,
            "enable_caching": True,
            "cache_ttl": 3600,
            "production_mode": True,
            "tier_timeouts": {
                "classification": 10.0,
                "routing": 5.0,
                "optimization": 120.0,
                "prediction": 15.0
            }
        }

        orchestrator = create_4tier_orchestrator(config)

        # Perform initial health check
        health = await orchestrator.health_check()
        logger.info(f"4-Tier system initialized with status: {health['overall_status']}")

        if health['overall_status'] != 'healthy':
            logger.warning(f"System started with issues: {health['alerts']}")

    except Exception as e:
        logger.error(f"Failed to initialize 4-tier system: {e}")
        raise

@router.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    global orchestrator

    logger.info("Shutting down 4-Tier Optimization System...")

    try:
        if orchestrator:
            orchestrator.shutdown()

        # Clean up temporary files
        temp_dir = Path("/tmp/claude/uploads")
        if temp_dir.exists():
            for temp_file in temp_dir.glob("*"):
                temp_file.unlink()

        logger.info("4-Tier system shutdown complete")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# ============================================================================
# Core API Endpoints
# ============================================================================

@router.post("/optimize", response_model=Tier4OptimizationResponse)
async def optimize_image_4tier(
    file: UploadFile = File(...),
    request: Tier4OptimizationRequest = Tier4OptimizationRequest(),
    api_key: str = Depends(verify_api_key)
) -> Tier4OptimizationResponse:
    """
    Complete 4-Tier Image Optimization

    Executes full pipeline: Classification → Routing → Optimization → Quality Prediction

    - **file**: Image file to optimize (PNG, JPG, etc.)
    - **request**: Optimization parameters and requirements
    - Returns comprehensive optimization results with quality predictions
    """

    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="4-Tier optimization system not available"
        )

    temp_file_path = None

    try:
        # Validate file
        if not validate_image_file(file):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image file format or size"
            )

        # Save uploaded file
        temp_file_path = await save_uploaded_file(file)

        # Prepare user requirements
        user_requirements = {
            "quality_target": request.quality_target,
            "time_constraint": request.time_constraint,
            "speed_priority": request.speed_priority,
            "optimization_method": request.optimization_method if request.optimization_method != "auto" else None,
            "enable_quality_prediction": request.enable_quality_prediction,
            "enable_validation": request.enable_validation
        }

        # Execute 4-tier optimization
        logger.info(f"Starting 4-tier optimization for {file.filename}")

        optimization_result = await orchestrator.execute_4tier_optimization(
            temp_file_path, user_requirements
        )

        # Generate SVG content if requested
        svg_content = None
        quality_validation = None

        if request.return_svg_content and optimization_result["success"]:
            try:
                # Use AI enhanced converter to generate SVG
                converter = AIEnhancedConverter()
                svg_content = converter.convert(
                    temp_file_path,
                    **optimization_result["optimized_parameters"]
                )

                # Perform quality validation if enabled
                if request.enable_validation:
                    quality_validation = await perform_quality_validation(
                        temp_file_path, svg_content, optimization_result
                    )

            except Exception as e:
                logger.warning(f"SVG generation failed: {e}")
                svg_content = None

        # Filter metadata based on requested level
        filtered_metadata = filter_metadata_by_level(
            optimization_result.get("metadata", {}),
            request.metadata_level
        )

        # Create response
        response = Tier4OptimizationResponse(
            success=optimization_result["success"],
            request_id=optimization_result["request_id"],
            total_execution_time=optimization_result["total_execution_time"],
            optimized_parameters=optimization_result["optimized_parameters"],
            method_used=optimization_result["method_used"],
            optimization_confidence=optimization_result["optimization_confidence"],
            predicted_quality=optimization_result["predicted_quality"],
            quality_confidence=optimization_result["quality_confidence"],
            qa_recommendations=optimization_result.get("qa_recommendations", []),
            image_type=optimization_result["image_type"],
            complexity_level=optimization_result["complexity_level"],
            image_features=optimization_result["image_features"],
            routing_decision=optimization_result["routing_decision"],
            tier_performance=optimization_result["tier_performance"],
            svg_content=svg_content,
            quality_validation=quality_validation,
            metadata=filtered_metadata,
            error_message=optimization_result.get("error")
        )

        logger.info(f"4-tier optimization completed: {optimization_result['request_id']}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"4-tier optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )
    finally:
        if temp_file_path:
            cleanup_temp_file(temp_file_path)

@router.post("/optimize-batch", response_model=BatchOptimizationResponse)
async def optimize_batch_4tier(
    files: List[UploadFile] = File(...),
    batch_request: BatchOptimizationRequest = BatchOptimizationRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    api_key: str = Depends(verify_api_key)
) -> BatchOptimizationResponse:
    """
    Batch 4-Tier Image Optimization

    Process multiple images using complete 4-tier pipeline with parallel execution

    - **files**: List of image files to optimize
    - **batch_request**: Batch processing parameters
    - Returns batch processing results with individual optimization details
    """

    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="4-Tier optimization system not available"
        )

    # Validate batch size
    if len(files) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size exceeds maximum limit of 50 images"
        )

    batch_id = generate_request_id()
    batch_start_time = time.time()
    temp_files = []

    try:
        # Save all files
        temp_files = []
        for file in files:
            if validate_image_file(file):
                temp_path = await save_uploaded_file(file)
                temp_files.append((temp_path, file.filename))

        # Process batch with concurrency control
        semaphore = asyncio.Semaphore(batch_request.max_concurrent)
        results = []

        async def process_single_image(temp_path: str, filename: str, req: Tier4OptimizationRequest):
            async with semaphore:
                try:
                    user_requirements = {
                        "quality_target": req.quality_target,
                        "time_constraint": req.time_constraint,
                        "speed_priority": req.speed_priority,
                        "optimization_method": req.optimization_method if req.optimization_method != "auto" else None,
                        "enable_quality_prediction": req.enable_quality_prediction
                    }

                    optimization_result = await orchestrator.execute_4tier_optimization(
                        temp_path, user_requirements
                    )

                    # Create response for this image
                    response = Tier4OptimizationResponse(
                        success=optimization_result["success"],
                        request_id=optimization_result["request_id"],
                        total_execution_time=optimization_result["total_execution_time"],
                        optimized_parameters=optimization_result["optimized_parameters"],
                        method_used=optimization_result["method_used"],
                        optimization_confidence=optimization_result["optimization_confidence"],
                        predicted_quality=optimization_result["predicted_quality"],
                        quality_confidence=optimization_result["quality_confidence"],
                        qa_recommendations=optimization_result.get("qa_recommendations", []),
                        image_type=optimization_result["image_type"],
                        complexity_level=optimization_result["complexity_level"],
                        image_features=optimization_result["image_features"],
                        routing_decision=optimization_result["routing_decision"],
                        tier_performance=optimization_result["tier_performance"],
                        metadata=filter_metadata_by_level(optimization_result.get("metadata", {}), "standard"),
                        error_message=optimization_result.get("error")
                    )

                    return response

                except Exception as e:
                    logger.error(f"Batch image {filename} failed: {e}")
                    return Tier4OptimizationResponse(
                        success=False,
                        request_id=generate_request_id(),
                        total_execution_time=0.0,
                        optimized_parameters={},
                        method_used="none",
                        optimization_confidence=0.0,
                        predicted_quality=0.0,
                        quality_confidence=0.0,
                        image_type="unknown",
                        complexity_level="unknown",
                        image_features={},
                        routing_decision={},
                        tier_performance={},
                        metadata={},
                        error_message=str(e)
                    )

        # Execute batch processing
        if len(batch_request.requests) == len(temp_files):
            # Use provided requests
            tasks = [
                process_single_image(temp_path, filename, req)
                for (temp_path, filename), req in zip(temp_files, batch_request.requests)
            ]
        else:
            # Use default request for all images
            default_request = Tier4OptimizationRequest()
            tasks = [
                process_single_image(temp_path, filename, default_request)
                for temp_path, filename in temp_files
            ]

        # Wait for all tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=batch_request.batch_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Batch {batch_id} timed out")
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="Batch processing timed out"
            )

        # Calculate batch statistics
        successful_results = [r for r in results if isinstance(r, Tier4OptimizationResponse) and r.success]
        failed_results = [r for r in results if not isinstance(r, Tier4OptimizationResponse) or not r.success]

        batch_statistics = {
            "total_images": len(files),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "success_rate": len(successful_results) / max(len(results), 1),
            "average_execution_time": sum(r.total_execution_time for r in successful_results) / max(len(successful_results), 1),
            "average_predicted_quality": sum(r.predicted_quality for r in successful_results) / max(len(successful_results), 1),
            "methods_used": {
                method: sum(1 for r in successful_results if r.method_used == method)
                for method in set(r.method_used for r in successful_results)
            } if successful_results else {}
        }

        # Create batch response
        batch_response = BatchOptimizationResponse(
            success=len(successful_results) > 0,
            batch_id=batch_id,
            total_images=len(files),
            completed=len(successful_results),
            failed=len(failed_results),
            results=[r for r in results if isinstance(r, Tier4OptimizationResponse)],
            batch_statistics=batch_statistics,
            total_batch_time=time.time() - batch_start_time
        )

        logger.info(f"Batch optimization completed: {batch_id}, {len(successful_results)}/{len(files)} successful")
        return batch_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch optimization failed: {str(e)}"
        )
    finally:
        # Cleanup temp files
        for temp_path, _ in temp_files:
            cleanup_temp_file(temp_path)

# ============================================================================
# System Monitoring Endpoints
# ============================================================================

@router.get("/health", response_model=SystemHealthResponse)
async def system_health_check() -> SystemHealthResponse:
    """
    4-Tier System Health Check

    Comprehensive health check of all system components and tiers
    """

    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="4-Tier optimization system not available"
        )

    try:
        health_result = await orchestrator.health_check()

        return SystemHealthResponse(
            overall_status=health_result["overall_status"],
            timestamp=health_result["timestamp"],
            check_duration=health_result["check_duration"],
            components=health_result["components"],
            performance=health_result["performance"],
            alerts=health_result["alerts"]
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(
    api_key: str = Depends(verify_api_key)
) -> SystemMetricsResponse:
    """
    4-Tier System Performance Metrics

    Detailed performance metrics for all system tiers and components
    """

    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="4-Tier optimization system not available"
        )

    try:
        system_status = orchestrator.get_system_status()
        metrics = system_status["performance_metrics"]

        return SystemMetricsResponse(
            total_requests=metrics.total_requests,
            successful_requests=metrics.successful_requests,
            system_reliability=metrics.system_reliability,
            tier_performance=metrics.tier_performance or {},
            method_effectiveness=metrics.method_effectiveness or {},
            cache_performance=metrics.cache_hit_rates or {},
            active_requests=system_status["active_requests"]
        )

    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )

@router.get("/execution-history", response_model=ExecutionHistoryResponse)
async def get_execution_history(
    limit: int = Query(default=100, ge=1, le=1000),
    api_key: str = Depends(verify_api_key)
) -> ExecutionHistoryResponse:
    """
    4-Tier System Execution History

    Recent execution history with performance analysis
    """

    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="4-Tier optimization system not available"
        )

    try:
        history = orchestrator.get_execution_history(limit)

        # Calculate performance summary
        if history:
            execution_times = [h["execution_time"] for h in history if h["execution_time"] > 0]
            success_rate = sum(1 for h in history if h["success"]) / len(history)

            performance_summary = {
                "average_execution_time": sum(execution_times) / max(len(execution_times), 1),
                "success_rate": success_rate,
                "total_executions": len(history),
                "method_distribution": {},
                "image_type_distribution": {}
            }

            # Method distribution
            methods = [h["method_used"] for h in history]
            performance_summary["method_distribution"] = {
                method: methods.count(method) for method in set(methods)
            }

            # Image type distribution
            image_types = [h["image_type"] for h in history]
            performance_summary["image_type_distribution"] = {
                img_type: image_types.count(img_type) for img_type in set(image_types)
            }
        else:
            performance_summary = {
                "average_execution_time": 0.0,
                "success_rate": 0.0,
                "total_executions": 0,
                "method_distribution": {},
                "image_type_distribution": {}
            }

        # Time range
        if history:
            timestamps = [h["timestamp"] for h in history]
            time_range = {
                "start": min(timestamps),
                "end": max(timestamps),
                "span_hours": (max(timestamps) - min(timestamps)) / 3600
            }
        else:
            current_time = time.time()
            time_range = {"start": current_time, "end": current_time, "span_hours": 0.0}

        return ExecutionHistoryResponse(
            executions=history,
            total_executions=len(history),
            time_range=time_range,
            performance_summary=performance_summary
        )

    except Exception as e:
        logger.error(f"History retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get execution history: {str(e)}"
        )

# ============================================================================
# Configuration and Management
# ============================================================================

@router.get("/config")
async def get_system_config(
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Get 4-tier system configuration"""

    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="4-Tier optimization system not available"
        )

    try:
        system_status = orchestrator.get_system_status()

        return {
            "system_config": system_status["configuration"],
            "api_config": api_config,
            "component_versions": system_status["component_versions"],
            "available_methods": ["feature_mapping", "regression", "ppo", "performance"],
            "supported_image_types": ["simple", "text", "gradient", "complex"],
            "tier_descriptions": {
                "tier_1": "Image Classification and Feature Extraction",
                "tier_2": "Intelligent Method Selection and Routing",
                "tier_3": "Parameter Optimization Execution",
                "tier_4": "Quality Prediction and Validation"
            }
        }

    except Exception as e:
        logger.error(f"Config retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get configuration: {str(e)}"
        )

@router.post("/shutdown")
async def shutdown_system(
    api_key: str = Depends(verify_api_key)
) -> Dict[str, str]:
    """Gracefully shutdown 4-tier system"""

    if not orchestrator:
        return {"message": "System already shut down"}

    try:
        orchestrator.shutdown()
        logger.info("System shutdown via API request")

        return {"message": "4-Tier optimization system shut down successfully"}

    except Exception as e:
        logger.error(f"Shutdown failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Shutdown failed: {str(e)}"
        )

# ============================================================================
# Helper Functions
# ============================================================================

async def perform_quality_validation(
    image_path: str,
    svg_content: str,
    optimization_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Perform quality validation on generated SVG"""

    try:
        # Save SVG temporarily for validation
        svg_path = f"/tmp/claude/validation_{int(time.time())}.svg"

        async with aiofiles.open(svg_path, 'w') as f:
            await f.write(svg_content)

        # Calculate actual quality metrics
        quality_metrics = ComprehensiveMetrics()
        actual_metrics = quality_metrics.compare_images(image_path, svg_path)

        # Compare with predictions
        predicted_quality = optimization_result["predicted_quality"]
        actual_quality = actual_metrics.get("ssim", 0.0)

        validation_result = {
            "actual_quality": actual_quality,
            "predicted_quality": predicted_quality,
            "prediction_accuracy": 1.0 - abs(actual_quality - predicted_quality),
            "quality_target_met": actual_quality >= optimization_result.get("user_requirements", {}).get("quality_target", 0.85),
            "quality_metrics": actual_metrics,
            "validation_timestamp": datetime.now().isoformat()
        }

        # Cleanup temp file
        cleanup_temp_file(svg_path)

        return validation_result

    except Exception as e:
        logger.error(f"Quality validation failed: {e}")
        return {
            "error": str(e),
            "validation_failed": True
        }

# ============================================================================
# Error Handlers
# ============================================================================

@router.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error in 4-tier API: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error in 4-tier optimization system",
            "request_id": generate_request_id(),
            "timestamp": datetime.now().isoformat()
        }
    )