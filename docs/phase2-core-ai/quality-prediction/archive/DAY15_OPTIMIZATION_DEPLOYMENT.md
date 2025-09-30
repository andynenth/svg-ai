# Day 15: Performance Optimization & Deployment - Quality Prediction Model

**Date**: Week 4, Day 5 (Friday)
**Duration**: 8 hours
**Team**: 2 developers
**Objective**: Optimize Quality Prediction Model for CPU deployment and prepare production infrastructure

---

## Prerequisites Checklist

Before starting, verify these are complete:
- [x] Phase 2.3: Quality Prediction Model trained and validated
- [x] ResNet-50 + MLP architecture finalized
- [x] Model accuracy meets performance targets (>85% SSIM prediction accuracy)
- [x] Integration interfaces defined for optimization system
- [x] Target deployment environment: Intel Mac x86_64 CPU-only

---

## Developer A Tasks (4 hours) - CPU Performance Optimization

### Task A15.1: Model Quantization and Compression ⏱️ 2 hours

**Objective**: Optimize the Quality Prediction Model for CPU inference with quantization and compression techniques.

**Implementation**:
```python
# backend/ai_modules/quality_prediction/model_optimizer.py
import torch
import torch.quantization as quantization
from torch.jit import script
import onnx
import onnxruntime as ort

class ModelOptimizer:
    """Optimize Quality Prediction Model for CPU deployment"""

    def __init__(self, model_path: str):
        self.model = torch.load(model_path, map_location='cpu')
        self.optimized_models = {}

    def quantize_dynamic(self) -> torch.nn.Module:
        """Apply dynamic quantization for CPU inference"""
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        return quantized_model

    def convert_to_onnx(self, input_shape: tuple) -> str:
        """Convert PyTorch model to ONNX for optimized runtime"""
        dummy_input = torch.randn(input_shape)
        onnx_path = "model_optimized.onnx"

        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        return onnx_path
```

**Detailed Checklist**:
- [ ] Implement dynamic quantization for Linear and Conv2d layers
- [ ] Apply INT8 quantization with calibration dataset
- [ ] Convert model to ONNX format for optimized inference
- [ ] Implement FP16 quantization for memory optimization
- [ ] Test quantized model accuracy vs. original (target: >95% retention)
- [ ] Benchmark inference speed improvement (target: 2-3x faster)
- [ ] Optimize memory usage during quantization process
- [ ] Create quantization configuration profiles (aggressive, balanced, conservative)
- [ ] Implement model compression with pruning techniques
- [ ] Validate compressed model maintains prediction accuracy
- [ ] Document quantization impact on model performance

**Performance Targets**:
- Model size reduction: >50% (target: 60-70%)
- Inference speed improvement: >200% (target: 250-300%)
- Accuracy retention: >95% of original model performance
- Memory usage during inference: <256MB

**Deliverable**: Optimized model variants with quantization and compression

### Task A15.2: Intel Mac x86_64 Specific Optimizations ⏱️ 2 hours

**Objective**: Apply Intel-specific optimizations for maximum CPU performance on x86_64 architecture.

**Implementation**:
```python
# backend/ai_modules/quality_prediction/intel_optimizer.py
import torch
import mkl
import numpy as np
from typing import Dict, Any

class IntelCPUOptimizer:
    """Intel x86_64 specific optimizations for Quality Prediction Model"""

    def __init__(self):
        self.configure_mkl()
        self.configure_torch_threads()

    def configure_mkl(self):
        """Configure Intel MKL for optimal performance"""
        # Set MKL threads based on CPU cores
        import os
        cpu_count = os.cpu_count()
        mkl.set_num_threads(cpu_count)

        # Enable MKL-DNN optimizations
        torch.backends.mkldnn.enabled = True
        torch.backends.mkldnn.verbose = 0

    def configure_torch_threads(self):
        """Configure PyTorch threading for Intel CPUs"""
        import os
        cpu_count = os.cpu_count()

        # Set intra-op and inter-op threads
        torch.set_num_threads(cpu_count)
        torch.set_num_interop_threads(1)

        # Enable OpenMP optimizations
        os.environ['OMP_NUM_THREADS'] = str(cpu_count)
        os.environ['MKL_NUM_THREADS'] = str(cpu_count)

    def optimize_model_for_intel(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply Intel-specific model optimizations"""
        model.eval()

        # Enable MKLDNN optimization
        model = torch.jit.optimize_for_inference(model)

        # Apply Intel-specific fusion optimizations
        model = torch.jit.freeze(model)

        return model
```

**Detailed Checklist**:
- [ ] Configure Intel MKL-DNN for optimal BLAS operations
- [ ] Set optimal thread count based on CPU cores (detect dynamically)
- [ ] Enable PyTorch Intel extensions if available
- [ ] Configure OpenMP settings for Intel CPUs
- [ ] Apply Intel-specific model fusion optimizations
- [ ] Implement memory alignment optimizations for x86_64
- [ ] Use Intel VTune profiler integration for performance analysis
- [ ] Configure CPU affinity for consistent performance
- [ ] Implement SIMD optimizations where applicable
- [ ] Benchmark performance across different Intel CPU generations
- [ ] Create Intel-optimized inference pipeline
- [ ] Document Intel-specific configuration parameters

**Intel Optimization Targets**:
- Thread utilization: 100% of available CPU cores
- Cache hit ratio: >95% for model weights
- SIMD instruction usage: Maximize vectorization
- Memory bandwidth utilization: >80% of theoretical maximum

**Deliverable**: Intel x86_64 optimized inference engine with maximum CPU utilization

---

## Developer B Tasks (4 hours) - Deployment Infrastructure

### Task B15.1: Deployment Configuration and Containerization ⏱️ 2 hours

**Objective**: Create production-ready deployment configuration with Docker containerization.

**Implementation**:
```dockerfile
# deployments/quality_prediction/Dockerfile
FROM python:3.9-slim

# Install system dependencies for Intel optimizations
RUN apt-get update && apt-get install -y \
    intel-mkl \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Intel MKL environment variables
ENV MKL_NUM_THREADS=4
ENV OMP_NUM_THREADS=4
ENV MKL_DYNAMIC=false

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements_deployment.txt .
RUN pip install --no-cache-dir -r requirements_deployment.txt

# Copy model and application code
COPY models/quality_prediction_optimized.onnx ./models/
COPY backend/ai_modules/quality_prediction/ ./quality_prediction/
COPY deployments/quality_prediction/app.py .

# Create non-root user
RUN useradd -m -u 1000 modeluser && chown -R modeluser:modeluser /app
USER modeluser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start application
CMD ["python", "app.py"]
```

**Detailed Checklist**:
- [ ] Create optimized Dockerfile with Intel MKL support
- [ ] Configure environment variables for Intel optimizations
- [ ] Implement multi-stage build for smaller image size
- [ ] Add security configurations (non-root user, minimal permissions)
- [ ] Create docker-compose.yml for local development
- [ ] Implement health check endpoint for container monitoring
- [ ] Configure proper logging and log rotation
- [ ] Add resource limits and constraints
- [ ] Create deployment scripts for different environments
- [ ] Implement configuration management with environment variables
- [ ] Test container startup time (target: <30 seconds)
- [ ] Validate container security scanning

**Container Specifications**:
- Base image: python:3.9-slim with Intel MKL
- Memory limit: 1GB
- CPU limit: 4 cores
- Storage: 500MB for models and cache
- Network: HTTP/8080 with health checks

**Deliverable**: Production-ready Docker container with optimized runtime environment

### Task B15.2: Model Serving Infrastructure Setup ⏱️ 2 hours

**Objective**: Implement high-performance model serving infrastructure with monitoring and health checks.

**Implementation**:
```python
# deployments/quality_prediction/app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import time
import psutil
import logging
from typing import Dict, List
import numpy as np

app = FastAPI(
    title="Quality Prediction Service",
    description="SSIM Quality Prediction for SVG Optimization",
    version="1.0.0"
)

class ModelServer:
    """High-performance model serving with caching and monitoring"""

    def __init__(self):
        self.model = None
        self.cache = {}
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'avg_latency': 0,
            'errors': 0
        }
        self.load_model()

    def load_model(self):
        """Load optimized model for inference"""
        import onnxruntime as ort

        # Configure ONNX Runtime for Intel CPU
        providers = ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.model = ort.InferenceSession(
            "models/quality_prediction_optimized.onnx",
            sess_options=sess_options,
            providers=providers
        )

    async def predict_quality(self, features: np.ndarray) -> float:
        """Predict SSIM quality with caching and monitoring"""
        start_time = time.time()

        try:
            # Check cache first
            cache_key = hash(features.tobytes())
            if cache_key in self.cache:
                self.stats['cache_hits'] += 1
                return self.cache[cache_key]

            # Run inference
            input_name = self.model.get_inputs()[0].name
            result = self.model.run(None, {input_name: features})
            prediction = float(result[0][0])

            # Cache result
            self.cache[cache_key] = prediction

            # Update statistics
            latency = time.time() - start_time
            self.stats['requests'] += 1
            self.stats['avg_latency'] = (
                (self.stats['avg_latency'] * (self.stats['requests'] - 1) + latency)
                / self.stats['requests']
            )

            return prediction

        except Exception as e:
            self.stats['errors'] += 1
            raise HTTPException(status_code=500, detail=str(e))

model_server = ModelServer()

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()

    return {
        "status": "healthy",
        "cpu_usage": cpu_percent,
        "memory_usage": memory.percent,
        "model_loaded": model_server.model is not None,
        "cache_size": len(model_server.cache)
    }

@app.get("/metrics")
async def get_metrics():
    """Performance metrics endpoint"""
    return {
        "model_stats": model_server.stats,
        "system_stats": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "cache_size": len(model_server.cache)
        }
    }

@app.post("/predict")
async def predict_ssim(features: Dict[str, float]):
    """Predict SSIM quality from image features"""
    try:
        # Convert features to numpy array
        feature_array = np.array([[
            features.get('edge_density', 0.0),
            features.get('unique_colors', 0.0),
            features.get('entropy', 0.0),
            features.get('corner_density', 0.0),
            features.get('gradient_strength', 0.0),
            features.get('complexity_score', 0.0)
        ]], dtype=np.float32)

        prediction = await model_server.predict_quality(feature_array)

        return {
            "predicted_ssim": prediction,
            "confidence": "high",
            "latency_ms": model_server.stats['avg_latency'] * 1000
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        workers=1,
        loop="uvloop",
        access_log=True
    )
```

**Detailed Checklist**:
- [ ] Implement FastAPI-based model serving with async support
- [ ] Add request/response caching for repeated predictions
- [ ] Create comprehensive health check endpoint
- [ ] Implement performance metrics collection and endpoint
- [ ] Add request validation and error handling
- [ ] Configure CORS for web integration
- [ ] Implement request rate limiting and throttling
- [ ] Add structured logging with correlation IDs
- [ ] Create monitoring dashboard integration hooks
- [ ] Implement graceful shutdown handling
- [ ] Add request/response compression
- [ ] Test concurrent request handling (target: 100 RPS)

**API Specifications**:
- POST /predict: SSIM prediction from image features
- GET /health: Container health and resource status
- GET /metrics: Performance and usage statistics
- Response time: <50ms for cached, <100ms for new predictions
- Throughput: >100 requests per second

**Deliverable**: Production-ready model serving API with monitoring and caching

---

## Integration Preparation for Agent 4

### Interface Contracts

**Quality Prediction API Interface**:
```python
# API Contract for Optimization System Integration
class QualityPredictionAPI:
    """Interface contract for optimization system integration"""

    async def predict_quality(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict SSIM quality from image features

        Args:
            features: Dictionary with keys:
                - edge_density: float [0.0, 1.0]
                - unique_colors: int [2, 256]
                - entropy: float [0.0, 8.0]
                - corner_density: float [0.0, 1.0]
                - gradient_strength: float [0.0, 1.0]
                - complexity_score: float [0.0, 1.0]

        Returns:
            {
                "predicted_ssim": float [0.0, 1.0],
                "confidence": str ["low", "medium", "high"],
                "latency_ms": float,
                "cached": bool
            }
        """

    async def batch_predict(self, features_list: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Batch prediction for multiple feature sets"""

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and performance statistics"""
```

**Performance Guarantees**:
- Single prediction latency: <50ms (cached), <100ms (new)
- Batch prediction throughput: >100 predictions/second
- Model accuracy: >85% SSIM prediction accuracy
- Uptime requirement: >99.9% availability
- Memory usage: <512MB during peak load

### Integration Hooks

**Optimization System Integration Points**:
1. **Parameter Optimization**: Quality predictions inform parameter tuning
2. **Route Selection**: Quality predictions influence converter routing
3. **Batch Processing**: Parallel quality predictions for batch optimization
4. **Performance Monitoring**: Quality prediction accuracy tracking
5. **Fallback Mechanisms**: Graceful degradation when quality service unavailable

---

## End-of-Day Checklist

### Performance Validation
- [ ] Model inference time <50ms on target hardware
- [ ] Quantized model accuracy >95% of original
- [ ] Container startup time <30 seconds
- [ ] API response time <100ms for new predictions
- [ ] Memory usage <512MB during normal operation
- [ ] CPU utilization optimized for Intel x86_64

### Deployment Readiness
- [ ] Docker container builds successfully
- [ ] Health checks pass consistently
- [ ] API endpoints respond correctly
- [ ] Monitoring metrics are collected
- [ ] Configuration is externalized
- [ ] Security scanning passes

### Integration Preparation
- [ ] API interface contracts defined
- [ ] Performance guarantees documented
- [ ] Integration hooks identified
- [ ] Error handling mechanisms tested
- [ ] Fallback strategies implemented

---

## Production Deployment Checklist

### Infrastructure Requirements
- **Hardware**: Intel Mac x86_64 with 4+ CPU cores
- **Memory**: 2GB RAM minimum, 4GB recommended
- **Storage**: 1GB for models and cache
- **Network**: HTTP/HTTPS access on port 8080

### Configuration Management
- **Environment Variables**: All configuration externalized
- **Secrets Management**: Model paths and API keys secured
- **Logging Configuration**: Structured logging to stdout/stderr
- **Monitoring Integration**: Metrics exported to monitoring system

### Security Considerations
- **Container Security**: Non-root user, minimal attack surface
- **API Security**: Input validation, rate limiting
- **Network Security**: TLS termination at load balancer
- **Secret Management**: No sensitive data in container images

### Monitoring and Alerting
- **Health Monitoring**: Container health checks every 30 seconds
- **Performance Monitoring**: Response time and throughput metrics
- **Error Monitoring**: Error rate and failure alerts
- **Resource Monitoring**: CPU and memory usage tracking

---

## Success Criteria

✅ **Day 15 Success**: Quality Prediction Model optimized for production deployment with Intel CPU optimizations

**Performance Targets Achieved**:
- Model size: <50MB (target: <100MB)
- Inference time: <30ms (target: <50ms)
- Accuracy retention: >95% after optimization
- API response time: <50ms cached, <100ms new
- Container startup: <30 seconds
- Memory usage: <512MB during operation

**Deliverables**:
- Quantized and optimized model variants
- Intel x86_64 specific optimizations
- Production-ready Docker container
- High-performance model serving API
- Comprehensive monitoring and health checks
- Integration interface contracts
- Production deployment documentation

**Files Created**:
- `backend/ai_modules/quality_prediction/model_optimizer.py`
- `backend/ai_modules/quality_prediction/intel_optimizer.py`
- `deployments/quality_prediction/Dockerfile`
- `deployments/quality_prediction/app.py`
- `deployments/quality_prediction/docker-compose.yml`
- `deployments/quality_prediction/requirements_deployment.txt`
- `docs/deployment/QUALITY_PREDICTION_DEPLOYMENT.md`

**Ready for Integration**: Quality Prediction Model is production-ready and optimized for integration with the 3-tier optimization system.