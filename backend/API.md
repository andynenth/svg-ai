# SVG-AI API Documentation v2.0

*Comprehensive API reference for the SVG-AI Enhanced Conversion Pipeline*

## Base Configuration

**Base URL:** `http://localhost:5000` (Flask) or `http://localhost:8000` (Web UI)
**API Version:** v2.0
**Architecture:** Consolidated (194+ files → 15 core files)

## Core Endpoints

### Health Check

```http
GET /health
```

System health and status check.

**Response:**
```json
{
  "status": "ok",
  "version": "2.0",
  "architecture": "consolidated"
}
```

**Example:**
```bash
curl http://localhost:5000/health
```

---

### File Upload

```http
POST /api/upload
```

Upload PNG, JPEG, or JPG image for conversion.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** File with field name `file`
- **Max Size:** 10MB (configurable)

**Supported Formats:**
- PNG (.png)
- JPEG (.jpg, .jpeg)

**Example Request:**
```bash
curl -X POST -F "file=@logo.png" http://localhost:5000/api/upload
```

**Success Response (200):**
```json
{
  "file_id": "28e6f76dcb3259ca3ad4b314d3a4d86f",
  "filename": "logo.png",
  "path": "uploads/28e6f76dcb3259ca3ad4b314d3a4d86f.png",
  "size_bytes": 45678,
  "uploaded_at": "2025-09-30T10:30:00Z"
}
```

**Error Responses:**
- `400` - No file provided, invalid format, or file too large
- `413` - Payload too large

---

### Standard Conversion

```http
POST /api/convert
```

Convert uploaded image to SVG using standard converters.

**Request Body:**
```json
{
  "file_id": "28e6f76dcb3259ca3ad4b314d3a4d86f",
  "converter_type": "vtracer",
  "color_precision": 6,
  "corner_threshold": 60,
  "max_iterations": 10
}
```

**Parameters:**
- `file_id` (required): File ID from upload endpoint
- `converter_type` (optional): Converter to use
  - `alpha` - Alpha-aware converter (default)
  - `vtracer` - VTracer vectorization
  - `potrace` - Potrace black & white
  - `smart_auto` - AI-powered auto routing
  - `ai_enhanced` - Full AI enhancement
- **VTracer Parameters** (optional):
  - `color_precision`: 1-10 (default: 6)
  - `layer_difference`: 1-32 (default: 16)
  - `corner_threshold`: 10-90 (default: 60)
  - `max_iterations`: 1-30 (default: 10)
  - `min_area`: 1-100 (default: 10)
  - `path_precision`: 1-15 (default: 8)
  - `length_threshold`: 1.0-10.0 (default: 4.0)
  - `splice_threshold`: 10-90 (default: 45)

**Example Request:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "28e6f76dcb3259ca3ad4b314d3a4d86f",
    "converter_type": "vtracer",
    "color_precision": 8,
    "corner_threshold": 30
  }' \
  http://localhost:5000/api/convert
```

**Success Response (200):**
```json
{
  "success": true,
  "svg": "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg>...</svg>",
  "size_bytes": 2048,
  "quality_metrics": {
    "ssim": 0.95,
    "mse": 0.023,
    "psnr": 45.2
  },
  "conversion_time_ms": 234,
  "parameters_used": {
    "color_precision": 8,
    "corner_threshold": 30
  },
  "converter": "vtracer"
}
```

**Error Responses:**
- `400` - Invalid file_id or parameters
- `404` - File not found
- `500` - Conversion failed

---

## AI-Enhanced Endpoints

### Logo Classification

```http
POST /api/classify-logo
```

Classify logo type using hybrid AI classifier.

**Request Body:**
```json
{
  "file_id": "28e6f76dcb3259ca3ad4b314d3a4d86f",
  "use_neural": true
}
```

**Parameters:**
- `file_id` (required): File ID from upload
- `use_neural` (optional): Use neural network classifier (default: false)

**Response:**
```json
{
  "success": true,
  "logo_type": "simple_geometric",
  "confidence": 0.87,
  "features": {
    "unique_colors": 8,
    "complexity": 0.23,
    "edge_density": 0.45,
    "aspect_ratio": 1.2,
    "has_text": false,
    "has_gradients": false
  },
  "classification_method": "hybrid",
  "processing_time_ms": 45
}
```

**Logo Types:**
- `simple_geometric` - Simple shapes, few colors
- `text_based` - Contains text elements
- `gradient` - Contains color gradients
- `complex` - Complex imagery

---

### Parameter Optimization

```http
POST /api/optimize
```

Optimize VTracer parameters using ML-based optimization.

**Request Body:**
```json
{
  "file_id": "28e6f76dcb3259ca3ad4b314d3a4d86f",
  "target_quality": 0.9,
  "use_ml": true,
  "fine_tune": false
}
```

**Parameters:**
- `file_id` (required): File ID from upload
- `target_quality` (optional): Target SSIM score (0.0-1.0, default: 0.9)
- `use_ml` (optional): Use ML model for optimization (default: true)
- `fine_tune` (optional): Enable fine-tuning for specific image (default: false)

**Response:**
```json
{
  "success": true,
  "optimized_parameters": {
    "color_precision": 6,
    "layer_difference": 12,
    "corner_threshold": 45,
    "max_iterations": 15,
    "min_area": 8,
    "path_precision": 10,
    "length_threshold": 3.5,
    "splice_threshold": 50
  },
  "predicted_quality": 0.92,
  "optimization_method": "ml_model",
  "processing_time_ms": 12
}
```

---

### AI-Enhanced Conversion

```http
POST /api/convert-ai
```

Full AI-enhanced conversion with intelligent routing and optimization.

**Request Body:**
```json
{
  "file_id": "28e6f76dcb3259ca3ad4b314d3a4d86f",
  "tier": "auto",
  "target_quality": 0.9,
  "time_budget": 5.0,
  "include_analysis": true
}
```

**Parameters:**
- `file_id` (required): File ID from upload
- `tier` (optional): AI processing tier
  - `auto` - Automatic tier selection (default)
  - `1` - Fast processing (statistical methods)
  - `2` - Balanced processing (hybrid methods)
  - `3` - High-quality processing (full ML)
- `target_quality` (optional): Target SSIM score (default: 0.9)
- `time_budget` (optional): Maximum processing time in seconds
- `include_analysis` (optional): Include detailed AI analysis (default: true)

**Response:**
```json
{
  "success": true,
  "svg": "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg>...</svg>",
  "ai_metadata": {
    "routing": {
      "selected_tier": 2,
      "selection_method": "auto",
      "confidence": 0.85
    },
    "conversion": {
      "logo_type": "simple_geometric",
      "optimization_method": "ml_model",
      "parameters_used": {
        "color_precision": 6,
        "corner_threshold": 45
      }
    },
    "tier_used": 2,
    "predicted_quality": 0.91,
    "actual_quality": 0.93
  },
  "processing_time": 1.234,
  "endpoint": "/api/convert-ai",
  "ai_enabled": true
}
```

---

## System Health & Monitoring

### AI Health Check

```http
GET /api/ai-health
```

Comprehensive AI system health check.

**Response:**
```json
{
  "timestamp": "2025-09-30T10:30:00Z",
  "overall_status": "healthy",
  "components": {
    "ai_initialized": true,
    "model_manager": {
      "status": "healthy",
      "loaded_models": ["classification", "optimization"],
      "memory_usage_mb": 245.6,
      "within_memory_limits": true
    },
    "quality_predictor": {
      "status": "healthy",
      "model_available": true,
      "fallback_enabled": true
    },
    "router": {
      "status": "healthy",
      "feature_extractor_available": true,
      "classifier_available": true
    },
    "converter": {
      "status": "healthy",
      "converter_type": "AIEnhancedConverter"
    }
  },
  "performance_metrics": {
    "memory": {
      "current_mb": 245.6,
      "peak_mb": 290.1,
      "within_limits": true
    },
    "routing": {
      "feature_extraction_available": true,
      "classification_available": true
    }
  },
  "recommendations": [
    "System is healthy - all components operational"
  ]
}
```

**Status Values:**
- `healthy` - All systems operational
- `degraded` - Some components have issues
- `unhealthy` - Major system problems
- `error` - Critical system failure

---

### Model Status

```http
GET /api/model-status
```

Detailed ML model status and memory usage.

**Response:**
```json
{
  "models_available": true,
  "models": {
    "classification_model": {
      "loaded": true,
      "type": "HybridClassifier",
      "memory_mb": 45.2
    },
    "optimization_model": {
      "loaded": true,
      "type": "OptimizationEngine",
      "memory_mb": 32.1
    },
    "quality_predictor": {
      "loaded": false,
      "type": null,
      "memory_mb": 0
    }
  },
  "memory_report": {
    "current_memory_mb": 234.5,
    "peak_memory_mb": 290.1,
    "within_limits": true
  },
  "cache_stats": {
    "hit_rate": 0.78,
    "total_requests": 1250,
    "cache_hits": 975
  }
}
```

---

## Error Handling

### Standard Error Response Format

```json
{
  "success": false,
  "error": "Detailed error message",
  "error_code": "CONVERSION_FAILED",
  "timestamp": "2025-09-30T10:30:00Z",
  "request_id": "req_12345",
  "suggestions": [
    "Try reducing image size",
    "Use a different converter type"
  ]
}
```

### HTTP Status Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| `200` | Success | Request completed successfully |
| `400` | Bad Request | Invalid parameters, missing file_id |
| `404` | Not Found | File not found, invalid file_id |
| `413` | Payload Too Large | File exceeds size limit (10MB) |
| `422` | Unprocessable Entity | Invalid file format |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Conversion failed, system error |
| `503` | Service Unavailable | AI components not available |

### Error Codes

| Error Code | Description |
|------------|-------------|
| `FILE_NOT_FOUND` | Uploaded file not found |
| `INVALID_FORMAT` | Unsupported file format |
| `CONVERSION_FAILED` | SVG conversion failed |
| `AI_UNAVAILABLE` | AI components not initialized |
| `OPTIMIZATION_FAILED` | Parameter optimization failed |
| `CLASSIFICATION_FAILED` | Logo classification failed |
| `MEMORY_LIMIT_EXCEEDED` | System memory limit exceeded |
| `TIMEOUT` | Processing timeout exceeded |

---

## Converter Comparison

| Converter | Best For | Speed | Quality | AI Features |
|-----------|----------|--------|---------|-------------|
| `alpha` | Transparent logos | ⚡ Fast | High | None |
| `vtracer` | Complex color images | 🔄 Medium | High | Parameter optimization |
| `potrace` | B&W graphics | ⚡ Fast | Medium | None |
| `smart_auto` | Any logo type | 🔄 Medium | High | Auto-routing |
| `ai_enhanced` | Optimal results | 🐌 Slow | Highest | Full AI pipeline |

---

## Quality Metrics System (Updated Day 1)

The Quality System provides comprehensive quality assessment with **dual API compatibility**:

### API Methods Available

#### `calculate_metrics()` - Integration Test Compatible
```python
from backend.ai_modules.quality import QualitySystem
quality = QualitySystem()
metrics = quality.calculate_metrics(original_path, converted_path)
```

**Parameters:**
- `original_path` (str): Path to original image file
- `converted_path` (str): Path to converted SVG file

**Returns:** Dictionary with quality metrics

#### `calculate_comprehensive_metrics()` - Full Analysis
```python
metrics = quality.calculate_comprehensive_metrics(original_path, svg_path)
```

**Both methods return identical structure:**
```json
{
  "ssim": 0.85,
  "mse": 100.0,
  "psnr": 30.0,
  "file_size_original": 12345,
  "file_size_svg": 6789,
  "compression_ratio": 1.82,
  "quality_score": 0.65
}
```

### Quality Metric Definitions

#### SSIM (Structural Similarity Index)
- **Range:** 0.0 - 1.0
- **Target:** >0.85 for simple logos, >0.70 for complex
- **Interpretation:** Higher values indicate better structural preservation

#### MSE (Mean Squared Error)
- **Range:** 0.0 - 1.0
- **Target:** <0.05 for good quality
- **Interpretation:** Lower values indicate better pixel-level accuracy

#### PSNR (Peak Signal-to-Noise Ratio)
- **Range:** 0 - 100+ dB
- **Target:** >30 dB for acceptable quality
- **Interpretation:** Higher values indicate better signal quality

#### Compression Ratio
- **Calculation:** `original_size / svg_size`
- **Target:** 2.0+ for efficient compression
- **Interpretation:** Higher values indicate better size reduction

#### Quality Score
- **Calculation:** `ssim * 0.7 + (compression_ratio / 10.0) * 0.3`
- **Range:** 0.0 - 1.0+
- **Interpretation:** Combined quality and efficiency metric

---

## Rate Limits

**Default Limits (per IP address):**
- Upload: 30 requests/minute
- Convert: 60 requests/minute
- AI endpoints: 20 requests/minute
- Health checks: 120 requests/minute

**Rate Limit Headers:**
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1609459200
```

---

## Example Workflows

### Basic Conversion

```javascript
// 1. Upload image
const formData = new FormData();
formData.append('file', imageFile);

const uploadResponse = await fetch('/api/upload', {
  method: 'POST',
  body: formData
});
const { file_id } = await uploadResponse.json();

// 2. Convert to SVG
const convertResponse = await fetch('/api/convert', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    file_id: file_id,
    converter_type: 'vtracer',
    color_precision: 8
  })
});
const { svg, quality_metrics } = await convertResponse.json();
```

### AI-Enhanced Workflow

```javascript
// 1. Upload and classify
const { file_id } = await uploadImage(imageFile);
const classification = await classifyLogo(file_id);

// 2. Optimize parameters
const optimization = await optimizeParameters(file_id, {
  target_quality: 0.9,
  use_ml: true
});

// 3. Convert with AI enhancement
const result = await convertWithAI(file_id, {
  tier: 'auto',
  target_quality: 0.9,
  include_analysis: true
});

console.log(`Logo: ${classification.logo_type}`);
console.log(`Quality: ${result.ai_metadata.actual_quality}`);
```

### Batch Processing

```python
import asyncio
import aiohttp

async def process_batch(image_files):
    async with aiohttp.ClientSession() as session:
        # Upload all files
        upload_tasks = [upload_file(session, file) for file in image_files]
        file_ids = await asyncio.gather(*upload_tasks)

        # Convert all files
        convert_tasks = [convert_ai(session, fid) for fid in file_ids]
        results = await asyncio.gather(*convert_tasks)

        return results
```

---

## Configuration

### Environment Variables

```bash
# Core settings
SVG_AI_DEBUG=false
SVG_AI_MAX_FILE_SIZE=10485760  # 10MB
SVG_AI_UPLOAD_FOLDER=./uploads
SVG_AI_PORT=5000

# AI features
SVG_AI_ENABLE_ML=true
SVG_AI_MODEL_CACHE=./models
SVG_AI_ONLINE_LEARNING=false
SVG_AI_MEMORY_LIMIT=512  # MB

# Performance
SVG_AI_ENABLE_CACHING=true
SVG_AI_CACHE_SIZE=1000
SVG_AI_WORKER_THREADS=4

# Rate limiting
SVG_AI_RATE_LIMIT_ENABLED=true
SVG_AI_UPLOAD_RATE_LIMIT=30
SVG_AI_CONVERT_RATE_LIMIT=60
```

---

## Security Features

- ✅ Input validation for all file uploads
- ✅ File type and size restrictions
- ✅ No execution of user-provided code
- ✅ Sanitized error messages
- ✅ CORS configuration
- ✅ Rate limiting on API endpoints
- ✅ Request ID tracking
- ✅ Memory usage monitoring

---

## Performance Benchmarks

| Operation | Avg Time | Target | Status |
|-----------|----------|--------|---------|
| File Upload | <100ms | <200ms | ✅ Excellent |
| Alpha Conversion | 100-300ms | <500ms | ✅ Excellent |
| VTracer Conversion | 200-800ms | <1000ms | ✅ Good |
| Logo Classification | 1-100ms | <200ms | ✅ Excellent |
| Parameter Optimization | <1ms | <10ms | ✅ Excellent |
| AI-Enhanced Conversion | 1-5s | <10s | ✅ Good |

---

**API Documentation v2.0** - *Last Updated: September 30, 2025*