# SVG-AI API Reference

## Base URL
```
http://your-domain/
```

## Authentication
Currently, no authentication is required. Rate limiting is applied per IP address.

## Content Type
All requests must use `Content-Type: application/json` for POST requests.

## Rate Limits
- `/api/convert`: 10 requests per minute
- `/api/batch-convert`: 2 requests per minute
- Other endpoints: 50 requests per hour

Rate limit headers are included in all responses:
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 9
X-RateLimit-Reset: 1640995200
```

## Error Responses

All errors follow a consistent format:

```json
{
  "error": "Error description",
  "error_type": "ErrorType",
  "details": {
    "field": "field_name",
    "message": "Detailed error message"
  },
  "timestamp": "2023-12-01T12:00:00Z",
  "request_id": "uuid"
}
```

### Error Types
- `ValidationError`: Invalid input data
- `ProcessingError`: Conversion failed
- `RateLimitError`: Rate limit exceeded
- `FileSizeError`: File too large
- `FormatError`: Unsupported format
- `ServerError`: Internal server error

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (validation error)
- `413`: Payload Too Large
- `422`: Unprocessable Entity
- `429`: Too Many Requests
- `500`: Internal Server Error

## Endpoints

### Health Check

#### GET /health
Check system health and status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T12:00:00Z",
  "version": "1.0.0",
  "uptime": 3600,
  "components": {
    "redis": "connected",
    "converter": "ready",
    "ai_classifier": "ready"
  }
}
```

**Status Values:**
- `healthy`: All systems operational
- `degraded`: Some components unavailable
- `unhealthy`: Critical components down

#### GET /api/classification-status
Check AI classification component status.

**Response:**
```json
{
  "classification_available": true,
  "components": {
    "traditional_classifier": "ready",
    "neural_network": "ready"
  },
  "performance": {
    "avg_classification_time": 0.12,
    "cache_hit_rate": 0.85
  }
}
```

### File Upload

Upload an image file for conversion processing.

**Endpoint:** `POST /api/upload`

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (required): Image file (PNG, JPEG)
  - Maximum file size: 16MB
  - Supported formats: PNG (.png), JPEG (.jpg, .jpeg)

**Response:**
```json
{
  "file_id": "abc123def456...",
  "filename": "original_filename.png",
  "path": "/path/to/uploaded/file.png"
}
```

**Example:**
```bash
curl -X POST \
  -F "file=@logo.png" \
  http://localhost:8001/api/upload
```

**Error Responses:**
- `400 Bad Request`: Missing file, invalid format, or file too large
- `413 Request Entity Too Large`: File exceeds maximum size limit
- `500 Internal Server Error`: Upload processing failed

### Image Conversion

Convert an uploaded image to SVG format using specified converter and parameters.

**Endpoint:** `POST /api/convert`

**Content-Type:** `application/json`

**Required Parameters:**
- `file_id` (string): File identifier returned from upload endpoint

**Optional Parameters:**

#### Common Parameters
- `converter` (string): Converter type
  - Options: `"alpha"`, `"vtracer"`, `"potrace"`, `"smart"`, `"smart_auto"`
  - Default: `"alpha"`
- `threshold` (integer): Color threshold for processing
  - Range: 0-255
  - Default: 128

#### VTracer Parameters
- `colormode` (string): Color processing mode
  - Options: `"color"`, `"binary"`
  - Default: `"color"`
- `color_precision` (integer): Color reduction level
  - Range: 1-10
  - Default: 6
- `layer_difference` (integer): Layer separation threshold
  - Range: 0-256
  - Default: 16
- `path_precision` (integer): Path coordinate precision
  - Range: 0-10
  - Default: 5
- `corner_threshold` (integer): Corner detection threshold
  - Range: 0-180
  - Default: 60
- `length_threshold` (float): Minimum path length
  - Range: 0-100
  - Default: 5.0
- `max_iterations` (integer): Maximum processing iterations
  - Range: 1-50
  - Default: 10
- `splice_threshold` (integer): Path splicing threshold
  - Range: 0-180
  - Default: 45

#### Potrace Parameters
- `turnpolicy` (string): Turn policy for path generation
  - Options: `"black"`, `"white"`, `"right"`, `"left"`, `"minority"`, `"majority"`
  - Default: `"black"`
- `turdsize` (integer): Filter out small components
  - Range: 0-100
  - Default: 2
- `alphamax` (float): Corner rounding parameter
  - Range: 0-1.34
  - Default: 1.0
- `opttolerance` (float): Optimization tolerance
  - Range: 0.0-1.0
  - Default: 0.2

#### Alpha Converter Parameters
- `use_potrace` (boolean): Use Potrace for binary regions
  - Default: true
- `preserve_antialiasing` (boolean): Preserve edge antialiasing
  - Default: false

**Request Example:**
```json
{
  "file_id": "abc123def456...",
  "converter": "vtracer",
  "colormode": "color",
  "color_precision": 6,
  "corner_threshold": 60
}
```

**Response:**
```json
{
  "success": true,
  "svg": "<svg xmlns=\"http://www.w3.org/2000/svg\"...>...</svg>",
  "size": 2048,
  "ssim": 0.95,
  "path_count": 15,
  "avg_path_length": 136,
  "converter_type": "vtracer"
}
```

**Response Fields:**
- `success` (boolean): Conversion success status
- `svg` (string): Generated SVG content
- `size` (integer): SVG file size in bytes
- `ssim` (float): Structural similarity index (quality metric)
- `path_count` (integer): Number of SVG paths generated
- `avg_path_length` (integer): Average path complexity
- `converter_type` (string): Converter used for conversion

**Example:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"file_id":"abc123","converter":"vtracer","color_precision":6}' \
  http://localhost:8001/api/convert
```

**Error Responses:**
- `400 Bad Request`: Invalid parameters or missing file_id
- `404 Not Found`: File not found for given file_id
- `500 Internal Server Error`: Conversion processing failed

## AI-Enhanced Conversion

The system supports AI-enhanced conversion when AI modules are available. The AI enhancement automatically:

1. **Analyzes image features:** Edge density, color complexity, entropy, corners, gradients
2. **Classifies logo type:** Simple geometric, text-based, gradient, or complex
3. **Optimizes parameters:** Automatically selects optimal VTracer parameters based on classification
4. **Provides metadata:** Includes AI analysis results in response

### AI Enhancement Usage

AI enhancement is automatically applied when:
- AI modules are installed and available
- The system detects the image can benefit from AI optimization
- No explicit `ai_disable` parameter is provided

To disable AI enhancement for a specific conversion:
```json
{
  "file_id": "abc123",
  "converter": "vtracer",
  "ai_disable": true
}
```

### AI Response Format

When AI enhancement is active, the response includes additional metadata:
```json
{
  "success": true,
  "svg": "...",
  "ai_enhanced": true,
  "classification": {
    "logo_type": "simple",
    "confidence": 0.85
  },
  "features": {
    "edge_density": 0.23,
    "unique_colors": 0.15,
    "entropy": 0.78,
    "corner_density": 0.42,
    "gradient_strength": 0.12,
    "complexity_score": 0.35
  },
  "ai_analysis_time": 0.125,
  "parameters_used": {
    "color_precision": 3,
    "corner_threshold": 30,
    "layer_difference": 32
  }
}
```

## Rate Limits

No explicit rate limiting is currently enforced, but consider:
- File upload size limits (16MB)
- Processing timeout for complex images
- Reasonable request intervals for production use

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": "Human-readable error message",
  "debug": {
    "technical_message": "Detailed technical information"
  }
}
```

Common HTTP status codes:
- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `413 Request Entity Too Large`: File too large
- `500 Internal Server Error`: Server processing error

## Security Considerations

- **File Validation:** Uploaded files are validated by content (magic bytes) and extension
- **Input Sanitization:** All parameters are validated and sanitized
- **Path Traversal Protection:** File IDs are validated to prevent directory traversal
- **Size Limits:** File upload size is limited to prevent DoS attacks
- **Security Headers:** CSP, X-Frame-Options, and other security headers are enforced

## SDK and Libraries

### Python SDK Example

```python
import requests
import json

class SVGConverter:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url

    def upload_file(self, file_path):
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/api/upload",
                files={'file': f}
            )
        return response.json()

    def convert_image(self, file_id, **params):
        data = {'file_id': file_id, **params}
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        return response.json()

# Usage
converter = SVGConverter()
upload_result = converter.upload_file("logo.png")
file_id = upload_result['file_id']

conversion_result = converter.convert_image(
    file_id,
    converter="vtracer",
    color_precision=6,
    corner_threshold=60
)

if conversion_result['success']:
    with open("output.svg", "w") as f:
        f.write(conversion_result['svg'])
```

### JavaScript/Node.js Example

```javascript
const FormData = require('form-data');
const fetch = require('node-fetch');
const fs = require('fs');

class SVGConverter {
    constructor(baseUrl = 'http://localhost:8001') {
        this.baseUrl = baseUrl;
    }

    async uploadFile(filePath) {
        const form = new FormData();
        form.append('file', fs.createReadStream(filePath));

        const response = await fetch(`${this.baseUrl}/api/upload`, {
            method: 'POST',
            body: form
        });

        return await response.json();
    }

    async convertImage(fileId, params = {}) {
        const response = await fetch(`${this.baseUrl}/api/convert`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_id: fileId, ...params })
        });

        return await response.json();
    }
}

// Usage
const converter = new SVGConverter();

(async () => {
    const uploadResult = await converter.uploadFile('logo.png');
    const fileId = uploadResult.file_id;

    const conversionResult = await converter.convertImage(fileId, {
        converter: 'vtracer',
        color_precision: 6,
        corner_threshold: 60
    });

    if (conversionResult.success) {
        fs.writeFileSync('output.svg', conversionResult.svg);
    }
})();
```

## Testing

Use the health endpoint to verify API availability:

```bash
# Check API health
curl http://localhost:8001/health

# Test file upload
curl -X POST -F "file=@test.png" http://localhost:8001/api/upload

# Test conversion
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"file_id":"<file_id>","converter":"vtracer"}' \
  http://localhost:8001/api/convert
```