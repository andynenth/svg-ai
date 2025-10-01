# SVG-AI User Guide

## Quick Start

### API Endpoint: Convert Image to SVG
```bash
curl -X POST http://your-domain/api/convert \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image_data",
    "format": "png",
    "options": {
      "optimize": true,
      "quality_target": 0.9
    }
  }'
```

### Response Format
```json
{
  "svg": "<svg>...</svg>",
  "quality": {
    "ssim": 0.95,
    "mse": 12.3,
    "psnr": 42.1
  },
  "parameters": {
    "color_precision": 6,
    "corner_threshold": 60
  },
  "processing_time": 1.23
}
```

## API Endpoints

### Health Check
- **GET** `/health` - System health status
- **GET** `/api/classification-status` - AI components status

### Conversion
- **POST** `/api/convert` - Convert single image
- **POST** `/api/batch-convert` - Convert multiple images
- **POST** `/api/optimize` - Get optimized parameters

### Classification
- **POST** `/api/classify-logo` - Classify logo type

## Error Handling

All API endpoints return structured error responses:
```json
{
  "error": "Error description",
  "error_type": "ValidationError",
  "details": {
    "field": "image",
    "message": "Invalid base64 encoding"
  }
}
```

## Rate Limits

- Convert endpoint: 10 requests/minute
- Batch endpoint: 2 requests/minute
- Other endpoints: 50 requests/hour

## Quality Guidelines

Expected quality scores by image type:
- Simple geometric: SSIM > 0.95
- Text-based: SSIM > 0.98
- Gradient logos: SSIM > 0.90
- Complex designs: SSIM > 0.80

## Detailed API Documentation

### Single Image Conversion

**Endpoint:** `POST /api/convert`

**Request Body:**
```json
{
  "image": "string", // Base64 encoded image data
  "format": "string", // Input format: "png", "jpg", "jpeg", "gif"
  "converter": "string", // Optional: "vtracer", "potrace", "smart", "alpha"
  "options": {
    "optimize": "boolean", // Enable parameter optimization
    "quality_target": "number", // Target SSIM score (0.0-1.0)
    "max_iterations": "number" // Max optimization iterations
  }
}
```

**Response:**
```json
{
  "svg": "string", // Generated SVG content
  "quality": {
    "ssim": "number", // Structural similarity index
    "mse": "number", // Mean squared error
    "psnr": "number" // Peak signal-to-noise ratio
  },
  "parameters": {
    // Converter-specific parameters used
  },
  "processing_time": "number", // Time in seconds
  "optimization_iterations": "number" // If optimization enabled
}
```

### Batch Image Conversion

**Endpoint:** `POST /api/batch-convert`

**Request Body:**
```json
{
  "images": [
    {
      "name": "string", // Filename
      "data": "string", // Base64 encoded image data
      "format": "string" // Input format
    }
  ],
  "options": {
    "batch_size": "number", // Processing batch size
    "max_workers": "number", // Parallel workers
    "optimize": "boolean" // Enable optimization
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "name": "string",
      "success": "boolean",
      "svg": "string", // If successful
      "quality": {}, // Quality metrics
      "error": "string" // If failed
    }
  ],
  "summary": {
    "total_images": "number",
    "successful_conversions": "number",
    "failed_conversions": "number",
    "processing_time": "number"
  }
}
```

### Logo Classification

**Endpoint:** `POST /api/classify-logo`

**Request Body:**
```json
{
  "image": "string", // Base64 encoded image data
  "format": "string" // Input format
}
```

**Response:**
```json
{
  "type": "string", // "simple_geometric", "text_based", "gradient", "complex"
  "confidence": "number", // Classification confidence (0.0-1.0)
  "features": {
    "edge_density": "number",
    "color_count": "number",
    "text_detected": "boolean",
    "has_gradients": "boolean"
  },
  "recommended_parameters": {
    // Suggested converter parameters for this type
  }
}
```

### Parameter Optimization

**Endpoint:** `POST /api/optimize`

**Request Body:**
```json
{
  "image": "string", // Base64 encoded image data
  "format": "string",
  "target_ssim": "number", // Target quality score
  "max_iterations": "number" // Optional: max iterations
}
```

**Response:**
```json
{
  "optimized_parameters": {
    // Best parameters found
  },
  "final_quality": {
    "ssim": "number",
    "mse": "number",
    "psnr": "number"
  },
  "iterations": "number",
  "optimization_time": "number"
}
```

## Usage Examples

### Python Example
```python
import requests
import base64

# Read and encode image
with open('logo.png', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Convert to SVG
response = requests.post('http://your-domain/api/convert', json={
    'image': image_data,
    'format': 'png',
    'options': {
        'optimize': True,
        'quality_target': 0.9
    }
})

if response.status_code == 200:
    result = response.json()

    # Save SVG
    with open('logo.svg', 'w') as f:
        f.write(result['svg'])

    print(f"Quality: {result['quality']['ssim']:.3f}")
    print(f"Processing time: {result['processing_time']:.2f}s")
else:
    print(f"Error: {response.json()['error']}")
```

### JavaScript Example
```javascript
async function convertImageToSVG(imageFile) {
    // Convert file to base64
    const reader = new FileReader();
    reader.readAsDataURL(imageFile);

    reader.onload = async function(event) {
        const base64Data = event.target.result.split(',')[1];

        try {
            const response = await fetch('/api/convert', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: base64Data,
                    format: 'png',
                    options: {
                        optimize: true,
                        quality_target: 0.9
                    }
                })
            });

            if (response.ok) {
                const result = await response.json();
                console.log('SVG:', result.svg);
                console.log('Quality:', result.quality.ssim);
            } else {
                const error = await response.json();
                console.error('Error:', error.error);
            }
        } catch (error) {
            console.error('Request failed:', error);
        }
    };
}
```

### cURL Examples

**Basic conversion:**
```bash
curl -X POST http://localhost/api/convert \
  -H "Content-Type: application/json" \
  -d @- << EOF
{
  "image": "$(base64 -i logo.png)",
  "format": "png"
}
EOF
```

**With optimization:**
```bash
curl -X POST http://localhost/api/convert \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$(base64 -w 0 logo.png)'",
    "format": "png",
    "options": {
      "optimize": true,
      "quality_target": 0.95
    }
  }'
```

**Batch conversion:**
```bash
curl -X POST http://localhost/api/batch-convert \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      {
        "name": "logo1.png",
        "data": "'$(base64 -w 0 logo1.png)'",
        "format": "png"
      },
      {
        "name": "logo2.png",
        "data": "'$(base64 -w 0 logo2.png)'",
        "format": "png"
      }
    ],
    "options": {
      "optimize": true
    }
  }'
```

## Best Practices

### Image Preparation
1. **Format:** PNG is recommended for best quality
2. **Resolution:** 500-2000px for optimal processing speed
3. **File Size:** Keep under 10MB per image
4. **Content:** Clean, high-contrast images work best

### Quality Optimization
1. Use `optimize: true` for automatic parameter tuning
2. Set realistic quality targets (0.85-0.95 for most cases)
3. Allow sufficient iterations for complex images
4. Consider logo type when setting expectations

### Performance Tips
1. Use batch processing for multiple images
2. Enable parallel processing with appropriate worker count
3. Cache frequently converted images
4. Monitor rate limits for high-volume usage

### Error Handling
1. Always check response status codes
2. Implement retry logic for temporary failures
3. Handle rate limiting gracefully
4. Validate image data before sending

## Troubleshooting

### Common Issues

**Invalid base64 encoding:**
```json
{
  "error": "Invalid base64 encoding",
  "error_type": "ValidationError"
}
```
Solution: Ensure proper base64 encoding without line breaks

**Image too large:**
```json
{
  "error": "File size exceeds maximum allowed (10MB)",
  "error_type": "FileSizeError"
}
```
Solution: Reduce image size or resolution

**Rate limit exceeded:**
```json
{
  "error": "Rate limit exceeded",
  "error_type": "RateLimitError"
}
```
Solution: Wait before retrying or reduce request frequency

**Low quality results:**
- Try different converter types
- Enable optimization
- Check input image quality
- Adjust quality target

**Slow processing:**
- Reduce image resolution
- Use simpler converter settings
- Consider batch processing for multiple images

### Getting Help

1. Check system health: `GET /health`
2. Verify component status: `GET /api/classification-status`
3. Review error messages carefully
4. Check rate limit headers
5. Monitor processing times for performance issues

For additional support, please refer to the operations manual or contact the system administrator.