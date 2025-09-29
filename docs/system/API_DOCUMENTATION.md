# PNG to SVG Converter API Documentation

Version: 1.0.0
Base URL: http://localhost:8000

## Health Check

Check if the server is running and healthy.

**GET** `/health`

**Response:**
- `200 OK`
```json
{
  "status": "ok"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

## Upload Image

Upload a PNG or JPEG image to convert to SVG.

**POST** `/api/upload`

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (required): Image file (PNG or JPEG format)

**Success Response:**
- `200 OK`
```json
{
  "file_id": "28e6f76dcb3259ca3ad4b314d3a4d86f",
  "filename": "logo.png",
  "path": "uploads/28e6f76dcb3259ca3ad4b314d3a4d86f.png"
}
```

**Error Responses:**
- `400 Bad Request`
```json
{
  "error": "No file provided"
}
```
- `413 Payload Too Large`
```json
{
  "error": "File too large (max 16MB)"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@logo.png"
```

---

## Convert Image

Convert an uploaded image to SVG format using specified converter.

**POST** `/api/convert`

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "file_id": "28e6f76dcb3259ca3ad4b314d3a4d86f",
  "converter": "alpha",
  "threshold": 128
}
```

**Parameters:**
- `file_id` (required): ID returned from upload endpoint
- `converter` (optional): Conversion algorithm
  - `alpha` (default): Alpha-aware converter, best for logos with transparency
  - `vtracer`: Color vectorization using VTracer
  - `potrace`: Black and white conversion using Potrace
- `threshold` (optional): Threshold for black/white conversion (0-255, default: 128)

**Success Response:**
- `200 OK`
```json
{
  "success": true,
  "svg": "<svg>...</svg>",
  "ssim": 0.95,
  "size": 2048
}
```

**Fields:**
- `success`: Whether conversion succeeded
- `svg`: SVG content as string
- `ssim`: Structural similarity index (0-1, higher is better)
- `size`: Size of SVG in bytes

**Error Responses:**
- `400 Bad Request`
```json
{
  "error": "No file_id provided"
}
```
- `404 Not Found`
```json
{
  "error": "File not found"
}
```
- `500 Internal Server Error`
```json
{
  "success": false,
  "error": "Conversion failed: error details"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/convert \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "28e6f76dcb3259ca3ad4b314d3a4d86f",
    "converter": "vtracer",
    "threshold": 128
  }'
```

---

## Error Responses

All endpoints may return these error codes:

- `400 Bad Request`: Invalid input parameters
- `404 Not Found`: Resource not found
- `413 Payload Too Large`: File exceeds size limit
- `429 Too Many Requests`: Rate limit exceeded (if configured)
- `500 Internal Server Error`: Server error during processing

---

## Rate Limits

Default rate limits (when configured):
- Upload: 10 requests per minute per IP
- Convert: 20 requests per minute per IP

---

## File Formats Supported

**Input:**
- PNG (.png)
- JPEG (.jpg, .jpeg)

**Output:**
- SVG (.svg)

---

## Converter Comparison

| Converter | Best For | Color Support | Quality | Speed |
|-----------|----------|---------------|---------|-------|
| alpha | Icons with transparency | Yes | High | Fast |
| vtracer | Complex color images | Yes | High | Medium |
| potrace | Simple B&W graphics | No | Medium | Fast |

---

## Usage Example

Complete workflow to convert a PNG to SVG:

```javascript
// 1. Upload image
const formData = new FormData();
formData.append('file', imageFile);

const uploadResponse = await fetch('http://localhost:8000/api/upload', {
  method: 'POST',
  body: formData
});

const { file_id } = await uploadResponse.json();

// 2. Convert to SVG
const convertResponse = await fetch('http://localhost:8000/api/convert', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    file_id: file_id,
    converter: 'vtracer',
    threshold: 128
  })
});

const { svg, ssim } = await convertResponse.json();
console.log(`Conversion complete! Quality: ${ssim}`);
```