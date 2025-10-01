# Backend API Documentation

## Base URL
```
http://localhost:8000
```

## Endpoints

### Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "ok"
}
```

---

### Upload File
```
POST /api/upload
```

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: File upload with field name `file`

**Example Request:**
```bash
curl -X POST -F "file=@test.png" http://localhost:8000/api/upload
```

**Example Response:**
```json
{
  "file_id": "28e6f76dcb3259ca3ad4b314d3a4d86f",
  "filename": "test.png",
  "path": "uploads/28e6f76dcb3259ca3ad4b314d3a4d86f.png"
}
```

---

### Convert to SVG
```
POST /api/convert
```

**Request:**
- Method: POST
- Content-Type: application/json

**Body:**
```json
{
  "file_id": "28e6f76dcb3259ca3ad4b314d3a4d86f",
  "converter": "alpha",
  "threshold": 128
}
```

**Parameters:**
- `file_id` (required): The file ID returned from upload endpoint
- `converter` (optional): Converter type - "alpha", "vtracer", or "potrace" (default: "alpha")
- `threshold` (optional): Threshold value for conversion (default: 128)

**Example Request:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"file_id": "28e6f76dcb3259ca3ad4b314d3a4d86f", "converter": "alpha"}' \
  http://localhost:8000/api/convert
```

**Example Response:**
```json
{
  "success": true,
  "svg": "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg>...</svg>",
  "size": 1024,
  "ssim": 0.95
}
```

---

## Error Codes

### 400 Bad Request
- No file provided in upload
- No file selected
- Only PNG files allowed
- No file_id provided

### 404 Not Found
- File not found (invalid file_id)
- Endpoint not found

### 500 Internal Server Error
- Conversion failed
- Unknown converter

---

## Example Workflow

1. Upload a PNG file:
```bash
curl -X POST -F "file=@logo.png" http://localhost:8000/api/upload
```

2. Get the file_id from response:
```json
{"file_id": "abc123...", "filename": "logo.png", "path": "uploads/abc123...png"}
```

3. Convert to SVG:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"file_id": "abc123...", "converter": "vtracer"}' \
  http://localhost:8000/api/convert
```

4. Receive SVG content:
```json
{"success": true, "svg": "<svg>...</svg>", "size": 2048, "ssim": 0.97}
```

---

## Supported Converters

- **alpha**: Alpha-aware converter for images with transparency
- **vtracer**: VTracer converter for complex color images
- **potrace**: Potrace converter for black and white conversion

---

## File Formats

- Input: PNG, JPG, JPEG
- Output: SVG

---

## Limits

- Maximum file size: 16MB
- Supported image formats: PNG, JPG, JPEG