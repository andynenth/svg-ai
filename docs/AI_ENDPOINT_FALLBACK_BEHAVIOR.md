# AI Endpoint Fallback Behavior Documentation

## Overview

The AI-enhanced SVG conversion endpoints implement a robust fallback mechanism to ensure service availability even when AI components are unavailable or fail. This document describes the fallback behavior and degraded mode operations.

## Fallback Triggers

The system will fall back to basic conversion when:

1. **AI Components Not Initialized**: Model manager, quality predictor, or router fail to initialize
2. **Models Not Found**: No AI models are available in the configured model directory
3. **Tier Processing Failure**: Selected tier encounters an error during conversion
4. **Timeout**: AI processing exceeds the specified time budget
5. **Resource Constraints**: Memory or CPU limits are exceeded

## Fallback Behavior

### `/api/convert-ai` Endpoint

When AI conversion fails, the endpoint:

1. **Logs Detailed Error Context**:
   - Converter attempted
   - Tier attempted
   - Target quality requested
   - Error type and message

2. **Attempts Basic VTracer Conversion**:
   - Falls back to non-AI optimized VTracer
   - Uses default parameters
   - Calculates real quality metrics (SSIM, MSE, PSNR)

3. **Verifies Fallback Success**:
   - Checks that SVG was generated
   - Validates conversion success flag
   - Returns error if fallback also fails

4. **Returns Structured Response**:
   ```json
   {
     "success": true,
     "svg": "<svg>...</svg>",
     "ai_metadata": {
       "fallback_used": true,
       "fallback_reason": "Error message",
       "tier_attempted": 2,
       "tier_used": "fallback",
       "error_context": {
         "converter": "AIEnhancedConverter",
         "tier_attempted": 2,
         "error_type": "RuntimeError",
         "error_message": "Model not loaded"
       },
       "quality_metrics": {
         "ssim": 0.95,
         "mse": 100.0,
         "psnr": 30.0
       }
     }
   }
   ```

### `/api/ai-health` Endpoint

The health endpoint provides actionable guidance when models are missing:

```json
{
  "model_manager": {
    "status": "degraded",
    "models_found": false,
    "model_directory": "models/production",
    "guidance": "No AI models found. To enable AI features, export models to: models/production",
    "instructions": [
      "1. Export quality_predictor.torchscript to the model directory",
      "2. Export logo_classifier.onnx to the model directory",
      "3. Export correlation_models.pkl to the model directory",
      "4. Restart the service to load models"
    ]
  }
}
```

## Degraded Mode Operations

### Available Features in Degraded Mode

- ✅ Basic PNG to SVG conversion
- ✅ Quality metrics calculation (SSIM, MSE, PSNR)
- ✅ File size optimization
- ✅ Path count analysis
- ✅ Standard VTracer parameters

### Unavailable Features in Degraded Mode

- ❌ AI-optimized parameter selection
- ❌ Logo type classification
- ❌ Quality prediction
- ❌ Intelligent tier routing
- ❌ Adaptive parameter optimization

## Error Handling Best Practices

### For API Consumers

1. **Check `fallback_used` Flag**: Determine if AI optimization was applied
2. **Examine `error_context`**: Understand why fallback was triggered
3. **Monitor Quality Metrics**: Compare actual quality against requirements
4. **Retry Logic**: Implement exponential backoff for transient failures

### For Operators

1. **Monitor `/api/ai-health`**: Regular health checks for AI component status
2. **Check Logs**: Review error contexts to identify patterns
3. **Model Deployment**: Follow instructions in health endpoint for model setup
4. **Resource Monitoring**: Ensure adequate memory and CPU for AI operations

## Configuration

### Environment Variables

```bash
# Model directory location
MODEL_DIR=/app/models/production

# Enable/disable fallback
ENABLE_FALLBACK=true

# Fallback timeout (seconds)
FALLBACK_TIMEOUT=10

# Log level for debugging
LOG_LEVEL=INFO
```

### Fallback Configuration in Code

```python
# In backend/api/ai_endpoints.py
FALLBACK_CONFIG = {
    'enabled': True,
    'timeout': 10,
    'default_converter': 'vtracer',
    'log_errors': True,
    'return_metrics': True
}
```

## Monitoring and Alerts

### Key Metrics to Monitor

- **Fallback Rate**: Percentage of requests using fallback
- **AI Success Rate**: Successful AI conversions / total requests
- **Model Load Status**: Binary indicator of model availability
- **Average Quality**: Mean SSIM across conversions
- **Error Types**: Distribution of error categories

### Recommended Alerts

1. **High Fallback Rate**: > 10% requests using fallback
2. **Models Not Loaded**: AI models unavailable for > 5 minutes
3. **Quality Degradation**: Average SSIM < 0.85
4. **Conversion Failures**: Both AI and fallback fail

## Testing Fallback Behavior

### Manual Testing

```bash
# Test with missing models
rm -rf models/production/*
curl -X POST http://localhost:8000/api/convert-ai \
  -H "Content-Type: application/json" \
  -d '{"file_id": "test.png", "tier": 2}'

# Check health endpoint
curl http://localhost:8000/api/ai-health
```

### Automated Tests

See `tests/test_ai_endpoints_fallbacks.py` for comprehensive fallback testing:

- Test missing model scenarios
- Test invalid tier requests
- Test fallback success paths
- Test error propagation

## Troubleshooting

### Common Issues

1. **"No models found" but models exist**
   - Check MODEL_DIR environment variable
   - Verify model file permissions
   - Ensure correct model formats (.torchscript, .onnx, .pkl)

2. **Fallback also fails**
   - Check input image validity
   - Verify VTracer installation
   - Review disk space for temp files

3. **High memory usage during fallback**
   - Implement memory limits
   - Use smaller batch sizes
   - Enable garbage collection

## Future Improvements

1. **Smart Fallback**: Use historical data to predict optimal fallback parameters
2. **Partial AI**: Use available models even if some are missing
3. **Caching**: Cache fallback results for repeated requests
4. **Progressive Enhancement**: Start with fallback, upgrade to AI when available

---

*Last Updated: Day 1 - AI Endpoint Stabilization*
*Version: 1.0.0*