# AI Modules Logging Conventions

This document defines logging conventions and best practices for AI modules.

## Logger Naming Convention

### Hierarchical Structure
```
backend.ai_modules                     # Root logger
├── classification                     # Classification components
│   ├── feature_extractor             # Specific component
│   ├── logo_classifier               # Specific component
│   └── rule_based_classifier         # Specific component
├── optimization                       # Optimization components
│   ├── feature_mapping               # Specific component
│   ├── rl_optimizer                  # Specific component
│   └── adaptive_optimizer            # Specific component
├── prediction                         # Prediction components
│   ├── quality_predictor             # Specific component
│   └── model_utils                   # Specific component
└── utils                             # Utility components
    ├── performance_monitor           # Performance monitoring
    └── logging_config               # Logging configuration
```

### Logger Naming Examples
```python
# Get loggers using the utility function
logger = get_ai_logger("classification.feature_extractor")
logger = get_ai_logger("optimization.rl_optimizer")
logger = get_ai_logger("prediction.quality_predictor")
```

## Log Levels

### DEBUG
- Internal state information
- Detailed parameter values
- Step-by-step operation progress
```python
logger.debug(f"Extracting features with params: {params}")
logger.debug(f"Intermediate result: shape={data.shape}")
```

### INFO
- Normal operation flow
- Successful completions
- Important state changes
```python
logger.info("Feature extraction completed successfully")
logger.info(f"Classified as {logo_type} with confidence {confidence:.3f}")
```

### WARNING
- Fallback behavior triggered
- Performance degradation
- Recoverable errors
```python
logger.warning("Model not found, using fallback predictor")
logger.warning(f"Operation took {duration:.3f}s (expected <0.1s)")
```

### ERROR
- Operation failures
- Exception handling
- Critical issues
```python
logger.error(f"Failed to load image: {image_path}")
logger.exception("Unexpected error during optimization")
```

## Structured Logging

### Operation Logging
Use `log_ai_operation()` for structured operation logging:
```python
log_ai_operation("feature_extraction",
                level="INFO",
                image_path=image_path,
                duration=duration,
                features_count=len(features))
```

### Performance Logging
Use `log_ai_performance()` for performance metrics:
```python
log_ai_performance("classification",
                  duration=duration,
                  memory_delta=memory_delta,
                  success=True,
                  logo_type=logo_type,
                  confidence=confidence)
```

### Custom Extra Data
Add structured data to any log message:
```python
logger.info("Operation completed",
           extra={
               'operation': 'optimize_parameters',
               'duration': 0.05,
               'parameters_count': 8,
               'logo_type': 'simple'
           })
```

## Environment Configurations

### Development
- File + Console logging
- DEBUG level for AI modules
- Human-readable format
```python
setup_ai_logging("development")
```

### Production
- File logging only
- INFO level for AI modules
- Structured JSON format
```python
setup_ai_logging("production")
```

### Testing
- Console logging only
- INFO level for AI modules
- Minimal format
```python
setup_ai_logging("testing")
```

## Log File Organization

### File Structure
```
logs/ai_modules/
├── main.log              # All AI module activities
├── classification.log    # Classification component logs
├── optimization.log      # Optimization component logs
├── prediction.log        # Prediction component logs
├── performance.log       # Performance metrics
└── errors.log           # Error-level messages only
```

### File Rotation
- Main logs: 10MB, 5 backups
- Component logs: 5MB, 3 backups
- Performance logs: 10MB, 5 backups
- Error logs: 5MB, 10 backups

## Message Format Guidelines

### Standard Messages
```python
# Good: Descriptive and informative
logger.info(f"Feature extraction completed: {len(features)} features in {duration:.3f}s")

# Bad: Too verbose or unclear
logger.info(f"The feature extraction process has now been completed successfully and we found {len(features)} features")
```

### Error Messages
```python
# Good: Clear error with context
logger.error(f"Image loading failed: {image_path} - {str(error)}")

# Bad: Vague error message
logger.error("Something went wrong")
```

### Performance Messages
```python
# Good: Relevant metrics
logger.info(f"Classification: {logo_type} (confidence: {confidence:.3f}, {duration*1000:.1f}ms)")

# Bad: Too much detail
logger.info(f"The classification algorithm determined with confidence {confidence} that the logo type is {logo_type} after {duration} seconds")
```

## Integration with Performance Monitoring

### Automatic Performance Logging
Performance decorators automatically log to performance logger:
```python
@monitor_performance("feature_extraction")
def extract_features(self, image_path):
    # Automatically logs performance metrics
    pass
```

### Manual Performance Logging
For operations without decorators:
```python
start_time = time.time()
# ... operation ...
duration = time.time() - start_time

log_ai_performance("custom_operation",
                  duration=duration,
                  memory_delta=memory_delta,
                  custom_metric=value)
```

## Best Practices

### 1. Context Information
Always include relevant context:
```python
logger.info(f"Starting optimization for {logo_type} logo: {image_path}")
```

### 2. Consistent Formatting
Use consistent formats for similar operations:
```python
# Classification results
logger.info(f"Classification: {logo_type} (confidence: {confidence:.3f})")

# Optimization results
logger.info(f"Optimization: {param_count} parameters optimized in {duration:.3f}s")
```

### 3. Avoid Sensitive Data
Never log sensitive information:
```python
# Good
logger.info(f"Processing image: {os.path.basename(image_path)}")

# Bad - might contain sensitive paths
logger.info(f"Processing image: {image_path}")
```

### 4. Exception Handling
Use `logger.exception()` for caught exceptions:
```python
try:
    result = risky_operation()
except Exception as e:
    logger.exception("Operation failed")
    # Handle error appropriately
```

### 5. Performance Awareness
Be mindful of logging performance impact:
```python
# Good: Only format if logging level allows
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Complex debug info: {expensive_operation()}")

# Better: Use lazy formatting
logger.debug("Complex debug info: %s", lambda: expensive_operation())
```

## Testing Logging

### Test Log Output
```python
def test_logging():
    with self.assertLogs('backend.ai_modules.test', level='INFO') as log:
        logger = get_ai_logger('test')
        logger.info('Test message')

    self.assertIn('Test message', log.output[0])
```

### Capture Performance Logs
```python
def test_performance_logging():
    setup_ai_logging("testing")
    log_ai_performance("test_operation", 0.05, 2.0)
    # Verify performance metrics were logged
```

## Troubleshooting

### Common Issues

1. **Logs not appearing**: Check log level configuration
2. **File permissions**: Ensure write access to log directory
3. **Disk space**: Monitor log file sizes and rotation
4. **Performance impact**: Use appropriate log levels in production

### Debug Logging Configuration
```python
# Enable debug logging for specific component
import logging
logging.getLogger('backend.ai_modules.classification').setLevel(logging.DEBUG)
```

## Configuration Examples

### Custom Log Directory
```python
setup_ai_logging("development", log_dir="/custom/log/path")
```

### Structured Logging Only
```python
ai_logging_config.setup_logging(
    enable_file_logging=True,
    enable_console_logging=False,
    structured_logging=True
)
```

This logging system provides comprehensive monitoring and debugging capabilities while maintaining performance and readability.