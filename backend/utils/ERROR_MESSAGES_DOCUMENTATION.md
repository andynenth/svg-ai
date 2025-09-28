# Error Message System Documentation

## Overview

The SVG-AI converter uses a standardized error message system that provides consistent, appropriate messages for different audiences:

- **User Messages**: Simple, actionable messages for end users
- **Developer Messages**: Detailed technical information for developers
- **Log Messages**: Comprehensive context for debugging and monitoring

## Usage

### Basic Error Creation

```python
from utils.error_messages import ErrorMessageFactory, create_api_error_response

# Create a standardized error
error = ErrorMessageFactory.create_error("FILE_NOT_FOUND",
                                        {"file_path": "/path/to/file.png"})

# Log the error
error.log(logger)

# Return as API response
return jsonify(create_api_error_response(error)), 404
```

### Quick Logging and Creation

```python
from utils.error_messages import log_error_with_context

# Create, log, and return error in one call
error = log_error_with_context("CONVERSION_FAILED",
                               {"converter": "vtracer", "image_path": "test.png"},
                               original_exception,
                               logger)
```

## Available Error Types

### File Operation Errors

- **FILE_NOT_FOUND**: File doesn't exist at specified path
- **FILE_PERMISSION_DENIED**: Insufficient permissions to access file
- **INVALID_FILE_FORMAT**: Unsupported file format

### Conversion Errors

- **CONVERSION_FAILED**: General conversion failure
- **CONVERTER_NOT_AVAILABLE**: Required converter tool not installed

### System Errors

- **INSUFFICIENT_MEMORY**: Out of memory or file too large
- **TIMEOUT_ERROR**: Operation exceeded time limit

### Validation Errors

- **INVALID_PARAMETERS**: Invalid or missing parameters

## API Response Format

The standardized API error response format is:

```json
{
  "success": false,
  "error": {
    "code": "FILE_NOT_FOUND",
    "message": "The file you selected could not be found. Please check the file path and try again.",
    "suggestions": [
      "Check that the file exists at the specified location",
      "Verify file permissions",
      "Try uploading the file again"
    ]
  },
  "debug": {
    "technical_message": "File not found at specified path: /uploads/test.png",
    "category": "file_operation",
    "severity": "error"
  }
}
```

**Note**: The `debug` section is only included when logging level is DEBUG.

## Error Categories and Severity

### Categories
- `file_operation`: File system related errors
- `conversion`: Image conversion related errors
- `validation`: Input validation errors
- `network`: Network and communication errors
- `configuration`: Configuration and setup errors
- `system`: System resource and environment errors

### Severity Levels
- `info`: Informational messages
- `warning`: Warning conditions
- `error`: Error conditions
- `critical`: Critical failures

## Adding New Error Types

To add a new error type, add it to the `ERROR_TEMPLATES` dictionary in `ErrorMessageFactory`:

```python
"NEW_ERROR_TYPE": {
    "user_message": "Simple message for users",
    "developer_message": "Technical details with {context_var}",
    "log_message": "Full context: {context_var} - {original_error}",
    "category": ErrorCategory.CONVERSION,
    "severity": ErrorSeverity.ERROR,
    "suggestions": [
        "Actionable suggestion 1",
        "Actionable suggestion 2"
    ]
}
```

## Context Variables

Error messages support context variable substitution:

```python
error = ErrorMessageFactory.create_error("CONVERSION_FAILED",
                                        {
                                            "converter": "vtracer",
                                            "image_path": "/path/to/image.png",
                                            "file_size": "2.5MB"
                                        },
                                        original_exception)
```

## Best Practices

1. **Always provide context**: Include relevant parameters like file paths, converter names, etc.
2. **Log errors appropriately**: Use the built-in logging to ensure errors are tracked
3. **Use appropriate severity**: Match the severity level to the actual impact
4. **Provide actionable suggestions**: Help users resolve the issue
5. **Include original exceptions**: Pass the original exception for technical context

## Integration Examples

### Flask Route Error Handling

```python
@app.route("/api/convert", methods=["POST"])
def convert():
    try:
        # ... conversion logic ...
        return jsonify(result)
    except FileNotFoundError as e:
        error = log_error_with_context("FILE_NOT_FOUND",
                                     {"file_path": file_path},
                                     e,
                                     app.logger)
        return jsonify(create_api_error_response(error)), 404
    except Exception as e:
        error = log_error_with_context("CONVERSION_FAILED",
                                     {"converter": converter_type},
                                     e,
                                     app.logger)
        return jsonify(create_api_error_response(error)), 500
```

### Converter Error Handling

```python
def convert(self, image_path: str, **kwargs) -> str:
    try:
        # ... conversion logic ...
        return svg_content
    except subprocess.CalledProcessError as e:
        error = log_error_with_context("CONVERTER_NOT_AVAILABLE",
                                     {"converter": self.get_name()},
                                     e,
                                     logger)
        raise RuntimeError(error.developer_message) from e
```

This standardized system ensures consistent error handling across the entire application while providing appropriate information for different audiences.