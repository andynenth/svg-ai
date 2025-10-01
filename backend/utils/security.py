import re
from typing import Any, Dict, Optional
import base64
from pathlib import Path

class SecurityValidator:
    def __init__(self):
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.allowed_image_types = ['image/png', 'image/jpeg', 'image/gif']
        self.dangerous_patterns = [
            r'\.\./',          # Path traversal
            r'<script',        # Script injection
            r'javascript:',    # JavaScript protocol
            r'data:.*base64',  # Suspicious data URLs
            r'file://',        # File protocol
        ]

    def validate_image_upload(self, image_data: str, filename: str = None) -> Dict[str, Any]:
        """Validate uploaded image data"""
        result = {'valid': True, 'errors': []}

        # Validate base64 format
        try:
            decoded = base64.b64decode(image_data)
            if len(decoded) > self.max_file_size:
                result['valid'] = False
                result['errors'].append(f"File too large: {len(decoded)} bytes")
        except Exception:
            result['valid'] = False
            result['errors'].append("Invalid base64 encoding")

        # Validate filename if provided
        if filename:
            if not self.validate_filename(filename):
                result['valid'] = False
                result['errors'].append("Invalid filename")

        return result

    def validate_filename(self, filename: str) -> bool:
        """Validate filename for security"""
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return False

        # Check filename length
        if len(filename) > 255:
            return False

        # Check for only safe characters
        safe_pattern = r'^[a-zA-Z0-9._-]+$'
        return bool(re.match(safe_pattern, filename))

    def sanitize_output(self, data: Any) -> Any:
        """Sanitize output data"""
        if isinstance(data, str):
            # Remove potentially dangerous content
            data = re.sub(r'<script.*?</script>', '', data, flags=re.IGNORECASE | re.DOTALL)
            data = re.sub(r'javascript:', '', data, flags=re.IGNORECASE)
        elif isinstance(data, dict):
            return {k: self.sanitize_output(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_output(item) for item in data]

        return data