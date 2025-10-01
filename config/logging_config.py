# config/logging_config.py
"""Production logging configuration"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON log formatter for production"""

    def format(self, record):
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)

        # Add custom fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename',
                          'funcName', 'levelname', 'levelno', 'lineno',
                          'module', 'msecs', 'message', 'pathname', 'process',
                          'processName', 'relativeCreated', 'thread',
                          'threadName', 'exc_info', 'exc_text', 'stack_info']:
                log_obj[key] = value

        return json.dumps(log_obj)


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None
):
    """Configure logging for production"""

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)

    if log_format == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )

    root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )

        if log_format == "json":
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )

        root_logger.addHandler(file_handler)

    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    return root_logger


# Structured logging helpers
class StructuredLogger:
    """Helper for structured logging"""

    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)

    def log_request(self, method: str, path: str, status: int, duration: float):
        """Log API request"""
        self.logger.info(
            "API Request",
            extra={
                'method': method,
                'path': path,
                'status': status,
                'duration_ms': duration * 1000
            }
        )

    def log_conversion(self, image_type: str, quality: float, duration: float):
        """Log conversion result"""
        self.logger.info(
            "Conversion completed",
            extra={
                'image_type': image_type,
                'quality': quality,
                'duration_ms': duration * 1000
            }
        )

    def log_error(self, error_type: str, message: str, **kwargs):
        """Log error with context"""
        self.logger.error(
            message,
            extra={
                'error_type': error_type,
                **kwargs
            }
        )