# backend/ai_modules/utils/logging_config.py
"""Logging configuration for AI modules"""

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format(self, record):
        # Create structured log entry
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "operation"):
            log_data["operation"] = record.operation
        if hasattr(record, "duration"):
            log_data["duration"] = record.duration
        if hasattr(record, "memory_delta"):
            log_data["memory_delta"] = record.memory_delta
        if hasattr(record, "image_path"):
            log_data["image_path"] = record.image_path
        if hasattr(record, "parameters"):
            log_data["parameters"] = record.parameters

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, indent=None, separators=(",", ":"))


class AILoggingConfig:
    """Centralized logging configuration for AI modules"""

    def __init__(self, log_dir: Optional[str] = None, level: str = "INFO"):
        self.log_dir = Path(log_dir or "logs")
        self.log_dir.mkdir(exist_ok=True)
        self.level = level.upper()

    def setup_logging(
        self,
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        structured_logging: bool = False,
    ) -> None:
        """Set up comprehensive logging configuration"""

        # Create log directory structure
        self.log_dir.mkdir(exist_ok=True)
        (self.log_dir / "ai_modules").mkdir(exist_ok=True)

        # Define log files
        log_files = {
            "main": self.log_dir / "ai_modules" / "main.log",
            "classification": self.log_dir / "ai_modules" / "classification.log",
            "optimization": self.log_dir / "ai_modules" / "optimization.log",
            "prediction": self.log_dir / "ai_modules" / "prediction.log",
            "performance": self.log_dir / "ai_modules" / "performance.log",
            "errors": self.log_dir / "ai_modules" / "errors.log",
        }

        # Configure formatters
        formatters = {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {"format": "%(levelname)s - %(name)s - %(message)s"},
            "structured": {"()": StructuredFormatter},
        }

        # Configure handlers
        handlers = {}

        # Console handler
        if enable_console_logging:
            handlers["console"] = {
                "class": "logging.StreamHandler",
                "level": self.level,
                "formatter": "structured" if structured_logging else "simple",
                "stream": "ext://sys.stdout",
            }

        # File handlers
        if enable_file_logging:
            # Main log file
            handlers["file_main"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "structured" if structured_logging else "detailed",
                "filename": str(log_files["main"]),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            }

            # Component-specific log files
            for component in ["classification", "optimization", "prediction"]:
                handlers[f"file_{component}"] = {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "structured" if structured_logging else "detailed",
                    "filename": str(log_files[component]),
                    "maxBytes": 5242880,  # 5MB
                    "backupCount": 3,
                }

            # Performance log file
            handlers["file_performance"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "structured" if structured_logging else "detailed",
                "filename": str(log_files["performance"]),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            }

            # Error log file
            handlers["file_errors"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "structured" if structured_logging else "detailed",
                "filename": str(log_files["errors"]),
                "maxBytes": 5242880,  # 5MB
                "backupCount": 10,
            }

        # Configure loggers
        loggers = {
            "backend.ai_modules": {
                "level": "DEBUG",
                "handlers": ["console", "file_main"] if enable_file_logging else ["console"],
                "propagate": False,
            },
            "backend.ai_modules.classification": {
                "level": "DEBUG",
                "handlers": ["file_classification"] if enable_file_logging else [],
                "propagate": True,
            },
            "backend.ai_modules.optimization": {
                "level": "DEBUG",
                "handlers": ["file_optimization"] if enable_file_logging else [],
                "propagate": True,
            },
            "backend.ai_modules.prediction": {
                "level": "DEBUG",
                "handlers": ["file_prediction"] if enable_file_logging else [],
                "propagate": True,
            },
            "backend.ai_modules.utils.performance_monitor": {
                "level": "INFO",
                "handlers": ["file_performance"] if enable_file_logging else [],
                "propagate": True,
            },
        }

        # Remove handlers that don't exist
        existing_handlers = list(handlers.keys())
        for logger_config in loggers.values():
            logger_config["handlers"] = [
                h for h in logger_config["handlers"] if h in existing_handlers
            ]

        # Build logging configuration
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": formatters,
            "handlers": handlers,
            "loggers": loggers,
            "root": {"level": "WARNING", "handlers": ["console"] if enable_console_logging else []},
        }

        # Apply configuration
        logging.config.dictConfig(config)

        # Log configuration success
        logger = logging.getLogger("backend.ai_modules")
        logger.info(
            f"AI modules logging configured - Level: {self.level}, "
            f"File logging: {enable_file_logging}, "
            f"Console logging: {enable_console_logging}, "
            f"Structured: {structured_logging}"
        )

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the appropriate configuration"""
        return logging.getLogger(f"backend.ai_modules.{name}")

    def log_operation(
        self, logger: logging.Logger, operation: str, level: str = "INFO", **kwargs
    ) -> None:
        """Log an AI operation with structured data"""
        extra = {"operation": operation}
        extra.update(kwargs)

        log_level = getattr(logging, level.upper())
        logger.log(log_level, f"AI operation: {operation}", extra=extra)

    def log_performance(
        self, operation: str, duration: float, memory_delta: float, success: bool = True, **kwargs
    ) -> None:
        """Log performance metrics"""
        perf_logger = logging.getLogger("backend.ai_modules.utils.performance_monitor")

        extra = {
            "operation": operation,
            "duration": duration,
            "memory_delta": memory_delta,
            "success": success,
        }
        extra.update(kwargs)

        level = logging.INFO if success else logging.WARNING
        message = f"Performance: {operation} - {duration:.3f}s, {memory_delta:+.1f}MB"

        perf_logger.log(level, message, extra=extra)

    def setup_development_logging(self) -> None:
        """Setup logging for development environment"""
        self.setup_logging(
            enable_file_logging=True, enable_console_logging=True, structured_logging=False
        )

    def setup_production_logging(self) -> None:
        """Setup logging for production environment"""
        self.setup_logging(
            enable_file_logging=True, enable_console_logging=False, structured_logging=True
        )

    def setup_testing_logging(self) -> None:
        """Setup minimal logging for testing"""
        self.setup_logging(
            enable_file_logging=False, enable_console_logging=True, structured_logging=False
        )


# Global logging configuration instance
ai_logging_config = AILoggingConfig()


# Convenience functions
def setup_ai_logging(environment: str = "development", log_dir: Optional[str] = None) -> None:
    """Setup AI logging for specified environment"""
    global ai_logging_config
    ai_logging_config = AILoggingConfig(log_dir=log_dir)

    if environment.lower() == "production":
        ai_logging_config.setup_production_logging()
    elif environment.lower() == "testing":
        ai_logging_config.setup_testing_logging()
    else:
        ai_logging_config.setup_development_logging()


def get_ai_logger(name: str) -> logging.Logger:
    """Get an AI module logger"""
    return ai_logging_config.get_logger(name)


def log_ai_operation(operation: str, level: str = "INFO", **kwargs) -> None:
    """Log an AI operation with structured data"""
    logger = logging.getLogger("backend.ai_modules")
    ai_logging_config.log_operation(logger, operation, level, **kwargs)


def log_ai_performance(operation: str, duration: float, memory_delta: float, **kwargs) -> None:
    """Log AI performance metrics"""
    ai_logging_config.log_performance(operation, duration, memory_delta, **kwargs)


# Initialize with development settings by default
if not logging.getLogger("backend.ai_modules").handlers:
    setup_ai_logging("development")
