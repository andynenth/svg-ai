import logging
import json
import time
from typing import Dict, Any

class StructuredLogger:
    def __init__(self, service_name: str = "svg-ai"):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        self._setup_handlers()

    def _setup_handlers(self):
        # Console handler with JSON formatting
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(JSONFormatter())

        # File handler for persistent logs
        file_handler = logging.FileHandler('logs/application.log')
        file_handler.setFormatter(JSONFormatter())

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

    def info(self, message: str, extra: Dict[str, Any] = None):
        self._log('info', message, extra)

    def warning(self, message: str, extra: Dict[str, Any] = None):
        self._log('warning', message, extra)

    def error(self, message: str, extra: Dict[str, Any] = None):
        self._log('error', message, extra)

    def _log(self, level: str, message: str, extra: Dict[str, Any] = None):
        log_data = {
            'timestamp': time.time(),
            'service': self.service_name,
            'level': level,
            'message': message,
            **(extra or {})
        }

        getattr(self.logger, level)(json.dumps(log_data))

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': record.created,
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        if hasattr(record, 'extra'):
            log_data.update(record.extra)

        return json.dumps(log_data)