import logging
import json
import os
from logging.handlers import RotatingFileHandler

class JsonFormatter(logging.Formatter):
    """Format log records as JSON with detailed context."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - simple serializer
        """Return the given record serialized as a JSON string."""
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt or "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "module": record.module,
            "funcName": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Include any extra attributes passed via ``logger.info(..., extra={})``
        standard_keys = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
            'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
            'processName', 'process'
        }
        for key, value in record.__dict__.items():
            if key not in standard_keys and key not in log_entry:
                log_entry[key] = value

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_json_file_logger(log_file: str, level: int = logging.INFO) -> RotatingFileHandler:
    """Attach a RotatingFileHandler with :class:`JsonFormatter` to the root logger.

    Parameters
    ----------
    log_file:
        Path to the JSON log file.
    level:
        Logging level for the handler and root logger (default: ``logging.INFO``).

    Returns
    -------
    RotatingFileHandler
        The handler that was added. Call ``root_logger.removeHandler`` on it when done.
    """

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10_485_760,
        backupCount=5,
        encoding="utf8",
    )
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    return handler
