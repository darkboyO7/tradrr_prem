"""Trading Intelligence Platform - Structured Logging Engine.

Provides high-performance, structured JSON and formatted console logging,
with correlation context, latency tracking, and strict credential redaction.
"""

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

# Context variable for request / correlation tracing
correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# Regular expressions for scrubbing sensitive keys and tokens
SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(password|secret|token|api[_-]?key|auth|bearer|passphrase)[\"']?\s*[:=]\s*[\"']?([^\"'\s,]+)[\"']?"),
]


def redact_sensitive_data(text: str) -> str:
    """Scrub sensitive credentials, passwords, and tokens from log messages."""
    for pattern in SENSITIVE_PATTERNS:
        text = pattern.sub(r'\1="[REDACTED]"', text)
    return text


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_sensitive_data(record.getMessage()),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include correlation ID if present
        cid = correlation_id_ctx.get()
        if cid:
            log_obj["correlation_id"] = cid

        # Include extra attributes attached to the record
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            for k, v in record.extra_fields.items():
                if k not in log_obj:
                    log_obj[k] = v

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """Formats log records for local human-readable terminal display."""

    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        time_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cid = correlation_id_ctx.get()
        cid_str = f" [{cid[:8]}]" if cid else ""
        raw_msg = redact_sensitive_data(record.getMessage())

        extra_str = ""
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            extra_str = " " + " ".join(f"{k}={v}" for k, v in record.extra_fields.items())

        msg = (
            f"{color}{time_str} | {record.levelname:<8} | "
            f"{record.name}:{record.lineno}{cid_str} | {raw_msg}{extra_str}{self.RESET}"
        )
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return msg


class StructuredLogger(logging.LoggerAdapter):
    """Logger adapter enabling structured key-value bindings."""

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        extra = kwargs.get("extra", {})
        # Extract extra fields passed as kwargs
        extra_fields = {k: v for k, v in kwargs.items() if k not in ("exc_info", "stack_info", "extra")}
        for k in extra_fields:
            del kwargs[k]
        extra["extra_fields"] = extra_fields
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance for a given module name."""
    logger = logging.getLogger(name)
    return StructuredLogger(logger, {})


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure root logger with chosen formatting and log level."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to prevent duplicate lines
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(ColoredConsoleFormatter())

    root_logger.addHandler(handler)

    # Silence excessively noisy 3rd party loggers
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
