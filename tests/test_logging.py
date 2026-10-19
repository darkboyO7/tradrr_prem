"""Tests for structured logging and credential redaction."""

import json
import logging

from core.logging import StructuredJsonFormatter, redact_sensitive_data


def test_redact_sensitive_data():
    raw_text = 'User login failed with password="SuperSecret123" and api_key: "abc-987-xyz"'
    redacted = redact_sensitive_data(raw_text)
    assert "SuperSecret123" not in redacted
    assert "abc-987-xyz" not in redacted
    assert "[REDACTED]" in redacted


def test_structured_json_formatter():
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_path.py",
        lineno=42,
        msg="Processing trade batch",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    parsed = json.loads(formatted)
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["message"] == "Processing trade batch"
    assert "timestamp" in parsed
