"""Tests for Unified Master CLI Controller."""

import pytest

from scripts.cli import main


def test_cli_help(capsys):
    """Verify CLI prints help and exits with code 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Institutional Trading Intelligence Platform" in captured.out


def test_cli_status(capsys):
    """Verify CLI executes status diagnostics cleanly."""
    res = main(["status"])
    assert res == 0
    captured = capsys.readouterr()
    assert "SYSTEM DIAGNOSTICS" in captured.out
    assert "Operating System" in captured.out


def test_cli_backtest_command(capsys):
    """Verify CLI runs backtest sub-command."""
    res = main(["backtest", "--symbol", "BTC-USD", "--candles", "60"])
    assert res == 0
    captured = capsys.readouterr()
    assert "EVENT-DRIVEN BACKTEST EXECUTION SUMMARY" in captured.out
