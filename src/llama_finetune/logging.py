"""Logging configuration."""

from loguru import logger
import sys
from pathlib import Path

def setup_logging(level: str = "INFO", log_file: str = None):
    """Configure logging for style-bench."""
    logger.remove()

    def format_record(record):
        """Some added pazazz to the log output"""
        level_icons = {
            "DEBUG": "🔍",
            "INFO": "📝",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
        }
        icon = level_icons.get(record["level"].name, "📝")
        return f"{icon} <level>{record['message']}</level>\n"

    logger.add(sys.stderr, level=level, format=format_record, colorize=True)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level)

    return logger