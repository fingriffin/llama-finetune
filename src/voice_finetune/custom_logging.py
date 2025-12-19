"""Logging configuration for style-bench."""

import sys

from loguru import logger
from loguru._logger import Logger


def setup_logging(level: str = "INFO", log_file: str | None = None) -> Logger:
    """
    Configure logging for style-bench.

    :param level: Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
    :param log_file: Optional file path to save logs.
    :return: Configured logger instance.
    """
    logger.remove()

    def format_record(record: dict) -> str:
        """
        Format log records with icons based on severity level.

        :param record: Log record.
        :return: Formatted log string.
        """
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
        from pathlib import Path

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level)

    return logger
