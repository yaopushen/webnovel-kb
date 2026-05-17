"""Logging configuration with file rotation and module-level control."""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def setup_logging(
    level: str = "INFO",
    log_dir: str = "",
    log_file: str = "webnovel-kb.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console_level: str = "",
    file_level: str = "",
) -> logging.Logger:
    """Setup logging with console and rotating file handlers.

    Args:
        level: Global log level (DEBUG/INFO/WARNING/ERROR)
        log_dir: Directory for log files. Empty string = no file logging.
        log_file: Log file name.
        max_bytes: Max size per log file before rotation (default 10MB).
        backup_count: Number of rotated backup files to keep.
        console_level: Override console log level (defaults to `level`).
        file_level: Override file log level (defaults to `level`).

    Returns:
        Root logger for webnovel-kb.
    """
    global _initialized
    if _initialized:
        return logging.getLogger("webnovel-kb")

    root_logger = logging.getLogger("webnovel-kb")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_lvl = getattr(logging, (console_level or level).upper(), logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_lvl)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_lvl = getattr(logging, (file_level or level).upper(), logging.INFO)
        file_handler = RotatingFileHandler(
            log_path / log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(file_lvl)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _initialized = True
    return root_logger


def get_logger(module_name: str) -> logging.Logger:
    """Get a hierarchical logger for a module.

    Usage:
        logger = get_logger("core")       -> webnovel-kb.core
        logger = get_logger("search")     -> webnovel-kb.search
        logger = get_logger("api.clients")-> webnovel-kb.api.clients
    """
    return logging.getLogger(f"webnovel-kb.{module_name}")
