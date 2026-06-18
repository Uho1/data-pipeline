from __future__ import annotations

import logging
from pathlib import Path


def setup_wrds_logger(log_dir: Path) -> logging.Logger:
    """Create a dedicated WRDS ingestion logger with file and stderr handlers."""

    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("market_data.wrds")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_dir / "wrds_ingest.log")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
