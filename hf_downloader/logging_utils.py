from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(log_dir: Path) -> logging.Logger:
    logger = logging.getLogger("hf_downloader")
    if logger.handlers:
        return logger

    log_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(log_dir / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
