from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(app_data_dir: str) -> logging.Logger:
    """
    Configure app logging (console + file).

    - Logs to console
    - Logs to APP_DATA_DIR/logs/app.log
    - Safe to call multiple times (no duplicate handlers)
    """
    logger = logging.getLogger("local_deep_research")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_dir = Path(app_data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"

    fmt = logging.Formatter(
        fmt="%(asctime)sZ %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    def _has_handler(cls: type[logging.Handler], *, filename: str | None = None) -> bool:
        for h in logger.handlers:
            if not isinstance(h, cls):
                continue
            if filename is None:
                return True
            if isinstance(h, logging.FileHandler) and str(getattr(h, "baseFilename", "")) == filename:
                return True
        return False

    if not _has_handler(logging.StreamHandler):
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if not _has_handler(logging.FileHandler, filename=str(log_path)):
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
