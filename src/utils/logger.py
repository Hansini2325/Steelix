import logging
import sys
from pathlib import Path
from typing import Optional

_FMT = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def setup_root_logger(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    logging.basicConfig(
        level=level,
        format=_FMT,
        datefmt=_DATE_FMT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if log_file is not None:
        root = logging.getLogger()
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
        root.addHandler(fh)
