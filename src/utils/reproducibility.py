import os
import random
from typing import Optional

from src.utils.logger import get_logger

log = get_logger(__name__)


def seed_everything(seed: int = 42) -> None:
    log.info(f"Setting global random seed to {seed}")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        log.warning("NumPy not installed — skipping numpy seed.")

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        log.info("CUDA seeds set and cuDNN deterministic mode enabled.")
    except ImportError:
        log.warning("PyTorch not installed — skipping torch seed.")


def get_seed_from_config(config: dict, key: str = "seed", default: int = 42) -> int:
    seed = config.get(key, default)
    if not isinstance(seed, int) or seed < 0:
        log.warning(f"Invalid seed value '{seed}'. Using default {default}.")
        return default
    return seed
