from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from src.utils.logger import get_logger

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


def get_project_root() -> Path:
    return PROJECT_ROOT


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    return data


def load_dataset_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "dataset.yaml"
    return load_yaml(config_path)


def load_train_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "train.yaml"
    return load_yaml(config_path)


def load_inference_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "inference.yaml"
    return load_yaml(config_path)


def load_tracking_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "tracking.yaml"
    return load_yaml(config_path)


def load_inspection_rules(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "inspection_rules.yaml"
    return load_yaml(config_path)


def resolve_path(path: Union[str, Path]) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
    log.debug(f"Config saved → {path}")
