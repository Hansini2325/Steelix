from __future__ import annotations

import os
from pathlib import Path
import tempfile
import yaml

from src.utils.logger import get_logger
from src.utils.config import load_yaml
from src.utils.hardware import get_hardware_info


def test_logger():
    log = get_logger("test_log")
    assert log.name == "test_log"


def test_load_yaml():
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as tmp:
        yaml.dump({"test_key": "test_val", "num": 42}, tmp)
        tmp_name = tmp.name
        
    try:
        data = load_yaml(tmp_name)
        assert data["test_key"] == "test_val"
        assert data["num"] == 42
    finally:
        os.remove(tmp_name)


def test_hardware_info():
    hw = get_hardware_info()
    assert "platform_system" in hw
    assert "python_version" in hw
    assert "cpu_cores_logical" in hw
    assert "ram_total_gb" in hw
