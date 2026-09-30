from __future__ import annotations

import platform
import sys
from typing import Any, Dict

from src.utils.logger import get_logger

log = get_logger(__name__)


def get_hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "platform_system": platform.system(),
        "platform_version": platform.version(),
        "platform_machine": platform.machine(),
        "python_version": sys.version,
        "cpu_brand": platform.processor() or "Unknown",
        "cpu_cores_physical": 1,
        "cpu_cores_logical": 1,
        "cpu_freq_mhz": 0.0,
        "ram_total_gb": 0.0,
        "ram_available_gb": 0.0,
        "ram_used_pct": 0.0,
        "torch_available": False,
        "torch_version": "N/A",
        "cuda_available": False,
        "cuda_version": "N/A",
        "cuda_count": 0,
        "gpu_count": 0,
        "gpu_names": [],
        "onnxruntime_available": False,
        "onnxruntime_version": "N/A",
        "onnxruntime_providers": [],
        "tensorrt_available": False,
        "device": "cpu",
    }

    try:
        import psutil

        phys = psutil.cpu_count(logical=False)
        logi = psutil.cpu_count(logical=True)
        info["cpu_cores_physical"] = phys if phys else 1
        info["cpu_cores_logical"] = logi if logi else 1

        freq = psutil.cpu_freq()
        if freq:
            info["cpu_freq_mhz"] = round(freq.current, 1)

        try:
            import cpuinfo
            cpu_data = cpuinfo.get_cpu_info()
            info["cpu_brand"] = cpu_data.get("brand_raw", info["cpu_brand"])
        except ImportError:
            pass

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1024 ** 3, 2)
        info["ram_available_gb"] = round(vm.available / 1024 ** 3, 2)
        info["ram_used_pct"] = round(vm.percent, 1)

    except ImportError:
        log.warning("psutil not installed — CPU/RAM details unavailable.")

    try:
        import torch

        info["torch_available"] = True
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "N/A"
            n = torch.cuda.device_count()
            info["cuda_count"] = n
            info["gpu_count"] = n
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(n)]
            info["device"] = "cuda"
        else:
            info["device"] = "cpu"

    except ImportError:
        log.warning("PyTorch not installed.")

    try:
        import onnxruntime as ort

        info["onnxruntime_available"] = True
        info["onnxruntime_version"] = ort.__version__
        info["onnxruntime_providers"] = ort.get_available_providers()

    except ImportError:
        log.info("ONNX Runtime not installed.")

    try:
        import tensorrt as trt

        info["tensorrt_available"] = True
        info["tensorrt_version"] = trt.__version__

    except ImportError:
        info["tensorrt_available"] = False

    return info


def print_hardware_info() -> None:
    info = get_hardware_info()
    sep = "─" * 60
    print(f"\n{sep}")
    print("  System Environment")
    print(sep)
    print(f"  OS           : {info['os']} {info['os_version']}")
    print(f"  Architecture : {info['architecture']}")
    print(f"  Python       : {info['python_version'].split()[0]}")
    print(f"  CPU          : {info['cpu_brand']}")
    print(f"  Cores        : {info['cpu_cores_physical']} phys / {info['cpu_cores_logical']} logical")
    print(f"  RAM          : {info['ram_total_gb']} GB total, {info['ram_available_gb']} GB free")
    print(f"  PyTorch      : {info['torch_version']}")
    cuda_str = (
        "Available (v" + info["cuda_version"] + ")"
        if info["cuda_available"]
        else "Not available (CPU only)"
    )
    print(f"  CUDA         : {cuda_str}")
    for i, gpu in enumerate(info["gpu_names"]):
        print(f"  GPU [{i}]      : {gpu}")
    ort_str = info["onnxruntime_version"] if info["onnxruntime_available"] else "Not installed"
    print(f"  ONNX RT      : {ort_str}")
    if info["onnxruntime_available"]:
        print(f"  ORT Providers: {', '.join(info['onnxruntime_providers'])}")
    trt_ver = info.get("tensorrt_version", "?")
    trt_str = f"Available (v{trt_ver})" if info["tensorrt_available"] else "Not available"
    print(f"  TensorRT     : {trt_str}")
    print(f"  Active Device: {info['device'].upper()}")
    print(f"{sep}\n")


def get_best_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
