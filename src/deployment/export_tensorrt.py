from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.utils.logger import get_logger

log = get_logger(__name__)


def check_tensorrt_available() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "available": False,
        "cuda_available": False,
        "tensorrt_version": None,
        "reason": "",
    }

    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        info["reason"] = "PyTorch not installed."
        return info

    if not info["cuda_available"]:
        info["reason"] = "CUDA not available on this machine."
        return info

    try:
        import tensorrt as trt
        info["tensorrt_version"] = trt.__version__
        info["available"] = True
        info["reason"] = f"TensorRT {trt.__version__} available."
    except ImportError:
        info["reason"] = (
            "TensorRT not installed. pip install tensorrt tensorrt-cu12"
        )

    return info


def export_to_tensorrt(
    weights_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    imgsz: int = 640,
    half: bool = False,
    workspace_gb: int = 4,
) -> Optional[Path]:

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    trt_status = check_tensorrt_available()
    if not trt_status["available"]:
        log.warning(f"TensorRT UNAVAILABLE: {trt_status['reason']}")
        log.info("TensorRT export: PENDING EXECUTION — requires compatible NVIDIA hardware.")
        return None

    if output_path is None:
        suffix = ".engine"
        output_path = weights_path.with_suffix(suffix)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Exporting {weights_path} → TensorRT  (half={half}, workspace={workspace_gb}GB)")

    try:
        from ultralytics import YOLO
        model = YOLO(str(weights_path))
        exported = model.export(
            format="engine",
            imgsz=imgsz,
            half=half,
            workspace=workspace_gb,
            device=0,
        )
        if exported and Path(str(exported)).exists():
            exported_path = Path(str(exported))
            if exported_path.resolve() != output_path.resolve():
                import shutil
                shutil.copy2(exported_path, output_path)
            log.info(f"TensorRT export successful → {output_path}")
            return output_path
        else:
            log.error("TensorRT export produced no output.")
            return None
    except Exception as exc:
        log.exception(f"TensorRT export failed: {exc}")
        return None
