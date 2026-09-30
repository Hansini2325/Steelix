from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.utils.logger import get_logger

log = get_logger(__name__)


def export_to_onnx(
    weights_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    imgsz: int = 640,
    opset: int = 17,
    half: bool = False,
    dynamic: bool = False,
) -> Optional[Path]:

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"PyTorch weights not found at: {weights_path}")

    if output_path is None:
        output_path = weights_path.with_suffix(".onnx")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Exporting {weights_path} -> ONNX (imgsz={imgsz}, opset={opset}, half={half})")

    try:
        from ultralytics import YOLO
        model = YOLO(str(weights_path))

        exported = model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            half=half,
            dynamic=dynamic,
            simplify=True,
        )

        if exported and Path(str(exported)).exists():
            exported_path = Path(str(exported))
            if exported_path.resolve() != output_path.resolve():
                import shutil
                shutil.copy2(exported_path, output_path)
            log.info(f"ONNX export successful -> {output_path}")
            return output_path
        else:
            log.error("ONNX export did not produce a file.")
            return None

    except Exception as exc:
        log.exception(f"ONNX export failed: {exc}")
        return None


def validate_onnx_model(onnx_path: Union[str, Path]) -> bool:
    try:
        import onnx
    except ImportError:
        log.error("ONNX is not installed. Run: pip install onnx")
        return False

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        log.error(f"Cannot validate onnx model - file not found: {onnx_path}")
        return False

    try:
        log.info(f"Validating ONNX model structure: {onnx_path}")
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

        inputs = model.graph.input
        outputs = model.graph.output
        log.info("ONNX Check Passed. Model signature:")
        for i, inp in enumerate(inputs):
            shape = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in inp.type.tensor_type.shape.dim]
            log.info(f"  Input[{i}]: {inp.name} shape={shape}")
        for i, out in enumerate(outputs):
            shape = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in out.type.tensor_type.shape.dim]
            log.info(f"  Output[{i}]: {out.name} shape={shape}")

        return True
    except Exception as exc:
        log.error(f"ONNX validation failed: {exc}")
        return False
