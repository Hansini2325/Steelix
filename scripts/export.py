from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.deployment.export_onnx import export_to_onnx, validate_onnx_model
from src.deployment.export_tensorrt import export_to_tensorrt, check_tensorrt_available
from src.utils.logger import get_logger

log = get_logger("export")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLOv8 model to deployment format.")
    p.add_argument("--weights", required=True, help="Path to trained .pt weights")
    p.add_argument("--format", required=True, choices=["onnx", "tensorrt", "both"],
                   help="Export format")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument("--half", action="store_true", help="FP16 export (TensorRT/CUDA only)")
    p.add_argument("--output", default=None, help="Override output path")
    p.add_argument("--validate", action="store_true", default=True,
                   help="Validate ONNX model after export")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        log.error(f"Weights not found: {weights}")
        return 1

    success = True

    if args.format in ("onnx", "both"):
        log.info("Exporting to ONNX …")
        onnx_path = export_to_onnx(
            weights_path=weights,
            output_path=Path(args.output) if args.output else None,
            imgsz=args.imgsz,
            opset=args.opset,
            half=args.half,
        )
        if onnx_path:
            log.info(f"✓ ONNX export: {onnx_path}")
            if args.validate:
                valid = validate_onnx_model(onnx_path)
                log.info(f"  ONNX validation: {'PASSED' if valid else 'FAILED'}")
        else:
            log.error("✗ ONNX export failed.")
            success = False

    if args.format in ("tensorrt", "both"):
        log.info("Checking TensorRT availability …")
        trt_status = check_tensorrt_available()
        log.info(f"  TensorRT: {trt_status['reason']}")

        if trt_status["available"]:
            log.info("Exporting to TensorRT …")
            engine_path = export_to_tensorrt(
                weights_path=weights,
                imgsz=args.imgsz,
                half=args.half,
            )
            if engine_path:
                log.info(f"✓ TensorRT export: {engine_path}")
            else:
                log.error("✗ TensorRT export failed.")
                success = False
        else:
            log.warning("TensorRT export: PENDING EXECUTION — TensorRT unavailable on this machine.")
            log.info("  Install NVIDIA TensorRT and retry on a CUDA-compatible GPU.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
