from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.deployment.benchmark import run_full_benchmark
from src.utils.hardware import get_hardware_info
from src.utils.logger import get_logger

log = get_logger("benchmark")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark YOLOv8 inference backends.")
    p.add_argument("--weights", default=None, help="PyTorch .pt weights path")
    p.add_argument("--onnx", default=None, help="ONNX model path")
    p.add_argument("--engine", default=None, help="TensorRT .engine path")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p.add_argument("--warmup", type=int, default=5, help="Warm-up iterations")
    p.add_argument("--iter", type=int, default=50, help="Measurement iterations")
    p.add_argument("--output", default="results/benchmarks", help="Output directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not any([args.weights, args.onnx, args.engine]):
        log.error("Provide at least one of: --weights, --onnx, --engine")
        return 1

    log.info("=" * 60)
    log.info("  SteelVision AI — Backend Benchmarking")
    log.info("=" * 60)

    hw = get_hardware_info()
    log.info(f"  OS      : {hw.get('os')}")
    log.info(f"  Python  : {hw.get('python')}")
    log.info(f"  PyTorch : {hw.get('pytorch_version')}")
    log.info(f"  CUDA    : {hw.get('cuda_available')}  {hw.get('cuda_version', '')}")
    log.info(f"  GPU     : {hw.get('gpu_names', ['N/A'])[0] if hw.get('gpu_names') else 'N/A'}")

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_full_benchmark(
        weights_path=Path(args.weights) if args.weights else None,
        onnx_path=Path(args.onnx) if args.onnx else None,
        engine_path=Path(args.engine) if args.engine else None,
        imgsz=args.imgsz,
        n_warmup=args.warmup,
        n_iter=args.iter,
        output_json=output_dir / "benchmark_results.json",
        output_csv=output_dir / "benchmark_results.csv",
    )

    log.info("\n── BENCHMARK RESULTS ──────────────────────────────")
    for bench in results.get("benchmarks", []):
        backend = bench.get("backend", "?")
        status = bench.get("status", "?")
        if status == "MEASURED":
            log.info(
                f"  {backend:<25}: "
                f"{bench.get('mean_ms', 0):.1f}ms / img => "
                f"{bench.get('fps', 0):.1f} FPS"
            )
        else:
            log.info(f"  {backend:<25}: {status}")
    log.info("────────────────────────────────────────────────────")
    log.info(f"NOTE: FPS target >= 35 is a GOAL only — not a guaranteed specification.")
    log.info(f"Results saved → {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
