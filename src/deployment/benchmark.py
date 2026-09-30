from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
BENCHMARK_DIR = PROJECT_ROOT / "results" / "benchmarks"


def _warmup_and_measure(
    run_fn,
    n_warmup: int = 5,
    n_iter: int = 50,
) -> Dict[str, float]:

    for _ in range(n_warmup):
        run_fn()

    latencies: List[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        run_fn()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    arr = np.array(latencies)
    mean_ms = float(np.mean(arr))
    return {
        "mean_ms": round(mean_ms, 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "fps": round(1000.0 / mean_ms if mean_ms > 0 else 0.0, 2),
        "n_iter": n_iter,
    }


def benchmark_pytorch(
    weights_path: Union[str, Path],
    imgsz: int = 640,
    n_warmup: int = 5,
    n_iter: int = 50,
    device: str = "",
) -> Dict[str, Any]:

    weights_path = Path(weights_path)
    if not weights_path.exists():
        log.warning(f"PyTorch benchmark: weights not found at {weights_path}. Marked PENDING.")
        return {"backend": "PyTorch", "status": "PENDING — weights not found"}

    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        return {"backend": "PyTorch", "status": "PENDING — ultralytics not installed"}

    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    log.info(f"Benchmarking PyTorch backend ({device}) …")
    model = YOLO(str(weights_path))

    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    def run_fn():
        model(dummy, imgsz=imgsz, device=device, verbose=False)

    stats = _warmup_and_measure(run_fn, n_warmup=n_warmup, n_iter=n_iter)

    model_size_mb = round(weights_path.stat().st_size / 1e6, 2)

    result = {
        "backend": "PyTorch",
        "device": device,
        "imgsz": imgsz,
        "model_size_mb": model_size_mb,
        "status": "MEASURED",
        **stats,
    }
    log.info(f"PyTorch benchmark: mean={stats['mean_ms']:.1f}ms  FPS={stats['fps']:.1f}")
    return result


def benchmark_onnx(
    onnx_path: Union[str, Path],
    imgsz: int = 640,
    n_warmup: int = 5,
    n_iter: int = 50,
) -> Dict[str, Any]:

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        log.warning(f"ONNX benchmark: model not found at {onnx_path}. Marked PENDING.")
        return {"backend": "ONNX Runtime", "status": "PENDING — model not found"}

    try:
        import onnxruntime as ort
    except ImportError:
        return {"backend": "ONNX Runtime", "status": "PENDING — onnxruntime not installed"}

    log.info("Benchmarking ONNX Runtime …")
    providers = ort.get_available_providers()
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name

    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    def run_fn():
        session.run(None, {input_name: dummy})

    stats = _warmup_and_measure(run_fn, n_warmup=n_warmup, n_iter=n_iter)

    model_size_mb = round(onnx_path.stat().st_size / 1e6, 2)
    result = {
        "backend": "ONNX Runtime",
        "device": providers,
        "model_size_mb": model_size_mb,
        "imgsz": imgsz,
        "status": "MEASURED",
        **stats,
    }
    log.info(f"ONNX benchmark: mean={stats['mean_ms']:.1f}ms  FPS={stats['fps']:.1f}")
    return result


def benchmark_tensorrt(
    engine_path: Union[str, Path],
    imgsz: int = 640,
    n_warmup: int = 5,
    n_iter: int = 50,
    half: bool = False,
) -> Dict[str, Any]:

    from src.deployment.export_tensorrt import check_tensorrt_available
    trt_status = check_tensorrt_available()
    if not trt_status["available"]:
        log.info(f"TensorRT benchmark: PENDING EXECUTION — {trt_status['reason']}")
        return {
            "backend": f"TensorRT {'FP16' if half else 'FP32'}",
            "status": f"PENDING EXECUTION — {trt_status['reason']}",
        }

    engine_path = Path(engine_path)
    if not engine_path.exists():
        return {
            "backend": f"TensorRT {'FP16' if half else 'FP32'}",
            "status": "PENDING — engine file not found",
        }

    log.info("Benchmarking TensorRT …")
    try:
        from ultralytics import YOLO
        model = YOLO(str(engine_path))
        dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        def run_fn():
            model(dummy, imgsz=imgsz, device=0, verbose=False)

        stats = _warmup_and_measure(run_fn, n_warmup=n_warmup, n_iter=n_iter)
        result = {
            "backend": f"TensorRT {'FP16' if half else 'FP32'}",
            "imgsz": imgsz,
            "device": half,
            "status": "MEASURED",
            **stats,
        }
        log.info(f"TensorRT benchmark: mean={stats['mean_ms']:.1f}ms  FPS={stats['fps']:.1f}")
        return result
    except Exception as exc:
        log.exception(f"TensorRT benchmark failed: {exc}")
        return {"backend": "TensorRT", "status": f"ERROR: {exc}"}


def run_full_benchmark(
    weights_path: Optional[Path] = None,
    onnx_path: Optional[Path] = None,
    engine_path: Optional[Path] = None,
    imgsz: int = 640,
    n_warmup: int = 5,
    n_iter: int = 50,
    output_json: Optional[Path] = None,
    output_csv: Optional[Path] = None,
) -> Dict[str, Any]:

    from src.utils.hardware import get_hardware_info
    hw = get_hardware_info()

    results: Dict[str, Any] = {
        "hardware": hw,
        "config": {"imgsz": imgsz, "n_warmup": n_warmup, "n_iter": n_iter},
        "target_fps": ">= 35 FPS on suitable compatible hardware (TARGET ONLY — not guaranteed)",
        "benchmarks": [],
    }

    if weights_path:
        pt_result = benchmark_pytorch(weights_path, imgsz, n_warmup, n_iter)
        results["benchmarks"].append(pt_result)

    if onnx_path:
        onnx_result = benchmark_onnx(onnx_path, imgsz, n_warmup, n_iter)
        results["benchmarks"].append(onnx_result)

    if engine_path:
        trt_result = benchmark_tensorrt(engine_path, imgsz, n_warmup, n_iter)
        results["benchmarks"].append(trt_result)

    if not results["benchmarks"]:
        trt_pending = benchmark_tensorrt(Path("dummy.engine"), imgsz)
        results["benchmarks"].append(trt_pending)

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    if output_json is None:
        output_json = BENCHMARK_DIR / "benchmark_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Benchmark results → {output_json}")

    if output_csv is None:
        output_csv = BENCHMARK_DIR / "benchmark_results.csv"
    _write_benchmark_csv(results["benchmarks"], output_csv)

    return results


def _write_benchmark_csv(benchmarks: List[Dict[str, Any]], path: Path) -> None:
    if not benchmarks:
        return
    fields = [
        "backend", "device", "imgsz", "status", "mean_ms", "median_ms",
        "p95_ms", "fps", "model_size_mb", "n_iter",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(benchmarks)
    log.info(f"Benchmark CSV → {path}")
