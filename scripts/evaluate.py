from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import Evaluator
from src.utils.logger import get_logger

log = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 model.")
    p.add_argument("--weights", required=True, help="Path to .pt weights file")
    p.add_argument("--dataset", default="data/processed/neu_det.yaml", help="YOLO dataset YAML")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    p.add_argument("--device", default="", help="Device string")
    p.add_argument("--split", default="test", choices=["val", "test"], help="Split to evaluate")
    p.add_argument("--output", default="results/experiments", help="Output directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        log.error(f"Weights not found: {weights}")
        return 1

    dataset_yaml = PROJECT_ROOT / args.dataset
    if not dataset_yaml.exists():
        log.error(f"Dataset YAML not found: {dataset_yaml}")
        return 1

    evaluator = Evaluator(
        weights_path=weights,
        dataset_yaml=dataset_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        output_dir=PROJECT_ROOT / args.output,
        split=args.split,
    )

    metrics = evaluator.evaluate()

    if metrics:
        log.info("\n── EVALUATION RESULTS ──────────────────────────")
        for k, v in metrics.items():
            log.info(f"  {k:<25}: {v}")
        log.info("────────────────────────────────────────────────")
        return 0
    else:
        log.error("Evaluation failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
