from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import UltralyticsPredictor
from src.utils.logger import get_logger

log = get_logger("predict")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run YOLOv8 inference on images.")
    p.add_argument("--weights", required=True, help="Path to .pt weights")
    p.add_argument("--source", required=True, help="Image path or directory")
    p.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", default="", help="Device string")
    p.add_argument("--save", action="store_true", help="Save annotated images")
    p.add_argument("--output", default="results/predictions", help="Output directory")
    p.add_argument("--json", action="store_true", help="Also save detections as JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        log.error(f"Weights not found: {weights}\nTrain first: python scripts/train.py")
        return 1

    source = Path(args.source)
    if not source.exists():
        log.error(f"Source not found: {source}")
        return 1

    if source.is_dir():
        images = sorted([f for f in source.rglob("*") if f.suffix.lower() in IMAGE_EXTS])
    else:
        images = [source]

    if not images:
        log.error(f"No images found in: {source}")
        return 1

    log.info(f"Found {len(images)} image(s). Running inference …")

    predictor = UltralyticsPredictor(
        weights_path=weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for img_path in images:
        save_path = (output_dir / img_path.name) if args.save else None
        try:
            result = predictor.predict_image(img_path, save_path=save_path)
            n = result["num_detections"]
            lat = result["latency_ms"]
            status = result["inspection"]
            log.info(f"  {img_path.name}: {n} detection(s)  {lat:.1f}ms  [{status}]")
            all_results.append(result)
        except Exception as exc:
            log.error(f"  {img_path.name}: FAILED — {exc}")

    if args.json:
        json_out = output_dir / "predictions.json"
        clean = [{k: v for k, v in r.items() if k != "annotated_image"} for r in all_results]
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, default=str)
        log.info(f"JSON results → {json_out}")

    log.info(f"\n✓ Inference complete. Processed {len(all_results)} image(s).")
    if args.save:
        log.info(f"Annotated images → {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
