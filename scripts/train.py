from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import train
from src.utils.logger import get_logger

log = get_logger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 on NEU-DET dataset.")
    p.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weights (e.g. yolov8n.pt, yolov8s.pt)")
    p.add_argument("--dataset", default="data/processed/neu_det.yaml", help="YOLO dataset YAML")
    p.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    p.add_argument("--optimizer", default="AdamW", help="Optimizer (AdamW | SGD | Adam)")
    p.add_argument("--device", default="", help="Device ('' = auto, 'cpu', 'cuda', '0')")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    p.add_argument("--project", default="results/training", help="Output project directory")
    p.add_argument("--name", default="neu_det", help="Experiment name")
    p.add_argument("--task", default="detect", choices=["detect", "classify"],
                   help="Task type: detect or classify")
    p.add_argument("--config", default=None, help="Override with train.yaml config file")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    dataset_yaml = PROJECT_ROOT / args.dataset
    if not dataset_yaml.exists():
        log.error(f"Dataset YAML not found: {dataset_yaml}\nPlease run scripts/prepare_dataset.py first.")
        return 1

    best_weights = train(
        dataset_yaml=dataset_yaml,
        model_weights=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        lr0=args.lr0,
        optimizer=args.optimizer,
        device=args.device,
        workers=args.workers,
        augmentation=not args.no_augment,
        project=args.project,
        name=args.name,
        config_path=Path(args.config) if args.config else None,
        task=args.task,
    )

    if best_weights:
        log.info(f"\n✓ Training complete. Best weights: {best_weights}")
        log.info(f"\nNext steps:")
        log.info(f"  python scripts/evaluate.py --weights {best_weights}")
        log.info(f"  python scripts/predict.py  --weights {best_weights} --source image.jpg")
        log.info(f"  python scripts/export.py   --weights {best_weights} --format onnx")
        return 0
    else:
        log.error("Training failed or no weights were produced.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
