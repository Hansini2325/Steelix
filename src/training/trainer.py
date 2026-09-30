from __future__ import annotations

import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.utils.logger import get_logger
from src.utils.reproducibility import seed_everything
from src.utils.hardware import get_hardware_info, get_best_device

log = get_logger(__name__)

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]
EXPERIMENTS_CSV = Path("results/experiments/experiments.csv")

CSV_FIELDNAMES = [
    "experiment_id", "timestamp", "model", "dataset", "task",
    "epochs", "epochs_completed", "imgsz", "batch",
    "optimizer", "lr0", "weight_decay", "patience", "seed",
    "augmentation", "device", "gpu_name",
    "precision", "recall", "map50", "map50_95",
    "model_size_mb", "best_weights",
    "fps", "latency_ms", "backend",
]


def _load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()


def _append_experiment(record: Dict[str, Any], path: Path = EXPERIMENTS_CSV) -> None:
    _ensure_csv(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writerow(record)
    log.info(f"Experiment record saved → {path}")


def train(
    dataset_yaml: Path,
    model_weights: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 20,
    seed: int = 42,
    lr0: float = 0.001,
    optimizer: str = "AdamW",
    device: str = "",
    workers: int = 4,
    augmentation: bool = True,
    project: str = "results/training",
    name: str = "neu_det_experiment",
    config_path: Optional[Path] = None,
    task: str = "detect",
) -> Optional[Path]:

    if config_path and Path(config_path).exists():
        cfg = _load_config(Path(config_path))
        model_weights = cfg.get("model", model_weights)
        epochs = cfg.get("epochs", epochs)
        imgsz = cfg.get("imgsz", imgsz)
        batch = cfg.get("batch", batch)
        patience = cfg.get("patience", patience)
        seed = cfg.get("seed", seed)
        lr0 = cfg.get("lr0", lr0)
        optimizer = cfg.get("optimizer", optimizer)
        device = cfg.get("device", device)
        workers = cfg.get("workers", workers)
        augmentation = cfg.get("augmentation", augmentation)
        project = cfg.get("project", project)
        name = cfg.get("name", name)

    seed_everything(seed)

    hw = get_hardware_info()
    if not device:
        device = get_best_device()

    log.info("=" * 60)
    log.info(f"  TRAINING START")
    log.info(f"  Model        : {model_weights}")
    log.info(f"  Dataset YAML : {dataset_yaml}")
    log.info(f"  Task         : {task}")
    log.info(f"  Epochs       : {epochs}  |  Batch: {batch}  |  ImgSz: {imgsz}")
    log.info(f"  Device       : {device.upper() if device else 'AUTO'}")
    log.info("=" * 60)

    experiment_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    aug_overrides: Dict[str, Any] = {}
    if not augmentation:
        aug_overrides = {
            "hsv_h": 0, "hsv_s": 0, "hsv_v": 0,
            "degrees": 0, "translate": 0, "scale": 0,
            "shear": 0, "fliplr": 0, "mosaic": 0,
        }

    try:
        from ultralytics import YOLO

        model = YOLO(model_weights, task=task)

        train_args: Dict[str, Any] = dict(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            seed=seed,
            lr0=lr0,
            optimizer=optimizer,
            device=device,
            workers=workers,
            project=project,
            name=name,
            exist_ok=True,
            plots=True,
            verbose=True,
            **aug_overrides,
        )

        results = model.train(**train_args)

        run_dir = Path(project) / name
        best_weights = run_dir / "weights" / "best.pt"
        if not best_weights.exists():
            candidates = list(run_dir.rglob("best.pt"))
            best_weights = candidates[0] if candidates else None

        metrics = results.results_dict if hasattr(results, "results_dict") else {}
        map50 = metrics.get("metrics/mAP50(B)", "PENDING")
        map5095 = metrics.get("metrics/mAP50-95(B)", "PENDING")
        prec = metrics.get("metrics/precision(B)", "PENDING")
        rec = metrics.get("metrics/recall(B)", "PENDING")

        model_size_mb = (
            round(best_weights.stat().st_size / 1e6, 2)
            if best_weights and best_weights.exists() else "N/A"
        )

        log.info(f"Training complete. Best weights → {best_weights}")
        log.info(f"mAP50={map50}  mAP50-95={map5095}  P={prec}  R={rec}")

        record = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "model": model_weights,
            "dataset": str(dataset_yaml),
            "task": task,
            "epochs": epochs,
            "epochs_completed": getattr(results, "epoch", "N/A"),
            "imgsz": imgsz,
            "batch": batch,
            "optimizer": optimizer,
            "lr0": lr0,
            "weight_decay": 0.0005,
            "patience": patience,
            "seed": seed,
            "augmentation": augmentation,
            "device": device or "auto",
            "gpu_name": hw["gpu_names"][0] if hw["gpu_names"] else "N/A",
            "precision": prec,
            "recall": rec,
            "map50": map50,
            "map50_95": map5095,
            "model_size_mb": model_size_mb,
            "best_weights": str(best_weights) if best_weights else "N/A",
            "fps": "PENDING",
            "latency_ms": "PENDING",
            "backend": "PyTorch",
        }
        _append_experiment(record)

        return best_weights

    except ImportError:
        log.error("Ultralytics is not installed. Run: pip install ultralytics")
        return None
    except Exception as exc:
        log.exception(f"Training failed: {exc}")
        return None
