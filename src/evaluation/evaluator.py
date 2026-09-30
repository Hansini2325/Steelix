from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.utils.logger import get_logger

log = get_logger(__name__)


class Evaluator:

    def __init__(
        self,
        weights_path: Union[str, Path],
        dataset_yaml: Union[str, Path],
        imgsz: int = 640,
        batch: int = 16,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "",
        output_dir: Union[str, Path] = "results/evaluation",
        split: str = "test",
    ) -> None:

        self.weights_path = Path(weights_path)
        self.dataset_yaml = Path(dataset_yaml)
        self.imgsz = imgsz
        self.batch = batch
        self.conf = conf
        self.iou = iou
        self.device = device
        self.output_dir = Path(output_dir)
        self.split = split

    def evaluate(self) -> Optional[Dict[str, Any]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Evaluating {self.weights_path} on {self.dataset_yaml} (split={self.split})")

        try:
            from ultralytics import YOLO
        except ImportError:
            log.error("ultralytics not installed.")
            return None

        if not self.weights_path.exists():
            log.error(f"Weights not found: {self.weights_path}")
            return None
        if not self.dataset_yaml.exists():
            log.error(f"Dataset YAML not found: {self.dataset_yaml}")
            return None

        try:
            model = YOLO(str(self.weights_path))
            results = model.val(
                data=str(self.dataset_yaml),
                imgsz=self.imgsz,
                batch=self.batch,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                split=self.split,
                project=str(self.output_dir),
                name="eval_run",
                exist_ok=True,
                verbose=True,
                plots=True,
            )

            metrics = results.results_dict if hasattr(results, "results_dict") else {}
            if not metrics:
                log.warning("No metrics dictionary found in results object.")

            map50 = metrics.get("metrics/mAP50(B)", 0.0)
            map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
            p = metrics.get("metrics/precision(B)", 0.0)
            r = metrics.get("metrics/recall(B)", 0.0)

            clean_metrics = {
                "mAP50": round(map50, 4),
                "mAP50-95": round(map50_95, 4),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "split": self.split,
                "weights": str(self.weights_path.name),
                "classes": {},
            }

            if hasattr(results, "box") and hasattr(results.box, "map50"):
                try:
                    maps = results.box.map50
                    names = model.names
                    if len(maps) == len(names):
                        for i, name in names.items():
                            clean_metrics["classes"][name] = round(maps[i], 4)
                except Exception:
                    pass

            out_json = self.output_dir / "eval_run" / "metrics.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w") as f:
                json.dump(clean_metrics, f, indent=2)

            log.info(f"Evaluation metrics saved -> {out_json}")
            return clean_metrics

        except Exception as exc:
            log.exception(f"Evaluation failed: {exc}")
            return None
