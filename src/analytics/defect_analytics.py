from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]

DISPLAY_NAMES = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted Surface",
    "rolled_in_scale": "Rolled-in Scale",
    "scratches": "Scratches",
}

GRID_LABELS = [
    "Top-left",    "Top-center",    "Top-right",
    "Center-left", "Center",        "Center-right",
    "Bottom-left", "Bottom-center", "Bottom-right",
]


class DefectAnalytics:

    def __init__(
        self,
        class_names: List[str] = CLASS_NAMES,
        low_conf_threshold: float = 0.35,
    ) -> None:
        self.class_names = class_names
        self.low_conf_threshold = low_conf_threshold
        self._detections: List[Dict[str, Any]] = []
        self._class_counts: Counter = Counter()
        self._region_counts: Counter = Counter()
        self._region_class_counts: Dict[str, Counter] = defaultdict(Counter)
        self._confidences: List[float] = []
        self._class_confidences: Dict[str, List[float]] = defaultdict(list)
        self._bbox_areas: List[float] = []
        self._frame_percentages: List[float] = []
        self._low_conf_count: int = 0

    def add_detection(self, det: Dict[str, Any]) -> None:
        self._detections.append(det)
        cls_name = det.get("class_name", "unknown")
        region = det.get("region", "Unknown")
        conf = det.get("confidence", 0.0)

        self._class_counts[cls_name] += 1
        self._region_counts[region] += 1
        self._region_class_counts[region][cls_name] += 1
        self._confidences.append(conf)
        self._class_confidences[cls_name].append(conf)

        if conf < self.low_conf_threshold:
            self._low_conf_count += 1

        area = det.get("bbox_area_px2", 0.0)
        pct = det.get("frame_percentage", 0.0)
        if area > 0:
            self._bbox_areas.append(area)
        if pct > 0:
            self._frame_percentages.append(pct)

    def add_detections(self, dets: List[Dict[str, Any]]) -> None:
        for det in dets:
            self.add_detection(det)

    def class_distribution(self) -> Dict[str, int]:
        result = {cls: 0 for cls in self.class_names}
        result.update(dict(self._class_counts))
        return result

    def class_distribution_display(self) -> Dict[str, int]:
        dist = self.class_distribution()
        return {DISPLAY_NAMES.get(k, k): v for k, v in dist.items()}

    def avg_confidence_per_class(self) -> Dict[str, float]:
        result = {}
        for cls in self.class_names:
            confs = self._class_confidences.get(cls, [])
            result[cls] = float(np.mean(confs)) if confs else 0.0
        return result

    def avg_confidence_overall(self) -> float:
        if not self._confidences:
            return 0.0
        return float(np.mean(self._confidences))

    def region_distribution(self) -> Dict[str, int]:
        result = {region: 0 for region in GRID_LABELS}
        result.update(dict(self._region_counts))
        return result

    def region_class_breakdown(self) -> Dict[str, Dict[str, int]]:
        result: Dict[str, Dict[str, int]] = {}
        for region in GRID_LABELS:
            cls_ctr = self._region_class_counts.get(region, Counter())
            result[region] = {cls: cls_ctr.get(cls, 0) for cls in self.class_names}
        return result

    def most_affected_region(self) -> Optional[str]:
        if not self._region_counts:
            return None
        return self._region_counts.most_common(1)[0][0]

    def bbox_area_stats(self) -> Dict[str, float]:
        if not self._bbox_areas:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(self._bbox_areas)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def frame_pct_stats(self) -> Dict[str, float]:
        if not self._frame_percentages:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(self._frame_percentages)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def get_summary(self) -> Dict[str, Any]:
        total = len(self._detections)
        return {
            "total_detections": total,
            "class_distribution": self.class_distribution(),
            "class_distribution_display": self.class_distribution_display(),
            "avg_confidence_overall": round(self.avg_confidence_overall(), 3),
            "avg_confidence_per_class": {
                k: round(v, 3)
                for k, v in self.avg_confidence_per_class().items()
            },
            "low_confidence_count": self._low_conf_count,
            "low_confidence_threshold": self.low_conf_threshold,
            "most_frequent_class": (
                self._class_counts.most_common(1)[0][0]
                if self._class_counts else None
            ),
            "most_affected_region": self.most_affected_region(),
            "region_distribution": self.region_distribution(),
            "region_class_breakdown": self.region_class_breakdown(),
            "bbox_area_stats_px2": self.bbox_area_stats(),
            "frame_percentage_stats": self.frame_pct_stats(),
        }

    def reset(self) -> None:
        self._detections.clear()
        self._class_counts.clear()
        self._region_counts.clear()
        self._region_class_counts.clear()
        self._confidences.clear()
        self._class_confidences.clear()
        self._bbox_areas.clear()
        self._frame_percentages.clear()
        self._low_conf_count = 0
