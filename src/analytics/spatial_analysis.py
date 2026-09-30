from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

GRID_LABELS = [
    "Top-left", "Top-center", "Top-right",
    "Center-left", "Center", "Center-right",
    "Bottom-left", "Bottom-center", "Bottom-right",
]


def get_spatial_region(cx_norm: float, cy_norm: float) -> str:
    col = min(int(cx_norm * 3), 2)
    row = min(int(cy_norm * 3), 2)
    return GRID_LABELS[row * 3 + col]


class SpatialAnalyzer:

    def __init__(self) -> None:
        self._coords: List[Tuple[float, float]] = []
        self._class_coords: Dict[str, List[Tuple[float, float]]] = {}
        self._region_counts: Counter = Counter()
        self._class_region_counts: Dict[str, Counter] = {}

    def add_detection(
        self,
        cx_norm: float,
        cy_norm: float,
        class_name: Optional[str] = None,
    ) -> str:
        region = get_spatial_region(cx_norm, cy_norm)
        self._coords.append((cx_norm, cy_norm))
        self._region_counts[region] += 1

        if class_name:
            if class_name not in self._class_coords:
                self._class_coords[class_name] = []
                self._class_region_counts[class_name] = Counter()
            self._class_coords[class_name].append((cx_norm, cy_norm))
            self._class_region_counts[class_name][region] += 1

        return region

    def region_distribution(self) -> Dict[str, int]:
        return {region: self._region_counts.get(region, 0) for region in GRID_LABELS}

    def most_affected_region(self) -> Optional[str]:
        if not self._region_counts:
            return None
        return self._region_counts.most_common(1)[0][0]

    def class_region_distribution(self, class_name: str) -> Dict[str, int]:
        ctr = self._class_region_counts.get(class_name, Counter())
        return {region: ctr.get(region, 0) for region in GRID_LABELS}

    def spatial_concentration_score(self) -> float:
        total = sum(self._region_counts.values())
        if total == 0:
            return 0.0
        top_count = self._region_counts.most_common(1)[0][1]
        return top_count / total

    def get_grid_matrix(self) -> np.ndarray:
        matrix = np.zeros((3, 3), dtype=int)
        for region, count in self._region_counts.items():
            idx = GRID_LABELS.index(region)
            row, col = divmod(idx, 3)
            matrix[row, col] = count
        return matrix

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_detections": len(self._coords),
            "region_distribution": self.region_distribution(),
            "most_affected_region": self.most_affected_region(),
            "spatial_concentration_score": round(self.spatial_concentration_score(), 3),
            "grid_matrix": self.get_grid_matrix().tolist(),
        }
