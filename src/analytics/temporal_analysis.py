from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


class TemporalAnalyzer:

    def __init__(self, bin_width_seconds: float = 5.0) -> None:
        self.bin_width = bin_width_seconds
        self._events: List[Dict[str, Any]] = []
        self._start_ts: Optional[float] = None

    def record_detection(
        self,
        timestamp: float,
        class_name: str,
        track_id: Optional[int] = None,
    ) -> None:
        if self._start_ts is None:
            self._start_ts = timestamp
        self._events.append({
            "timestamp": timestamp,
            "class_name": class_name,
            "track_id": track_id,
            "relative_ts": timestamp - self._start_ts,
        })

    def detection_rate_over_time(self) -> Dict[str, Any]:
        if not self._events:
            return {"bin_edges": [], "counts_per_bin": [], "labels": []}

        start = self._events[0]["relative_ts"]
        end = self._events[-1]["relative_ts"]
        duration = max(end - start, self.bin_width)

        n_bins = max(int(duration / self.bin_width) + 1, 1)
        counts = [0] * n_bins

        for evt in self._events:
            rel = evt["relative_ts"]
            bin_idx = min(int(rel / self.bin_width), n_bins - 1)
            counts[bin_idx] += 1

        bin_edges = [i * self.bin_width for i in range(n_bins)]
        labels = [f"{e:.0f}s" for e in bin_edges]

        return {
            "bin_edges": bin_edges,
            "counts_per_bin": counts,
            "labels": labels,
        }

    def class_rate_over_time(self, class_name: str) -> Dict[str, Any]:
        class_events = [e for e in self._events if e["class_name"] == class_name]
        if not class_events or self._start_ts is None:
            return {"bin_edges": [], "counts_per_bin": [], "labels": []}

        end = self._events[-1]["relative_ts"]
        duration = max(end, self.bin_width)
        n_bins = max(int(duration / self.bin_width) + 1, 1)
        counts = [0] * n_bins

        for evt in class_events:
            rel = evt["relative_ts"]
            bin_idx = min(int(rel / self.bin_width), n_bins - 1)
            counts[bin_idx] += 1

        bin_edges = [i * self.bin_width for i in range(n_bins)]
        return {
            "bin_edges": bin_edges,
            "counts_per_bin": counts,
            "labels": [f"{e:.0f}s" for e in bin_edges],
        }

    def unique_defects_over_time(self) -> Dict[str, Any]:
        if not self._events or self._start_ts is None:
            return {"bin_edges": [], "cumulative_unique": [], "labels": []}

        end = self._events[-1]["relative_ts"]
        duration = max(end, self.bin_width)
        n_bins = max(int(duration / self.bin_width) + 1, 1)

        seen_ids: set = set()
        new_per_bin = [0] * n_bins

        for evt in sorted(self._events, key=lambda e: e["relative_ts"]):
            tid = evt.get("track_id")
            if tid is not None and tid > 0 and tid not in seen_ids:
                seen_ids.add(tid)
                rel = evt["relative_ts"]
                bin_idx = min(int(rel / self.bin_width), n_bins - 1)
                new_per_bin[bin_idx] += 1

        cumulative = []
        running = 0
        for count in new_per_bin:
            running += count
            cumulative.append(running)

        bin_edges = [i * self.bin_width for i in range(n_bins)]
        return {
            "bin_edges": bin_edges,
            "cumulative_unique": cumulative,
            "labels": [f"{e:.0f}s" for e in bin_edges],
        }

    def get_summary(self) -> Dict[str, Any]:
        total = len(self._events)
        if total == 0:
            return {"total_events": 0}

        class_ctr: Dict[str, int] = defaultdict(int)
        for e in self._events:
            class_ctr[e["class_name"]] += 1

        duration = (
            self._events[-1]["relative_ts"] if self._events else 0.0
        )
        return {
            "total_events": total,
            "duration_seconds": round(duration, 2),
            "class_counts": dict(class_ctr),
            "avg_detections_per_minute": round(total / max(duration / 60.0, 1e-6), 2),
        }
