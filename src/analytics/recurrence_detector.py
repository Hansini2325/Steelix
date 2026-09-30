from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

log = get_logger(__name__)

DISPLAY_NAMES = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted Surface",
    "rolled_in_scale": "Rolled-in Scale",
    "scratches": "Scratches",
}


@dataclass
class RecurrencePattern:
    pattern_id: str
    pattern_type: str
    class_name: str
    display_name: str
    unique_track_ids: List[int] = field(default_factory=list)
    region: Optional[str] = None
    first_ts: float = 0.0
    latest_ts: float = 0.0
    triggered_ts: float = field(default_factory=time.time)

    @property
    def unique_occurrences(self) -> int:
        return len(self.unique_track_ids)

    @property
    def duration_seconds(self) -> float:
        return self.latest_ts - self.first_ts

    def avg_interval_seconds(self) -> Optional[float]:
        n = len(self.unique_track_ids)
        if n <= 1:
            return None
        return self.duration_seconds / (n - 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "class_name": self.class_name,
            "display_name": self.display_name,
            "unique_occurrences": self.unique_occurrences,
            "region": self.region,
            "first_ts": round(self.first_ts, 2),
            "latest_ts": round(self.latest_ts, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "avg_interval_seconds": (
                round(self.avg_interval_seconds(), 2)
                if self.avg_interval_seconds() is not None else None
            ),
            "triggered_ts": round(self.triggered_ts, 2),
        }


class RecurrenceDetector:

    def __init__(
        self,
        min_unique_occurrences: int = 3,
        time_window_seconds: float = 60.0,
        spatial_concentration_threshold: float = 0.60,
        alert_cooldown_seconds: float = 30.0,
    ) -> None:
        self.min_unique_occurrences = min_unique_occurrences
        self.time_window_seconds = time_window_seconds
        self.spatial_concentration_threshold = spatial_concentration_threshold
        self.alert_cooldown_seconds = alert_cooldown_seconds

        self._class_tracks: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)

        self._patterns: List[RecurrencePattern] = []
        self._pattern_counter: int = 0

        self._last_alert_ts: Dict[Tuple[str, Optional[str]], float] = {}

    def update_track(
        self,
        track_id: int,
        class_name: str,
        region: str,
        timestamp: float,
    ) -> None:
        if track_id <= 0:
            return

        tracks = self._class_tracks[class_name]
        if track_id not in tracks:
            tracks[track_id] = {
                "first_ts": timestamp,
                "latest_ts": timestamp,
                "region": region,
            }
        else:
            tracks[track_id]["latest_ts"] = timestamp

        tracks[track_id]["region"] = region

    def check_patterns(self, now: Optional[float] = None) -> List[RecurrencePattern]:
        if now is None:
            now = time.time()

        new_patterns: List[RecurrencePattern] = []

        for class_name, tracks in self._class_tracks.items():

            window_tracks = {
                tid: info for tid, info in tracks.items()
                if (now - info["first_ts"]) <= self.time_window_seconds
            }

            if len(window_tracks) < self.min_unique_occurrences:
                continue

            region_ctr: Counter = Counter(info["region"] for info in window_tracks.values())
            dominant_region, dominant_count = region_ctr.most_common(1)[0]
            concentration = dominant_count / len(window_tracks)

            ptype = (
                "spatial_concentration"
                if concentration >= self.spatial_concentration_threshold
                else "repeated_class"
            )

            alert_key = (class_name, dominant_region)
            last_alert = self._last_alert_ts.get(alert_key, 0.0)
            if (now - last_alert) < self.alert_cooldown_seconds:
                continue

            self._pattern_counter += 1
            pattern_id = f"PAT-{self._pattern_counter:04d}"

            first_ts = min(info["first_ts"] for info in window_tracks.values())
            latest_ts = max(info["latest_ts"] for info in window_tracks.values())
            display_name = DISPLAY_NAMES.get(class_name, class_name.replace("_", " ").title())

            pattern = RecurrencePattern(
                pattern_id=pattern_id,
                pattern_type=ptype,
                class_name=class_name,
                display_name=display_name,
                unique_track_ids=list(window_tracks.keys()),
                region=dominant_region,
                first_ts=first_ts,
                latest_ts=latest_ts,
                triggered_ts=now,
            )
            self._patterns.append(pattern)
            self._last_alert_ts[alert_key] = now
            new_patterns.append(pattern)

            log.info(
                f"Pattern detected: {ptype} — {display_name} x{len(window_tracks)} in {dominant_region}"
            )

        return new_patterns

    @property
    def patterns(self) -> List[RecurrencePattern]:
        return list(self._patterns)

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    def most_recurring_class(self) -> Optional[str]:
        if not self._patterns:
            return None
        ctr: Counter = Counter()
        for p in self._patterns:
            ctr[p.class_name] = max(ctr[p.class_name], p.unique_occurrences)
        return ctr.most_common(1)[0][0]

    def highest_recurrence_region(self) -> Optional[str]:
        if not self._patterns:
            return None
        region_ctr: Counter = Counter()
        for p in self._patterns:
            if p.region:
                region_ctr[p.region] += p.unique_occurrences
        if not region_ctr:
            return None
        return region_ctr.most_common(1)[0][0]

    def longest_recurring_sequence(self) -> int:
        if not self._patterns:
            return 0
        return max(p.unique_occurrences for p in self._patterns)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "pattern_count": self.pattern_count,
            "patterns": [p.to_dict() for p in self._patterns],
            "most_recurring_class": self.most_recurring_class(),
            "highest_recurrence_region": self.highest_recurrence_region(),
            "longest_recurring_sequence": self.longest_recurring_sequence(),
        }
